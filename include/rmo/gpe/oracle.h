#ifndef RMO_GPE_ORACLE_H
#define RMO_GPE_ORACLE_H

#include <rmo/gpe/gpe.h>
#include <rmo/gpe/kernels.h>
#include <rmo/gpe/metric.h>
#include <rmo/gpe/residual.h>

#include <rmo/ropt/oracle_base.h>

#include <deal.II/base/timer.h>

namespace rmo::gpe
{

// Common methods for GP oracles (only distinction in used metric for Riemannian gradient)
// System: GrossPitaevskiiSystem or GrossPitaevskiiLumpedSystem
template <typename System>
class GrossPitaevskiiOracle : public OracleBase
{
public:
    static constexpr int dimension = System::dimension;
    static constexpr auto metric_t = MetricKind::NONE;

    using Functional = GrossPitaevskiiFunctional<System>;
    using Operator   = typename Functional::Operator;
    using InverseM   = typename Functional::InverseM;
    using InverseA   = typename Functional::InverseA;

    // TODO: move options to base class constructor? (used for all but Frobenius -> no-op)
    explicit GrossPitaevskiiOracle(Functional& func)
        : m_func(func)
          , m_res(func)
    {}

    ~GrossPitaevskiiOracle() override = default;

    // Function evaluation
    void update(const Vector<double>& x) override
    {
        m_func.update(x);
    }

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        return m_func.value(x);
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_func.directional_derivative(x, z);
    }

    [[nodiscard]] unsigned n_dofs() const override
    {
        return m_func.n_dofs();
    }

    // Shared accessors
    const Functional& get_functional() const { return m_func; }
    const GrossPitaevskiiResidual<System>& get_residual() const { return m_res; }
    const Operator& get_M() const { return m_func.get_M(); }
    const Operator& get_A() const { return m_func.get_A(); }
    const SparseMatrix<double>& get_A0() const { return m_func.get_A0(); }

    // Oracle getters must remain const so they can be called inside gradient(...) const
    InverseM& get_M_inv() const { return m_func.get_M_inv(); }
    InverseA& get_A_inv() const { return m_func.get_A_inv(); }

protected:
    Functional& m_func;

    const GrossPitaevskiiResidual<System> m_res;
};


template <typename System>
class MassOracle : public GrossPitaevskiiOracle<System>
{
public:
    using Base       = GrossPitaevskiiOracle<System>;
    using Functional = typename Base::Functional;
    using Operator   = typename Base::Operator;
    using InverseM   = typename Base::InverseM;

    const char* id() const override { return "M"; }
    static constexpr auto metric_t = MetricKind::MASS;

    MassOracle(Functional& func, SolverOptions options)
        : Base(func)
          , options(options)
          , m_metric(func.get_M(), metric_t)
    {}

    /* @brief Computes the Riemannian gradient in the M-metric. */
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double x_residual = this->m_res.residual(x);
        Assert(x_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, x_residual);
        info.residual = x_residual;

        return info;
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};
        InverseM& M_inv = this->get_M_inv();

        if (residual > 0)
        {
            M_inv.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        kernels::grad_mass(M_inv, this->get_A(), this->get_M(), x, output);

        info.num_iter = M_inv.control().last_step();
        info.tolerance = M_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

    [[nodiscard]] const MetricBase& metric() const override { return m_metric; }

    SolverOptions get_options() const { return options; }

private:
    SolverOptions options;

    OperatorMetric<Operator> m_metric;
};


template <typename System>
class EnergyOracle : public GrossPitaevskiiOracle<System>
{
public:
    using Base       = GrossPitaevskiiOracle<System>;
    using Functional = typename Base::Functional;
    using Operator   = typename Base::Operator;
    using InverseA   = typename Base::InverseA;

    const char* id() const override { return "A"; }
    static constexpr auto metric_t = MetricKind::ENERGY_ADAPTIVE;

    EnergyOracle(Functional& func, SolverOptions options)
        : Base(func)
          , options(options)
          , m_metric(func.get_A(), metric_t)
    {}

    /**
     * @brief Computes the Riemannian gradient in the A-metric.
     * Solves the inner linear system \f$ A^{-1} \nabla E \f$ using the PreconditionInverse wrapper.
     */
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double x_residual = this->m_res.residual(x);
        Assert(x_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, x_residual);
        info.residual = x_residual;

        return info;
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};
        InverseA& A_inv = this->get_A_inv();

        if (residual > 0)
        {
            A_inv.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        kernels::grad_energy_adaptive(A_inv, this->get_M(), x, output);

        info.num_iter = A_inv.control().last_step();
        info.tolerance = A_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

    [[nodiscard]] const MetricBase& metric() const override { return m_metric; }

    SolverOptions get_options() const { return options; }

private:
    SolverOptions options;

    OperatorMetric<Operator> m_metric;
};


template <typename System>
class FrobeniusOracle : public GrossPitaevskiiOracle<System>
{
public:
    using Base       = GrossPitaevskiiOracle<System>;
    using Functional = typename Base::Functional;

    const char* id() const override { return "F"; }
    static constexpr auto metric_t = MetricKind::FROBENIUS;

    FrobeniusOracle(Functional& func, SolverOptions)
        : Base(func)
    {}

    /**
     * @brief Computes the Riemannian gradient in the F-metric.
     * \f[ \grad_{\rm F} E^{\rm GP}(\phi) = A_{\phi}\phi - \frac{\phi^\top M A_{\phi}\phi}{\phi^\top M^2 \phi} M \phi \f]
     */
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        timer.start();
        kernels::grad_frobenius(this->get_A(), this->get_M(), x, output);
        timer.stop();

        // F-gradient evaluation does not involve a linear solver.
        info.elapsed_time = timer.cpu_time();
        return info;
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double) const override
    {
        return gradient(x, output); // no-op
    }

    [[nodiscard]] const MetricBase& metric() const override { return m_metric; }

private:
    EuclideanMetric m_metric;
};

} // namespace rmo::gpe

#endif //RMO_GPE_ORACLE_H
