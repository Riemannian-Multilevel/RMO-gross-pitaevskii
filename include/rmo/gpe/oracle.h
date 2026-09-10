#ifndef RMO_GPE_ORACLE_H
#define RMO_GPE_ORACLE_H

#include <rmo/gpe/gpe.h>
#include <rmo/gpe/metric.h>

#include <rmo/ropt/oracle_base.h>

#include <deal.II/base/timer.h>

namespace rmo::gpe
{

namespace detail
{

/**
 * @brief Computes the Riemannian gradient for the Gross-Pitaevskii energy on the unit-mass manifold.
 *
 * This function calculates the gradient of the energy functional $E^{GP}(\phi)$
 * restricted to the sphere $S^{n-1}$ with an energy-adaptive metric $A_\phi$. The mathematical
 * formulation for the Riemannian gradient is:
 * $$ \grad_{A} E^{GP}(\phi) = \phi-\frac{1}{\phi^\top MA_\phi^{-1} M\phi} A_\phi^{-1}M\phi $$
 *
 * @tparam MatrixType A matrix-free operator or sparse matrix type providing a `vmult(dst, src)` method.
 * @tparam InverseMatrixType A solver wrapper or inverse operator type providing a `vmult(dst, src)` method.
 *
 * @param A_inv The inverse linear operator ($A_\phi^{-1}$).
 * @param M The mass matrix ($M$).
 * @param x The current state vector ($\phi$).
 * @param output The vector where the computed Riemannian gradient will be stored.
 */
template <typename MatrixType, typename InverseMatrixType>
void grad_energy_adaptive(const InverseMatrixType& A_inv, const MatrixType& M,
                          const Vector<double>& x, Vector<double>& output)
{
    // \Pi_x(x): R^n -> T_x S^{n-1}
    metric::energy::project_onto_tangent_space(A_inv, x, M, output);
}

/**
 * @brief Computes the Riemannian gradient for the Gross-Pitaevskii energy on the unit-mass manifold.
 *
 * This function calculates the gradient of the energy functional $E^{GP}(\phi)$
 * restricted to the sphere $S^{n-1}$ with a mass metric $M$. The mathematical
 * formulation for the Riemannian gradient is:
 * $$ \nabla_M E^{GP}(\phi) = M^{-1}\big(A_\phi\,\phi - (\phi^\top A_\phi\,\phi)M\phi\big) $$
 *
 * @tparam MatrixType A matrix-free operator or sparse matrix type providing a `vmult(dst, src)` method.
 * @tparam InverseMatrixType A solver wrapper or inverse operator type providing a `vmult(dst, src)` method.
 *
 * @param Minv The inverse mass operator ($M^{-1}$).
 * @param A The state-dependent total linear operator ($A_\phi$).
 * @param M The mass matrix ($M$).
 * @param x The current state vector ($\phi$).
 * @param output The vector where the computed Riemannian gradient will be stored.
 */
template <typename MatrixType, typename InverseMatrixType>
void grad_mass(const InverseMatrixType& Minv, const MatrixType& A, const MatrixType& M,
               const Vector<double>& x, Vector<double>& output)
{
    Vector<double> Ax(x.size());
    A.vmult(Ax, x);

    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    Ax.add(-(x * Ax), Mx);
    Minv.vmult(output, Ax);
}


/**
 * @brief Computes the Riemannian gradient in the F-metric.
 * \grad_{\rm F} E^{\rm GP}(\phi) = A_{\phi}\phi - \frac{\phi^\top M A_{\phi}\phi}{\phi^\top M^2 \phi} M \phi
*/
template <typename MatrixType>
void grad_frobenius(const MatrixType& A, const MatrixType& M,
                    const Vector<double>& x, Vector<double>& output)
{
    const unsigned int n_dofs = x.size();

    Vector<double> Ax(n_dofs);
    A.vmult(Ax, x);

    Vector<double> Mx(n_dofs);
    M.vmult(Mx, x);

    const double Mx_sq = Mx * Mx; // x^T M^2 x
    const double num = Mx * Ax; // x^T M A x

    output = Ax;
    output.add(-num / Mx_sq, Mx);
}

} // namespace detail


template <int dim>
class GrossPitaevskiiResidual
{
public:
    explicit GrossPitaevskiiResidual(const GrossPitaevskiiFunctional<dim>& m_func)
        : m_func(m_func)
          , m_norm(m_func.get_M())
    {
    }

    GrossPitaevskiiResidual(const GrossPitaevskiiFunctional<dim>& m_func, OperatorType op)
        : m_func(m_func)
          , m_norm(op)
    {
    }

    Vector<double> residual_vector(const Vector<double>& x) const
    {
        Vector<double> Mx(x.size());
        m_func.get_M().vmult(Mx, x);

        const double mass = x * Mx; // should be ~ 1 (energy constraint)
        //AssertThrow(std::abs(mass - 1) < 1e-12, dealii::ExcInternalError("mass constraint not fulfilled"));

        Vector<double> Ax(x.size()); // A x
        m_func.get_A().vmult(Ax, x);

        const double lambda = x * Ax / mass; // Rayleigh quotient (x'Ax / x'Mx)

        Vector<double> r(Ax);
        r.add(-lambda, Mx); // r = A x - lambda M x

        return r;
    }

    [[nodiscard]] double residual(const Vector<double>& x) const
    {
        return m_norm(residual_vector(x));
    }

private:
    const GrossPitaevskiiFunctional<dim>& m_func;

    SpdNorm<OperatorType> m_norm;
};


// Common methods for GP oracles (only distinction in used metric for Riemannian gradient)
template <int dim>
class GrossPitaevskiiOracle : public OracleBase
{
public:
    static constexpr int dimension = dim;
    static constexpr auto metric_t = MetricKind::NONE;

    // TODO: move options to base class constructor? (used for all but Frobenius -> no-op)
    explicit GrossPitaevskiiOracle(GrossPitaevskiiFunctional<dim>& func)
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

    // Residual evaluation
    [[nodiscard]] double residual(const Vector<double>& x) const override
    {
        return m_res.residual(x);
    }

    // Shared accessors
    const auto& get_M() const { return m_func.get_M(); }
    const auto& get_A() const { return m_func.get_A(); }
    const auto& get_A0() const { return m_func.get_A0(); }

    // Oracle getters must remain const so they can be called inside gradient(...) const
    InverseOpType& get_M_inv() const { return m_func.get_M_inv(); }
    InverseOpType& get_A_inv() const { return m_func.get_A_inv(); }

protected:
    GrossPitaevskiiFunctional<dim>& m_func;

    const GrossPitaevskiiResidual<dim> m_res;
};


template <int dim>
class MassOracle : public GrossPitaevskiiOracle<dim>
{
public:
    const char* id() const override { return "M"; }
    static constexpr auto metric_t = MetricKind::MASS;

    MassOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions options)
        : GrossPitaevskiiOracle<dim>(func)
          , options(options)
          , m_norm(this->get_M())
    {}

    /* @brief Computes the Riemannian gradient in the M-metric. */
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double x_residual = this->residual(x);
        Assert(x_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, x_residual);
        info.residual = x_residual;

        return info;
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};
        auto& M_inv = this->get_M_inv();

        if (residual > 0)
        {
            M_inv.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        detail::grad_mass(M_inv, this->get_A(), this->get_M(), x, output);

        info.num_iter = M_inv.control().last_step();
        info.tolerance = M_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

    [[nodiscard]] double norm(const Vector<double>& v) const override
    {
        return m_norm(v);
    }

    [[nodiscard]] double inner(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_norm(x, z);
    }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        this->get_M().vmult(dst, src);
    }

    MetricKind get_metric() const override { return metric_t; }

    SolverOptions get_options() const { return options; }

private:
    SolverOptions options;

    SpdNorm<OperatorType> m_norm;
};


template <int dim>
class EnergyOracle : public GrossPitaevskiiOracle<dim>
{
public:
    const char* id() const override { return "A"; }
    static constexpr auto metric_t = MetricKind::ENERGY_ADAPTIVE;

    EnergyOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions options)
        : GrossPitaevskiiOracle<dim>(func)
          , options(options)
          , m_norm(this->get_A())
    {}

    /**
     * @brief Computes the Riemannian gradient in the A-metric.
     * Solves the inner linear system $ A^{-1} \nabla E $ using the PreconditionInverse wrapper.
     */
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double x_residual = this->residual(x);
        Assert(x_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, x_residual);
        info.residual = x_residual;

        return info;
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};
        auto& A_inv = this->get_A_inv();

        if (residual > 0)
        {
            A_inv.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        detail::grad_energy_adaptive(A_inv, this->get_M(), x, output);

        info.num_iter = A_inv.control().last_step();
        info.tolerance = A_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

    [[nodiscard]] double norm(const Vector<double>& x) const override
    {
        return m_norm(x);
    }

    [[nodiscard]] double inner(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_norm(x, z);
    }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        this->get_A().vmult(dst, src);
    }

    MetricKind get_metric() const override { return metric_t; }

    SolverOptions get_options() const { return options; }

private:
    SolverOptions options;

    SpdNorm<OperatorType> m_norm;
};


template <int dim>
class FrobeniusOracle : public GrossPitaevskiiOracle<dim>
{
public:
    const char* id() const override { return "F"; }
    static constexpr auto metric_t = MetricKind::FROBENIUS;

    FrobeniusOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions)
        : GrossPitaevskiiOracle<dim>(func)
    {}

    /**
     * @brief Computes the Riemannian gradient in the F-metric.
     * \grad_{\rm F} E^{\rm GP}(\phi) = A_{\phi}\phi - \frac{\phi^\top M A_{\phi}\phi}{\phi^\top M^2 \phi} M \phi
     */
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        timer.start();
        detail::grad_frobenius(this->get_A(), this->get_M(), x, output);
        timer.stop();

        // F-gradient evaluation does not involve a linear solver.
        info.elapsed_time = timer.cpu_time();
        return info;
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double) const override
    {
        return gradient(x, output); // no-op
    }

    [[nodiscard]] double norm(const Vector<double>& v) const override
    {
        return std::sqrt(v * v);
    }

    [[nodiscard]] double inner(const Vector<double>& x, const Vector<double>& z) const override
    {
        return x * z;
    }

    MetricKind get_metric() const override { return metric_t; }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        dst = src;
    }
};

} // namespace rmo::gpe

#endif //RMO_GPE_ORACLE_H
