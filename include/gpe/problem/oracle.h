#ifndef GPE_ORACLE_H
#define GPE_ORACLE_H

#include <gpe/lac.h>
#include <gpe/problem/gpe.h>
#include <gpe/ropt/manifold.h>
#include <gpe/ropt/transport.h>

#include <deal.II/base/timer.h>

namespace gpe
{
/**
 * @brief Functor computing the square of the Euclidean norm of a point.
 *
 * Computes \f$ f(p) = \sum_{d=0}^{dim-1} p_d^2 \f$.
 * Used primarily for initializing test cases or potentials.
 *
 * @tparam dim The spatial dimension of the point.
 */
template <int dim>
class Square
{
public:
    double operator()(const Point<dim>& p) const {
        typename Point<dim>::value_type out = 0.0;

        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    }
};


// Fields for gradient computation with inner solver
// TODO: mvoe to option_types.h?
struct GradInfo
{
    double residual;
    unsigned num_iter;
    double tolerance;
    double elapsed_time;
};


template <int dim>
class GrossPitaevskiiResidual
{
public:
    GrossPitaevskiiResidual(const GrossPitaevskiiFunctional<dim>& m_func)
        : m_func(m_func)
        , m_norm(m_func.get_M())
    {}

    GrossPitaevskiiResidual(const GrossPitaevskiiFunctional<dim>& m_func, OperatorType op)
        : m_func(m_func)
        , m_norm(op)
    {}

    Vector<double> residual_vector(const Vector<double>& x) const
    {
        Vector<double> Mx(x.size());
        m_func.get_M().vmult(Mx, x);

        const double mass = x * Mx;             // should be ~ 1 (energy constraint)
        //AssertThrow(std::abs(mass - 1) < 1e-12, dealii::ExcInternalError("mass constraint not fulfilled"));

        Vector<double> Ax(x.size());         // A x
        m_func.get_A().vmult(Ax, x);

        const double lambda = x * Ax / mass;    // Rayleigh quotient (x'Ax / x'Mx)

        Vector<double> r(Ax);
        r.add(-lambda, Mx);                     // r = A x - lambda M x

        return r;
    }

    double residual(const Vector<double>& x) const
    {
        return m_norm(residual_vector(x));
    }

private:
    const GrossPitaevskiiFunctional<dim>& m_func;

    EnergyNorm<OperatorType> m_norm;
};


// Basic oracle interface
class OracleBase
{
public:
    virtual ~OracleBase() = default;

    virtual void update(const Vector<double>& x) = 0;

    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_assembly`
    virtual double value(const Vector<double>&) const = 0;

    // TODO: Compute directional derivative and Riemannian gradient successively
    virtual double directional_derivative(const Vector<double>&, const Vector<double>&) const = 0;

    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_gradient
    virtual GradInfo gradient(const Vector<double>&, Vector<double>&) const = 0;  // Riemannian gradient - metric-dependent

    // TODO: move this to a separate interface? (-> class Metric)
    virtual double norm(const Vector<double>&) const = 0;  // for (coarse) condition evaluation - metric-dependent
    virtual double metric(const Vector<double>&, const Vector<double>&) const = 0;
    // TODO: improve name
    virtual void apply_metric(const Vector<double>&, Vector<double>&) const = 0;
    virtual unsigned n_dofs() const = 0;

    // TODO: move this to a separate interface? (-> class Residual or GrossPitaevskiiFunctional)
    virtual double residual(const Vector<double>&) const = 0;
};


// Common methods for GP oracles (only distinction in used metric for Riemannian gradient)
template <int dim>
class GrossPitaevskiiOracle : public OracleBase
{
public:
    static constexpr int dimension = dim;
    static constexpr auto metric_t = MetricKind::NONE;

    GrossPitaevskiiOracle(GrossPitaevskiiFunctional<dim>& func)
        : m_func(func)
        , m_res(func)
    {}

    virtual ~GrossPitaevskiiOracle() = default;

    // Function evaluation
    void update(const Vector<double>& x) override
    {
        m_func.update(x);
    }

    double value(const Vector<double>& x) const override
    {
        return m_func.value(x);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_func.directional_derivative(x, z);
    }

    unsigned n_dofs() const override
    {
        return m_func.n_dofs();
    }

    // Residual evaluation
    double residual(const Vector<double>& x) const override
    {
        return m_res.residual(x);
    }

    // Shared accessors
    const auto& get_M()  const { return m_func.get_M(); }
    const auto& get_A()  const { return m_func.get_A(); }
    const auto& get_A0() const { return m_func.get_A0(); }

protected:
    GrossPitaevskiiFunctional<dim>& m_func;

    const GrossPitaevskiiResidual<dim> m_res;
};


template <int dim>
class MassOracle : public GrossPitaevskiiOracle<dim>
{
public:
    static constexpr const char* id = "M";
    static constexpr auto metric_t = MetricKind::MASS;

    MassOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions options)
        : GrossPitaevskiiOracle<dim>(func)
        , options(options)
        , M_inv(this->get_M(), options)
        , m_norm(this->get_M())
    {}

    void update(const Vector<double>& x) override
    {
        GrossPitaevskiiOracle<dim>::update(x);
    }

    /* @brief Computes the Riemannian gradient in the M-metric. */
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double x_residual = this->residual(x);
        Assert(x_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto inv_tol = x_residual * options.tol_inner_res;
        auto info = gradient(x, output, inv_tol);
        info.residual = x_residual;

        return info;
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double inv_tol) const
    {
        dealii::Timer timer;
        GradInfo info{};

        if (inv_tol > 0) {
            M_inv.set_tol(inv_tol);
        }

        timer.start();
        ellipsoid::mass::gradient(M_inv, this->get_A(), this->get_M(), x, output);

        info.num_iter     = M_inv.control().last_step();
        info.tolerance    = M_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

    double norm(const Vector<double>& v) const override
    {
        return m_norm(v);
    }

    double metric(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_norm(x, z);
    }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        this->get_M().vmult(dst, src);
    }

private:
    SolverOptions options;
    InverseOpType M_inv;

    EnergyNorm<OperatorType> m_norm;
};


template <int dim>
class EnergyOracle : public GrossPitaevskiiOracle<dim>
{
public:
    static constexpr const char* id = "A";
    static constexpr auto metric_t = MetricKind::ENERGY_ADAPTIVE;

    EnergyOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions options)
        : GrossPitaevskiiOracle<dim>(func)
        , options(options)
        , A_inv(this->get_A(), options)
        , m_norm(this->get_A())
    {
        A_inv.update_static(this->get_A0());
    }

    void update(const Vector<double>& x) override
    {
        GrossPitaevskiiOracle<dim>::update(x);

        A_inv.update_dynamic(this->get_A().diagonal());
    }

    /**
     * @brief Computes the Riemannian gradient in the A-metric.
     * Solves the inner linear system $ A^{-1} \nabla E $ using the PreconditionInverse wrapper.
     */
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double x_residual = this->residual(x);
        Assert(x_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto inv_tol = x_residual * options.tol_inner_res;
        auto info = gradient(x, output, inv_tol);
        info.residual = x_residual;

        return info;
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double inv_tol) const
    {
        dealii::Timer timer;
        GradInfo info{};

        if (inv_tol > 0) {
            A_inv.set_tol(inv_tol);
        }

        timer.start();
        ellipsoid::energy::gradient(A_inv, this->get_M(), x, output);

        info.num_iter     = A_inv.control().last_step();
        info.tolerance    = A_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

    double norm(const Vector<double>& x) const override
    {
        return m_norm(x);
    }

    double metric(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_norm(x, z);
    }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        this->get_A().vmult(dst, src);
    }

private:
    SolverOptions options;
    InverseOpType A_inv;

    EnergyNorm<OperatorType> m_norm;
};


template <int dim>
class FrobeniusOracle : public GrossPitaevskiiOracle<dim>
{
public:
    static constexpr const char* id = "F";
    static constexpr auto metric_t = MetricKind::FROBENIUS;

    FrobeniusOracle(GrossPitaevskiiFunctional<dim>& func)
        : GrossPitaevskiiOracle<dim>(func)
    {}

    void update(const Vector<double>& x) override
    {
        GrossPitaevskiiOracle<dim>::update(x);
    }

    /**
     * @brief Computes the Riemannian gradient in the F-metric.
     * \grad_{\rm F} E^{\rm GP}(\phi) = A_{\phi}\phi - \frac{\phi^\top M A_{\phi}\phi}{\phi^\top M^2 \phi} M \phi
     */
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        timer.start();
        ellipsoid::frobenius::gradient(this->get_A(), this->get_M(), x, output);
        timer.stop();

        // F-gradient evaluation does not involve a linear solver.
        info.elapsed_time = timer.cpu_time();
        return info;
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double inv_tol) const
    {
        return gradient(x, output);  // no-op
    }

    double norm(const Vector<double>& v) const override
    {
        return std::sqrt(v*v);
    }

    double metric(const Vector<double>& x, const Vector<double>& z) const override
    {
        return x*z;
    }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        dst = src;
    }
};

} // namespace gpe

#endif //GPE_ORACLE_H