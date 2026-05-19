#ifndef GPE_ORACLE_H
#define GPE_ORACLE_H

#include <gpe/lac.h>
#include <gpe/problem/gpe.h>
#include <gpe/ropt/manifold.h>
#include <gpe/ropt/transport.h>

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


// Basic oracle interface
// TODO: include residual(x)?
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
    virtual unsigned gradient(const Vector<double>&, Vector<double>&) const = 0;  // Riemannian gradient - metric-dependent

    virtual double norm(const Vector<double>&) const = 0;  // for (coarse) condition evaluation - metric-dependent
    virtual unsigned n_dofs() const = 0;
};


class Residual
{
public:
    virtual ~Residual() = default;

    double residual(const Vector<double>& x) const
    {
        // If the cache is empty, compute and store it
        if (m_residual < 0) {
            const auto r = residual_vector(x);
            m_residual = norm(r);
        }
        AssertThrow(m_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        // Otherwise, just return the cached value
        return m_residual;
    }

    void reinit() const { m_residual = -1.0; }  // invalidate cached value

protected:
    mutable double m_residual = -1.0;

    virtual double norm(const Vector<double>& x) const = 0;
    virtual Vector<double> residual_vector(const Vector<double>& x) const = 0;
};


template <int dim>
class GrossPitaevskiiResidual : public Residual
{
public:
    GrossPitaevskiiResidual(const GrossPitaevskiiFunctional<dim>& m_func)
        : m_func(m_func)
    {}

protected:
    double norm(const Vector<double>& x) const override
    {
        EnergyNorm norm(m_func.get_M());

        return norm(x);
    }

    Vector<double> residual_vector(const Vector<double>& x) const override
    {
        Vector<double> Mx(x.size());
        m_func.get_M().vmult(Mx, x);

        const double mass = x * Mx;             // should be ~ 1 (energy constraint)
        AssertThrow(std::abs(mass - 1) < 1e-12, dealii::ExcInternalError("mass constraint not fulfilled"));

        Vector<double> Ax(x.size());         // A x
        m_func.get_A().vmult(Ax, x);

        const double lambda = x * Ax / mass;    // Rayleigh quotient (x'Ax / x'Mx)

        Vector<double> r(Ax);
        r.add(-lambda, Mx);                     // r = A x - lambda M x

        return r;
    }

private:
    const GrossPitaevskiiFunctional<dim>& m_func;
};


// Metric-aware wrapper for GrossPitaevskii energy evaluation
template <int dim>
class GrossPitaevskiiOracle : public OracleBase
{
public:
    static constexpr int dimension = dim;

    GrossPitaevskiiOracle(GrossPitaevskiiFunctional<dim>& func)
        : m_func(func)
        , m_res_func(func)
    {}

    virtual ~GrossPitaevskiiOracle() = default;

    // Common lifecycle management
    void update(const Vector<double>& x) override
    {
        m_func.update(x);
        m_res_func.reinit();
    }

    // Common metric-independent GP evaluation
    double value(const Vector<double>& x) const override
    {
        return m_func.value(x);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_func.directional_derivative(x, z);
    }

    double residual(const Vector<double>& x) const
    {
        return m_res_func.residual(x);
    }

    unsigned n_dofs() const override
    {
        return m_func.n_dofs();
    }

    // Methods to be defined in child classes (metric-dependent)
    double norm(const Vector<double>&) const override
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    unsigned gradient(const Vector<double>&, Vector<double>&) const override
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    // Shared domain-specific accessors
    const auto& get_M()  const { return m_func.get_M(); }
    const auto& get_A()  const { return m_func.get_A(); }
    const auto& get_A0() const { return m_func.get_A0(); }

protected:
    GrossPitaevskiiFunctional<dim>& m_func;
    const GrossPitaevskiiResidual<dim> m_res_func;
};


template <int dim>
class MassOracle : public GrossPitaevskiiOracle<dim>
{
public:
    static constexpr const char* id = "M";

    MassOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions options)
        : GrossPitaevskiiOracle<dim>(func)
        , options(options)
        , M_inv(this->get_M(), options)
    {}

    void update(const Vector<double>& x) override
    {
        GrossPitaevskiiOracle<dim>::update(x);
    }

    /* @brief Computes the Riemannian gradient in the M-metric. */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        const double x_residual = this->residual(x);

        if (x_residual > 0) {
            M_inv.set_tol(x_residual * options.tol_inner_res);
        }
        ellipsoid::mass::gradient(M_inv, this->get_A(), this->get_M(), x, output);

        return this->M_inv.control().last_step();
    }

    double norm(const Vector<double>& v) const override
    {
        EnergyNorm norm(this->get_M());

        return norm(v);
    }

private:
    SolverOptions options;
    InverseOpType M_inv;
};


template <int dim>
class EnergyOracle : public GrossPitaevskiiOracle<dim>
{
public:
    static constexpr const char* id = "A";
    static constexpr int dimension = dim;

    EnergyOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions options)
        : GrossPitaevskiiOracle<dim>(func)
        , options(options)
        , A_inv(this->get_A(), options)
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
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        const double x_residual = this->residual(x);

        if (x_residual > 0) {
            A_inv.set_tol(x_residual * this->options.tol_inner_res);
        }
        ellipsoid::energy::gradient(A_inv, this->get_M(), x, output);

        return A_inv.control().last_step();
    }

    double norm(const Vector<double>& x) const override
    {
        EnergyNorm norm(this->get_A());

        return norm(x);
    }

private:
    SolverOptions options;
    InverseOpType A_inv;
};


template <int dim>
class FrobeniusOracle : public GrossPitaevskiiOracle<dim>
{
public:
    static constexpr const char* id = "F";

    FrobeniusOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions)
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
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        ellipsoid::frobenius::gradient(this->get_A(), this->get_M(), x, output);

        // F-gradient evaluation does not involve a linear solver.
        return 0;
    }

    double norm(const Vector<double>& v) const override
    {
        return std::sqrt(v*v);
    }
};

} // namespace gpe

#endif //GPE_ORACLE_H