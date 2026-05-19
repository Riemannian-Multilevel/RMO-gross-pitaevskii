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


template <int dim>
class MassOracle : public OracleBase
{
public:
    static constexpr const char* id = "M";
    static constexpr int dimension = dim;

    MassOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions options)
        : m_func(func)
        , options(options)
        , M(m_func.get_M())
        , A(m_func.get_A())
        , M_inv(m_func.get_M(), options)
    {}

    void update(const Vector<double>& x) override
    {
        m_func.update(x);
        // Invalidate the residual when the evaluation point changes
        m_func_res = -1.0;
    }

    double value(const Vector<double>& x) const override
    {
        return m_func.value(x);  // metric-independent
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_func.directional_derivative(x, z);  // metric-independent
    }

    // Lazy evaluation for residual
    double residual(const Vector<double>& x) const
    {
        // The residual is evaluated in the M-norm regardless of the metric chosen for the Riemannian gradient
        const auto norm = EnergyNorm(M);

        // If the cache is empty, compute and store it
        if (m_func_res < 0) {
            m_func_res = norm(m_func.residual(x));
        }
        AssertThrow(m_func_res >= 0, "residual must be positive");

        // Otherwise, just return the cached value
        return m_func_res;
    }

    /* @brief Computes the Riemannian gradient in the M-metric. */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        m_func_res = this->residual(x);

        if (m_func_res > 0) {
            M_inv.set_tol(m_func_res * options.tol_inner_res);
        }
        ellipsoid::mass::gradient(M_inv, A, M, x, output);

        return this->M_inv.control().last_step();
    }

    double norm(const Vector<double>& v) const override
    {
        Vector<double> Mv(m_func.n_dofs());
        M.vmult(Mv, v);

        return std::sqrt(v*Mv);
    }

    unsigned n_dofs() const override
    {
        return m_func.n_dofs();
    }

    // Domain-specific methods NOT in OracleBase
    // Called in <Metric>CoarseOracle, which encodes the used fine/coarse oracle as a template parameter
    const auto& get_M()  const
    {
        return m_func.get_M();
    }
    const auto& get_A()  const
    {
        return m_func.get_A();
    }
    const auto& get_A0() const
    {
        return m_func.get_A0();
    }

private:
    GrossPitaevskiiFunctional<dim>& m_func;
    SolverOptions options;

    const OperatorType &M, &A;  // operators owned by GrossPitaevskiiFunctional
    InverseOpType M_inv;

    mutable double m_func_res = -1.0;
};


template <int dim>
class EnergyOracle : public OracleBase
{
public:
    static constexpr const char* id = "A";
    static constexpr int dimension = dim;

    EnergyOracle(GrossPitaevskiiFunctional<dim>& func, SolverOptions options)
        : m_func(func)
        , options(options)
        , M(m_func.get_M())
        , A(m_func.get_A())
        , M_inv(m_func.get_M(), options)
        , A_inv(m_func.get_A(), options)
    {
        A_inv.update_static(m_func.get_A0());
    }

    void update(const Vector<double>& x) override
    {
        m_func.update(x);
        m_func_res = -1.0;

        A_inv.update_dynamic(A.diagonal());
    }

    double value(const Vector<double>& x) const override
    {
        return m_func.value(x);
    }

    // Metric-free implementation
    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_func.directional_derivative(x, z);
    }

    // Lazy evaluation for residual
    double residual(const Vector<double>& x) const
    {
        // The residual is evaluated in the M-norm regardless of the metric chosen for the Riemannian gradient
        const auto norm = EnergyNorm(M);

        // If the cache is empty, compute and store it
        if (m_func_res < 0) {
            m_func_res = norm(m_func.residual(x));
        }
        AssertThrow(m_func_res >= 0, "residual must be positive");

        // Otherwise, just return the cached value
        return m_func_res;
    }

    /**
     * @brief Computes the Riemannian gradient in the A-metric.
     * Solves the inner linear system $ A^{-1} \nabla E $ using the PreconditionInverse wrapper.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        m_func_res = this->residual(x);

        if (m_func_res > 0) {
            A_inv.set_tol(m_func_res * this->options.tol_inner_res);
        }
        ellipsoid::energy::gradient(A_inv, M, x, output);

        return A_inv.control().last_step();
    }

    double norm(const Vector<double>& v) const override
    {
        Vector<double> Av(m_func.n_dofs());
        A.vmult(Av, v);

        return std::sqrt(v*Av);
    }

    unsigned n_dofs() const override
    {
        return m_func.n_dofs();
    }

    // Domain-specific methods NOT in OracleBase
    // Called in <Metric>CoarseOracle, which encodes the used fine/coarse oracle as a template parameter
    const auto& get_M()  const
    {
        return m_func.get_M();
    }
    const auto& get_A()  const
    {
        return m_func.get_A();
    }
    const auto& get_A0() const
    {
        return m_func.get_A0();
    }

private:
    GrossPitaevskiiFunctional<dim>& m_func;
    SolverOptions options;

    const OperatorType &M, &A;  // operators owned by GrossPitaevskiiFunctional
    InverseOpType M_inv, A_inv;

    mutable double m_func_res = -1.0;
};


template <int dim>
class FrobeniusOracle : public OracleBase
{
public:
    static constexpr const char* id = "F";
    static constexpr int dimension = dim;

    FrobeniusOracle(GrossPitaevskiiFunctional<dim>& func)
            : m_func(func)
            , M(m_func.get_M())
            , A(m_func.get_A())
    {}

    void update(const Vector<double>& x) override
    {
        m_func.update(x);
        m_func_res = -1.0;
    }

    double value(const Vector<double>& x) const override
    {
        return m_func.value(x);  // metric-independent
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_func.directional_derivative(x, z);  // metric-independent
    }

    // Lazy evaluation for residual
    double residual(const Vector<double>& x) const
    {
        // The residual is evaluated in the M-norm regardless of the metric chosen for the Riemannian gradient
        const auto norm = EnergyNorm(M);

        // If the cache is empty, compute and store it
        if (m_func_res < 0) {
            m_func_res = norm(m_func.residual(x));
        }
        AssertThrow(m_func_res >= 0, "residual must be positive");

        // Otherwise, just return the cached value
        return m_func_res;
    }

    /**
     * @brief Computes the Riemannian gradient in the F-metric.
     * \grad_{\rm F} E^{\rm GP}(\phi) = A_{\phi}\phi - \frac{\phi^\top M A_{\phi}\phi}{\phi^\top M^2 \phi} M \phi
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: strictly not required for Frobenius gradient (no matrix inversion with adjutable tolerance.)
        //       compute here anyway for consistency with EnergyOracle/MassOracle.
        m_func_res = this->residual(x);

        ellipsoid::frobenius::gradient(this->A, this->M, x, output);

        // F-gradient evaluation does not involve a linear solver.
        return 0;
    }

    double norm(const Vector<double>& v) const override
    {
        return std::sqrt(v*v);
    }

    unsigned n_dofs() const override
    {
        return m_func.n_dofs();
    }

    // Domain-specific methods NOT in OracleBase
    // Called in <Metric>CoarseOracle, which encodes the used fine/coarse oracle as a template parameter
    const auto& get_M()  const
    {
        return m_func.get_M();
    }
    const auto& get_A()  const
    {
        return m_func.get_A();
    }
    const auto& get_A0() const
    {
        return m_func.get_A0();
    }

private:
    GrossPitaevskiiFunctional<dim>& m_func;
    const OperatorType &M, &A;  // operators owned by GrossPitaevskiiFunctional

    mutable double m_func_res = -1.0;
};

} // namespace gpe

#endif //GPE_ORACLE_H