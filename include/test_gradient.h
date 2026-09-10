//
// Created by Ferdinand Vanmaele on 04.04.26.
//
#ifndef RMO_TEST_GRADIENT_H
#define RMO_TEST_GRADIENT_H

#include <rmo/gpe/gpe.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/oracle_coarse.h>

#include <rmo/util/random.h>

#include <boost/math/special_functions/math_fwd.hpp>


namespace rmo::gpe
{
// TODO: check first order-coherence

// Contains all metric-independent components for testing gradients on the ellipsoid
template <int dim>
class GradientTestBase
{
public:
    const double mean = 0.0;
    const double stddev = 1.0;

    using MatrixType    = SparseMatrix<double>;
    using OperatorType  = LinearCombination<MatrixType, Vector<double>>;
    using InverseOpType = PreconditionInverse<OperatorType, MatrixType>;

    GradientTestBase(GrossPitaevskiiSystem<dim>& system, double beta, SolverOptions options)
        : m_system(system)
        , m_eval(system, beta, options)
        , m_beta(beta)
    {}
    virtual ~GradientTestBase() = default;

    void assemble(const Vector<double>& x)
    {
        m_system.assemble_nonlinear_term(x);  // updates Mpp -> A (mutable) for underlying operators
    }

    // Defined for all functions and metrics on S^n
    void retract(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_retr) const
    {
        v_retr = x;
        ellipsoid::retract_by_norm(get_M(), v, v_retr);  // input-output vector
    }

    // Special case for x == v
    void retract(const Vector<double>& x, Vector<double>& x_retr) const
    {
        x_retr = x;
        ellipsoid::retract_by_norm(get_M(), x_retr);
    }

    //! @brief Applies the linear constraints to @p z.
    //! The assembled operators represent the energy only on admissible vectors, so test points
    //! and tangent vectors must satisfy the constraints.
    void distribute(Vector<double>& z) const
    {
        m_system.get_constraints().distribute(z);
    }

    //! @brief Constrains a base point and puts it back on the manifold.
    void make_admissible(Vector<double>& x) const
    {
        distribute(x);
        ellipsoid::retract_by_norm(get_M(), x);
    }

    [[nodiscard]] double constraint_value(const Vector<double>& x) const
    {
        Vector<double> Mx(x.size());
        get_M().vmult(Mx, x);

        return x*Mx;
    }

    const auto& get_A() const { return m_eval.get_A(); }
    const auto& get_M() const { return m_eval.get_M(); }
    const auto& get_A_inv() const { return m_eval.get_A_inv(); }
    const auto& get_M_inv() const { return m_eval.get_M_inv(); }
    [[nodiscard]] unsigned n_dofs() const { return m_eval.n_dofs(); }

    virtual void random_point(Vector<double>& x) const = 0;  // virtual for problems with parameters (i.e. coarse model, w, phi)
    virtual void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const = 0;
    virtual void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const = 0;

    [[nodiscard]] virtual double value(const Vector<double>&) const = 0;
    [[nodiscard]] virtual double directional_derivative(const Vector<double>& x, const Vector<double>& z) const = 0;
    [[nodiscard]] virtual Vector<double> gradient(const Vector<double>&) const = 0;
    [[nodiscard]] virtual double inner(const Vector<double>&, const Vector<double>&) const = 0;


protected:
    GrossPitaevskiiSystem<dim> &m_system;
    GrossPitaevskiiFunctional<dim> m_eval;
    double m_beta;
};


template <int dim>
class GradientTest : public GradientTestBase<dim>
{
public:
    GradientTest(GrossPitaevskiiSystem<dim>& system, double beta, SolverOptions options)
        : GradientTestBase<dim>(system, beta, options)
    {}

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        return this->m_eval.value(x);
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return this->m_eval.directional_derivative(x, z);
    }

    void random_point(Vector<double>& x) const override
    {
        ellipsoid::random_point(x, this->get_M());
    }
};


template <int dim>
class GradientTestEnergy : public GradientTest<dim>
{
public:
    using GradientTest<dim>::GradientTest;

    [[nodiscard]] Vector<double> gradient(const Vector<double>& x) const final
    {
        Vector<double> x_grad(x.size());
        detail::grad_energy_adaptive(this->get_A_inv(), this->get_M(), x, x_grad);

        return x_grad;
    }

    [[nodiscard]] double inner(const Vector<double>& y, const Vector<double>& z) const final
    {
        AssertDimension(y.size(), z.size());
        Vector<double> Az(z.size());
        this->get_A().vmult(Az, z);

        return y*Az;
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const final
    {
        metric::energy::project_onto_tangent_space(this->get_A_inv(), x, this->get_M(), v, v_proj);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const final
    {
        metric::energy::random_tangent_vector(this->get_A_inv(), x, this->get_M(), v);
    }
};


template <int dim>
class GradientTestMass : public GradientTest<dim>
{
public:
    using GradientTest<dim>::GradientTest;

    [[nodiscard]] Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> x_grad(x.size());
        detail::grad_mass(this->get_M_inv(), this->get_A(), this->get_M(), x, x_grad);

        return x_grad;
    }

    [[nodiscard]] double inner(const Vector<double>& y, const Vector<double>& z) const override
    {
        AssertDimension(y.size(), z.size());
        Vector<double> Mz(z.size());
        this->get_M().vmult(Mz, z);

        return y*Mz;
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const override
    {
        metric::mass::project_onto_tangent_space(x, this->get_M(), v, v_proj);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const override
    {
        metric::mass::random_tangent_vector(x, this->get_M(), v);
    }
};


template <int dim>
class GradientTestFrobenius : public GradientTest<dim>
{
public:
    using GradientTest<dim>::GradientTest;

    [[nodiscard]] Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> x_grad(x.size());
        detail::grad_frobenius(this->get_A(), this->get_M(), x, x_grad);

        return x_grad;
    }

    [[nodiscard]] double inner(const Vector<double>& y, const Vector<double>& z) const override
    {
        // The Frobenius metric is exactly the standard Euclidean L2 inner product
        AssertDimension(y.size(), z.size());
        return y * z;
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const override
    {
        metric::frobenius::random_tangent_vector(x, this->get_M(), v);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const override
    {
        metric::frobenius::project_onto_tangent_space(x, this->get_M(), v, v_proj);
    }
};

struct CheckGradInfo
{
    double x_constr;            // constraint
    double grad_xv;             // <grad x, v>_x
    double dir_xv;              // DE(x)[v]
    double grad_res;            // |v-Proj(v)|_x
    double dir_xv_c;            // DE(x)[v] by central difference, O(h^2) accurate
    double slope;               // slope of the linear piece of log E(t) vs. log t; 2 iff correct

    std::vector<double> ts;
    std::vector<double> Ets;
};


template <int dim>
class GradientTestCoarse : public GradientTestBase<dim>
{
public:
    GradientTestCoarse(GrossPitaevskiiSystem<dim>& system, double beta, SolverOptions options,
                       const Vector<double>& phi,   // base point (restricted point)
                       const Vector<double>& w)     // correction term (restricted gradient difference)
        : GradientTestBase<dim>(system, beta, options)
        , m_phi(phi)
        , m_w(w)
    {}

    GradientTestCoarse(GrossPitaevskiiSystem<dim>& system, double beta, SolverOptions options)
        : GradientTestBase<dim>(system, beta, options)
        , m_phi(system.n_dofs())
        , m_w(system.n_dofs())
    {}

    void update_parameters(const Vector<double>& w, const Vector<double>& phi)
    {
        m_phi = phi;
        m_w = w;
    }

    // Unlike GradientTest, value(), random_point(), random_tangent_vector() and directional_derivative()
    // have different implementations per coarse-model, due to the correction vector computed
    // in different metrics.


protected:
    Vector<double> m_phi, m_w;  // copy stored for flipping sign
};


// This would usually be implemented on a coarser grid than the original problem,
// and an available restriction operator for computing `w`.
// For testing gradients, it suffices to consider some level of discretization,
// and consider random base points `phi` and `w`.
template <int dim>
class GradientTestCoarseMass : public GradientTestCoarse<dim>
{
public:
    using GradientTestCoarse<dim>::GradientTestCoarse;

    [[nodiscard]] double value(const Vector<double>& x) const final
    {
        const double energy = this->m_eval.value(x);

        return detail::coarse_mass_value(x, this->m_phi, this->m_w, this->get_M(), energy);
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const final
    {
        return detail::coarse_mass_dir_deriv(x, this->m_phi, this->m_w, z, this->get_M(), this->get_A());
    }

    [[nodiscard]] Vector<double> gradient(const Vector<double>& x) const final
    {
        Vector<double> q_grad(x.size());
        detail::coarse_mass_grad(this->get_M(), this->get_M_inv(), this->get_A(), x, this->m_phi, this->m_w, q_grad);

        return q_grad;
    }

    // Metric and gradient should correspond for testing identities <grad_x f(x), v>_x = Df(x)[v]
    [[nodiscard]] double inner(const Vector<double>& y, const Vector<double>& z) const final
    {
        AssertDimension(y.size(), z.size());
        Vector<double> Mz(z.size());
        this->get_M().vmult(Mz, z);

        return y*Mz;
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const final
    {
        metric::mass::random_tangent_vector(x, this->get_M(), v);
    }

    // Generate a random point x safely in the neighborhood of phi
    void random_point(Vector<double>& x) const final
    {
        // Generate a random tangent vector at phi
        Vector<double> v(x.size());
        this->random_tangent_vector(this->m_phi, v);

        v /= std::sqrt(this->inner(v, v));

        // Retract to find an x that is safely near phi
        this->retract(this->m_phi, v, x);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const final
    {
        metric::mass::project_onto_tangent_space(x, this->get_M(), v, v_proj);
    }
};


template <int dim>
class GradientTestCoarseFrobenius : public GradientTestCoarse<dim>
{
public:
    using GradientTestCoarse<dim>::GradientTestCoarse;

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        const double energy = this->m_eval.value(x);

        return detail::coarse_frobenius_value(x, this->m_phi, this->m_w, this->get_M(), energy);
    }

    [[nodiscard]] double directional_derivative(const dealii::Vector<double>& x, const dealii::Vector<double>& z) const override
    {
        return detail::coarse_frobenius_dir_deriv(x, this->m_phi, this->m_w, z, this->get_M(), this->get_A());
    }

    [[nodiscard]] Vector<double> gradient(const Vector<double>& x) const final
    {
        Vector<double> q_grad(x.size());
        // Pure F-metric gradient of the coarse model
        detail::coarse_frobenius_grad(this->get_M(), this->get_A(),
            x, this->m_phi, this->m_w, q_grad);

        return q_grad;
    }

    [[nodiscard]] double inner(const Vector<double>& y, const Vector<double>& z) const final
    {
        AssertDimension(y.size(), z.size());
        return y * z; // F-metric inner product
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const final
    {
        Vector<double> tmp(v.size());
        normrnd(this->mean, this->stddev, tmp);

        metric::frobenius::project_onto_tangent_space(x, this->get_M(), tmp, v);
    }

    void random_point(Vector<double>& x) const final
    {
        // Generate a random tangent vector at phi
        Vector<double> v(x.size());
        this->random_tangent_vector(this->m_phi, v);
        v /= std::sqrt(this->inner(v, v));

        // Retract to find an x that is safely near phi
        this->retract(this->m_phi, v, x);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const final
    {
        metric::frobenius::project_onto_tangent_space(x, this->get_M(), v, v_proj);
    }
};

}


#endif //RMO_TEST_GRADIENT_H