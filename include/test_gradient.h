//
// Created by Ferdinand Vanmaele on 04.04.26.
//
#ifndef GPE_TEST_GRADIENT_H
#define GPE_TEST_GRADIENT_H

#include <gpe/problem/gpe.h>
#include <gpe/util/random.h>
#include <gpe/ropt/manifold.h>

#include <boost/math/special_functions/math_fwd.hpp>


namespace gpe::ellipsoid
{
template <typename MatrixType>
void random_point(Vector<double>& x, const MatrixType& M,
                  double mean = 0.0, double stddev = 1.0)
{
    normrnd(mean, stddev, x);
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    const double factor = x*Mx;
    x /= std::sqrt(factor);
}

namespace mass
{
template <typename MatrixType>
void random_tangent_vector(const Vector<double>& x, const MatrixType& M,
                           Vector<double>& v,
                           double mean = 0.0, double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(v.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the mass metric
    mass::project_onto_tangent_space(x, M, tmp, v);
}

} // namespace mass


namespace energy
{
template <typename MatrixType, typename InverseMatrixType>
void random_tangent_vector(const InverseMatrixType& A_inv, const Vector<double>& x, const MatrixType& M,
                           Vector<double>& v,
                           const double mean = 0.0, const double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(v.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the energy-based metric
    energy::project_onto_tangent_space(A_inv, x, M, tmp, v);
}

} // namespace energy


namespace frobenius
{
template <typename MatrixType>
void random_tangent_vector(const Vector<double>& x, const MatrixType& M,
                           Vector<double>& v,
                           const double mean = 0.0, const double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(v.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the energy-based metric
    frobenius::project_onto_tangent_space(x, M, tmp, v);
}

} // namespace frobenius

} // namespace gpe::ellipdoid


namespace gpe
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

    GradientTestBase(const GrossPitaevskiiSystem<dim>& problem, double beta,
                     SolverOptions options)
        : m_problem(problem)
        , A(problem.get_operator_A(beta))   // all arguments are lazily evaluated
        , M(problem.get_operator_M())
        , A_inv(InverseOpType(A, options))
        , M_inv(InverseOpType(M, options))
        , m_beta(beta)
    {}
    virtual ~GradientTestBase() = default;

    void assemble(const Vector<double>& x) const
    {
        m_problem.assemble_nonlinear_term(x);  // updates Mpp -> A (mutable) for underlying operators
    }

    // Defined for all functions and metrics on S^n
    void retract(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_retr) const
    {
        v_retr = x;
        ellipsoid::retract_by_norm(M, v, v_retr);  // input-output vector
    }

    // Special case for x == v
    void retract(const Vector<double>& x, Vector<double>& x_retr) const
    {
        x_retr = x;
        ellipsoid::retract_by_norm(M, x_retr);
    }

    double constraint_value(const Vector<double>& x) const
    {
        Vector<double> Mx(x.size());
        this->M.vmult(Mx, x);
        return x*Mx;
    }

    auto get_A()     const { return A; }
    auto get_M()     const { return M; }
    auto get_A_inv() const { return A_inv; }
    auto get_M_inv() const { return M_inv; }
    auto n_dofs()    const { return A.m(); }

    virtual void random_point(Vector<double>& x) const = 0;  // virtual for problems with parameters (i.e. coarse model, w, phi)
    virtual void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const = 0;
    virtual void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const = 0;

    virtual double value(const Vector<double>&) const = 0;
    virtual double directional_derivative(const Vector<double>& x, const Vector<double>& z) const = 0;
    virtual Vector<double> gradient(const Vector<double>&) const = 0;
    virtual double metric(const Vector<double>&, const Vector<double>&) const = 0;

protected:
    const GrossPitaevskiiSystem<dim>& m_problem;
    OperatorType A, M;
    InverseOpType A_inv, M_inv;
    double m_beta;
};


template <int dim>
class GradientTest : public GradientTestBase<dim>
{
public:
    GradientTest(const GrossPitaevskiiSystem<dim>& problem, double beta, SolverOptions options)
        : GradientTestBase<dim>(problem, beta, options)
    {}

    double value(const Vector<double>& x) const override
    {
        return this->m_problem.value(x, this->m_beta);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return this->m_problem.directional_derivative(x, z, this->m_beta);
    }

    void random_point(Vector<double>& x) const override
    {
        ellipsoid::random_point(x, this->M);
    }
};


template <int dim>
class GradientTestEnergy : public GradientTest<dim>
{
public:
    using GradientTest<dim>::GradientTest;

    Vector<double> gradient(const Vector<double>& x) const final
    {
        Vector<double> x_grad(x.size());
        ellipsoid::energy::gradient(this->A_inv, this->M, x, x_grad);
        return x_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const final
    {
        AssertDimension(y.size(), z.size());
        Vector<double> Az(z.size());
        this->A.vmult(Az, z);
        return y*Az;
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const final
    {
        ellipsoid::energy::project_onto_tangent_space(this->A_inv, x, this->M, v, v_proj);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const final
    {
        ellipsoid::energy::random_tangent_vector(this->A_inv, x, this->M, v);
    }
};


template <int dim>
class GradientTestMass : public GradientTest<dim>
{
public:
    using GradientTest<dim>::GradientTest;

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> x_grad(x.size());
        ellipsoid::mass::gradient(this->M_inv, this->A, this->M, x, x_grad);
        return x_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        AssertDimension(y.size(), z.size());
        Vector<double> Mz(z.size());
        this->M.vmult(Mz, z);
        return y*Mz;
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const override
    {
        ellipsoid::mass::project_onto_tangent_space(x, this->M, v, v_proj);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const override
    {
        ellipsoid::mass::random_tangent_vector(x, this->M, v);
    }
};


template <int dim>
class GradientTestFrobenius : public GradientTest<dim>
{
public:
    using GradientTest<dim>::GradientTest;

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> x_grad(x.size());
        ellipsoid::frobenius::gradient(this->A, this->M, x, x_grad);
        return x_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        // The Frobenius metric is exactly the standard Euclidean L2 inner product
        AssertDimension(y.size(), z.size());
        return y * z;
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const override
    {
        ellipsoid::frobenius::random_tangent_vector(x, this->M, v);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const override
    {
        ellipsoid::frobenius::project_onto_tangent_space(x, this->M, v, v_proj);
    }
};

struct CheckGradInfo
{
    double x_constr;            // constraint
    double grad_xv;             // <grad x, v>_x
    double dir_xv;              // DE(x)[v]
    double grad_res;            // |v-Proj(v)|_x

    std::vector<double> ts;
    std::vector<double> Ets;
};


template <int dim>
class GradientTestCoarse : public GradientTestBase<dim>
{
public:
    GradientTestCoarse(const GrossPitaevskiiSystem<dim>& problem, double beta, SolverOptions options,
                       const Vector<double>& phi,   // base point (restricted point)
                       const Vector<double>& w)     // correction term (restricted gradient difference)
        : GradientTestBase<dim>(problem, beta, options)
        , m_phi(phi)
        , m_w(w)
    {}

    GradientTestCoarse(const GrossPitaevskiiSystem<dim>& problem, double beta, SolverOptions options)
        : GradientTestBase<dim>(problem, beta, options)
        , m_phi(problem.n_dofs())
        , m_w(problem.n_dofs())
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

    double value(const Vector<double>& x) const final
    {
        const double energy = this->m_problem.value(x, this->m_beta);

        return coarse::mass::function_value(x, this->m_phi, this->m_w, this->M, energy);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const final
    {
        return coarse::mass::directional_derivative(x, this->m_phi, this->m_w, z, this->M, this->A);
    }

    Vector<double> gradient(const Vector<double>& x) const final
    {
        Vector<double> q_grad(x.size());
        coarse::mass::gradient(this->M, this->M_inv, this->A, x, this->m_phi, this->m_w, q_grad);

        return q_grad;
    }

    // Metric and gradient should correspond for testing identities <grad_x f(x), v>_x = Df(x)[v]
    double metric(const Vector<double>& y, const Vector<double>& z) const final
    {
        AssertDimension(y.size(), z.size());
        Vector<double> Mz(z.size());
        this->M.vmult(Mz, z);
        return y*Mz;
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const final
    {
        ellipsoid::mass::random_tangent_vector(x, this->M, v);
    }

    // Generate a random point x safely in the neighborhood of phi
    void random_point(Vector<double>& x) const final
    {
        // Generate a random tangent vector at phi
        Vector<double> v(x.size());
        this->random_tangent_vector(this->m_phi, v);

        v /= std::sqrt(this->metric(v, v));

        // Retract to find an x that is safely near phi
        this->retract(this->m_phi, v, x);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const final
    {
        ellipsoid::mass::project_onto_tangent_space(x, this->M, v, v_proj);
    }
};


template <int dim>
class GradientTestCoarseFrobenius : public GradientTestCoarse<dim>
{
public:
    using GradientTestCoarse<dim>::GradientTestCoarse;

    double value(const Vector<double>& x) const override
    {
        const double energy = this->m_problem.value(x, this->m_beta);

        return coarse::frobenius::function_value(x, this->m_phi, this->m_w, this->M, energy);
    }

    double directional_derivative(const dealii::Vector<double>& x, const dealii::Vector<double>& z) const override
    {
        return coarse::frobenius::directional_derivative(x, this->m_phi, this->m_w, z, this->M, this->A);
    }

    Vector<double> gradient(const Vector<double>& x) const final
    {
        Vector<double> q_grad(x.size());
        // Pure F-metric gradient of the coarse model
        coarse::frobenius::gradient(this->M, this->A,
            x, this->m_phi, this->m_w, q_grad);
        return q_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const final
    {
        AssertDimension(y.size(), z.size());
        return y * z; // F-metric inner product
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const final
    {
        Vector<double> tmp(v.size());
        normrnd(this->mean, this->stddev, tmp);
        ellipsoid::frobenius::project_onto_tangent_space(x, this->M, tmp, v);
    }

    void random_point(Vector<double>& x) const final
    {
        // Generate a random tangent vector at phi
        Vector<double> v(x.size());
        this->random_tangent_vector(this->m_phi, v);
        v /= std::sqrt(this->metric(v, v));

        // Retract to find an x that is safely near phi
        this->retract(this->m_phi, v, x);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const final
    {
        ellipsoid::frobenius::project_onto_tangent_space(x, this->M, v, v_proj);
    }
};

}


#endif //GPE_TEST_GRADIENT_H