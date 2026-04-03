//
// Created by Ferdinand Vanmaele on 24.02.26.
//
#include "gpe.h"
#include "random.h"
#include "manifold.h"
#include "function.h"

#include <fstream>
#include <boost/math/special_functions/math_fwd.hpp>
#include <fmt/format.h>

#define NUM_TRIALS 50

using namespace gpe;

// TODO: long double doesn't do much here, since F() is evaluated in double
template <typename FuncType>
long double finite_difference(const FuncType& F, const Vector<double>& x,
                              const Vector<double>& v, long double h, long double Fx)
{
    // Evaluate at x + hv
    Vector<double> tmp(x);
    tmp.add(h, v);

    const long double F_tmp = F(tmp);
    return (F_tmp - Fx) / h;
}

std::vector<double>
logspace(double start_exp, double end_exp, int num) {
    std::vector<double> values;
    if (num <= 0) return values;
    if (num == 1) {
        values.push_back(std::pow(10, start_exp));
        return values;
    }

    double step = (end_exp - start_exp) / (num - 1);
    for (int i = 0; i < num; ++i) {
        values.push_back(std::pow(10, start_exp + i * step));
    }
    return values;
}

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

template <typename MatrixType>
void random_tangent_vector(const Vector<double>& x, const MatrixType& M,
                           Vector<double>& v,
                           double mean = 0.0, double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(v.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the mass metric
    ellipsoid::mass::project_onto_tangent_space(x, M, tmp, v);
}

template <typename MatrixType, typename InverseMatrixType>
void random_tangent_vector(const InverseMatrixType& A_inv, const Vector<double>& x,
                           const MatrixType& M, Vector<double>& v,
                           double mean = 0.0, double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(v.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the energy-based metric
    ellipsoid::energy::project_onto_tangent_space(A_inv, x, M, tmp, v);
}

}


// TODO: check first order-coherence

// Contains all metric-independent components for testing gradients on the ellipsoid
// TODO: reflect in class name?
template <int dim>
class GradientTestBase
{
public:
    const double mean = 0.0;
    const double stddev = 1.0;

    using MatrixType    = SparseMatrix<double>;
    using OperatorType  = LinearCombination<MatrixType, Vector<double>>;
    using InverseOpType = PreconditionInverse<OperatorType, MatrixType>;

    GradientTestBase(const GrossPitaevskiiProblem<dim>& problem, double beta,
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

    auto get_A() const { return A; }
    auto get_M() const { return M; }
    auto get_A_inv() const { return A_inv; }
    auto get_M_inv() const { return M_inv; }
    auto n_dofs() const { return A.m(); }

    virtual void random_point(Vector<double>& x) const = 0;
    virtual void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const = 0;
    virtual void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const = 0;

    virtual double value(const Vector<double>&) const = 0;
    virtual Vector<double> gradient(const Vector<double>&) const = 0;
    virtual double metric(const Vector<double>&, const Vector<double>&) const = 0;

protected:
    const GrossPitaevskiiProblem<dim>& m_problem;
    OperatorType A, M;
    InverseOpType A_inv, M_inv;
    double m_beta;
};

template <int dim>
class GradientTestEnergy : public GradientTestBase<dim>
{
public:
    GradientTestEnergy(const GrossPitaevskiiProblem<dim>& problem, double beta, SolverOptions options)
        : GradientTestBase<dim>(problem, beta, options)
    {}

    double value(const Vector<double>& x) const override
    {
        const auto& A0  = this->m_problem.get_A0();
        const auto& Mpp = this->m_problem.get_Mpp();

        return ellipsoid::function_value(x, A0, Mpp, this->m_beta);
    }

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> x_grad(x.size());
        ellipsoid::energy::gradient(this->A_inv, this->M, x, x_grad);
        return x_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        Vector<double> Az(z.size());
        this->A.vmult(Az, z);
        return y*Az;
    }

    void random_point(Vector<double>& x) const override
    {
        ellipsoid::random_point(x, this->M);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const override
    {
        ellipsoid::random_tangent_vector(this->A_inv, x, this->M, v);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const override
    {
        ellipsoid::energy::project_onto_tangent_space(this->A_inv, x, this->M, v, v_proj);
    }
};

template <int dim>
class GradientTestMass : public GradientTestBase<dim>
{
public:
    GradientTestMass(const GrossPitaevskiiProblem<dim>& problem, double beta, SolverOptions options)
        : GradientTestBase<dim>(problem, beta, options)
    {}

    double value(const Vector<double>& x) const override
    {
        const auto& A0  = this->m_problem.get_A0();
        const auto& Mpp = this->m_problem.get_Mpp();

        return ellipsoid::function_value(x, A0, Mpp, this->m_beta);
    }

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> x_grad(x.size());
        ellipsoid::mass::gradient(this->M_inv, this->A, this->M, x, x_grad);
        return x_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        Vector<double> Mz(z.size());
        this->M.vmult(Mz, z);
        return y*Mz;
    }

    void random_point(Vector<double>& x) const override
    {
        ellipsoid::random_point(x, this->M);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const override
    {
        ellipsoid::random_tangent_vector(x, this->M, v);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const override
    {
        ellipsoid::mass::project_onto_tangent_space(x, this->M, v, v_proj);
    }
};

// This would usually be implemented on a coarser grid than the original problem,
// and an available restriction operator for computing `w`.
// For testing gradients, it suffices to consider some level of discretization,
// and consider random base points `phi` and `w`.
template <int dim>
class GradientTestCoarseEnergy : public GradientTestBase<dim>
{
public:
    GradientTestCoarseEnergy(const GrossPitaevskiiProblem<dim>& problem, double beta, SolverOptions options,
                             const Vector<double>& phi,   // base point (restricted point)
                             const Vector<double>& w)     // correction term (restricted gradient difference)
        : GradientTestBase<dim>(problem, beta, options)
        , m_phi(phi)
        , m_w(w)
    {}

    GradientTestCoarseEnergy(const GrossPitaevskiiProblem<dim>& problem, double beta, SolverOptions options)
        : GradientTestBase<dim>(problem, beta, options)
        , m_phi(problem.n_dofs())
        , m_w(problem.n_dofs())
    {}

    void update_parameters(const Vector<double>& phi, const Vector<double>& w)
    {
        m_phi = phi;
        m_w = w;
    }

    double value(const Vector<double>& x) const override
    {
        const auto& A0  = this->m_problem.get_A0();
        const auto& Mpp = this->m_problem.get_Mpp();
        const auto& M   = this->m_problem.get_M();

        // Both energy-based and mass-weighed gradients use the mass-weighed coarse model
        return coarse::mass::function_value(x, m_phi, m_w, M, A0, Mpp, this->m_beta);
    }

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> q_grad(x.size());
        coarse::mass::energy_adaptive_gradient(this->M, this->A_inv, x, m_phi, m_w, q_grad);
        return q_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        Vector<double> Az(z.size());
        this->A.vmult(Az, z);
        return y*Az;
    }

    // Generate a random point x safely in the neighborhood of phi
    void random_point(Vector<double>& x) const override
    {
        // Generate a random tangent vector at phi
        Vector<double> v(x.size());
        ellipsoid::random_tangent_vector(this->A_inv, m_phi, this->M, v);
        v /= std::sqrt(this->metric(v, v));

        // Retract to find an x that is safely near phi
        this->retract(m_phi, v, x);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const override
    {
        ellipsoid::random_tangent_vector(this->A_inv, x, this->M, v);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const override
    {
        ellipsoid::energy::project_onto_tangent_space(this->A_inv, x, this->M, v, v_proj);
    }

private:
    Vector<double> m_phi, m_w;   // copy for flipping sign
};


template <int dim>
class GradientTestCoarseMass : public GradientTestBase<dim>
{
public:
    GradientTestCoarseMass(const GrossPitaevskiiProblem<dim>& problem, double beta, SolverOptions options,
                           const Vector<double>& phi,   // base point (restricted point)
                           const Vector<double>& w)     // correction term (restricted gradient difference)
        : GradientTestBase<dim>(problem, beta, options)
        , m_phi(phi)
        , m_w(w)
    {}

    GradientTestCoarseMass(const GrossPitaevskiiProblem<dim>& problem, double beta, SolverOptions options)
        : GradientTestBase<dim>(problem, beta, options)
        , m_phi(problem.n_dofs())
        , m_w(problem.n_dofs())
    {}

    void update_parameters(const Vector<double>& phi, const Vector<double>& w)
    {
        m_phi = phi;
        m_w = w;
    }

    double value(const Vector<double>& x) const override
    {
        const auto& A0  = this->m_problem.get_A0();
        const auto& Mpp = this->m_problem.get_Mpp();
        const auto& M   = this->m_problem.get_M();

        return coarse::mass::function_value(x, m_phi, m_w, M, A0, Mpp, this->m_beta);
    }

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> q_grad(x.size());
        coarse::mass::gradient(this->M, this->M_inv, this->A, x, m_phi, m_w, q_grad);
        return q_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        Vector<double> Mz(z.size());
        this->M.vmult(Mz, z);
        return y*Mz;
    }

    // Generate a random point x safely in the neighborhood of phi
    void random_point(Vector<double>& x) const override
    {
        // Generate a random tangent vector at phi
        Vector<double> v(x.size());
        ellipsoid::random_tangent_vector(m_phi, this->M, v);
        v /= std::sqrt(this->metric(v, v));

        // Retract to find an x that is safely near phi
        this->retract(m_phi, v, x);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const override
    {
        ellipsoid::random_tangent_vector(x, this->M, v);
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const override
    {
        ellipsoid::mass::project_onto_tangent_space(x, this->M, v, v_proj);
    }

private:
    Vector<double> m_phi, m_w;  // copy stored for flipping sign
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

struct EmptyStrategy
{
    void operator()() {};
};

// Test correctness of gradients for random samples
// TODO: additional data (h=1-8, n_trial_points=100, start_exp=-8)
template <int dim>
CheckGradInfo check_gradient_trial(const GradientTestBase<dim>& test_grad)
{
    CheckGradInfo check;
    const unsigned int n_dofs = test_grad.n_dofs();

    // 1. Generate a random point at x
    Vector<double> x(n_dofs);
    test_grad.random_point(x);
    test_grad.assemble(x);    // initializes M_pp

    // Check x fulfills |x|_M = 1
    double x_constr = test_grad.constraint_value(x);
    check.x_constr = x_constr;
    
    // 2. Generate a random tangent vector v at x with |v|_x = 1
    Vector<double> v(n_dofs);
    test_grad.random_tangent_vector(x, v);
    v /= std::sqrt(test_grad.metric(v, v));  // tangent vector with |v|_x = 1

    // Verify g_x(\grad(x), v) = DE(x)[v] for finite difference (directional derivative)
    double fx = test_grad.value(x);
    Vector<double> x_grad = test_grad.gradient(x);
    double g_xv = test_grad.metric(x_grad, v);
    check.grad_xv = g_xv;

    // --- FINITE DIFFERENCE CHECK ---
    // TODO: integrate this with value() (or remove the point argument)
    auto value_with_assembly = [&](const Vector<double>& z) {
        test_grad.assemble(z);         // Explicitly assemble for the trial point
        return test_grad.value(z);     // Evaluate
    };
    double dir_xv8 = finite_difference(value_with_assembly, x, v, 1e-8, fx);
    check.dir_xv = dir_xv8;

    // RESTORE BASE STATE! The finite difference mutated the matrices.
    test_grad.assemble(x);

    // 3. Check that grad is in T_x S
    Vector<double> x_grad_proj(x_grad.size());
    test_grad.to_tangent_space(x, x_grad, x_grad_proj);

    // Residual of difference between gradient, and projected gradient in T_x S
    Vector<double> x_grad_res(x_grad);
    x_grad_res.add(-1.0, x_grad_proj);
    double grad_res = std::sqrt(test_grad.metric(x_grad_res,x_grad_res));
    check.grad_res  = grad_res;

    // 4. Compute E(t) for several values of t logarithmically spaced on the interval [10−8,0]
    check.ts  = logspace(-8,0,100);
    check.Ets = std::vector<double>{};

    for (auto t : check.ts) {
        auto tv = Vector(v);
        tv *= t;

        Vector<double> Rx_tv(x.size());
        test_grad.retract(x, tv, Rx_tv);  // only uses M (no assembly required)

        // --- EXPLICIT ASSEMBLY FOR RETRACTED POINT ---
        // TODO: integrate this with value() (or remove the point argument)
        test_grad.assemble(Rx_tv);

        long double Et = std::abs(-test_grad.value(Rx_tv) + fx + t*g_xv);
        check.Ets.push_back(Et);
    }
    AssertDimension(check.ts.size(), check.Ets.size());
    return check;
}

template <int dim, typename Strategy = EmptyStrategy>
void check_gradient(const GradientTestBase<dim>& test_grad, unsigned n_trials, std::string prefix,
                    Strategy&& setup_trial = {})
{
    dealii::ConvergenceTable convergence_table;

    for (unsigned int trial = 0; trial < n_trials; trial++) {
        setup_trial();  // initialization method for GradientTestBase (-> base points for coarse model)
        auto info = check_gradient_trial(test_grad);
        
        convergence_table.add_value("x_constr", info.x_constr);
        convergence_table.add_value("grad_xv",info.grad_xv);
        convergence_table.add_value("dir_xv",info.dir_xv);
        convergence_table.add_value("grad_res",info.grad_res);

        // 5. Plot E(t) as a function of t, in a log–log plot;
        std::string filename;
        if (n_trials > 1) {
            filename = prefix + fmt::format("_{:03}.dat",trial);
        } else {
            filename = prefix + ".dat";
        }
        std::cerr << "Writing " << filename << std::endl;
        std::ofstream outfile(filename);
        
        for (unsigned i = 0; i < info.ts.size(); i++) {
            outfile << info.ts[i] << "\t" << info.Ets[i] << std::endl;
        }
        outfile.close();
    }
    
    convergence_table.set_precision("x_constr", 6);
    convergence_table.set_precision("dir_xv", 6);
    convergence_table.set_precision("grad_xv", 6);
    convergence_table.set_precision("grad_res", 6);

    convergence_table.set_scientific("x_constr", true);
    convergence_table.set_scientific("dir_xv", true);
    convergence_table.set_scientific("grad_xv", true);
    convergence_table.set_scientific("grad_res", true);

    convergence_table.write_text(std::cout, dealii::TableHandler::TextOutputFormat::table_with_headers);
}

int main()
{
    GPE_Options options{};
    options.dimension = 2;
    options.degree    = 1;  // piecewise linear (1) or quadratic (2) elements
    options.radius    = 10;
    options.beta      = 100;
    options.bc        = BoundaryCondition::DIRICHLET;
    options.mesh_kind = MeshKind::QUADRILATERAL;
    options.order     = Ordering::CUTHILL_MCKEE;

    SolverOptions options_slv{};
    options_slv.solver    = SolverMethod::CG;
    options_slv.max_inner = 2000;
    options_slv.precond   = Precondition::NONE;
    options_slv.tol_inner = 1e-12;

    constexpr unsigned int n_levels = 8;
    GrossPitaevskiiPackage<2> GS(options, n_levels);
    GrossPitaevskiiProblem<2> problem = GS.problem(Square<2>());
    const unsigned n_dofs = GS.n_dofs();

    {
        std::cerr << "--- GRADIENT CHECK - ENERGY\n";
        GradientTestEnergy<2> test_energy(problem, options.beta, options_slv);
        check_gradient(test_energy, NUM_TRIALS, "checkgradient_energy_2d");
        std::cerr << "\n";
    }

    {
        std::cerr << "--- GRADIENT CHECK - MASS\n";
        GradientTestMass<2> test_mass(problem, options.beta, options_slv);
        check_gradient(test_mass, NUM_TRIALS, "checkgradient_mass_2d");
        std::cerr << "\n";
    }

    {
        std::cerr << "--- GRADIENT CHECK - COARSE (MASS)\n";
        GradientTestCoarseMass<2> test_coarse_mass(problem, options.beta, options_slv);
        GradientTestMass<2> test_mass(problem, options.beta, options_slv);

        Vector<double> w(n_dofs);  // fixed correction term
        w = 1.0;

        auto setup_base_points_mass = [&w,&test_mass,&test_coarse_mass,n_dofs]()
        {
            Vector<double> phi(n_dofs); // random base point
            test_mass.random_point(phi);

            Vector<double> w_proj(n_dofs);
            ellipsoid::mass::project_onto_tangent_space(phi, test_mass.get_M(), w, w_proj);

            test_coarse_mass.update_parameters(phi, w_proj);
        };

        check_gradient(test_coarse_mass, NUM_TRIALS,
            "checkgradient_coarse_mass_2d", setup_base_points_mass);
    }

    {
        std::cerr << "--- GRADIENT CHECK - COARSE (ENERGY)\n";
        GradientTestCoarseEnergy<2> test_coarse_energy(problem, options.beta, options_slv);
        GradientTestEnergy<2> test_energy(problem, options.beta, options_slv);

        Vector<double> w(n_dofs);  // fixed correction term
        w = 1.0;

        auto setup_base_points_energy = [&w,&test_energy,&test_coarse_energy,n_dofs]()
        {
            Vector<double> phi(n_dofs);
            test_energy.random_point(phi);

            Vector<double> w_proj(n_dofs);
            ellipsoid::energy::project_onto_tangent_space(test_energy.get_A_inv(), phi, test_energy.get_M(), w, w_proj);

            test_coarse_energy.update_parameters(phi, w_proj);
        };

        check_gradient(test_coarse_energy, NUM_TRIALS,
            "checkgradient_coarse_energy_2d", setup_base_points_energy);
    }
}