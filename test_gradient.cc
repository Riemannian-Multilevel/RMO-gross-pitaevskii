//
// Created by Ferdinand Vanmaele on 24.02.26.
//
#include "gpe.h"
#include "random.h"
#include "manifold.h"

#include <fstream>
#include <fmt/format.h>

#define NUM_TRIALS 100

using namespace gpe;

template <typename FuncType>
long double finite_difference(const FuncType& F, const Vector<double>& x,
    const Vector<double>& v, long double h, long double Fx)
{
    AssertThrow(h > 0, dealii::ExcMessage("h must be positive"));

    // y <- x + hv, v direction
    Vector<double> tmp(x);
    tmp.add(h, v);

    return static_cast<long double>(F(tmp) - Fx) / h;
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

// TODO: check first order-coherence
using OperatorType = LinearCombination<SparseMatrix<double>, Vector<double>>;
template <typename PrecondType>
using InverseOpType = InverseMatrix<OperatorType, PrecondType>;

// Contains all metric-independent components for testing gradients on the ellipsoid
// TODO: reflect in class name?
template <int dim>
class GradientTestBase
{
public:
    const double mean = 0.0;
    const double stddev = 1.0;

    GradientTestBase(const GrossPitaevskiiProblem<dim>& problem, double beta)
        : m_problem(problem)
        , A(problem.get_operator_A(beta))   // all arguments are lazily evaluated
        , M(problem.get_operator_M())
        , A_inv(InverseMatrix(A, SolverMethod::CG, {}, 2000, 1e-12))
        , M_inv(InverseMatrix(M, SolverMethod::CG, {}, 2000, 1e-12))
    {}
    virtual ~GradientTestBase() {};

    void random_point(Vector<double>& x) const
    {
        normrnd(mean, stddev, x);
        Vector<double> Mx(x.size());
        M.vmult(Mx, x);

        const double factor = x*Mx;
        x /= std::sqrt(factor);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const
    {
        // 1. generate random vector in ambient space
        Vector<double> tmp(v.size());
        normrnd(mean, stddev, tmp);

        // 2. project orthogonally onto tangent space at x, wrt. the mass metric
        ellipsoid::project_onto_tangent_space(x, M, tmp, v);
    }

    void assemble(const Vector<double>& x) const
    {
        m_problem.assemble_nonlinear_term(x);  // updates Mpp -> A (mutable) for underlying operators
    }

    void to_tangent_space(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const
    {
        // project orthogonally onto tangent space at x, wrt. the mass metric
        ellipsoid::project_onto_tangent_space(x, M, v, v_proj);
    }

    void retract(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_retr) const
    {
        v_retr = x;
        ellipsoid::retract_by_norm(M, v, v_retr);  // input-output vector
    }

    void retract(const Vector<double>& x, Vector<double>& x_retr) const
    {
        x_retr = x;
        ellipsoid::retract_by_norm(M, x_retr);
    }

    auto get_A() const { return A; }
    auto get_M() const { return M; }
    auto get_A_inv() const { return A_inv; }
    auto get_M_inv() const { return M_inv; }
    auto n_dofs() const { return A.m(); }

    virtual double value(const Vector<double>&) const = 0;
    virtual Vector<double> gradient(const Vector<double>&) const = 0;
    virtual double metric(const Vector<double>&, const Vector<double>&) const = 0;

protected:
    const GrossPitaevskiiProblem<dim>& m_problem;
    OperatorType A, M;
    InverseOpType<dealii::PreconditionIdentity> A_inv, M_inv;
};

template <int dim>
class GradientTestEnergy : public GradientTestBase<dim>
{
public:
    GradientTestEnergy(const GrossPitaevskiiProblem<dim>& problem, double beta)
        : GradientTestBase<dim>(problem, beta)
    {}

    double value(const Vector<double>& x) const override
    {
        Vector<double> Ax(x.size());
        this->A.vmult(Ax, x);
        return 0.5 * (x * Ax);
    }

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> x_grad(x.size());
        ellipsoid::gradient(this->A_inv, this->M, x, x_grad);
        return x_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        Vector<double> Az(z.size());
        this->A.vmult(Az, z);
        return y*Az;
    }
};

template <int dim>
class GradientTestMass : public GradientTestBase<dim>
{
public:
    GradientTestMass(const GrossPitaevskiiProblem<dim>& problem, double beta)
        : GradientTestBase<dim>(problem, beta)
    {}

    double value(const Vector<double>& x) const override
    {
        Vector<double> Ax(x.size());
        this->A.vmult(Ax, x);
        return 0.5 * (x * Ax);
    }

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> x_grad(x.size());
        ellipsoid::gradient(this->M_inv, this->A, this->M, x, x_grad);
        return x_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        Vector<double> Mz(z.size());
        this->M.vmult(Mz, z);
        return y*Mz;
    }
};

// This would usually be implemented on a coarser grid than the original problem,
// and an available restriction operator for computing `w`.
// For testing gradients, it suffices to consider some level of discretization,
// and consider random base points `phi` and `w`.
template <int dim>
class GradientTestEnergyCoarse : public GradientTestBase<dim>
{
public:
    GradientTestEnergyCoarse(const GrossPitaevskiiProblem<dim>& problem, double beta,
                             const Vector<double>& phi,   // base point (restricted point)
                             const Vector<double>& w)     // correction term (restricted gradient difference))
        : GradientTestBase<dim>(problem, beta)
        , m_phi(phi)
        , m_w(w)
    {}

    double value(const Vector<double>& x) const override
    {
        // Both energy-based and mass-weighed gradients use the mass-weighed coarse model
        return coarse::function_value(x, m_phi, m_w, this->M, this->A);
    }

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> q_grad(x.size());
        coarse::gradient(this->M, this->A_inv, x, m_phi, m_w, q_grad);
        return q_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        Vector<double> Az(z.size());
        this->A.vmult(Az, z);
        return y*Az;
    }

private:
    const Vector<double>& m_phi, m_w;   // copy for flipping sign
};


template <int dim>
class GradientTestMassCoarse : public GradientTestBase<dim>
{
public:
    GradientTestMassCoarse(const GrossPitaevskiiProblem<dim>& problem, double beta,
                           const Vector<double>& phi,   // base point (restricted point)
                           const Vector<double>& w)     // correction term (restricted gradient difference)
        : GradientTestBase<dim>(problem, beta)
        , m_phi(phi)
        , m_w(w)
    {}

    double value(const Vector<double>& x) const override
    {
        return coarse::function_value(x, m_phi, m_w, this->M, this->A);
    }

    Vector<double> gradient(const Vector<double>& x) const override
    {
        Vector<double> q_grad(x.size());
        coarse::gradient(this->M, this->M_inv, this->A, x, m_phi, m_w, q_grad);
        return q_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        Vector<double> Mz(z.size());
        this->M.vmult(Mz, z);
        return y*Mz;
    }

private:
    const Vector<double>& m_phi, m_w;  // copy stored for flipping sign
};


// Test correctness of gradients for random samples
// TODO: proper test for testing coarse gradient in a neighborhood of \phi
template <int dim>
void check_gradient(const GradientTestBase<dim>& test_grad, unsigned int n_trials, std::string prefix,
                    const Vector<double>& phi_coarse_fix = {})
{
    dealii::ConvergenceTable convergence_table;

    for (unsigned int trial = 0; trial < n_trials; trial++) {
        // TODO: filename for both regular and coarse tests
        //auto filename = prefix + fmt::format("_{:03}.dat",trial);
        auto filename = prefix + ".dat";
        std::cerr << "Writing " << filename << std::endl;
        std::ofstream outfile(filename);  // rename according to function/metric used
        const unsigned int n_dofs = test_grad.n_dofs();

        Vector<double> x(n_dofs);
        if (phi_coarse_fix.empty()) {
            // 1. Generate a random point x in the manifold S
            test_grad.random_point(x);
        }
        else {
            // 1. Generate a random point x safely in the neighborhood of phi
            // Generate a random tangent vector at phi
            Vector<double> v_init(n_dofs);
            test_grad.random_tangent_vector(phi_coarse_fix, v_init);
            v_init /= std::sqrt(test_grad.metric(v_init, v_init));

            // Retract to find an x that is safely near phi
            // (e.g., a step size of 0.1 ensures phi^T M x > 0)
            //v_init *= 0.1;
            test_grad.retract(phi_coarse_fix, v_init, x);
        }
        test_grad.assemble(x);    // initializes M_pp

        // 2. Generate a random tangent vector v at x with |v|_x = 1
        Vector<double> v(n_dofs);
        test_grad.random_tangent_vector(x, v);
        v /= std::sqrt(test_grad.metric(v, v));  // tangent vector with |v|_x = 1

        // Check x fulfills |x|_M = 1
        // TODO: merge to GradientTestBase
        Vector<double> Mx(x.size());
        test_grad.get_M().vmult(Mx, x);
        convergence_table.add_value("x_constr", x*Mx);

        // The Riemannian gradient of E at x is defined as,
        // the unique element in the tangent space at x, T_x S,
        // such that for all v in T_x S,
        // g_x(\grad(x), v) = DE(x)[v]
        // 2b. Verify this condition for finite difference (directional derivative)
        double fx = test_grad.value(x);
        Vector<double> x_grad = test_grad.gradient(x);
        double g_xv = test_grad.metric(x_grad, v);
        convergence_table.add_value("grad_xv",g_xv);

        // TODO: merge to GradientTestBase
        auto value = std::bind(&GradientTestBase<dim>::value, &test_grad, std::placeholders::_1);
        double dir_xv8 = finite_difference(value, x, v, 1e-8, fx);
        convergence_table.add_value("dir_xv_1e-8",dir_xv8);

        // 3. Compute f(x) and grad_x f(x)
        //    Check that grad is in T_x S
        //    Compute <grad f(x), v>_x
        Vector<double> x_grad_proj(x_grad.size());
        // TODO: Do we care about which metric for the projected gradient?
        test_grad.to_tangent_space(x, x_grad, x_grad_proj);

        // Residual of difference between gradient, and projected gradient
        Vector<double> x_grad_res(x_grad);
        x_grad_res.add(-1.0, x_grad_proj);
        convergence_table.add_value("grad_proj_res",std::sqrt(test_grad.metric(x_grad_res,x_grad_res)));

        // 4. Compute E(t) for several values of t logarithmically spaced on the interval [10−8,0]
        auto ts = logspace(-8,0,100);
        // auto Ets = std::vector<double>();
        // Ets.reserve(ts.size());

        for (auto t : ts) {
            auto tv = Vector(v);
            tv *= t;

            Vector<double> Rx_tv(x.size());
            test_grad.retract(x, tv, Rx_tv);

            long double Et = std::abs(-test_grad.value(Rx_tv) + fx + t*g_xv);
            //Ets.push_back(Et);
            outfile << t << "\t" << Et << std::endl;
        }

        // 5. Plot E(t) as a function of t, in a log–log plot;
        outfile.close();
    }

    convergence_table.set_precision("x_constr", 6);
    convergence_table.set_precision("dir_xv_1e-8", 6);
    convergence_table.set_precision("grad_xv", 6);
    convergence_table.set_precision("grad_proj_res", 6);

    convergence_table.set_scientific("x_constr", true);
    convergence_table.set_scientific("dir_xv_1e-8", true);
    convergence_table.set_scientific("grad_xv", true);
    convergence_table.set_scientific("grad_proj_res", true);

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

    constexpr unsigned int n_levels = 8;
    GrossPitaevskiiPackage<2> GS(options, n_levels);
    GrossPitaevskiiProblem<2> problem = GS.problem(Square<2>());
    const unsigned n_dofs = GS.n_dofs();

    // std::cerr << "--- GRADIENT CHECK - ENERGY\n";
    // GradientTestEnergy<2> test_energy(problem, options.beta);
    // check_gradient(test_energy, NUM_TRIALS, "checkgradient_energy_2d");
    // std::cerr << "\n";
    //
    // std::cerr << "--- GRADIENT CHECK - MASS\n";
    GradientTestMass<2> test_mass(problem, options.beta);
    // check_gradient(test_mass, NUM_TRIALS, "checkgradient_mass_2d");
    // std::cerr << "\n";

    std::cerr << "--- GRADIENT CHECK - COARSE (MASS)\n";
    Vector<double> w(n_dofs);
    w = 1.0;

    for (unsigned int trial = 0; trial < NUM_TRIALS; trial++) {
        Vector<double> phi(n_dofs);
        test_mass.random_point(phi);

        Vector<double> w_proj(n_dofs);
        ellipsoid::project_onto_tangent_space(test_mass.get_A_inv(), phi, test_mass.get_M(), w, w_proj);

        GradientTestEnergyCoarse<2> test_coarse_energy(problem, options.beta, phi, w_proj);
        check_gradient(test_coarse_energy, 1,
            fmt::format("checkgradient_coarse_energy_2d_{:03}",trial), phi);
    }
    std::cerr << "\n";

    std::cerr << "--- GRADIENT CHECK - COARSE (ENERGY)\n";
    for (unsigned int trial = 0; trial < NUM_TRIALS; trial++) {
        Vector<double> phi(n_dofs);
        test_mass.random_point(phi);

        Vector<double> w_proj(n_dofs);
        ellipsoid::project_onto_tangent_space(phi, test_mass.get_M(), w, w_proj);

        GradientTestMassCoarse<2> test_coarse_mass(problem, options.beta, phi, w_proj);
        check_gradient(test_coarse_mass, 1,
            fmt::format("checkgradient_coarse_mass_2d_{:03}",trial), phi);
    }


    std::cerr << "\n";
}