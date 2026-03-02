//
// Created by Ferdinand Vanmaele on 24.02.26.
//
#include "gpe.h"
#include "random.h"
#include "manifold.h"

#include <fstream>
#include <fmt/format.h>

#define NUM_TRIALS 20

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

        m_problem.assemble_nonlinear_term(x);  // updates Mpp -> A for underlying operators
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const
    {
        // 1. generate random vector in ambient space
        Vector<double> tmp(v.size());
        normrnd(mean, stddev, tmp);

        // 2. project orthogonally onto tangent space at x, wrt. the mass metric
        ellipsoid::project_onto_tangent_space(x, M, tmp, v);
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

    auto get_A() const { return A; }
    auto get_M() const { return M; }
    auto get_A_inv() const { return A_inv; }
    auto get_M_inv() const { return M_inv; }

    virtual double value(const Vector<double>&) {}
    virtual Vector<double> gradient(const Vector<double>&) {}
    virtual double metric(const Vector<double>&, const Vector<double>&) {}

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

    double value(const Vector<double>& x) override
    {
        Vector<double> Ax(x.size());
        this->A.vmult(Ax, x);
        return 0.5 * (x * Ax);
    }

    Vector<double> gradient(const Vector<double>& x) override
    {
        Vector<double> x_grad(x.size());
        ellipsoid::gradient(this->A_inv, this->M, x, x_grad);
        return x_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) override
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

    double value(const Vector<double>& x) override
    {
        Vector<double> Ax(x.size());
        this->A.vmult(Ax, x);
        return 0.5 * (x * Ax);
    }

    Vector<double> gradient(const Vector<double>& x) override
    {
        Vector<double> x_grad(x.size());
        ellipsoid::gradient(this->M_inv, this->A, this->M, x, x_grad);
        return x_grad;
    }

    double metric(const Vector<double>& y, const Vector<double>& z) override
    {
        Vector<double> Mz(z.size());
        this->M.vmult(Mz, z);
        return y*Mz;
    }
};

template <int dim>
class GradientTestEnergyCoarse : public GradientTestBase<dim>
{
public:
    GradientTestEnergyCoarse(const GrossPitaevskiiProblem<dim>& problem, double beta)
        : GradientTestBase<dim>(problem, beta)
    {}

    double value(const Vector<double>& x) override
    {
        return 0;
    }

    Vector<double> gradient(const Vector<double>& x) override
    {
        return {};
    }

    double metric(const Vector<double>& y, const Vector<double>& z) override
    {
        return {};
    }
};

template <int dim>
class GradientTestMassCoarse : public GradientTestBase<dim>
{
public:
    GradientTestMassCoarse(const GrossPitaevskiiProblem<dim>& problem, double beta)
        : GradientTestBase<dim>(problem, beta)
    {}

    double value(const Vector<double>& x) override
    {
        return 0;
    }

    Vector<double> gradient(const Vector<double>& x) override
    {
        return {};
    }

    double metric(const Vector<double>& y, const Vector<double>& z) override
    {
        return {};
    }
};

template <int dim>
void check_gradient(const GradientTestBase<dim>& test_grad, std::string filename)
{

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

    // Problem definition
    // TODO: std::unique_ptr
    GradientTestMass<2>         test_mass(problem, options.beta);
    GradientTestEnergy<2>       test_energy(problem, options.beta);
    GradientTestEnergyCoarse<2> test_energy_coarse(problem, options.beta);
    GradientTestMassCoarse<2>   test_mass_coarse(problem, options.beta);

    dealii::ConvergenceTable convergence_table;

    // Test correctness of gradients for random samples
    // TODO: input: GradientTestBase
    for (unsigned int trial = 0; trial < NUM_TRIALS; trial++) {
        auto filename = fmt::format("checkgradient_trial_{:03}.dat", trial);
        std::cerr << "Writing " << filename << std::endl;
        std::ofstream outfile(filename);  // rename according to function/metric used

        // 1. Generate a random point x in the manifold S
        Vector<double> x(n_dofs);
        test_energy.random_point(x);  // initializes M_pp

        // Check x fulfills |x|_M = 1
        // TODO: merge to GradientTestBase
        Vector<double> Mx(x.size());
        test_energy.get_M().vmult(Mx, x);
        convergence_table.add_value("x_constr", x*Mx);

        // 2. Generate a random tangent vector v at x with |v|_x = 1
        Vector<double> v(n_dofs);
        test_energy.random_tangent_vector(x, v);
        v /= std::sqrt(test_energy.metric(v, v));  // tangent vector with |v|_x = 1

        // The Riemannian gradient of E at x is defined as,
        // the unique element in the tangent space at x, T_x S,
        // such that for all v in T_x S,
        // g_x(\grad(x), v) = DE(x)[v]
        // 2b. Verify this condition for finite difference (directional derivative)
        double fx = test_energy.value(x);
        Vector<double> x_grad = test_energy.gradient(x);
        double g_xv = test_energy.metric(x_grad, v);
        convergence_table.add_value("grad_xv",g_xv);

        // TODO: merge to GradientTestBase
        auto value = std::bind(&GradientTestEnergy<2>::value, &test_energy, std::placeholders::_1);
        double dir_xv8 = finite_difference(value, x, v, 1e-8, fx);
        convergence_table.add_value("dir_xv_1e-8",dir_xv8);

        // 3. Compute f(x) and grad_x f(x)
        //    Check that grad is in T_x S
        //    Compute <grad f(x), v>_x
        Vector<double> x_grad_proj(x_grad.size());
        // TODO: Do we care about which metric for the projected gradient?
        test_energy.to_tangent_space(x, x_grad, x_grad_proj);

        // Residual of difference between gradient, and projected gradient
        Vector<double> x_grad_res(x_grad);
        x_grad_res.add(-1.0, x_grad_proj);
        convergence_table.add_value("grad_proj_res",std::sqrt(test_energy.metric(x_grad_res,x_grad_res)));

        // 4. Compute E(t) for several values of t logarithmically spaced on the interval [10−8,0]
        auto ts = logspace(-8,0,100);
        // auto Ets = std::vector<double>();
        // Ets.reserve(ts.size());

        for (auto t : ts) {
            auto tv = Vector(v);
            tv *= t;

            Vector<double> Rx_tv(x.size());
            test_energy.retract(x, tv, Rx_tv);

            long double Et = std::abs(-test_energy.value(Rx_tv) + fx + t*g_xv);
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