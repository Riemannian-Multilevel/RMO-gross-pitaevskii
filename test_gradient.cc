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

// This function can be used for tangent vectors in the (mass-weighed) sphere,
// for both mass and energy-based metrics
template <typename MatrixType>
void random_tangent_vector(const Vector<double>& x, const MatrixType& M,
                           Vector<double>& output,
                           double mean = 0.0, double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(output.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the mass metric
    ellipsoid::project_onto_tangent_space(x, M, tmp, output);
}

template <typename MatrixType>
double tangent_condition(const Vector<double>& x, const MatrixType& M, const Vector<double>& v)
{
    Vector<double> Mv(v.size());
    M.vmult(Mv, v);
    return x*Mv;

}
template <typename MatrixType>
void random_point(const MatrixType& M, Vector<double>& output,
                  double mean = 0.0, double stddev = 1.0)
{
    normrnd(mean, stddev, output);

    Vector<double> Mx(output.size());
    M.vmult(Mx, output);

    const double factor = output*Mx;
    output *= std::sqrt(factor);
}

template <typename FuncType>
double finite_difference(const FuncType& F, const Vector<double>& x,
    const Vector<double>& v, double h, double Fx)
{
    AssertThrow(h > 0, dealii::ExcMessage("h must be positive"));

    // y <- x + hv, v direction
    Vector<double> tmp(x);
    tmp.add(h, v);

    return (F(tmp) - Fx) / h;
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

template <typename FuncF, typename FuncG, typename FuncM>
void check_gradient(const FuncF& f,
                    const FuncG& g,
                    const FuncM& metric,
                    const Vector<double>& x,
                    dealii::ConvergenceTable& table)
{

}

// TODO: check first order-coherence

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

    // GPE minimization
    constexpr unsigned int n_levels = 8;
    GrossPitaevskiiPackage<2> GS(options, n_levels);
    GrossPitaevskiiProblem<2> problem = GS.problem(Square<2>());
    const unsigned n_dofs = GS.n_dofs();

    // 1. Check that Riemannian gradient is in the tangential space

    // The Riemannian gradient of E at x is defined as,
    // the unique element in the tangent space at x, T_x S,
    // such that for all v in T_x S,
    // g_x(\grad(x), v) = DE(x)[v]
    // 2. verify this condition for finite difference (directional derivative)

    // -> error with Taylor expansion

    dealii::ConvergenceTable convergence_table;
    using OperatorType  = LinearCombination<SparseMatrix<double>, Vector<double>>;
    using InverseOpType = InverseMatrix<OperatorType>;

    // TODO: functor with f(.), grad(.), metric(.) arguments
    //       input: GrossPitaevskiiProblem<dim>
    for (unsigned int trial = 0; trial < NUM_TRIALS; trial++) {
        auto filename = fmt::format("checkgradient_trial_{:03}.dat", trial);
        std::cerr << "Writing " << filename << std::endl;
        std::ofstream outfile(filename);  // rename according to function/metric used

        // 1. Generate a random point x in the manifold S
        // TODO: this is necessarily done beforehand, since f/grad f require A_x
        Vector<double> x(n_dofs);
        random_point(problem.get_operator_M(), x, 0.0, 1.0);

        // --- Assemble operators
        // Update metric for x
        problem.assemble_nonlinear_term(x);
        const auto& M = problem.get_operator_M();
        auto Minv = InverseOpType(M, SolverMethod::MINRES);

        const auto& A = problem.get_operator_A(options.beta);
        auto Ainv = InverseOpType(A, SolverMethod::MINRES);
        const auto& Ac = problem.get_operator_A(options.beta/2);  // for function value

        // TODO: check for A-metric
        auto f = [&Ac](const Vector<double>& x)
        {
            return ellipsoid::function_value(x, Ac);  // needs to evaluated at multiple points
        };
        auto g = [&Minv, &A, &M](const Vector<double>& x)
        {
            Vector<double> x_grad(x.size());
            ellipsoid::gradient(Minv, A, M, x, x_grad);
            return x_grad;
        };
        auto metric = [&M](const Vector<double>& y, const Vector<double>& z)
        {
            Vector<double> Mz(z.size());
            M.vmult(Mz, z);
            return y*Mz;
        };

        // --- Perform verification
        // 2. Generate a random tangent vector v at x with |v|_x = 1
        Vector<double> v(n_dofs);
        random_tangent_vector(x, M, v, 0.0, 1.0);
        v /= std::sqrt(metric(v, v));  // tangent vector of norm 1

        Vector<double> x_grad = g(x);
        double fx = f(x);

        // The Riemannian gradient of E at x is defined as,
        // the unique element in the tangent space at x, T_x S,
        // such that for all v in T_x S,
        // g_x(\grad(x), v) = DE(x)[v]
        // 2b. Verify this condition for finite difference (directional derivative)
        double dir_xv = finite_difference(f, x, x_grad, 1e-8, fx);
        double g_xv = metric(x_grad, v);
        convergence_table.add_value("direct_xv",dir_xv);
        convergence_table.add_value("grad_xv",g_xv);

        // 3. Compute f(x) and grad_x f(x)
        //    Check that grad is in T_x S
        //    Compute <grad f(x), v>_x
        double tan_cond = tangent_condition(x, M, x_grad);   // only dependent on manifold
        convergence_table.add_value("tan_cond",tan_cond);

        // 4. Compute E(t) for several values of t logarithmically spaced on the interval [10−8,0]
        auto ts = logspace(-8,0,100);
        auto Ets = std::vector<double>();
        Ets.reserve(ts.size());

        for (auto t : ts) {
            auto tv = Vector(v);
            tv *= t;

            auto Rx_tv = Vector(x);
            ellipsoid::retract_by_norm(M, tv, Rx_tv);  // input-output vector

            double Et = std::abs(f(Rx_tv) - fx - t*g_xv);
            outfile << std::scientific << std::setprecision(10) << t << "\t" << Et << std::endl;
        }

        // 5. Plot E(t) as a function of t, in a log–log plot;
        outfile.close();

        // A-metric
        // Vector<double> Ax(n_dofs);
        // A.vmult(Ax, x_rand);
        // double Ax_norm = x_rand*Ax;
        // v_rand /= Ax_norm;
        //
    }

    convergence_table.set_precision("direct_xv", 6);
    convergence_table.set_precision("grad_xv", 6);
    convergence_table.set_precision("tan_cond", 6);

    convergence_table.set_scientific("direct_xv", true);
    convergence_table.set_scientific("grad_xv", true);
    convergence_table.set_scientific("tan_cond", true);

    convergence_table.write_text(std::cout, dealii::TableHandler::TextOutputFormat::table_with_headers);
}