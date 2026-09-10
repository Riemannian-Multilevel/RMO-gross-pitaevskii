//
// Verification of the linear case (beta = 0) against an ARPACK eigenvalue solve.
//
// For beta = 0 the Gross-Pitaevskii energy reduces to the Rayleigh quotient of a linear pencil:
//
//     E(x) = 1/2 x^T A0 x     on   { x : x^T M x = 1 },
//
// so its minimizer is the eigenvector of  A0 u = lambda M u  belonging to the smallest eigenvalue
// lambda_1, and the minimal energy is lambda_1 / 2. This program minimizes E with Riemannian
// gradient descent, computes lambda_1 independently with ARPACK, and compares the two.
//
// Note on boundary conditions: with --boundary dirichlet the constrained rows of A0 and M carry
// artificial diagonal entries and contribute spurious eigenvalues to the pencil. The default
// (neumann) leaves the system unconstrained, which is what makes the comparison clean.
//
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/manifold.h>

#include <rmo/ropt/observer_table.h>
#include <rmo/ropt/solver.h>

#include <rmo/option.h>
#include <rmo/util/util.h>

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_ARPACK
#  include <deal.II/lac/arpack_solver.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

using namespace rmo;
using namespace rmo::gpe;
using namespace dealii;

//! Return code reported to ctest when the build cannot run this test (SKIP_RETURN_CODE)
constexpr int SKIP_TEST = 77;

#ifdef DEAL_II_WITH_ARPACK

//! @brief Eigenvalues of A0 u = lambda M u closest to zero, by shift-and-invert with sigma = 0.
//!
//! With sigma = 0 the Arnoldi iteration runs on A0^{-1} M, whose largest-magnitude eigenvalues
//! 1/lambda are the smallest lambda of the original pencil; ArpackSolver transforms them back.
//! @p A_inv only has to provide vmult(dst, src), i.e. solve A0 dst = src.
template <int dim>
std::vector<double>
smallest_eigenvalues(const GrossPitaevskiiSystem<dim>& system,
                     const InverseOpType& A_inv,
                     std::vector<Vector<double>>& eigenvectors,
                     unsigned n_eigen)
{
    const unsigned n_dofs = system.n_dofs();

    SolverControl solver_control(n_dofs, 1e-12, false, false);
    // ARPACK requires strictly more Arnoldi vectors than 2*n_eigen + 1
    ArpackSolver::AdditionalData data(2 * n_eigen + 2, ArpackSolver::largest_magnitude,
                                      /*symmetric=*/true);
    ArpackSolver eigensolver(solver_control, data);

    std::vector<std::complex<double>> lambda(n_eigen);
    eigenvectors.assign(n_eigen, Vector<double>(n_dofs));

    eigensolver.solve(system.get_A0(), system.get_M(), A_inv, lambda, eigenvectors, n_eigen);

    std::vector<double> lambda_re;
    for (const auto& l : lambda) {
        lambda_re.push_back(l.real());
    }
    return lambda_re;
}


//! @brief Scales u to u^T M u = 1 and returns |x^T M u|, which is 1 iff x and u are parallel.
template <typename MatrixType>
double mass_overlap(const MatrixType& M, const Vector<double>& x, Vector<double> u)
{
    Vector<double> Mu(u.size());

    M.vmult(Mu, u);
    u /= std::sqrt(u * Mu);

    M.vmult(Mu, u);
    return std::abs(x * Mu);
}


template <int dim>
bool run_check(GPE_Options options, SolverOptions options_slv, DescentOptions options_gd,
               unsigned level, unsigned n_eigen, double tol)
{
    auto potential_v = potential::get_potential<dim>(options.potential, options.potential_expr);
    auto builder = std::visit([&](auto&& V) {
        return ModelBuilder<dim>(V, options, level);
    }, potential_v);

    auto& system = builder.get_system();
    GrossPitaevskiiFunctional<dim> objective(system, options.beta, options_slv);

    // 1. Minimize E on the unit mass sphere
    UnitMassSphere<OperatorType> manifold(objective.get_M());
    EnergyOracle<dim> oracle(objective, options_slv);

    Vector<double> x(system.n_dofs());
    x = 1.0;
    builder.distribute(x);
    ellipsoid::retract_by_norm(objective.get_M(), x);

    GradientDescent solver(oracle, manifold, options_gd);
    ConvergenceTableObserver conv_observer;
    solver.set_observer(conv_observer);
    solver.cycle(x, std::cout);

    // 2. Independent eigenvalue solve. A = A0 + beta*Mpp equals A0 here, so A_inv applies A0^{-1}.
    auto& A_inv = objective.get_A_inv();
    A_inv.set_tol(1e-13);  // shift-invert accuracy limits what ARPACK can converge to

    std::vector<Vector<double>> eigenvectors;
    const auto lambda = smallest_eigenvalues<dim>(system, A_inv, eigenvectors, n_eigen);

    const auto  it       = std::ranges::min_element(lambda);
    const auto  index    = std::distance(lambda.begin(), it);
    const double lambda_1 = *it;

    // 3. Compare
    Vector<double> A0x(x.size()), Mx(x.size());
    system.get_A0().vmult(A0x, x);
    system.get_M().vmult(Mx, x);

    const double energy   = objective.value(x);     // 1/2 x^T A0 x for beta = 0
    const double rayleigh = (x * A0x) / (x * Mx);
    const double overlap  = mass_overlap(system.get_M(), x, eigenvectors[index]);

    const double err_energy   = std::abs(energy - 0.5 * lambda_1) / std::abs(0.5 * lambda_1);
    const double err_rayleigh = std::abs(rayleigh - lambda_1) / std::abs(lambda_1);

    std::cout << "\ndim = " << dim << ", level = " << level << ", n_dofs = " << system.n_dofs() << "\n";
    std::cout << "eigenvalues (ARPACK):";
    for (double l : lambda) {
        std::cout << " " << l;
    }
    std::cout << "\n\n";
    std::cout << "  lambda_1                 " << lambda_1        << "\n"
              << "  Rayleigh quotient of x   " << rayleigh        << "  (rel. err " << err_rayleigh << ")\n"
              << "  E(x)                     " << energy          << "\n"
              << "  lambda_1 / 2             " << 0.5 * lambda_1  << "  (rel. err " << err_energy   << ")\n"
              << "  |<x, u_1>_M|             " << overlap         << "\n\n";

    bool ok = true;
    for (const auto& [name, err] : {std::pair{"energy", err_energy}, std::pair{"Rayleigh quotient", err_rayleigh}}) {
        if (!(err < tol)) {
            std::cerr << "FAIL: " << name << " differs from the ARPACK reference by " << err
                      << " > " << tol << "\n";
            ok = false;
        }
    }
    // Reported but not enforced: a degenerate lambda_1 makes the individual eigenvector arbitrary
    if (std::abs(overlap - 1.0) > 1e-3) {
        std::cerr << "warning: |<x, u_1>_M| = " << overlap
                  << " differs from 1; lambda_1 may be degenerate\n";
    }
    if (ok) {
        std::cout << "PASS: minimizer of the linear problem matches the ARPACK eigenpair\n";
    }
    return ok;
}

#endif // DEAL_II_WITH_ARPACK


int main(int argc, char* argv[])
{
#ifndef DEAL_II_WITH_ARPACK
    (void)argc; (void)argv;
    std::cerr << "deal.II was built without ARPACK (DEAL_II_WITH_ARPACK undefined); skipping.\n";
    return SKIP_TEST;
#else
    GPE_Options    options    {};
    DescentOptions options_gd {};
    SolverOptions  options_slv{};

    try {
        po::options_description all("Verification of the linear case (beta = 0) against ARPACK");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(inner_cli_options());
        all.add_options()
            ("help", "print this message")
            ("level", po::value<unsigned>()->default_value(6),
                "number of global refinements")
            ("n-eigen", po::value<unsigned>()->default_value(4),
                "number of eigenvalues to compute")
            ("tol-check", po::value<double>()->default_value(1e-6),
                "relative tolerance of the comparison");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << all << std::endl;
            return 0;
        }

        apply_gpe_options(vm, options);
        apply_descent_options(vm, options_gd);
        apply_inner_options(vm, options_slv);

        const auto level     = vm["level"].as<unsigned>();
        const auto n_eigen   = vm["n-eigen"].as<unsigned>();
        const auto tol_check = vm["tol-check"].as<double>();

        // The identity being verified only holds for the linear problem
        if (options.beta != 0.0) {
            std::cerr << "note: --beta " << options.beta << " ignored, the check requires beta = 0\n";
        }
        options.beta = 0.0;

        bool ok = false;
        with_dimension(options.dimension, [&]<typename T0>(T0)
        {
            constexpr int dim = T0::value;
            ok = run_check<dim>(options, options_slv, options_gd, level, n_eigen, tol_check);
        });
        return ok ? 0 : 1;
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }
#endif
}
