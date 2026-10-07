//
// Checks the linear case (beta = 0) against an ARPACK eigenvalue solve.
//
// For beta = 0, the energy is E(x) = 1/2 x^T A0 x on { x : x^T M x = 1 }. Its minimizer is the
// eigenvector of A0 u = lambda M u for the smallest eigenvalue lambda_1, and the minimal energy is
// lambda_1 / 2. The test minimizes E by Riemannian gradient descent, computes lambda_1 with ARPACK, and
// compares the two.
//
// With --boundary dirichlet, the constrained rows of A0 and M add artificial eigenvalues; the default
// (neumann) has no constrained rows.
//
#include "check.h"

#include <rmo/gpe/manifold.h>
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>

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
#include <optional>
#include <string>
#include <vector>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;
using namespace dealii;

#ifdef DEAL_II_WITH_ARPACK
namespace
{

struct Config
{
    GPE_Options    gpe{};
    SolverOptions  solver{};
    DescentOptions descent{};
    unsigned       level   = 0;    ///< number of global refinements
    unsigned       n_eigen = 0;    ///< number of eigenvalues computed by ARPACK
    double         tol     = 0.0;  ///< relative tolerance of the comparison
};

//! Configuration from the command line, or nothing for --help.
std::optional<Config> parse_options(int argc, char* argv[])
{
    po::options_description all("Verification of the linear case (beta = 0) against ARPACK");
    all.add(gpe_cli_options());
    all.add(descent_cli_options());
    all.add(inner_cli_options());
    all.add_options()
        ("help", "print this message")
        ("level", po::value<unsigned>()->default_value(6), "number of global refinements")
        ("n-eigen", po::value<unsigned>()->default_value(4), "number of eigenvalues to compute")
        ("tol-check", po::value<double>()->default_value(1e-6), "relative tolerance of the comparison");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, all), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << all << std::endl;
        return std::nullopt;
    }

    Config config;
    apply_gpe_options(vm, config.gpe);
    apply_descent_options(vm, config.descent);
    apply_inner_options(vm, config.solver);
    config.level   = vm["level"].as<unsigned>();
    config.n_eigen = vm["n-eigen"].as<unsigned>();
    config.tol     = vm["tol-check"].as<double>();

    if (config.gpe.beta != 0.0) {
        std::cerr << "note: --beta " << config.gpe.beta << " ignored, the check requires beta = 0\n";
    }
    config.gpe.beta = 0.0;
    return config;
}

struct Eigenpairs
{
    std::vector<double>         lambda;
    std::vector<Vector<double>> u;
};

//! @brief Eigenpairs of A0 u = lambda M u closest to zero, by ARPACK in shift-and-invert mode with shift 0.
//!
//! ARPACK finds the largest eigenvalues 1/lambda of A0^{-1} M, i.e. the smallest lambda. @p A_inv applies A0^{-1}.
template <int dim>
Eigenpairs smallest_eigenpairs(const GrossPitaevskiiSystem<dim>& system, const InverseOpType& A_inv, unsigned n_eigen)
{
    SolverControl control(system.n_dofs(), 1e-12, false, false);
    // ARPACK requires more than 2 * n_eigen + 1 Arnoldi vectors
    const ArpackSolver::AdditionalData data(2 * n_eigen + 2, ArpackSolver::largest_magnitude, /*symmetric=*/true);
    ArpackSolver eigensolver(control, data);

    std::vector<std::complex<double>> lambda(n_eigen);
    Eigenpairs pairs{{}, std::vector<Vector<double>>(n_eigen, Vector<double>(system.n_dofs()))};
    eigensolver.solve(system.get_A0(), system.get_M(), A_inv, lambda, pairs.u, n_eigen);

    for (const auto& l : lambda) {
        pairs.lambda.push_back(l.real());
    }
    return pairs;
}

//! |x^T M u| for u scaled to u^T M u = 1; equal to 1 iff x and u are parallel (for x^T M x = 1).
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
void check_linear_minimizer(CheckReport& report, const Config& config)
{
    auto builder = std::visit([&](auto&& V) {
        return ModelBuilder<GrossPitaevskiiSystem<dim>>(V, config.gpe, config.level);
    }, potential::get_potential<dim>(config.gpe.potential, config.gpe.potential_expr));
    auto& system = builder.get_system();
    GrossPitaevskiiFunctional<GrossPitaevskiiSystem<dim>> objective(system, 0.0, config.solver);

    // Minimize E on the unit mass sphere, starting from a constant
    Vector<double> x(system.n_dofs());
    x = 1.0;
    builder.distribute(x);
    ellipsoid::retract_by_norm(objective.get_M(), x);

    const UnitMassSphere<OperatorType> manifold(objective.get_M());
    EnergyOracle<GrossPitaevskiiSystem<dim>> oracle(objective, config.solver);
    GradientDescent solver(oracle, oracle.get_residual(), manifold, config.descent);
    ConvergenceTableObserver<CycleInfo> observer;
    solver.set_observer(observer);
    solver.cycle(x, std::cout);

    // For beta = 0, A = A0, so A_inv applies A0^{-1}; its accuracy limits that of ARPACK
    auto& A_inv = objective.get_A_inv();
    A_inv.set_tol(1e-13);
    const Eigenpairs pairs = smallest_eigenpairs<dim>(system, A_inv, config.n_eigen);
    const auto   i_min     = std::ranges::min_element(pairs.lambda) - pairs.lambda.begin();
    const double lambda_1  = pairs.lambda[i_min];

    Vector<double> A0x(x.size()), Mx(x.size());
    system.get_A0().vmult(A0x, x);
    system.get_M().vmult(Mx, x);
    const double energy   = objective.value(x);
    const double rayleigh = (x * A0x) / (x * Mx);
    const double err_energy   = std::abs(energy - 0.5 * lambda_1) / std::abs(0.5 * lambda_1);
    const double err_rayleigh = std::abs(rayleigh - lambda_1) / std::abs(lambda_1);

    std::string eigenvalues;
    for (double l : pairs.lambda) {
        eigenvalues += (eigenvalues.empty() ? "" : ", ") + std::to_string(l);
    }
    report.info("dim " + std::to_string(dim) + ", " + std::to_string(system.n_dofs()) + " DoFs, eigenvalues (ARPACK)",
                eigenvalues);
    report.check(err_energy < config.tol, "E(x) equals lambda_1 / 2", "rel. error " + sci(err_energy));
    report.check(err_rayleigh < config.tol, "x^T A0 x / x^T M x equals lambda_1", "rel. error " + sci(err_rayleigh));

    // Not checked: for a multiple eigenvalue lambda_1, the computed eigenvector is arbitrary
    report.info("|<x, u_1>_M| (1 unless lambda_1 is multiple)", sci(mass_overlap(system.get_M(), x, pairs.u[i_min])));
}

} // namespace
#endif // DEAL_II_WITH_ARPACK


int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
#ifndef DEAL_II_WITH_ARPACK
    std::cerr << "deal.II was built without ARPACK (DEAL_II_WITH_ARPACK undefined); skipping.\n";
    return skip_code;
#else
    std::optional<Config> config;
    try {
        config = parse_options(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    if (!config) {
        return 0;
    }

    return run_tests([&](CheckReport& report) {
        with_dimension(config->gpe.dimension, [&]<typename T0>(T0) {
            check_linear_minimizer<T0::value>(report, *config);
        });
    });
#endif
}
