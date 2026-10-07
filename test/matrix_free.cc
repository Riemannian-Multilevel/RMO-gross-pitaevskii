//
// Compares two evaluations of the Gross-Pitaevskii energy at random points: with the assembled sparse
// matrices (GrossPitaevskiiFunctional::value() after assembling Mpp(x)), and with the nonlinear term
// integrated by a loop over the cells. Checks that both agree, and reports the average time of each.
//
#include "check.h"

#include <rmo/gpe/manifold.h>
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>

#include <rmo/option.h>
#include <rmo/option_types.h>
#include <rmo/util/util.h>

#include <deal.II/base/timer.h>

#include <iostream>
#include <optional>
#include <string>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;
using namespace dealii;

namespace
{
constexpr double tol = 1e-10;  ///< largest admissible difference of the two energies

struct Config
{
    GPE_Options gpe{};
    unsigned    min_level = 0;  ///< coarsest refinement level
    unsigned    max_level = 0;  ///< finest refinement level
    unsigned    n_trials  = 0;  ///< random points per level
};

//! Configuration from the command line, or nothing for --help.
std::optional<Config> parse_options(int argc, char* argv[])
{
    po::options_description all("Cell-loop vs. sparse matrix-vector evaluation of the GP energy");
    all.add(gpe_cli_options());
    all.add_options()
        ("help", "print this message")
        ("min-level", po::value<unsigned>()->default_value(8), "coarsest refinement level")
        ("max-level", po::value<unsigned>()->default_value(11), "finest refinement level")
        ("trials", po::value<unsigned>()->default_value(200), "random points evaluated per level");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, all), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << all << std::endl;
        return std::nullopt;
    }

    Config config;
    apply_gpe_options(vm, config.gpe);
    config.min_level = vm["min-level"].as<unsigned>();
    config.max_level = vm["max-level"].as<unsigned>();
    config.n_trials  = vm["trials"].as<unsigned>();
    AssertThrow(config.min_level <= config.max_level, ExcMessage("--min-level must not exceed --max-level"));
    return config;
}

//! \f$ \int_\Omega x^4 \f$, by a loop over the cells instead of the assembled \f$ M_{\phi\phi}(x) \f$.
template <int dim>
double integrate_fourth_power(const DoFHandler<dim>& dofs, const Vector<double>& x)
{
    const QGauss<dim> quadrature(dofs.get_fe().degree + 1);
    FEValues<dim> fe_values(dofs.get_fe(), quadrature, update_values | update_JxW_values);
    std::vector<double> x_values(quadrature.size());
    double integral = 0.0;

    for (const auto& cell : dofs.active_cell_iterators()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(x, x_values);
        for (const unsigned q : fe_values.quadrature_point_indices()) {
            integral += std::pow(x_values[q], 4) * fe_values.JxW(q);
        }
    }
    return integral;
}

//! \f$ E(x) = \frac12 x^\top A_0 x + \frac\beta4 \int_\Omega x^4 \f$, see integrate_fourth_power().
template <int dim, typename MatrixType>
double energy_cell_loop(const DoFHandler<dim>& dofs, const MatrixType& A0, double beta, const Vector<double>& x)
{
    Vector<double> A0x(x.size());
    A0.vmult(A0x, x);
    return 0.5 * (x * A0x) + 0.25 * beta * integrate_fourth_power(dofs, x);
}

//! CPU time of @p f() in seconds.
template <typename Function>
double cpu_time(Function&& f)
{
    Timer timer;
    f();
    timer.stop();
    return timer.cpu_time();
}

template <int dim>
void check_level(CheckReport& report, const Config& config, unsigned level)
{
    ModelBuilder<GrossPitaevskiiSystem<dim>> model(potential::Square<dim>(), config.gpe, level);
    auto& system = model.get_system();
    const auto func = model.get_eval(config.gpe.beta, SolverOptions{});

    double time_matrix = 0.0, time_cell_loop = 0.0, max_diff = 0.0;
    for (unsigned trial = 0; trial < config.n_trials; trial++) {
        Vector<double> x(model.n_dofs());
        ellipsoid::random_point(x, model.get_M(), 0.0, 1.0);

        double value = 0.0, value_cell_loop = 0.0;
        time_matrix += cpu_time([&] {
            system.assemble_nonlinear_term(x);
            value = func.value(x);
        });
        time_cell_loop += cpu_time([&] {
            value_cell_loop = energy_cell_loop(model.get_dofs(), system.get_A0(), config.gpe.beta, x);
        });
        max_diff = std::max(max_diff, std::abs(value - value_cell_loop));
    }

    const std::string name = "level " + std::to_string(level);
    report.check(max_diff < tol, name + ": energy with sparse matrices equals the cell loop", "max diff " + sci(max_diff));
    report.info(name + ": average time", "sparse matrices " + sci(time_matrix / config.n_trials) + " s, cell loop "
                                         + sci(time_cell_loop / config.n_trials) + " s");
}

} // namespace


int main(int argc, char* argv[])
{
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
            for (unsigned level = config->min_level; level <= config->max_level; level++) {
                check_level<T0::value>(report, *config, level);
            }
        });
    });
}
