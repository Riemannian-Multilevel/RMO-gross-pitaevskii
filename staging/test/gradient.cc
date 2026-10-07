/**
 * @file
 * @brief Taylor test of the Riemannian gradient kernels: the energy in the energy-adaptive, M- and F-metric, and the
 * coarse models in their own metric and in the energy-adaptive metric.
 *
 * At random points \f$ x \f$ and unit tangent vectors \f$ v \f$, the Taylor error
 * \f$ |f(R_x(t v)) - f(x) - t \langle \grad f(x), v \rangle_x| \f$ is \f$ O(t^2) \f$ exactly when the gradient is
 * correct, so its slope in log-log scale must be 2 (checked with `--slope-tol`). The table of the trials also lists
 * the normal part of the gradient and difference quotients of \f$ f \f$ along \f$ v \f$.
 *
 * The Taylor errors are written to `checkgradient_<problem>_<dim>d_<trial>.dat` (columns \f$ t \f$, error; without
 * `_<trial>` for a single trial).
 */
#include "check.h"
#include "finite_difference.h"
#include "gradient_problems.h"

#include <rmo/gpe/model.h>
#include <rmo/option.h>
#include <rmo/util/util.h>

#include <deal.II/base/convergence_table.h>

#include <fmt/format.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::gpe::test;
using namespace rmo::test;

namespace
{

struct Config
{
    GPE_Options gpe{};
    unsigned    level     = 0;    ///< number of global refinements
    unsigned    n_trials  = 0;    ///< random points per problem
    double      slope_tol = 0.0;  ///< allowed deviation of the Taylor error slope from 2; 0 disables the check
};

//! Configuration from the command line, or nothing for --help.
std::optional<Config> parse_options(int argc, char* argv[])
{
    po::options_description all("Finite-difference check of the Riemannian gradients");
    all.add(gpe_cli_options());
    all.add_options()
        ("help", "print this message")
        ("level", po::value<unsigned>()->default_value(8), "number of global refinements")
        ("trials", po::value<unsigned>()->default_value(50), "random base points checked per gradient")
        ("slope-tol", po::value<double>()->default_value(0.05),
            "allowed deviation of the Taylor-error slope from 2 (0 disables the check)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, all), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << all << std::endl;
        return std::nullopt;
    }

    Config config;
    apply_gpe_options(vm, config.gpe);
    config.level     = vm["level"].as<unsigned>();
    config.n_trials  = vm["trials"].as<unsigned>();
    config.slope_tol = vm["slope-tol"].as<double>();
    return config;
}

//! Not on the command line: the difference quotients resolve the gradient only if the inner solves are far
//! more accurate than them.
SolverOptions solver_options()
{
    SolverOptions options{};
    options.solver    = SolverMethod::CG;
    options.max_inner = 2000;
    options.precond   = Precondition::NONE;
    options.tol_inner = 1e-12;
    return options;
}

//! @p n values from \f$ 10^{a} \f$ to \f$ 10^{b} \f$, equally spaced in log scale.
std::vector<double> logspace(double a, double b, unsigned n)
{
    std::vector<double> values;
    for (unsigned i = 0; i < n; ++i) {
        values.push_back(std::pow(10.0, a + i * (b - a) / (n - 1)));
    }
    return values;
}

//! @brief Slope of the most linear piece of the curve (x, y): fits a line by least squares to each window of
//! @p window_len consecutive points, and returns the slope of the window with the smallest residual.
double linear_piece_slope(const std::vector<double>& x, const std::vector<double>& y, unsigned window_len = 10)
{
    AssertDimension(x.size(), y.size());
    AssertThrow(x.size() > window_len, dealii::ExcMessage("not enough points to identify a linear piece"));

    double best_residual = std::numeric_limits<double>::infinity();
    double best_slope    = std::numeric_limits<double>::quiet_NaN();

    for (unsigned i = 0; i + window_len <= x.size(); i++) {
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (unsigned k = i; k < i + window_len; k++) {
            sx += x[k];  sy += y[k];  sxx += x[k]*x[k];  sxy += x[k]*y[k];
        }
        const double n   = window_len;
        const double den = n*sxx - sx*sx;
        if (std::abs(den) < 1e-300) {
            continue;
        }
        const double slope     = (n*sxy - sx*sy) / den;
        const double intercept = (sy - slope*sx) / n;

        double residual = 0.0;
        for (unsigned k = i; k < i + window_len; k++) {
            const double r = y[k] - (slope*x[k] + intercept);
            residual += r*r;
        }
        if (residual < best_residual) {
            best_residual = residual;
            best_slope    = slope;
        }
    }
    return best_slope;
}

//! Slope of the Taylor error in log-log scale, between the round-off at small t and the end of the
//! asymptotic range at large t.
double taylor_slope(const std::vector<double>& t, const std::vector<double>& error)
{
    std::vector<double> log_t, log_error;
    for (unsigned i = 0; i < t.size(); i++) {
        if (error[i] > 0.0) {  // exact cancellation would give log(0)
            log_t.push_back(std::log10(t[i]));
            log_error.push_back(std::log10(error[i]));
        }
    }
    return linear_piece_slope(log_t, log_error);
}

struct TrialResult
{
    double grad_v;       ///< <grad f(x), v>_x
    double fd_central;   ///< Df(x)[v] by a central difference
    double normal_part;  ///< |grad f(x) - Pi_x grad f(x)|_x / |grad f(x)|_x, not seen by the Taylor error
    double slope;        ///< slope of the Taylor error, 2 for a correct gradient
    std::vector<double> t, taylor_error;
};

/** @brief One trial of the gradient check: a random point and tangent vector, and the quantities of TrialResult. */
template <int dim>
class GradientCheck
{
public:
    GradientCheck(ConstrainedSphere<dim>& sphere, GradientProblem& problem) : sphere(sphere), problem(problem) {}

    TrialResult trial()
    {
        const unsigned n = sphere.n_dofs();
        TrialResult r{};
        problem.sample_parameters();

        Vector<double> x(n);
        problem.random_point(x);
        sphere.make_admissible(x);
        sphere.update(x);

        // Tangent vector with |v|_x = 1; the constraints move v off the tangent space, so project afterwards
        Vector<double> v_ambient(n), v(n);
        problem.random_tangent_vector(x, v_ambient);
        sphere.distribute(v_ambient);
        problem.project(x, v_ambient, v);
        v /= problem.metric().norm(v);

        const double fx = problem.value(x);
        Vector<double> g(n);
        problem.gradient(x, g);
        r.grad_v = problem.metric().inner(g, v);

        // h ~ eps^(1/3) balances truncation and round-off
        r.fd_central = central_difference([this](const auto& z) { return value(z); }, x, v, 1e-5);
        sphere.update(x);

        Vector<double> g_tangent(n), g_normal(g);
        problem.project(x, g, g_tangent);
        g_normal -= g_tangent;
        r.normal_part = problem.metric().norm(g_normal) / problem.metric().norm(g);

        r.t = logspace(-8, 0, 100);
        Vector<double> tv(n), x_t(n);
        for (const double t : r.t) {
            tv.equ(t, v);
            sphere.retract(x, tv, x_t);
            r.taylor_error.push_back(std::abs(value(x_t) - fx - t * r.grad_v));
        }
        r.slope = taylor_slope(r.t, r.taylor_error);
        return r;
    }

private:
    //! Value at @p z, after updating the operators to @p z.
    double value(const Vector<double>& z)
    {
        sphere.update(z);
        return problem.value(z);
    }

    ConstrainedSphere<dim>& sphere;
    GradientProblem& problem;
};

/** @brief Table of the trials, and the Taylor error of each trial in `<prefix>_<trial>.dat` (`<prefix>.dat` for one). */
class TrialLog
{
public:
    TrialLog(std::string prefix, unsigned n_trials) : prefix(std::move(prefix)), n_trials(n_trials) {}

    void add(unsigned trial, const TrialResult& r)
    {
        table.add_value("<grad, v>", r.grad_v);
        table.add_value("fd_central", r.fd_central);
        table.add_value("normal_part", r.normal_part);
        table.add_value("slope", r.slope);

        std::ofstream out(n_trials > 1 ? fmt::format("{}_{:03}.dat", prefix, trial) : prefix + ".dat");
        for (unsigned i = 0; i < r.t.size(); i++) {
            out << r.t[i] << "\t" << r.taylor_error[i] << "\n";
        }
    }

    void write_table(std::ostream& os)
    {
        for (const auto* column : {"<grad, v>", "fd_central", "normal_part"}) {
            table.set_precision(column, 6);
            table.set_scientific(column, true);
        }
        table.set_precision("slope", 4);
        table.write_text(os, dealii::TableHandler::TextOutputFormat::table_with_headers);
    }

private:
    const std::string prefix;
    const unsigned n_trials;
    dealii::ConvergenceTable table;
};

template <int dim>
void check_problem(CheckReport& report, ConstrainedSphere<dim>& sphere, GradientProblem&& problem,
                   const std::string& name, const Config& config)
{
    GradientCheck<dim> check(sphere, problem);
    TrialLog log(fmt::format("checkgradient_{}_{}d", name, dim), config.n_trials);
    double slope_min =  std::numeric_limits<double>::infinity();
    double slope_max = -std::numeric_limits<double>::infinity();
    double max_normal = 0.0;

    for (unsigned trial = 0; trial < config.n_trials; trial++) {
        const TrialResult r = check.trial();
        log.add(trial, r);
        slope_min = std::min(slope_min, r.slope);
        slope_max = std::max(slope_max, r.slope);
        max_normal = std::max(max_normal, r.normal_part);
    }
    std::cout << "\n" << name << "\n";
    log.write_table(std::cout);

    report.check(max_normal < 1e-8, name + ": gradient is tangent", "max |normal part| / |grad| " + sci(max_normal));

    // Only a slope below 2 fails: above 2, the second-order term happens to vanish along v
    const std::string detail = fmt::format("slope in [{:.4f}, {:.4f}]", slope_min, slope_max);
    if (config.slope_tol > 0.0) {
        report.check(slope_min > 2.0 - config.slope_tol, name + ": Taylor error is O(t^2)", detail);
    } else {
        report.info(name + ": Taylor error", detail);
    }
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
            constexpr int dim = T0::value;
            ModelBuilder<GrossPitaevskiiSystem<dim>> builder(potential::Square<dim>(), config->gpe, config->level);
            ConstrainedSphere<dim> sphere(builder.get_system(), config->gpe.beta, solver_options());

            // Fixed vector for the corrections of the coarse models
            Vector<double> w(sphere.n_dofs());
            w = 1.0;
            sphere.distribute(w);

            // TODO: check first-order coherence
            check_problem(report, sphere, FineProblem<dim, EnergyAdaptive>(sphere), "energy", *config);
            check_problem(report, sphere, FineProblem<dim, Mass>(sphere), "mass", *config);
            check_problem(report, sphere, FineProblem<dim, Frobenius>(sphere), "frob", *config);
            check_problem(report, sphere, CoarseProblem<dim, MassModel, Mass>(sphere, w), "coarse_mass", *config);
            check_problem(report, sphere, CoarseProblem<dim, FrobeniusModel, Frobenius>(sphere, w),
                          "coarse_frob", *config);
            check_problem(report, sphere, CoarseProblem<dim, MassModel, EnergyAdaptive>(sphere, w),
                          "coarse_mass_energy", *config);
            check_problem(report, sphere, CoarseProblem<dim, FrobeniusModel, EnergyAdaptive>(sphere, w),
                          "coarse_frob_energy", *config);
        });
    });
}
