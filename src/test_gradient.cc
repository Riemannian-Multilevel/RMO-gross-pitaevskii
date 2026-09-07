//
// Created by Ferdinand Vanmaele on 24.02.26.
//
#include "test_gradient.h"

#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>
#include <rmo/option.h>
#include <rmo/util/util.h>
#include <fstream>
#include <fmt/format.h>
#include <deal.II/base/convergence_table.h>

#include <limits>


using namespace rmo;
using namespace rmo::gpe;

// TODO: long double doesn't do much here, since F() is evaluated in double
// Finite difference: O(h) accurate
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


// Central difference: O(h^2) accurate
template <typename FuncType>
long double central_difference(const FuncType& F, const Vector<double>& x,
                               const Vector<double>& v, long double h)
{
    Vector<double> tmp(x);
    tmp.add(h, v);
    const long double F_plus = F(tmp);

    tmp = x;
    tmp.add(-h, v);
    const long double F_minus = F(tmp);

    return (F_plus - F_minus) / (2*h);
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

struct EmptyStrategy
{
    void operator()() {};
};


//! @brief Slope of the best-fitting linear piece of a curve, after Manopt's identify_linear_piece.
//!
//! Slides a window of @p window_len points over the (log t, log E) curve, fits a line to each by
//! least squares, and returns the slope of the window with the smallest residual. This picks out
//! the region where the Taylor error follows a clean power law, ignoring both the round-off noise
//! at small t and the loss of the asymptotic regime at large t.
inline double identify_linear_piece(const std::vector<double>& x, const std::vector<double>& y,
                                    unsigned window_len = 10)
{
    AssertDimension(x.size(), y.size());
    AssertThrow(x.size() > window_len,
        dealii::ExcMessage("not enough usable points to identify a linear piece"));

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
        residual = std::sqrt(residual);

        if (residual < best_residual) {
            best_residual = residual;
            best_slope    = slope;
        }
    }
    return best_slope;
}


// Test correctness of gradients for random samples
// TODO: additional data (h=1-8, n_trial_points=100, start_exp=-8)
//       include exact directional derivative
template <int dim>
CheckGradInfo check_gradient_trial(GradientTestBase<dim>& test_grad)
{
    CheckGradInfo check;
    const unsigned int n_dofs = test_grad.n_dofs();

    // 1. Generate a random point at x
    Vector<double> x(n_dofs);
    test_grad.random_point(x);
    test_grad.make_admissible(x);
    test_grad.assemble(x);    // initializes M_pp

    // Check x fulfills |x|_M = 1
    double x_constr = test_grad.constraint_value(x);
    check.x_constr = x_constr;
    
    // 2. Generate a random tangent vector v at x with |v|_x = 1
    Vector<double> v(n_dofs);
    test_grad.random_tangent_vector(x, v);

    test_grad.distribute(v);
    Vector<double> v_ambient(v);
    test_grad.to_tangent_space(x, v_ambient, v);  // constraining v moved it off T_x

    v /= std::sqrt(test_grad.metric(v, v));  // tangent vector with |v|_x = 1

    // Verify g_x(\grad(x), v) = DE(x)[v] for finite difference (directional derivative)
    // metric() and gradient() should match
    double fx             = test_grad.value(x);
    Vector<double> x_grad = test_grad.gradient(x);
    double g_xv           = test_grad.metric(x_grad, v);
    check.grad_xv         = g_xv;

    // --- FINITE DIFFERENCE CHECK ---
    // TODO: integrate this with value() (or remove the point argument)
    auto value_with_assembly = [&](const Vector<double>& z) {
        test_grad.assemble(z);         // Explicitly assemble for the trial point
        return test_grad.value(z);     // Evaluate
    };
    double dir_xv8 = finite_difference(value_with_assembly, x, v, 1e-8, fx);
    check.dir_xv = dir_xv8;

    // h ~ eps^(1/3) balances truncation against round-off for a central difference
    check.dir_xv_c = central_difference(value_with_assembly, x, v, 1e-5);

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

    // 5. E(t) = |E(R_x(tv)) - E(x) - t <grad E(x), v>_x| is O(t^2) exactly when the gradient is
    //    correct, so the log-log curve has slope 2 over the range where truncation dominates.
    std::vector<double> log_t, log_Et;
    for (unsigned i = 0; i < check.ts.size(); i++) {
        if (check.Ets[i] > 0.0) {   // exact cancellation would give log(0)
            log_t.push_back(std::log10(check.ts[i]));
            log_Et.push_back(std::log10(check.Ets[i]));
        }
    }
    check.slope = identify_linear_piece(log_t, log_Et);

    return check;
}


template <int dim, typename Strategy = EmptyStrategy>
void check_gradient(GradientTestBase<dim>& test_grad, unsigned n_trials, std::string prefix,
                    double slope_tol, Strategy&& setup_trial = {})
{
    dealii::ConvergenceTable convergence_table;

    double slope_min =  std::numeric_limits<double>::infinity();
    double slope_max = -std::numeric_limits<double>::infinity();

    for (unsigned int trial = 0; trial < n_trials; trial++) {
        setup_trial();  // initialization method for GradientTestBase (-> base points for coarse model)
        auto info = check_gradient_trial(test_grad);
        
        convergence_table.add_value("x_constr", info.x_constr);
        convergence_table.add_value("grad_xv",info.grad_xv);
        convergence_table.add_value("dir_xv",info.dir_xv);
        convergence_table.add_value("grad_res",info.grad_res);
        convergence_table.add_value("dir_xv_c",info.dir_xv_c);
        convergence_table.add_value("slope",info.slope);

        slope_min = std::min(slope_min, info.slope);
        slope_max = std::max(slope_max, info.slope);

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
    convergence_table.set_precision("dir_xv_c", 6);
    convergence_table.set_precision("slope", 4);

    convergence_table.set_scientific("x_constr", true);
    convergence_table.set_scientific("dir_xv", true);
    convergence_table.set_scientific("grad_xv", true);
    convergence_table.set_scientific("grad_res", true);
    convergence_table.set_scientific("dir_xv_c", true);

    convergence_table.write_text(std::cout, dealii::TableHandler::TextOutputFormat::table_with_headers);

    // E(t) is O(t^2) exactly when <grad E(x), v>_x is the directional derivative, so the slope
    // replaces the visual inspection of the log-log plots written above.
    std::cerr << fmt::format("{}: slope should be 2, observed in [{:.4f}, {:.4f}]\n",
                             prefix, slope_min, slope_max);

    // Only a slope below 2 is a failure. A slope above 2 means the second-order term happens to
    // vanish along v, which says nothing about the gradient.
    if (slope_tol > 0.0) {
        AssertThrow(slope_min > 2.0 - slope_tol,
            dealii::ExcMessage(fmt::format(
                "{}: smallest Taylor error slope {} is below {} -- "
                "gradient or directional derivative is likely wrong", prefix, slope_min, 2.0 - slope_tol)));
    }
}


int main(int argc, char* argv[])
{
    GPE_Options options{};
    unsigned n_levels = 0;
    unsigned n_trials = 0;
    double   slope_tol = 0.0;

    try {
        po::options_description all("Finite-difference check of the Riemannian gradients");
        all.add(gpe_cli_options());
        all.add_options()
            ("help", "print this message")
            ("level", po::value<unsigned>()->default_value(8),
                "number of global refinements")
            ("trials", po::value<unsigned>()->default_value(50),
                "random base points checked per gradient")
            ("slope-tol", po::value<double>()->default_value(0.05),
                "allowed deviation of the Taylor-error slope from 2 (0 disables the check)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << all << std::endl;
            return 0;
        }

        apply_gpe_options(vm, options);
        n_levels = vm["level"].as<unsigned>();
        n_trials  = vm["trials"].as<unsigned>();
        slope_tol = vm["slope-tol"].as<double>();
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    // Not exposed on the command line: the finite differences resolve the gradient, only if the
    // inner solves are far more accurate than the difference quotients they are compared against.
    SolverOptions options_slv{};
    options_slv.solver    = SolverMethod::CG;
    options_slv.max_inner = 2000;
    options_slv.precond   = Precondition::NONE;
    options_slv.tol_inner = 1e-12;

    with_dimension(options.dimension, [&]<typename T0>(T0)
    {
        constexpr int dim = T0::value;

        ModelBuilder<dim> builder(potential::Square<dim>(), options, n_levels);
        auto& system = builder.get_system();
        const unsigned n_dofs = builder.n_dofs();

        {
            std::cerr << "--- GRADIENT CHECK - ENERGY\n";
            GradientTestEnergy<dim> test_energy(system, options.beta, options_slv);
            check_gradient(test_energy, n_trials, fmt::format("checkgradient_energy_{}d", dim), slope_tol);
            std::cerr << "\n";
        }

        {
            std::cerr << "--- GRADIENT CHECK - MASS\n";
            GradientTestMass<dim> test_mass(system, options.beta, options_slv);
            check_gradient(test_mass, n_trials, fmt::format("checkgradient_mass_{}d", dim), slope_tol);
            std::cerr << "\n";
        }

        {
            std::cerr << "--- GRADIENT CHECK - FROBENIUS\n";
            GradientTestFrobenius<dim> test_frob(system, options.beta, options_slv);
            check_gradient(test_frob, n_trials, fmt::format("checkgradient_frob_{}d", dim), slope_tol);
            std::cerr << "\n";
        }

        {
            //std::cerr << "--- GRADIENT CHECK - COARSE (ENERGY)\n";
            // TODO
        }

        {
            std::cerr << "--- GRADIENT CHECK - COARSE (MASS)\n";
            GradientTestCoarseMass<dim> test_coarse_mass(system, options.beta, options_slv);
            GradientTestMass<dim> test_mass(system, options.beta, options_slv);

            Vector<double> w(n_dofs);  // fixed correction term
            w = 1.0;
            test_mass.distribute(w);

            auto setup_base_points_mass = [&w,&test_mass,&test_coarse_mass,n_dofs]()
        {
                Vector<double> phi(n_dofs); // random base point
                test_mass.random_point(phi);
                test_mass.make_admissible(phi);

                Vector<double> w_proj(n_dofs);
                ellipsoid::mass::project_onto_tangent_space(phi, test_mass.get_M(), w, w_proj);

                test_coarse_mass.update_parameters(w_proj, phi);
            };

            check_gradient(test_coarse_mass, n_trials,
                fmt::format("checkgradient_coarse_mass_{}d", dim), slope_tol, setup_base_points_mass);
        }

        {
            std::cerr << "--- GRADIENT CHECK - COARSE (FROBENIUS)\n";
            GradientTestCoarseFrobenius<dim> test_coarse_frob(system, options.beta, options_slv);
            GradientTestFrobenius<dim> test_frob(system, options.beta, options_slv);

            Vector<double> w(n_dofs);  // fixed correction term
            w = 1.0;
            test_frob.distribute(w);

            auto setup_base_points_frob = [&w, &test_frob, &test_coarse_frob, n_dofs]()
        {
                Vector<double> phi(n_dofs);
                test_frob.random_point(phi);
                test_frob.make_admissible(phi);

                // Project the ambient tilt vector w onto the F-metric tangent space
                Vector<double> w_proj(n_dofs);
                ellipsoid::frobenius::project_onto_tangent_space(phi, test_frob.get_M(), w, w_proj);

                test_coarse_frob.update_parameters(w_proj, phi);
            };

            check_gradient(test_coarse_frob, n_trials,
                fmt::format("checkgradient_coarse_frob_{}d", dim), slope_tol, setup_base_points_frob);
        }
    });

    return 0;
}
