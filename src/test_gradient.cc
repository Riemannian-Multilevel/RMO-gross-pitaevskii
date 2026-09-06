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


using namespace rmo;
using namespace rmo::gpe;

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

struct EmptyStrategy
{
    void operator()() {};
};


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
    test_grad.assemble(x);    // initializes M_pp

    // Check x fulfills |x|_M = 1
    double x_constr = test_grad.constraint_value(x);
    check.x_constr = x_constr;
    
    // 2. Generate a random tangent vector v at x with |v|_x = 1
    Vector<double> v(n_dofs);
    test_grad.random_tangent_vector(x, v);
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
void check_gradient(GradientTestBase<dim>& test_grad, unsigned n_trials, std::string prefix,
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


int main(int argc, char* argv[])
{
    GPE_Options options{};
    unsigned n_levels = 0;
    unsigned n_trials = 0;

    try {
        po::options_description all("Finite-difference check of the Riemannian gradients");
        all.add(gpe_cli_options());
        all.add_options()
            ("help", "print this message")
            ("level", po::value<unsigned>()->default_value(8),
                "number of global refinements")
            ("trials", po::value<unsigned>()->default_value(50),
                "random base points checked per gradient");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << all << std::endl;
            return 0;
        }

        apply_gpe_options(vm, options);
        n_levels = vm["level"].as<unsigned>();
        n_trials = vm["trials"].as<unsigned>();
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    // Not exposed on the command line: the finite differences resolve the gradient only if the
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
            check_gradient(test_energy, n_trials, fmt::format("checkgradient_energy_{}d", dim));
            std::cerr << "\n";
        }

        {
            std::cerr << "--- GRADIENT CHECK - MASS\n";
            GradientTestMass<dim> test_mass(system, options.beta, options_slv);
            check_gradient(test_mass, n_trials, fmt::format("checkgradient_mass_{}d", dim));
            std::cerr << "\n";
        }

        {
            std::cerr << "--- GRADIENT CHECK - FROBENIUS\n";
            GradientTestFrobenius<dim> test_frob(system, options.beta, options_slv);
            check_gradient(test_frob, n_trials, fmt::format("checkgradient_frob_{}d", dim));
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

            auto setup_base_points_mass = [&w,&test_mass,&test_coarse_mass,n_dofs]()
        {
                Vector<double> phi(n_dofs); // random base point
                test_mass.random_point(phi);

                Vector<double> w_proj(n_dofs);
                ellipsoid::mass::project_onto_tangent_space(phi, test_mass.get_M(), w, w_proj);

                test_coarse_mass.update_parameters(w_proj, phi);
            };

            check_gradient(test_coarse_mass, n_trials,
                fmt::format("checkgradient_coarse_mass_{}d", dim), setup_base_points_mass);
        }

        {
            std::cerr << "--- GRADIENT CHECK - COARSE (FROBENIUS)\n";
            GradientTestCoarseFrobenius<dim> test_coarse_frob(system, options.beta, options_slv);
            GradientTestFrobenius<dim> test_frob(system, options.beta, options_slv);

            Vector<double> w(n_dofs);  // fixed correction term
            w = 1.0;

            auto setup_base_points_frob = [&w, &test_frob, &test_coarse_frob, n_dofs]()
        {
                Vector<double> phi(n_dofs);
                test_frob.random_point(phi);

                // Project the ambient tilt vector w onto the F-metric tangent space
                Vector<double> w_proj(n_dofs);
                ellipsoid::frobenius::project_onto_tangent_space(phi, test_frob.get_M(), w, w_proj);

                test_coarse_frob.update_parameters(w_proj, phi);
            };

            check_gradient(test_coarse_frob, n_trials,
                fmt::format("checkgradient_coarse_frob_{}d", dim), setup_base_points_frob);
        }
    });

    return 0;
}
