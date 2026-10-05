//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/manifold.h>

#include <rmo/ropt/solver.h>
#include <rmo/ropt/observer_table.h>
#include <rmo/option.h>
#include <rmo/util/util.h>

#include <iostream>
#include <fmt/format.h>

using namespace rmo;
using namespace rmo::gpe;
using namespace dealii;


// Option groups of this program
struct ProgramOptions
{
    GPE_Options    gpe;
    DescentOptions descent;
    SolverOptions  solver;
    MG_Options     mg;
    OutputOptions  output;
};


// Gradient descent on each level of --levels, for the system type selected by --mass-lumping
template <typename System>
void solve(const ProgramOptions& opts)
{
    constexpr int dim = System::dimension;
    auto potential_v = potential::get_potential<dim>(opts.gpe.potential, opts.gpe.potential_expr);

    for (unsigned int level : opts.mg.v_levels) {
        // Set up the grid (Package) and finite element space
        auto context = std::visit([&](auto&& arg) {
            return ModelBuilder<System>(arg, opts.gpe, level);
        }, potential_v);

        // Set starting value, sufficiently far from an optimal solution
        Vector<double> x0(context.n_dofs());
        x0 = 1.0;
        context.distribute(x0);
        // auto M_norm = SpdNorm(context.get_M());
        // x0 /= M_norm(x0);

        // Define objective in ambient space
        auto gp = context.get_eval(opts.gpe.beta, opts.solver);
        // Define manifold
        auto manifold = UnitMassSphere<typename System::MassMatrix>(context.get_M());
        // Define Riemannian metric
        EnergyOracle<System> oracle(gp, opts.solver);

        // Termination criterion for gradient descent
        GradientDescent solver(oracle, manifold, opts.descent);
        ConvergenceTableObserver conv_observer;
        solver.set_observer(conv_observer);

        Vector<double> x(x0);
        solver.cycle(x, std::cout);

        if (opts.output.output_vtk) {
            const std::string name = opts.output.vtk_filename.empty() ? fmt::format("solution_{}d", dim) : opts.output.vtk_filename;
            output_vtk(solver.history(), context.get_package().get_dofs(), fmt::format("{}_lvl{}", name, level),
                       opts.output.output_every);
        }
    }
}


int main(int argc, char* argv[])
{
    ProgramOptions opts{};

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(mg_cli_options());
        all.add(inner_cli_options());
        all.add(output_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, opts.gpe);
        apply_descent_options(vm, opts.descent);
        apply_mg_options(vm, opts.mg);
        apply_inner_options(vm, opts.solver);
        apply_output_options(vm, opts.output);

        // TODO: use multiresolution if multilevel=true
        //       timer carried on across levels
        with_dimension(opts.gpe.dimension, [&]<typename T0>(T0)
        {
            with_system<T0::value>(opts.gpe.mass_lumping, [&]<typename System>()
            {
                solve<System>(opts);
            });
        });
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }
}
