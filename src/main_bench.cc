#include <rmo/gpe/oracle.h>
#include <rmo/gpe/model.h>
#include <rmo/gpe/manifold.h>

#include <rmo/fe/interpolate.h>
#include <rmo/ropt/solver.h>
#include <rmo/ropt/observer_table.h>

#include <rmo/option.h>
#include <rmo/util/util.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/mg_level_object.h>

#include <fstream>
#include <memory>
#include <iomanip>
#include <sstream>

using namespace dealii;
using namespace rmo;
using namespace rmo::gpe;


template <int dim>
static void
prolongate_between_meshes(const ModelBuilder<dim>& coarse, const Vector<double>& x_coarse,
                          const ModelBuilder<dim>& fine, Vector<double>& y0_fine)
{
    const fe::LinearTransfer<dim> transfer(coarse.get_package().get_dofs(),
                                           fine.get_package().get_dofs(),
                                           coarse.get_package().get_constraints(),
                                           fine.get_package().get_constraints());
    transfer.to_fine_mesh(x_coarse, y0_fine);
}

int main(int argc, char* argv[])
{
    // Global timer
    TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

    GPE_Options    options    {};
    DescentOptions options_gd {};
    SolverOptions  options_slv{};

    unsigned int ref_min = 0;  // coarse
    unsigned int ref_max = 0;  // fine

    try {
        po::options_description all("Cascadic nested iteration benchmark");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(inner_cli_options());
        all.add_options()
            ("help", "print this message")
            ("ref-min", po::value<unsigned>()->default_value(8),
                "coarsest refinement level of the cascade")
            ("ref-max", po::value<unsigned>()->default_value(11),
                "finest refinement level of the cascade");

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

        ref_min = vm["ref-min"].as<unsigned>();
        ref_max = vm["ref-max"].as<unsigned>();
        AssertThrow(ref_min <= ref_max,
            dealii::ExcMessage("--ref-min must not exceed --ref-max"));

        with_dimension(options.dimension, [&]<typename T0>(T0)
        {
            constexpr int dim = T0::value;
            auto potential_v = potential::get_potential<dim>(options.potential, options.potential_expr);

            // Adjust tolerances per level
            MGLevelObject<SolverOptions> options_slv_level(ref_min, ref_max);
            MGLevelObject<DescentOptions> options_gd_level(ref_min, ref_max);

            for (unsigned int ref = ref_min; ref <= ref_max; ++ref) {
                options_slv_level[ref] = options_slv;
                options_gd_level[ref]  = options_gd;

                if (ref < ref_max) {
                    // Lower precision for intermediate coarse grids
                    const double factor = std::pow(10, ref_max - ref);
                    options_slv_level[ref].tol_inner = options_slv.tol_inner * factor;
                }
            }

            // 1) Setup + assemble each refinement using ModelBuilder
            MGLevelObject<std::unique_ptr<ModelBuilder<dim>>> builder(ref_min, ref_max);
            for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
            {
                std::cout << "---- ASSEMBLY REF " << ref << " ----\n";
                TimerOutput::Scope t(timer, "Assembly - ref " + std::to_string(ref));
                builder[ref] = std::visit([&](auto&& V) {
                    return std::make_unique<ModelBuilder<dim>>(V, options, ref);
                }, potential_v);
            }

            // 2) Hierarchy of starting vectors and solution vectors
            MGLevelObject<Vector<double>> y0(ref_min, ref_max);
            MGLevelObject<Vector<double>> x (ref_min, ref_max);

            for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
            {
                y0[ref].reinit(builder[ref]->n_dofs());
                x [ref].reinit(builder[ref]->n_dofs());
            }

            {   // --- MULTIRESOLUTION (COARSE TO FINE) ---
                TimerOutput::Scope u(timer, "Solve - coarse to fine");
                TimerOutput timer_ref(std::cout, TimerOutput::summary, TimerOutput::wall_times);

                // Coarsest guess
                y0[ref_min] = 1.0;
                builder[ref_min]->distribute(y0[ref_min]);

                std::cout << "\n---- COARSE -> FINE (BY REFINEMENT COUNT) ----\n";
                const unsigned int width = std::to_string(ref_max).size();

                for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
                {
                    std::cout << "\nSOLVE REF " << ref << "\n";

                    std::ostringstream name;
                    name << "solve_ref_" << std::setw(width) << std::setfill('0') << ref << ".csv";
                    std::ofstream file(name.str());

                    {
                        TimerOutput::Scope t(timer_ref, "Solve - ref " + std::to_string(ref));

                        // New Architecture Pipeline
                        auto gp_func = builder[ref]->get_eval(options.beta, options_slv_level[ref]);
                        UnitMassSphere<OperatorType> manifold(gp_func.get_M());
                        EnergyOracle<dim> oracle(gp_func, options_slv_level[ref]);

                        GradientDescent solver(oracle, manifold, options_gd_level[ref]);
                        ConvergenceTableObserver conv_observer;
                        solver.set_observer(conv_observer);

                        x[ref] = y0[ref];  // cycle() updates the iterate in place
                        solver.cycle(x[ref], file);
                    }

                    // Prolongate to the next finer grid
                    if (ref < ref_max)
                    {
                        prolongate_between_meshes(*builder[ref], x[ref], *builder[ref + 1], y0[ref + 1]);
                        builder[ref + 1]->distribute(y0[ref + 1]); // Apply constraints on interpolated vector
                    }
                }
            }

            {   // --- COMPARISON TO GD ON MOST-REFINED LEVEL (COLD START) ---
                TimerOutput::Scope t(timer, "Solve - ref_max cold start");

                Vector<double> y0_fine(builder[ref_max]->n_dofs());
                y0_fine = 1.0;
                builder[ref_max]->distribute(y0_fine);

                std::cout << "\nSOLVE - fine (Cold Start)\n";
                std::ofstream file("solve_ref_max_cold.csv");

                auto gp_func = builder[ref_max]->get_eval(options.beta, options_slv_level[ref_max]);
                UnitMassSphere<OperatorType> manifold(gp_func.get_M());
                EnergyOracle<dim> oracle(gp_func, options_slv);

                GradientDescent solver(oracle, manifold, options_gd);
                ConvergenceTableObserver conv_observer;
                solver.set_observer(conv_observer);
                solver.cycle(y0_fine, file);  // cycle() updates the iterate in place
            }
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

    return 0;
}