#include <gpe/problem/oracle.h>
#include <gpe/main/model.h>
#include <gpe/util/util.h>

#include <gpe/ropt/manifold.h>
#include <gpe/ropt/descent.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <memory>
#include <iomanip>
#include <sstream>

using namespace dealii;
using namespace gpe;


template <int dim>
static void
prolongate_between_meshes(const ModelBuilder<dim>& coarse, const Vector<double>& x_coarse,
                          const ModelBuilder<dim>& fine, Vector<double>& y0_fine)
{
    y0_fine.reinit(fine.n_dofs());
    y0_fine = 0.0;

    VectorTools::interpolate_to_finer_mesh(
        coarse.get_package().get_dofs(), x_coarse,
        fine.get_package().get_dofs(),
        fine.get_package().get_constraints(), y0_fine);
}

int main()
{
    // Global timer
    TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

    // --- Hardcoded Options ---
    SolverOptions options_slv{};
    options_slv.max_inner    = 500;
    options_slv.solver       = SolverMethod::MINRES;
    options_slv.tol_inner    = 1e-6;

    DescentOptions options_gd{};
    options_gd.step_size    = 1.0;
    options_gd.tol_lambda   = 1e-8;
    options_gd.tol_residual = 1e-4;
    options_gd.max_iter     = 20;

    GPE_Options options{};
    options.dimension = 2;
    options.degree    = 1;
    options.radius    = 10;
    options.beta      = 100;
    options.bc        = BoundaryCondition::DIRICHLET;
    options.mesh_kind = MeshKind::QUADRILATERAL;
    options.order     = Ordering::CUTHILL_MCKEE;

    // Refinement-count hierarchy
    const unsigned int ref_min = 8;   // coarse
    const unsigned int ref_max = 11;  // fine

    try {
        // Dispatch to the correct dimension using your lambda wrapper
        with_dimension(options.dimension, [&]<typename T0>(T0)
        {
            constexpr int dim = T0::value;
            potential::Square<dim> V;

            // Adjust tolerances per level
            MGLevelObject<SolverOptions> options_slv_level(ref_min, ref_max);
            MGLevelObject<DescentOptions> options_gd_level(ref_min, ref_max);

            for (unsigned int ref = ref_min; ref <= ref_max; ++ref) {
                options_slv_level[ref] = options_slv;
                options_gd_level[ref]  = options_gd;

                if (ref < ref_max) {
                    double factor = std::pow(10, ref_max - ref);
                    // Lower precision for intermediate coarse grids
                    options_slv_level[ref].tol_inner = 1e-6 * factor;
                }
            }

            // 1) Setup + assemble each refinement using ModelBuilder
            MGLevelObject<std::unique_ptr<ModelBuilder<dim>>> builder(ref_min, ref_max);
            for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
            {
                std::cout << "---- ASSEMBLY REF " << ref << " ----\n";
                TimerOutput::Scope t(timer, "Assembly - ref " + std::to_string(ref));
                builder[ref] = std::make_unique<ModelBuilder<dim>>(V, options, ref);
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
                        UnitMassSphere<dim, OperatorType> manifold(gp_func.get_M());
                        EnergyOracle<dim> oracle(gp_func, options_slv_level[ref]);

                        x[ref] = gradient_descent(oracle, manifold, y0[ref], options_gd_level[ref], file);
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
                UnitMassSphere<dim, OperatorType> manifold(gp_func.get_M());
                EnergyOracle<dim> oracle(gp_func, options_slv);

                gradient_descent(oracle, manifold, y0_fine, options_gd, file);
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