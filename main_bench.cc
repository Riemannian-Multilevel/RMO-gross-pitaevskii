#include "manifold.h"
#include "main.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <memory>

using namespace dealii;
using namespace gpe;

// TODO: use multigrid transfer/matrices
//       multiple runs with average time/stddev
template <int dim>
static void
prolongate_between_meshes(const EnergySimulator<dim> &coarse, const Vector<double> &x_coarse,
                          const EnergySimulator<dim> &fine, Vector<double> &y0_fine)
{
    y0_fine.reinit(fine.n_dofs());
    y0_fine = 0.0;

    VectorTools::interpolate_to_finer_mesh(coarse.get_dofs(),
        x_coarse, fine.get_dofs(),
        fine.get_constraints(), y0_fine);
}

int main()
{
    // Global timer
    TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

    // --- options as before ---
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
    options.degree    = 1;  // piecewise linear (1) or quadratic (2) elements
    options.radius    = 10;
    options.beta      = 100;
    options.bc        = BoundaryCondition::DIRICHLET;
    options.mesh_kind = MeshKind::QUADRILATERAL;
    options.order     = Ordering::CUTHILL_MCKEE;

    constexpr int dim = 2;
    Square<dim> V;

    // Refinement-count hierarchy
    const unsigned int ref_min = 8;   // coarse
    const unsigned int ref_max = 11;  // fine

    // Adjust tolerances per level
    MGLevelObject<SolverOptions> options_slv_level(ref_min, ref_max);
    MGLevelObject<DescentOptions> options_gd_level(ref_min, ref_max);

    for (unsigned int ref = ref_min; ref <= ref_max; ++ref) {
        options_slv_level[ref] = options_slv;
        options_gd_level[ref]  = options_gd;

        if (ref < ref_max) {
            double factor = std::pow(10,ref_max-ref);  // 1.0 for ref_max
            options_slv_level[ref].tol_inner  = 1e-6*factor;  // lower precision for multigrid
            //options_gd_level[ref].tol_lambda   = 1e-8/factor;
            //options_gd_level[ref].tol_residual = 1e-4/factor;
        }
        //dump_options(options_gd_level[ref], std::cerr);
    }

    // 1) One solver per refinement count
    MGLevelObject<std::unique_ptr<EnergySimulator<dim>>> solver(ref_min, ref_max);

    // for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
    //     solver[ref - ref_min] = std::make_unique<GPE<dim>>(options, ref);

    // 2) Setup + assemble each refinement
    for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
    {
        std::cout << "---- ASSEMBLY REF " << ref << " ----\n";
        TimerOutput::Scope t(timer, "Assembly - ref " + std::to_string(ref));

        solver[ref] = std::make_unique<EnergySimulator<dim>>(V, options, ref);
    }

    // 3) Hierarchy of starting vectors (indexed by refinement count!)
    MGLevelObject<Vector<double>> y0(ref_min, ref_max);
    MGLevelObject<Vector<double>> x (ref_min, ref_max);

    for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
    {
        y0[ref].reinit(solver[ref]->n_dofs());
        x [ref].reinit(solver[ref]->n_dofs());
    }

    {   // Multiresolution
        TimerOutput::Scope u(timer, "Solve - coarse to fine");
        // Local timer
        TimerOutput timer_ref(std::cout, TimerOutput::summary, TimerOutput::wall_times);

        // Coarsest guess
        y0[ref_min] = 1.0; // or 0.0

        // 4) Nested iteration: solve and prolongate to next refinement
        std::cout << "\n---- COARSE -> FINE (BY REFINEMENT COUNT) ----\n";

        const unsigned int width = std::to_string(ref_max).size();

        for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
        {
            std::cout << "\nSOLVE REF " << ref << "\n";

            std::ostringstream name;
            name << "solve_ref_"
                 << std::setw(width) << std::setfill('0') << ref
                 << ".csv";

            std::ofstream file(name.str());
            {
                TimerOutput::Scope t(timer_ref, "Solve - ref " + std::to_string(ref));
                x[ref] = solver[ref]->run(y0[ref], options.beta, options_slv_level[ref], options_gd_level[ref], file);
            }

            if (ref < ref_max)
            {
                prolongate_between_meshes<dim>(*solver[ref], x[ref],
                    *solver[ref + 1], y0[ref + 1]);
            }
        }
    }

    {   // Comparison to GD on most-refined level
        TimerOutput::Scope t(timer, "Solve - ref_max");

        Vector<double> y0_fine(solver[ref_max]->n_dofs());
        y0_fine = 1.0;

        std::cout << "\nSOLVE - fine\n";

        std::ofstream file("solve_ref_max.csv");
        {
            solver[ref_max]->run(y0_fine, options.beta, options_slv, options_gd, file);
        }
    }

}
