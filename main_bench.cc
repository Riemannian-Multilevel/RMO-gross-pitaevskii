#include "function.h"
#include "main.h"
#include "option.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <memory>
#include <vector>

using namespace dealii;
using namespace gpe;

// TODO: use multigrid transfer/matrices
template <int dim>
static void
prolongate_between_meshes(const GPE<dim> &coarse,
                          const Vector<double> &x_coarse,
                          const GPE<dim> &fine,
                          Vector<double> &y0_fine)
{
    y0_fine.reinit(fine.n_dofs());
    y0_fine = 0.0;

    VectorTools::interpolate_to_finer_mesh(coarse.get_dofs(), x_coarse,
                                           fine.get_dofs(), fine.get_constraints(),
                                           y0_fine);
}

int main()
{
    TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

    // --- options as before ---
    GdOptions options_gd{};
    options_gd.tol_inner    = 1e-6;
    options_gd.tol_lambda   = 1e-8;
    options_gd.tol_residual = 1e-6;
    options_gd.step_size    = 1.0;
    options_gd.max_iter     = 20;
    options_gd.max_inner    = 500;
    options_gd.solver       = SolverMethod::MINRES;

    GPE_Options options{};
    options.dimension = 2;
    options.degree    = 1;  // piecewise linear (1) or quadratic (2) elements
    options.radius    = 10;
    options.beta      = 100;
    options.bc        = BoundaryCondition::DIRICHLET;

    constexpr int dim = 2;
    Square<dim> V;

    // Refinement-count hierarchy
    const unsigned int ref_min = 8;   // coarse
    const unsigned int ref_max = 11;  // fine

    // 1) One solver per refinement count
    // We'll store them in a vector, indexed by (ref - ref_min)
    std::vector<std::unique_ptr<GPE<dim>>> solver(ref_max - ref_min + 1);

    // for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
    //     solver[ref - ref_min] = std::make_unique<GPE<dim>>(options, ref);

    // 2) Setup + assemble each refinement
    for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
    {
        std::cout << "---- ASSEMBLY REF " << ref << " ----\n";
        TimerOutput::Scope t(timer, "Assembly - ref " + std::to_string(ref));

        solver[ref - ref_min] = std::make_unique<GPE<dim>>(options, ref);
        solver[ref - ref_min]->assemble(V);
    }

    // 3) Hierarchy of starting vectors (indexed by refinement count!)
    MGLevelObject<Vector<double>> y0(ref_min, ref_max);
    MGLevelObject<Vector<double>> x (ref_min, ref_max);

    for (unsigned int ref = ref_min; ref <= ref_max; ++ref)
    {
        y0[ref].reinit(solver[ref - ref_min]->n_dofs());
        x[ref].reinit(solver[ref - ref_min]->n_dofs());
    }

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
            TimerOutput::Scope t(timer, "Solve - ref " + std::to_string(ref));
            x[ref] = solver[ref - ref_min]->run(y0[ref], options.beta, options_gd, file);
        }

        if (ref < ref_max)
        {
            prolongate_between_meshes<dim>(*solver[ref - ref_min], x[ref],
                *solver[(ref + 1) - ref_min],
                y0[ref + 1]);
        }
    }
    return 0;
}
