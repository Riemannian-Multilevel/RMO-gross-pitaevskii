//
// Created by Ferdinand Vanmaele on 08.01.26.
//
#include "function.h"
#include "main.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <memory>
#include <vector>

using namespace dealii;
using namespace gpe;

template <int dim>
static void
prolongate_between_meshes(const DoFHandler<dim> &coarse,
                          const Vector<double> &x_coarse,
                          const DoFHandler<dim> &fine,
                          const AffineConstraints<double> &fine_constraints,
                          Vector<double> &y0_fine)
{
    y0_fine.reinit(fine.n_dofs());
    y0_fine = 0.0;

    VectorTools::interpolate_to_finer_mesh(coarse, x_coarse,
        fine, fine_constraints, y0_fine);

    // Good practice: enforce constraints explicitly
    fine_constraints.distribute(y0_fine);
}

// orthogonal projection on tangent space T_\phi S^{n-1}
template <int dim>
Vector<double> project()
{

}

// energy gradient \grad E(\phi)
template <int dim>
Vector<double> grad()
{

}

template <int dim>
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
    options.degree    = 1;
    options.radius    = 10;
    options.beta      = 100;
    options.bc        = BoundaryCondition::DIRICHLET;

    constexpr int dim = 2;
    Square<dim> V;

    using Exec = execution::seq_t;

}