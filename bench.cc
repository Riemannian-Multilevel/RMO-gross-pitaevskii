#include "function.h"
#include "main.h"

#include <deal.II/base/timer.h>
#include <iostream>
#include <fmt/format.h>

using namespace gpe;
using namespace dealii;

// struct GPE_Options
// {
//     int dimension;          // dimension of domain
//     int n_levels;           // number of levels for global refinement
//     int degree;             // degree of shape functions
//     double radius;          // radius of the cube (square, line) domain
//     double beta;            // factor for the non-linear term in GPE
//     Ordering order;         // ordering for degrees of freedom
//     BoundaryCondition bc;   // problem boundary conditions (dirichlet or neumann)
// };

// struct GdOptions
// {
//     double tol_inner;     // relative tolerance for inner solver
//     double tol_lambda;    // tolerance for rayleigh quotients
//     double tol_residual;  // tolerance for M-residual
//     double step_size;     // fixed step-size used in iteration steps
//     int max_iter;         // maximum GD iterations
//     int max_inner;        // maximum sparse solver iterations
//     SolverMethod solver;  // method for solving sparse linear equations
// };

int main()
{
    TimerOutput timer (std::cout, TimerOutput::summary, TimerOutput::wall_times);

    GdOptions options_gd{};
    options_gd.tol_inner = 1e-6;
    options_gd.tol_lambda = 1e-8;
    options_gd.tol_residual = 1e-4;
    options_gd.step_size = 1.0;
    options_gd.max_iter = 20;
    options_gd.max_inner = 500;
    options_gd.solver = SolverMethod::MINRES;

    // TODO: compare matrix operators
    // * mesh size
    // * sequential assembly
    // * parallel assembly
    // * matrix-free assembly

    GPE_Options options{};
    options.dimension = 2;
    options.degree = 1;
    options.radius = 10;
    options.beta = 100;
    options.bc = BoundaryCondition::DIRICHLET;

    GPE_Options options_coarse(options);
    constexpr int dim = 2;

    Square<dim> V;
    int n_levels_fine = 10;
    int n_levels_coarse = 9;
    GPE_Solve<dim, execution::seq_t> solver_fine(options, n_levels_fine);
    GPE_Solve<dim, execution::seq_t> solver_coarse(options_coarse, n_levels_coarse);

    std::cout << "---- FINE ASSEMBLY ----" << std::endl;
    {
        TimerOutput::Scope timer_section(timer, "Assembly - fine");
        solver_fine.setup();
        solver_fine.assemble_matrix(V);
    }

    std::cout << std::endl;
    std::cout << "---- COARSE ASSEMBLY ----" << std::endl;
    {
        TimerOutput::Scope timer_section(timer, "Assembly - coarse");
        solver_coarse.setup();
        solver_coarse.assemble_matrix(V);
    }

    std::cout << std::endl;
    std::cout << "---- FINE SOLVE ----" << std::endl;
    {
        TimerOutput::Scope timer_section(timer, "Solve - fine");
        solver_fine.run(1.0, options.beta, options_gd, 1);
    }

    std::cout << std::endl;
    std::cout << "---- COARSE SOLVE ----" << std::endl;
    {
        TimerOutput::Scope timer_section(timer, "Solve - coarse");
        solver_coarse.run(1.0, options.beta, options_gd, 1);
    }

    // Multiresolution
    std::cout << std::endl;
    std::cout << "---- COARSE THEN FINE SOLVE" << std::endl;

    // LevelMatrix lm_fine, lm_coarse;
    // solver_fine.get_matrix(V, lm_fine); // for normalization
    // solver_coarse.get_matrix(V, lm_coarse);

    {
        TimerOutput::Scope timer_section(timer, "Solve - coarse then fine");
        auto x = solver_coarse.run(1.0, options.beta, options_gd, 1);
        auto y0 = Vector<double>(solver_fine.n_dofs());

        VectorTools::interpolate_to_finer_mesh(solver_coarse.get_dofs(), x,
            solver_fine.get_dofs(), solver_fine.get_constraints(), y0);

        // Vector<double> My0(y0.size());
        // lm_fine.M.vmult(My0, y0);
        // y0 /= std::sqrt(y0 * My0);

        std::cout << std::endl;
        auto y = solver_fine.run(y0, options.beta, options_gd, 1);
    }
}