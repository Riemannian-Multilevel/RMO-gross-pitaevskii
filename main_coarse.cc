#include "manifold.h"
#include "function.h"
#include "main.h"

using namespace dealii;
using namespace gpe;

int main()
{
    // --- options as before ---
    // TODO: command-line interface
    DescentOptions options_gd{};
    options_gd.step_size    = 1.0;
    options_gd.max_inner    = 500;
    options_gd.solver       = SolverMethod::MINRES;
    options_gd.tol_inner    = 1e-6;
    options_gd.tol_lambda   = 1e-8;
    options_gd.tol_residual = 1e-4;
    options_gd.max_iter     = 10;
    options_gd.line_search  = false;
    // options_gd.ls_alpha    = 1.0;
    // options_gd.ls_beta     = 0.6;
    // options_gd.ls_sigma    = 0.2;
    // options_gd.max_search  = 6;

    DescentOptions options_gd_coarse = options_gd;
    options_gd_coarse.max_iter    = 5;
    options_gd_coarse.step_size   = 1.0;
    options_gd_coarse.line_search = false; // TODO: armijo line search

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

    unsigned int n_coarse_levels = 8;
    unsigned int n_fine_levels = 9;
    EnergySimulator<dim> GP_coarse(V, options, n_coarse_levels);
    EnergySimulator<dim> GP_fine(V, options, n_fine_levels);

    Vector<double> y0(GP_fine.n_dofs());
    y0 = 1.0;  // starting value should be non-zero

    FullApproximationScheme<dim> FAS(GP_coarse, GP_fine, options.beta);
    FAS.cycle(y0, std::cout, options_gd, options_gd_coarse);
}

