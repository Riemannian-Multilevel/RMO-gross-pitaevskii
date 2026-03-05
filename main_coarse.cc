#include "manifold.h"
#include "main.h"

using namespace dealii;
using namespace gpe;

int main()
{
    // --- options as before ---
    DescentOptions options_gd{};
    options_gd.step_size    = 1.0;
    options_gd.max_inner    = 500;
    options_gd.solver       = SolverMethod::MINRES;
    options_gd.tol_inner    = 1e-6;
    options_gd.tol_lambda   = 1e-8;
    options_gd.tol_residual = 1e-4;
    options_gd.max_iter     = 1;
    options_gd.line_search  = false;

    DescentOptions options_gd_coarse = options_gd;
    options_gd_coarse.max_iter    = 5;
    options_gd_coarse.step_size   = 1.0;
    options_gd_coarse.line_search = false; // TODO: armijo line search
    options_gd_coarse.ls_alpha    = 1.0;
    options_gd_coarse.ls_beta     = 0.5;
    options_gd_coarse.ls_sigma    = 0.4;
    options_gd_coarse.max_search  = 20;

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

    EnergySimulator<dim> GP_coarse(V, options, 8);
    EnergySimulator<dim> GP_fine(V, options, 9);
    Vector<double> y0(GP_fine.n_dofs());
    y0 = 1.0;  // starting value should be non-zero
    unsigned int n_cycles = 10;

    EnergyOracle oracle_fine(GP_fine.get_problem(), options.beta);
    EnergyOracle oracle_coarse(GP_coarse.get_problem(), options.beta);

    full_approximation_scheme(oracle_coarse, oracle_fine,
        GP_coarse.get_package(), GP_fine.get_package(),
        GP_coarse.get_problem(), y0, options_gd, options_gd_coarse,
        n_cycles, std::cout);
}

