#include "manifold.h"
#include "oracle.h"
#include "main.h"

using namespace dealii;
using namespace gpe;

int main()
{
    // --- options as before ---
    // TODO: command-line interface
    DescentOptions options_gd{};   // outer solver options
    options_gd.step_size     = 1.0;
    options_gd.tol_lambda    = 1e-8;
    options_gd.tol_residual  = 1e-15;
    options_gd.max_iter      = 30;
    options_gd.line_search   = true;
    // options_gd.line_search   = false;

    options_gd.ls.alpha      = 1.0;
    options_gd.ls.beta       = 0.6;
    options_gd.ls.sigma      = 0.2;
    options_gd.ls.min        = 1e-1;   // threshold
    options_gd.ls.max_iter   = 3;

    SolverOptions  options_slv{};  // inner solver options
    options_slv.max_inner     = 500;
    options_slv.solver        = SolverMethod::CG;
    options_slv.precond       = Precondition::NONE;
    options_slv.tol_inner     = 1e-6;
    options_slv.tol_inner_res = 1e-2;

    SolverOptions options_slv_coarse = options_slv;
    // TODO: reduce tolerance for coarse level

    DescentOptions options_gd_coarse = options_gd;
    // TODO: heuristic: take steps UNTIL armijo line search fails OR max_iter encountered
    options_gd_coarse.max_iter    = 4;
    options_gd_coarse.step_size   = 1.0;
    options_gd_coarse.line_search = false;

    GPE_Options options{};
    options.dimension = 2;
    options.degree    = 1;  // piecewise linear (1) or quadratic (2) elements
    options.radius    = 10;
    options.beta      = 100;
    options.bc        = BoundaryCondition::DIRICHLET;
    options.mesh_kind = MeshKind::QUADRILATERAL;
    options.order     = Ordering::DEFAULT;

    constexpr int dim = 2;
    Square<dim> V;
    unsigned int n_coarse_levels = 8;
    unsigned int n_fine_levels = 9;

    GrossPitaevskiiSimulator<dim, EnergyOracle<dim>> GP_coarse(V, options, n_coarse_levels);
    const auto& problem_coarse = GP_coarse.get_problem();

    GrossPitaevskiiSimulator<dim, EnergyOracle<dim>> GP_fine(V, options, n_fine_levels);
    const auto& problem_fine = GP_fine.get_problem();

    LinearTransferMatrix<dim> transfer(GP_coarse.get_dofs(), GP_fine.get_dofs(),
        GP_coarse.get_constraints(), GP_fine.get_constraints());

    Vector<double> y0(GP_fine.n_dofs());
    y0 = 1.0;  // starting value should be non-zero

    // TODO: separate preconditioners for gradient descent, and inverse of M (coarse gradients)
    using OperatorType    = LinearCombination<SparseMatrix<double>, Vector<double>>;
    using InverseOpType   = PreconditionInverse<OperatorType, SparseMatrix<double>>;
    using SmoothOracle    = EnergyOracle<dim>;                          // for solutions on the fine level
    using CoarseOracle    = MassCoarseOracleEnergyAdaptive<dim>;        // for solutions on the coarse level
    // using CoarseOracle    = FrobeniusCoarseOracleEnergyAdaptive<dim>;
    using TiltOracle      = MassOracle<dim>;                            // for computing correction term w
    // using TiltOracle      = FrobeniusOracle<dim>;
    using VectorTransport = MassProjectionTransport<OperatorType>; // for transferring coarse directions
    // using VectorTransport = FrobeniusProjectionTransport<OperatorType>;
    using CoarseModel     = CoarseModel<dim, TiltOracle, CoarseOracle, VectorTransport>;

    FullApproximationScheme<dim, SmoothOracle, CoarseModel> FAS(
        problem_coarse,
        problem_fine,
        transfer,
        options.beta,
        options_slv,
        options_slv_coarse
    );

    //FAS.cycle(y0, std::cout, options_gd, options_gd_coarse);
    double kappa = 0.8;
    double eps = 1e-4;
    unsigned coarse_every = 2;
    FAS.cycle_condition(y0, std::cout, options_gd, options_gd_coarse, kappa, eps, coarse_every);
}

