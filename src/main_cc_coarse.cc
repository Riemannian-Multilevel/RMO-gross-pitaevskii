//
// Two-level adaptive Riemannian multilevel optimization for the continuous-cuts
// problem, wiring oracle_coarse.h through FullApproximationScheme. Uses the same
// synthetic disk image as main_cc.cc, at a 21x21 coarse / 41x41 fine resolution
// (one exact doubling apart, as interpolate.h requires).
//
// rho is computed once at the finest resolution and coarsened by averaging,
// matching the reference implementation's problem.py (which computes rho once
// and average-pools it down to each coarser level, rather than recomputing the
// color model at coarser resolution). "Averaging" here means full-weighting,
// F_h^H = (1/4) Tfine (\S 6.3.2) -- the framework's own averaging restriction --
// rather than a plain disjoint 2x2 block mean, since this grid's coarse pixel I
// coincides exactly with fine pixel 2I (one mesh-refinement step), not the
// center of a 2x2 block the way a plain image pyramid's would.
//
// Regularization parameters match the paper's two-level continuous-cuts
// experiment (Sec. 6.3.4): alpha = 0.1 fine / 0.4 coarse; the coarse solver
// runs for 10 iterations per call, eta = 0.6 (Fig. 15, Option 1). eps follows
// the reference's convention (eps_paper = sqrt(eps_ref), see the eps= comments
// below and CONTINUOUS_CUTS.md \S1's "eps convention"): 1e-2 fine, ~3.16e-2 coarse.
//
// This now reproduces the reference's speedup: with the reference's Armijo
// (coarse ls.alpha = 1, undamped) and its lockout on consecutive coarse
// corrections (coarse_every = 2), the driver reaches E = -109.62 after the
// first fine iteration -- what main_cc.cc's single-level baseline needs 16
// iterations to reach -- and E = -112.23 by iteration 300, matching
// single-level to 5 digits. The earlier committed state (coarse ls.alpha =
// 0.01, coarse_every = 1) instead stalled at E = -94.01 with every fine step
// forced to be a rejected coarse step (2464 "Step rejected" lines): those two
// parameters, not the vector transport's DC gain, were the cause -- see
// CONTINUOUS_CUTS.md \S3 (bugs 1-2) and \S4.1 for the corrected measurement,
// cross-checked against the Python reference.
//
#include <rmo/cc/cc.h>
#include <rmo/cc/condition.h>
#include <rmo/cc/grid.h>
#include <rmo/cc/interpolate.h>
#include <rmo/cc/manifold.h>
#include <rmo/cc/oracle.h>
#include <rmo/cc/oracle_coarse.h>
#include <rmo/cc/synthetic.h>
#include <rmo/cc/transport.h>

#include <rmo/ropt/fas.h>
#include <rmo/ropt/observer_table.h>

#include <iostream>

using namespace rmo;
using namespace rmo::cc;

int main()
{
    constexpr unsigned int min_level = 0, max_level = 1;   // 0 = coarse, 1 = fine
    const std::vector<unsigned> level_indices{min_level, max_level};

    const SyntheticImage img_fine(41);

    const PixelGrid grid_coarse(21, 21);
    const PixelGrid grid_fine(img_fine.rows, img_fine.cols);

    const ForwardDifference D_coarse(grid_coarse);
    const ForwardDifference D_fine(grid_fine);

    auto grid_transfer    = std::make_shared<BernoulliGridTransfer>(grid_coarse, grid_fine);
    auto point_transfer   = std::make_shared<BernoulliPointTransfer>(
        *grid_transfer, BernoulliPointTransfer::Restriction::INJECTION);   // Table 10, Option 1
    auto vector_transport = std::make_shared<GeometricBilinearTransport>(*grid_transfer);
    auto manifold         = std::make_shared<BernoulliManifold>();

    const Vector<double> rho_fine = grid_fine.to_dof_order(build_rho(img_fine));

    // rho_coarse = F_h^H rho_fine = (1/4) Tfine(rho_fine): average-pool rho itself down to
    // the coarse level (see the file comment above), rather than recomputing it from a
    // separately-built coarse image.
    Vector<double> rho_coarse(grid_coarse.n_dofs());
    grid_transfer->Tfine(rho_fine, rho_coarse);
    rho_coarse *= 0.25;

    // eps convention: cc.h squares eps under the root (eq. (42)'s eps^2), but the reference
    // (objective.py) adds its eps argument unsquared -- and \S6.3.4 states the paper's figures
    // were produced by the reference. So eps_paper = sqrt(eps_ref), with eps_ref = 1e-4 fine /
    // 1e-3 coarse being the reference's own numbers; passing eps_ref directly here (as before)
    // made the smoothing 100x sharper than any run the paper reports (CONTINUOUS_CUTS.md \S1).
    auto obj_coarse = std::make_shared<ContinuousCutsFunctional>(D_coarse, rho_coarse, /*alpha=*/0.4, /*eps=*/3.1622776601683795e-2);
    auto obj_fine   = std::make_shared<ContinuousCutsFunctional>(D_fine, rho_fine, /*alpha=*/0.1, /*eps=*/1e-2);

    dealii::MGLevelObject<std::shared_ptr<ManifoldBase>>                manifold_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<ManifoldTransferBase>>        point_transfer_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<VectorTransportBase>>         vector_transport_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<ContinuousCutsFunctional>>    objective_mg(min_level, max_level);
    dealii::MGLevelObject<DescentOptions>                               options_descent_mg(min_level, max_level);
    dealii::MGLevelObject<SolverOptions>                                options_solver_mg(min_level, max_level);

    manifold_mg[min_level] = manifold_mg[max_level] = manifold;   // stateless, shared across levels
    point_transfer_mg[max_level]   = point_transfer;              // only the fine level's entry is ever read
    vector_transport_mg[max_level] = vector_transport;
    objective_mg[min_level] = obj_coarse;
    objective_mg[max_level] = obj_fine;

    DescentOptions options_fine{};
    options_fine.tol_residual = 1e-6;
    options_fine.step_size    = 1.0;
    options_fine.max_iter     = 300;
    options_fine.line_search  = true;
    options_fine.ls           = {50, 1.0, 0.5, 1e-4, 1e-12};

    DescentOptions options_coarse = options_fine;
    options_coarse.max_iter  = 5;    // paper/reference: 10 iterations per call (Sec. 6.3.4); halved
                                      // here since the CPU-time benchmark (main_cc_bench.cc,
                                      // CONTINUOUS_CUTS.md \S4.1) found the fixed 10-iteration
                                      // coarse solve costs about as much as the fine iterations it
                                      // replaces on this problem, wiping out the iteration-count
                                      // speedup once priced in CPU time -- see \S4.1 for whether 5
                                      // iterations changes that.
    options_coarse.ls.alpha  = 1.0;  // reference (optimizer.py:8): Armijo starts at alpha=1 on
                                      // every level and never damps. The coarse model q_k is
                                      // indeed unbounded below as z -> boundary (its correction
                                      // term is +/- w_k^T logit(z)), so a full first step can
                                      // retract close to it -- but the reference lives with this
                                      // by clamping iterates (optimizer.py:15,49) rather than
                                      // damping alpha, and this port's own MIN_WEIGHT clamp
                                      // (metric.h) does the same job. Damping to 0.01 instead
                                      // left the coarse model unminimised (10 steps of exactly
                                      // 0.01) and was the main cause of the "no speedup" result
                                      // this driver used to report (CONTINUOUS_CUTS.md \S3,
                                      // bugs 1-2, and \S4.1).

    options_descent_mg[min_level] = options_coarse;
    options_descent_mg[max_level] = options_fine;
    options_solver_mg[min_level] = options_solver_mg[max_level] = SolverOptions{};   // unused (no linear solve)

    FAS_Options options_fas{};
    options_fas.kappa                  = 0.6;    // eta in eq. (16); paper's Fig. 15 value for Option 1
    options_fas.eps                    = 0.5;    // mu in eq. (16); bounds the coarse norm, see
                                                  // cc::ScaledCoarseCondition below
    options_fas.coarse_every           = 2;      // paper \S6.4 / reference multilevel.py:148
                                                  // "lockouts": consecutive coarse corrections are
                                                  // not permitted -- every coarse correction is
                                                  // preceded by a fine-level gradient step. With
                                                  // coarse_every = 1 a rejected coarse step was
                                                  // followed by another rejected coarse step
                                                  // forever once the trigger fired, freezing the
                                                  // driver (CONTINUOUS_CUTS.md \S3, bug 2).
    options_fas.coarse_energy_adaptive = false;  // GP-specific, unused here

    // grid_scale = 2 is a reference-side deviation (operators.py, get_grid_scale("Option 1",
    // n_pools=1)) absent from the paper's eq. (16); it compensates for one factor-2 step's
    // norm contraction under R (see cc/condition.h / CONTINUOUS_CUTS.md \S1).
    dealii::MGLevelObject<std::shared_ptr<CoarseConditionBase>> condition_mg(min_level, max_level);
    condition_mg[max_level] = std::make_shared<ScaledCoarseCondition>(2.0);

    FullApproximationScheme<ContinuousCutsFunctional> fas_solver(
        manifold_mg, point_transfer_mg, vector_transport_mg, objective_mg,
        level_indices, options_descent_mg, options_solver_mg, options_fas,
        std::nullopt, condition_mg);

    ConvergenceTableObserver observer(min_level, max_level);
    fas_solver.set_observer(observer);

    Vector<double> phi(grid_fine.n_dofs());
    phi = 0.5;   // uniform initialization

    FisherRaoOracle O_fine(*obj_fine);
    FisherRaoOracle T_fine(*obj_fine);

    fas_solver.cycle<FisherRaoOracle, ContinuousCutsCoarseOracle, ContinuousCutsCoarseOracle>(
        O_fine, T_fine, phi, level_indices.size() - 1, std::cout);

    return 0;
}
