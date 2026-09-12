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
// experiment (Sec. 6.3.4): (alpha,eps) = (0.1,1e-4) fine, (0.4,1e-3) coarse; the
// coarse solver runs for 10 iterations per call, eta=0.6 (Fig. 15, Option 1).
//
// This still does not demonstrate a speedup over main_cc.cc's single-level
// baseline: final energy after 300 iterations is -94.01 here (was -93.99 with
// the earlier injected-image rho), against -112.23 in 147 iterations
// single-level, and the run-time behavior (grad_norm ~2.5 vs grad_restr_norm
// ~548 by the end -- the same ~220x gap as before) is essentially unchanged.
// So the earlier hypothesis that recomputing rho from an injected coarse image
// (rather than averaging it, as here) was contributing to the poor convergence
// was tested directly and ruled out: switching the rho construction changes the
// numbers but not the qualitative behavior. The scale mismatch between
// grad_norm and grad_restr_norm that eq. (16)'s trigger condition compares (see
// doc/plan_continuous_cuts.tex) is squarely a property of Option 1's vector
// transport (R = 4F_h^H), independent of how the coarse data term is built.
//
#include <rmo/cc/cc.h>
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

    auto obj_coarse = std::make_shared<ContinuousCutsFunctional>(D_coarse, rho_coarse, /*alpha=*/0.4, /*eps=*/1e-3);
    auto obj_fine   = std::make_shared<ContinuousCutsFunctional>(D_fine, rho_fine, /*alpha=*/0.1, /*eps=*/1e-4);

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
    options_coarse.max_iter  = 10;    // paper: coarse-level solver runs for 10 iterations per call
    options_coarse.ls.alpha  = 0.01;  // the coarse model q_k is unbounded below as z -> boundary
                                       // (its correction term is +/- w_k^T logit(z)), so a full
                                       // (alpha=1) first step can retract right up to the boundary
                                       // while still satisfying Armijo, in one shot -- damp the
                                       // starting step so backtracking gets a chance to engage

    options_descent_mg[min_level] = options_coarse;
    options_descent_mg[max_level] = options_fine;
    options_solver_mg[min_level] = options_solver_mg[max_level] = SolverOptions{};   // unused (no linear solve)

    FAS_Options options_fas{};
    options_fas.kappa                  = 0.6;    // eta in eq. (16); paper's Fig. 15 value for Option 1
    options_fas.eps                    = 0.5;    // mu in eq. (16)
    options_fas.coarse_every           = 1;
    options_fas.coarse_energy_adaptive = false;  // GP-specific, unused here

    FullApproximationScheme<ContinuousCutsFunctional> fas_solver(
        manifold_mg, point_transfer_mg, vector_transport_mg, objective_mg,
        level_indices, options_descent_mg, options_solver_mg, options_fas);

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
