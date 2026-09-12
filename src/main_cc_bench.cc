//
// CPU-time comparison between the single-level (main_cc.cc) and two-level
// (main_cc_coarse.cc) continuous-cuts drivers, across a range of problem sizes.
//
// Both solvers already report cumulative CPU time per iteration via CycleInfo::elapsed
// (a dealii::Timer restarted once at the start of the outermost cycle and read with
// cpu_time() on every evaluated iterate -- for the two-level driver this includes time
// spent inside coarse corrections, since the timer is not paused during the recursive
// call into FullApproximationScheme::cycle for the coarse level). RecordingObserver below
// captures that history directly instead of going through ConvergenceTableObserver's text
// tables, so it can be post-processed here: for each problem size, find the first
// iteration (of either solver) whose energy is at or below the single-level solver's own
// final (max_iter-iteration) energy, and report the CPU time spent to reach it. That is
// the actual claim a multilevel method makes -- less CPU time for the same solution
// quality -- as opposed to "fewer iterations", which does not account for the extra cost
// of each coarse correction (REVIEW-continuous-cuts.md \S4).
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
#include <rmo/ropt/solver.h>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace rmo;
using namespace rmo::cc;

namespace {

// Records the finest level's iteration history without any text formatting, so it can be
// post-processed instead of only printed (cf. ConvergenceTableObserver).
class RecordingObserver : public IterationObserver
{
public:
    explicit RecordingObserver(unsigned top_level) : m_top_level(top_level) {}

    void begin_level(unsigned level) override
    {
        if (level == m_top_level) history.clear();
    }

    void add(const CycleInfo& info) override
    {
        if (info.level == m_top_level) history.push_back(info);
    }

    std::vector<CycleInfo> history;

private:
    unsigned m_top_level;
};

std::vector<CycleInfo> run_single_level(unsigned n, unsigned max_iter)
{
    const SyntheticImage img(n);
    const PixelGrid grid(img.rows, img.cols);
    const ForwardDifference D(grid);

    ContinuousCutsFunctional functional(D, grid.to_dof_order(build_rho(img)), /*alpha=*/0.1, /*eps=*/1e-2);
    FisherRaoOracle oracle(functional);
    const BernoulliManifold manifold;

    Vector<double> phi(grid.n_dofs());
    phi = 0.5;

    DescentOptions options{};
    options.tol_residual = 1e-6;
    options.step_size    = 1.0;
    options.max_iter     = max_iter;
    options.line_search  = true;
    options.ls           = {50, 1.0, 0.5, 1e-4, 1e-12};

    GradientDescent solver(oracle, manifold, options);
    RecordingObserver observer(0);
    solver.set_observer(observer);
    solver.cycle(phi);

    return observer.history;
}

std::vector<CycleInfo> run_multilevel(unsigned n_fine, unsigned max_iter)
{
    constexpr unsigned int min_level = 0, max_level = 1;
    const std::vector<unsigned> level_indices{min_level, max_level};

    const unsigned n_coarse = (n_fine + 1) / 2;   // fine = 2*coarse - 1

    const SyntheticImage img_fine(n_fine);
    const PixelGrid grid_coarse(n_coarse, n_coarse);
    const PixelGrid grid_fine(img_fine.rows, img_fine.cols);

    const ForwardDifference D_coarse(grid_coarse);
    const ForwardDifference D_fine(grid_fine);

    auto grid_transfer    = std::make_shared<BernoulliGridTransfer>(grid_coarse, grid_fine);
    auto point_transfer   = std::make_shared<BernoulliPointTransfer>(
        *grid_transfer, BernoulliPointTransfer::Restriction::INJECTION);
    auto vector_transport = std::make_shared<GeometricBilinearTransport>(*grid_transfer);
    auto manifold         = std::make_shared<BernoulliManifold>();

    const Vector<double> rho_fine = grid_fine.to_dof_order(build_rho(img_fine));

    Vector<double> rho_coarse(grid_coarse.n_dofs());
    grid_transfer->Tfine(rho_fine, rho_coarse);
    rho_coarse *= 0.25;

    auto obj_coarse = std::make_shared<ContinuousCutsFunctional>(D_coarse, rho_coarse, /*alpha=*/0.4, /*eps=*/3.1622776601683795e-2);
    auto obj_fine   = std::make_shared<ContinuousCutsFunctional>(D_fine, rho_fine, /*alpha=*/0.1, /*eps=*/1e-2);

    dealii::MGLevelObject<std::shared_ptr<ManifoldBase>>                manifold_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<ManifoldTransferBase>>        point_transfer_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<VectorTransportBase>>         vector_transport_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<ContinuousCutsFunctional>>    objective_mg(min_level, max_level);
    dealii::MGLevelObject<DescentOptions>                               options_descent_mg(min_level, max_level);
    dealii::MGLevelObject<SolverOptions>                                options_solver_mg(min_level, max_level);

    manifold_mg[min_level] = manifold_mg[max_level] = manifold;
    point_transfer_mg[max_level]   = point_transfer;
    vector_transport_mg[max_level] = vector_transport;
    objective_mg[min_level] = obj_coarse;
    objective_mg[max_level] = obj_fine;

    DescentOptions options_fine{};
    options_fine.tol_residual = 1e-6;
    options_fine.step_size    = 1.0;
    options_fine.max_iter     = max_iter;
    options_fine.line_search  = true;
    options_fine.ls           = {50, 1.0, 0.5, 1e-4, 1e-12};

    DescentOptions options_coarse = options_fine;
    options_coarse.max_iter  = 10;
    options_coarse.ls.alpha  = 1.0;

    options_descent_mg[min_level] = options_coarse;
    options_descent_mg[max_level] = options_fine;
    options_solver_mg[min_level] = options_solver_mg[max_level] = SolverOptions{};

    FAS_Options options_fas{};
    options_fas.kappa                  = 0.6;
    options_fas.eps                    = 0.5;
    options_fas.coarse_every           = 2;
    options_fas.coarse_energy_adaptive = false;

    dealii::MGLevelObject<std::shared_ptr<CoarseConditionBase>> condition_mg(min_level, max_level);
    condition_mg[max_level] = std::make_shared<ScaledCoarseCondition>(2.0);

    FullApproximationScheme<ContinuousCutsFunctional> fas_solver(
        manifold_mg, point_transfer_mg, vector_transport_mg, objective_mg,
        level_indices, options_descent_mg, options_solver_mg, options_fas,
        std::nullopt, condition_mg);

    RecordingObserver observer(max_level);
    fas_solver.set_observer(observer);

    Vector<double> phi(grid_fine.n_dofs());
    phi = 0.5;

    FisherRaoOracle O_fine(*obj_fine);
    FisherRaoOracle T_fine(*obj_fine);

    fas_solver.cycle<FisherRaoOracle, ContinuousCutsCoarseOracle, ContinuousCutsCoarseOracle>(
        O_fine, T_fine, phi, level_indices.size() - 1);

    return observer.history;
}

// First entry whose energy is <= target (both trajectories are monotonically decreasing
// in practice, though not guaranteed by construction); returns history.back() if the
// target is never reached.
const CycleInfo& first_reaching(const std::vector<CycleInfo>& history, double target)
{
    for (const auto& info : history) {
        if (info.energy <= target) return info;
    }
    return history.back();
}

unsigned coarse_steps(const std::vector<CycleInfo>& history)
{
    unsigned n = 0;
    for (const auto& info : history) n += info.coarse ? 1 : 0;
    return n;
}

} // namespace

// Report, for one convergence target, the iteration and CPU time each solver first
// reaches it, and the CPU-time speedup ratio SL/ML (>1: multilevel wins).
void report_target(const char* label, const std::vector<CycleInfo>& sl, const std::vector<CycleInfo>& ml,
                   double target)
{
    const auto& sl_hit = first_reaching(sl, target);
    const auto& ml_hit = first_reaching(ml, target);
    const double speedup = sl_hit.elapsed / std::max(ml_hit.elapsed, 1e-12);

    std::cout << "  " << label << ": target E=" << target
              << "  SL it=" << sl_hit.iter << " cpu=" << sl_hit.elapsed
              << "  ML it=" << ml_hit.iter << " cpu=" << ml_hit.elapsed
              << "  speedup=" << speedup << "\n";
}

int main()
{
    const std::vector<unsigned> sizes{41, 81, 161, 321};
    constexpr unsigned max_iter = 300;
    constexpr unsigned n_repeat = 3;   // report the fastest of n_repeat runs per (size, solver)

    std::cout << std::scientific << std::setprecision(4);

    for (unsigned n : sizes) {
        std::vector<CycleInfo> sl, ml;
        for (unsigned r = 0; r < n_repeat; r++) {
            auto sl_r = run_single_level(n, max_iter);
            auto ml_r = run_multilevel(n, max_iter);
            if (sl.empty() || sl_r.back().elapsed < sl.back().elapsed) sl = std::move(sl_r);
            if (ml.empty() || ml_r.back().elapsed < ml.back().elapsed) ml = std::move(ml_r);
        }

        const double e0         = sl.front().energy;
        const double sl_final_e = sl.back().energy;
        // "Mostly converged": within 10% of the total energy drop of SL's own 300-iteration
        // run -- a scale-independent milestone reachable well before either solver's residual
        // plateau. "Fully converged": matches SL's own 300-iteration result outright.
        const double target_90   = e0 - 0.9 * (e0 - sl_final_e);
        const double target_full = sl_final_e;

        std::cout << "n_fine=" << n << " (n_dofs=" << n * n << "), best of " << n_repeat
                  << " runs, single-threaded (DEAL_II_NUM_THREADS=1 recommended):\n"
                  << "  SL: final E=" << sl_final_e << " cpu(300)=" << sl.back().elapsed << "\n"
                  << "  ML: final E=" << ml.back().energy << " cpu(300)=" << ml.back().elapsed
                  << " (" << coarse_steps(ml) << " coarse corrections)\n";
        report_target("90% converged", sl, ml, target_90);
        report_target("fully converged (=SL(300))", sl, ml, target_full);
        std::cout << "\n";
    }

    return 0;
}
