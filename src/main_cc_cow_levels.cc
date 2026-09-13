//
// 2-, 3-, and 4-level continuous cuts on the paper's cow-image problem, replicating
// examples/levels_2_3_4.py's own "best configurations" (RMO-continuous-cuts): the same
// finest-level objective throughout, with progressively deeper coarse hierarchies inserted
// between it and the same 240x320 (2-level) / 120x160 (4-level) coarsest levels. Single-
// level is run once and shared as the baseline for all three CPU-time comparisons.
//
// The 2-level hierarchy composes two factor-2 average-pool steps into one coarse-model
// transition (n_pools=2, as main_cc_cow.cc), since that is the reference's own "2-level"
// definition (960x1280 -> 240x320 directly, no intermediate solve). The 3- and 4-level
// hierarchies instead insert genuine intermediate levels -- 480x640 (3-level), and both
// 480x640 and 120x160 (4-level) -- each one factor-2 step from its neighbor
// (cumulative_to_delta_pools([2,1]) == [1,1], [3,2,1] == [1,1,1] in the reference), so they
// use plain BernoulliGridTransfer at each transition and FullApproximationScheme's ordinary
// N-level recursion, no ComposedGridTransfer needed.
//
// All three multilevel configurations use the min_fine_norm floor added to
// cc::ScaledCoarseCondition (see condition.h and CONTINUOUS_CUTS.md \S4.2-\S4.3): on
// the cow image the scaled restricted gradient stays orders of magnitude above the fine
// gradient for the entire run, so eq. (16)'s own gate never shuts corrections off on its
// own once the fine level is well past the point where a correction still helps.
//
// rho at each resolution is precomputed by test/cc/prepare_cow_data.py (run that first).
//
#include <rmo/cc/cc.h>
#include <rmo/cc/condition.h>
#include <rmo/cc/grid.h>
#include <rmo/cc/interpolate.h>
#include <rmo/cc/manifold.h>
#include <rmo/cc/oracle.h>
#include <rmo/cc/oracle_coarse.h>
#include <rmo/cc/rawio.h>
#include <rmo/cc/transport.h>

#include <rmo/ropt/fas.h>
#include <rmo/ropt/solver.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace rmo;
using namespace rmo::cc;

namespace {

std::string data_dir()
{
    if (const char* env = std::getenv("CC_COW_DATA_DIR")) return env;
    return "test/cc/data";
}

class RecordingObserver : public IterationObserver
{
public:
    explicit RecordingObserver(unsigned top_level) : m_top_level(top_level) {}
    void begin_level(unsigned level) override { if (level == m_top_level) history.clear(); }
    void add(const CycleInfo& info) override { if (info.level == m_top_level) history.push_back(info); }
    std::vector<CycleInfo> history;
private:
    unsigned m_top_level;
};

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

// eps_paper = sqrt(eps_ref), this port's established convention (REVIEW \S3.3).
double eps_of(double eps_ref) { return std::sqrt(eps_ref); }

struct LevelSpec {
    unsigned rows, cols;
    double alpha, eps_ref;   // eps_ref: the reference's own (unsquared) eps at this level
    unsigned coarse_max_iter;   // ignored for the finest entry (uses top_max_iter instead)
};

// Builds and runs an N-level FAS hierarchy (specs ordered coarsest..finest, one rho per
// spec in the same order), sharing this file's min_fine_norm floor at every transition.
// Every adjacent pair here is one factor-2 step apart (delta_pools=1), so grid_scale=2 at
// every transition (2^1, get_grid_scale("Option 1", 1) -- not the 2-level driver's 4).
std::vector<CycleInfo> run_n_level(const std::vector<LevelSpec>& specs,
                                   const std::vector<Vector<double>>& rhos,
                                   unsigned top_max_iter, double min_fine_norm)
{
    const unsigned n = static_cast<unsigned>(specs.size());
    Assert(n >= 2, dealii::ExcMessage("need at least 2 levels"));
    AssertDimension(rhos.size(), n);
    const unsigned min_level = 0, max_level = n - 1;

    std::vector<PixelGrid> grids;
    grids.reserve(n);
    for (const auto& s : specs) grids.emplace_back(s.rows, s.cols);

    std::vector<ForwardDifference> Ds;
    Ds.reserve(n);
    for (const auto& g : grids) Ds.emplace_back(g);

    std::vector<std::shared_ptr<ContinuousCutsFunctional>> objs;
    objs.reserve(n);
    for (unsigned l = 0; l < n; l++) {
        objs.push_back(std::make_shared<ContinuousCutsFunctional>(
            Ds[l], rhos[l], specs[l].alpha, eps_of(specs[l].eps_ref)));
    }

    // transfers[k]: between grids[k] (coarse) and grids[k+1] (fine), k = 0 .. n-2.
    std::vector<std::unique_ptr<BernoulliGridTransfer>> transfers;
    transfers.reserve(n - 1);
    for (unsigned k = 0; k + 1 < n; k++) {
        transfers.push_back(std::make_unique<BernoulliGridTransfer>(grids[k], grids[k + 1]));
    }

    auto manifold = std::make_shared<BernoulliManifold>();

    dealii::MGLevelObject<std::shared_ptr<ManifoldBase>>             manifold_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<ManifoldTransferBase>>     point_transfer_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<VectorTransportBase>>      vector_transport_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<ContinuousCutsFunctional>> objective_mg(min_level, max_level);
    dealii::MGLevelObject<DescentOptions>                            options_descent_mg(min_level, max_level);
    dealii::MGLevelObject<SolverOptions>                             options_solver_mg(min_level, max_level);
    dealii::MGLevelObject<std::shared_ptr<CoarseConditionBase>>      condition_mg(min_level, max_level);

    for (unsigned l = 0; l <= max_level; l++) {
        manifold_mg[l]  = manifold;
        objective_mg[l] = objs[l];
        options_solver_mg[l] = SolverOptions{};

        DescentOptions opt{};
        opt.tol_residual = 1e-6;
        opt.step_size    = 1.0;
        opt.line_search  = true;
        opt.ls           = {50, 1.0, 0.5, 1e-4, 1e-12};
        opt.max_iter     = (l == max_level) ? top_max_iter : specs[l].coarse_max_iter;
        options_descent_mg[l] = opt;
    }
    for (unsigned l = 1; l <= max_level; l++) {
        BernoulliGridTransfer* hop = transfers[l - 1].get();
        point_transfer_mg[l]   = std::make_shared<BernoulliPointTransfer>(*hop, BernoulliPointTransfer::Restriction::INJECTION);
        vector_transport_mg[l] = std::make_shared<GeometricBilinearTransport>(*hop);
        condition_mg[l]        = std::make_shared<ScaledCoarseCondition>(2.0, min_fine_norm);
    }

    std::vector<unsigned> level_indices(n);
    for (unsigned l = 0; l < n; l++) level_indices[l] = l;

    FAS_Options options_fas{};
    options_fas.kappa                  = 0.6;
    options_fas.eps                    = 0.5;
    options_fas.coarse_every           = 2;
    options_fas.coarse_energy_adaptive = false;

    FullApproximationScheme<ContinuousCutsFunctional> fas_solver(
        manifold_mg, point_transfer_mg, vector_transport_mg, objective_mg,
        level_indices, options_descent_mg, options_solver_mg, options_fas,
        std::nullopt, condition_mg);

    Vector<double> phi(grids[max_level].n_dofs());
    phi = 0.5;
    RecordingObserver recorder(max_level);
    fas_solver.set_observer(recorder);

    FisherRaoOracle O_fine(*objs[max_level]);
    FisherRaoOracle T_fine(*objs[max_level]);
    fas_solver.cycle<FisherRaoOracle, ContinuousCutsCoarseOracle, ContinuousCutsCoarseOracle>(
        O_fine, T_fine, phi, level_indices.size() - 1);

    return recorder.history;
}

void report(const char* label, const std::vector<CycleInfo>& sl_hist, const std::vector<CycleInfo>& ml_hist,
           double target_90, double target_full)
{
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "\n--- " << label << " ---\n"
              << "final E=" << ml_hist.back().energy << "  cpu(" << ml_hist.back().iter << ")=" << ml_hist.back().elapsed
              << "  (" << coarse_steps(ml_hist) << " coarse corrections)\n";
    for (auto [name, target] : {std::pair{"90% converged", target_90}, std::pair{"fully converged (=SL final)", target_full}}) {
        const auto& sl_hit = first_reaching(sl_hist, target);
        const auto& ml_hit = first_reaching(ml_hist, target);
        const double speedup = sl_hit.elapsed / std::max(ml_hit.elapsed, 1e-12);
        std::cout << "  " << name << ": SL it=" << sl_hit.iter << " cpu=" << sl_hit.elapsed
                  << "  ML it=" << ml_hit.iter << " cpu=" << ml_hit.elapsed << "  speedup=" << speedup << "\n";
    }
}

} // namespace

int main()
{
    constexpr unsigned max_iter = 300;
    constexpr double min_fine_norm = 10.0;   // see condition.h / REVIEW \S8.2, \S9

    const std::string dir = data_dir();
    const Vector<double> rho_960 = load_raster(dir + "/rho_fine_960x1280.bin", 960u * 1280u);
    const Vector<double> rho_480 = load_raster(dir + "/rho_480x640.bin",       480u * 640u);
    const Vector<double> rho_240 = load_raster(dir + "/rho_coarse_240x320.bin", 240u * 320u);
    const Vector<double> rho_120 = load_raster(dir + "/rho_120x160.bin",       120u * 160u);

    // --- Single-level baseline (shared across all hierarchy depths) ---
    const PixelGrid grid_fine(960, 1280);
    const ForwardDifference D_fine(grid_fine);
    ContinuousCutsFunctional obj_fine(D_fine, rho_960, 0.1, eps_of(1e-4));
    FisherRaoOracle sl_oracle(obj_fine);
    const BernoulliManifold sl_manifold;

    DescentOptions sl_options{};
    sl_options.tol_residual = 1e-6;
    sl_options.step_size    = 1.0;
    sl_options.max_iter     = max_iter;
    sl_options.line_search  = true;
    sl_options.ls           = {50, 1.0, 0.5, 1e-4, 1e-12};

    Vector<double> phi_sl(grid_fine.n_dofs());
    phi_sl = 0.5;
    GradientDescent sl_solver(sl_oracle, sl_manifold, sl_options);
    RecordingObserver sl_recorder(0);
    sl_solver.set_observer(sl_recorder);
    std::cout << "=== Single-level RGD (960x1280), shared baseline ===\n";
    sl_solver.cycle(phi_sl);

    const double e0         = sl_recorder.history.front().energy;
    const double sl_final_e = sl_recorder.history.back().energy;
    const double target_90   = e0 - 0.9 * (e0 - sl_final_e);
    const double target_full = sl_final_e;
    std::cout << "SL: E0=" << e0 << " final E=" << sl_final_e << " cpu(" << max_iter << ")=" << sl_recorder.history.back().elapsed << "\n";

    // --- 3-level: 240x320 / 480x640 / 960x1280 (examples/levels_2_3_4.py's "3-level") ---
    std::cout << "\n=== 3-level (240x320 / 480x640 / 960x1280) ===\n";
    const auto hist3 = run_n_level(
        {{240, 320, 0.4, 1e-3, 10}, {480, 640, 0.2, 5e-4, 4}, {960, 1280, 0.1, 1e-4, 0}},
        {rho_240, rho_480, rho_960}, max_iter, min_fine_norm);
    report("3-level", sl_recorder.history, hist3, target_90, target_full);

    // --- 4-level: 120x160 / 240x320 / 480x640 / 960x1280 ("4-level") ---
    std::cout << "\n=== 4-level (120x160 / 240x320 / 480x640 / 960x1280) ===\n";
    const auto hist4 = run_n_level(
        {{120, 160, 0.8, 2e-3, 10}, {240, 320, 0.4, 1e-3, 4}, {480, 640, 0.2, 5e-4, 3}, {960, 1280, 0.1, 1e-4, 0}},
        {rho_120, rho_240, rho_480, rho_960}, max_iter, min_fine_norm);
    report("4-level", sl_recorder.history, hist4, target_90, target_full);

    return 0;
}
