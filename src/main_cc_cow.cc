//
// Single-level vs. two-level continuous cuts on the paper's own cow-image problem
// (Sec. 6.3, examples/compare_variants.py in RMO-continuous-cuts): 960x1280 fine grid,
// 240x320 coarse grid, i.e. two composed factor-2 average-pool steps (n_pools=2) rather
// than the one exact mesh-doubling main_cc_coarse.cc uses on the synthetic disk.
//
// rho is precomputed by test/cc/prepare_cow_data.py (grayscale, 2x bilinear enlarge,
// Gaussian blur, foreground/background seed-rectangle means -- all delegated to skimage,
// the same code path the paper's own figures were produced with, rather than a fresh
// C++ image decoder/resizer/blur risking a silent divergence from it) and loaded here as
// two raw float64 rasters. Run that script first:
//   python3 test/cc/prepare_cow_data.py /path/to/RMO-continuous-cuts
// then point CC_COW_DATA_DIR at its --out-dir (default test/cc/data) if not running this
// binary from the repository root.
//
// Regularization matches the paper's cow-image experiment (compare_variants.py):
// alpha = 0.1 fine / 0.4 coarse, eta = 0.6 (Fig. 15, Option 1), mu = 0.5
// (gnorm_c_threshold), coarse solver 5 iterations per call (halved from the paper's 10,
// see REVIEW-continuous-cuts.md \S7.1: found not to matter for either final energy or
// CPU time on the synthetic disk), coarse_every = 2 (\S6.4 lockout). eps follows this
// port's established convention (eps_paper = sqrt(eps_ref), REVIEW-continuous-cuts.md
// \S3.3): 1e-2 fine, ~3.16e-2 coarse, matching the reference's own eps = 1e-4 / 1e-3
// unsquared. grid_scale = 2^n_pools = 4 (not 2 as on the synthetic disk's single hop),
// per the reference's operators.get_grid_scale("Option 1", n_pools=2).
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
#include <rmo/ropt/observer_table.h>
#include <rmo/ropt/solver.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
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

// Mirrors main_cc_bench.cc's RecordingObserver: captures the finest level's iteration
// history directly, so CPU time (CycleInfo::elapsed) can be compared without going
// through ConvergenceTableObserver's text tables.
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

} // namespace

int main()
{
    constexpr unsigned rows_fine = 960, cols_fine = 1280;
    constexpr unsigned rows_mid = 480, cols_mid = 640;
    constexpr unsigned rows_coarse = 240, cols_coarse = 320;
    constexpr unsigned max_iter = 300;

    const std::string dir = data_dir();
    const Vector<double> rho_fine   = load_raster(dir + "/rho_fine_960x1280.bin", rows_fine * cols_fine);
    const Vector<double> rho_coarse = load_raster(dir + "/rho_coarse_240x320.bin", rows_coarse * cols_coarse);

    const PixelGrid grid_fine(rows_fine, cols_fine);
    const PixelGrid grid_mid(rows_mid, cols_mid);
    const PixelGrid grid_coarse(rows_coarse, cols_coarse);
    const ForwardDifference D_fine(grid_fine);
    const ForwardDifference D_coarse(grid_coarse);

    constexpr double alpha_fine = 0.1, alpha_coarse = 0.4;
    constexpr double eps_fine = 1e-2, eps_coarse = 3.1622776601683795e-2;   // sqrt(1e-4), sqrt(1e-3)

    auto obj_fine   = std::make_shared<ContinuousCutsFunctional>(D_fine, rho_fine, alpha_fine, eps_fine);
    auto obj_coarse = std::make_shared<ContinuousCutsFunctional>(D_coarse, rho_coarse, alpha_coarse, eps_coarse);

    Vector<double> phi0(grid_fine.n_dofs());
    phi0 = 0.5;

    // --- Single-level baseline (same problem, no coarse correction) ---
    std::cout << "=== Single-level RGD (960x1280) ===\n";
    FisherRaoOracle sl_oracle(*obj_fine);
    const BernoulliManifold sl_manifold;

    DescentOptions sl_options{};
    sl_options.tol_residual = 1e-6;
    sl_options.step_size    = 1.0;
    sl_options.max_iter     = max_iter;
    sl_options.line_search  = true;
    sl_options.ls           = {50, 1.0, 0.5, 1e-4, 1e-12};

    Vector<double> phi_sl(phi0);
    GradientDescent sl_solver(sl_oracle, sl_manifold, sl_options);
    RecordingObserver sl_recorder(0);
    sl_solver.set_observer(sl_recorder);
    sl_solver.cycle(phi_sl);

    // Cross-check against the Python reference's solve_single_level on the same rho
    // (see test/cc/prepare_cow_data.py); reference values at these iterations, eps=1e-4
    // unsquared: it1=37965.978081, it2=20493.208684, it3=5555.309023, it5=-14429.485188,
    // it10=-37795.208389, it20=-52970.249212.
    std::cout << std::setprecision(6);
    for (unsigned it : {1u, 2u, 3u, 5u, 10u, 20u}) {
        if (it < sl_recorder.history.size()) {
            std::cout << "  it " << it << ": E=" << sl_recorder.history[it].energy << "\n";
        }
    }

    // --- Two-level FAS with a composed (n_pools=2) coarse transition ---
    std::cout << "\n=== Two-level FAS, composed n_pools=2 (960x1280 -> [480x640] -> 240x320) ===\n";
    constexpr unsigned int min_level = 0, max_level = 1;
    const std::vector<unsigned> level_indices{min_level, max_level};

    // The two single-hop transfers a composed coarse transition chains through; kept as
    // plain locals (ComposedGridTransfer/ComposedVectorTransport only borrow pointers to
    // them, same as test_interpolate.cc), since nothing here needs shared ownership of them.
    const BernoulliGridTransfer hop_fine_mid(grid_mid, grid_fine);      // fine <-> mid
    const BernoulliGridTransfer hop_mid_coarse(grid_coarse, grid_mid);  // mid  <-> coarsest
    // BernoulliPointTransfer stores its transfer by reference, so this needs a name (and a
    // lifetime past this point) rather than being a temporary passed inline.
    const ComposedGridTransfer composed_transfer({&hop_fine_mid, &hop_mid_coarse});

    auto manifold         = std::make_shared<BernoulliManifold>();
    auto point_transfer   = std::make_shared<BernoulliPointTransfer>(
        composed_transfer, BernoulliPointTransfer::Restriction::INJECTION);
    auto vector_transport = std::make_shared<ComposedVectorTransport>(
        std::vector<const BernoulliGridTransfer*>{&hop_fine_mid, &hop_mid_coarse});

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

    DescentOptions ml_options_fine{};
    ml_options_fine.tol_residual = 1e-6;
    ml_options_fine.step_size    = 1.0;
    ml_options_fine.max_iter     = max_iter;
    ml_options_fine.line_search  = true;
    ml_options_fine.ls           = {50, 1.0, 0.5, 1e-4, 1e-12};

    DescentOptions ml_options_coarse = ml_options_fine;
    ml_options_coarse.max_iter = 5;    // paper/reference: 10; halved, see REVIEW \S7.1
    ml_options_coarse.ls.alpha = 1.0;  // reference optimizer.py:8; undamped, see main_cc_coarse.cc

    options_descent_mg[min_level] = ml_options_coarse;
    options_descent_mg[max_level] = ml_options_fine;
    options_solver_mg[min_level] = options_solver_mg[max_level] = SolverOptions{};

    FAS_Options options_fas{};
    options_fas.kappa                  = 0.6;
    options_fas.eps                    = 0.5;
    options_fas.coarse_every           = 2;
    options_fas.coarse_energy_adaptive = false;

    // min_fine_norm=10: the fine gradient norm decays steadily (139 -> 6.5 over 300
    // iterations, REVIEW-continuous-cuts.md \S8.2) while the scaled restricted norm stays
    // 5000-9000x larger throughout, so the mu/kappa gate alone never shuts off (145 of 150
    // eligible iterations trigger a correction). 10 is past the halfway point of that decay
    // (crossed between it=121 and it=141 in the ungated run) -- see \S9 for the effect.
    dealii::MGLevelObject<std::shared_ptr<CoarseConditionBase>> condition_mg(min_level, max_level);
    condition_mg[max_level] = std::make_shared<ScaledCoarseCondition>(4.0, 10.0);   // 2^n_pools, n_pools=2

    FullApproximationScheme<ContinuousCutsFunctional> fas_solver(
        manifold_mg, point_transfer_mg, vector_transport_mg, objective_mg,
        level_indices, options_descent_mg, options_solver_mg, options_fas,
        std::nullopt, condition_mg);

    Vector<double> phi_ml(phi0);
    RecordingObserver ml_recorder(max_level);
    fas_solver.set_observer(ml_recorder);

    FisherRaoOracle O_fine(*obj_fine);
    FisherRaoOracle T_fine(*obj_fine);
    fas_solver.cycle<FisherRaoOracle, ContinuousCutsCoarseOracle, ContinuousCutsCoarseOracle>(
        O_fine, T_fine, phi_ml, level_indices.size() - 1, std::cout);

    // --- CPU-time comparison (main_cc_bench.cc's methodology, applied to this one problem) ---
    const auto& sl_hist = sl_recorder.history;
    const auto& ml_hist = ml_recorder.history;

    std::cout << "\n=== Fine-level ML iterations where the trigger was evaluated ===\n"
              << "  it  coarse  grad_norm  grad_restr_norm*4  energy\n";
    unsigned printed = 0;
    for (const auto& info : ml_hist) {
        if (info.grad_norm <= 0.0) continue;   // not evaluated this iteration
        if (printed < 20 || printed % 10 == 0) {
            std::cout << "  " << info.iter << "   " << (info.coarse ? "*" : " ")
                      << "   " << info.grad_norm << "   " << 4.0 * info.grad_restr_norm
                      << "   " << info.energy << "\n";
        }
        printed++;
    }

    const double e0         = sl_hist.front().energy;
    const double sl_final_e = sl_hist.back().energy;
    const double target_90   = e0 - 0.9 * (e0 - sl_final_e);
    const double target_full = sl_final_e;

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "\n=== CPU-time comparison (single-threaded: run with DEAL_II_NUM_THREADS=1) ===\n"
              << "SL: E0=" << e0 << " final E=" << sl_final_e << " cpu(" << max_iter << ")=" << sl_hist.back().elapsed << "\n"
              << "ML: final E=" << ml_hist.back().energy << " cpu(" << max_iter << ")=" << ml_hist.back().elapsed
              << " (" << coarse_steps(ml_hist) << " coarse corrections)\n";

    for (auto [label, target] : {std::pair{"90% converged", target_90}, std::pair{"fully converged (=SL final)", target_full}}) {
        const auto& sl_hit = first_reaching(sl_hist, target);
        const auto& ml_hit = first_reaching(ml_hist, target);
        const double speedup = sl_hit.elapsed / std::max(ml_hit.elapsed, 1e-12);
        std::cout << "  " << label << ": target E=" << target
                  << "  SL it=" << sl_hit.iter << " cpu=" << sl_hit.elapsed
                  << "  ML it=" << ml_hit.iter << " cpu=" << ml_hit.elapsed
                  << "  speedup=" << speedup << "\n";
    }

    return 0;
}
