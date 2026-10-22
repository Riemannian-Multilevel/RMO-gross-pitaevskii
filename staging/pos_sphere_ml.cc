/**
 * @file
 * @brief Multilevel FAS on the positive sphere with the density coarse model, and its single-level reference: the
 * driver of the positive blocks of the multilevel benchmark (md/positive_multilevel_benchmark.md, steps 5 and 6).
 *
 * One run per call. With one level (--levels l, or --multilevel l) GradientDescent on the finest level (case `sl`);
 * with two or more (--multilevel l_1,...,l_n) FullApproximationScheme (case `ml`). The single-level reference of a
 * multilevel run is its own run with the same smoother, as with main_coarse.
 *
 * - Finest level: smoother --smoother (FlooredPosSphere with cap A, PosUnitMassSphere, ExpPosSphere with the growth
 *   cap, or the latter with ClippedEnergyOracle), which is also the retraction of the coarse correction, and the
 *   descent metric --metric-smooth (energy-adaptive or mass).
 * - Coarse levels: a manifold with the density lifting (--coarse-retraction floor or plain), DensityCoarseOracle with
 *   the gradient --coarse-grad (default `model`, energy-adaptive, as the unit-sphere FAS), w in the mass metric, the
 *   point maps --point-transfer (linear interpolation or log-threshold), TangentProjectedTransport of the adjoint
 *   differential transport. Step rule (Armijo or --ls-exact) on every level; --coarse-max-iter iterations per coarse
 *   solve.
 * - Start as pos_sphere_study: constant, thresholded, normalized.
 *
 * Writes to --out:
 * - `summary.csv`, one line per run (appended; header if the file is new): configuration, problem, status
 *   (converged, stalled, max_iter, stopped, error), iterations, final residual, energy and minimum, CPU and wall time
 *   of the solve (setup excluded), and for `ml` the coarse corrections of the finest level and summed over the middle
 *   levels (`_mid`): attempted, accepted, rejected, fallback, failed, skipped (attempted, fallback, failed, skipped from
 *   FullApproximationScheme::correction_stats(), accepted and rejected from the iteration rows), the rows of the
 *   coarser levels (coarse_iter), and with --correction-control ratio the decrease ratios of the corrections of the
 *   finest level (number finite, number NaN, minimum, median);
 * - `iter.csv`: every row of the run, all levels (column `level`; for `sl` the finest level).
 * stdout: one line per run; stderr: the warnings of the solvers. The tables of the coarse levels are not written
 * (their rows are in iter.csv).
 *
 * Status `stalled` (unless converged): the last row of the finest level has step 0 (the line search returned 0; FAS
 * and GradientDescent stop), or energy and residual did not change over 20 consecutive rows of the finest level
 * somewhere in the run (steps of 1e-302 that are not 0; also when the run left the plateau later and reached
 * max_iter). `stopped`: none of the others (not expected). For `error` the counts are those up to the exception (an
 * attempted correction may have no outcome).
 */
#include "pos_sphere_ml.h"

#include <rmo/ropt/solver.h>
#include <rmo/util/util.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/format.h>

using namespace rmo;
using namespace rmo::gpe;
using namespace dealii;

namespace
{
using namespace pos_sphere_ml;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct DriverOptions : Options
{
    std::filesystem::path out;
};

//! All rows of a run, every level
struct Recorder : IterationObserver<CycleInfo>
{
    void add(const CycleInfo& info) override { rows.push_back(info); }
    std::vector<CycleInfo> rows;
};


//! The solve of one run: rows, statistics of the corrections, times; error message if it threw
struct Solve
{
    Recorder rec;
    MGLevelObject<CorrectionStats> stats;
    double cpu = 0, wall = 0;
    std::string error;
};

//! Times @p f (CPU and wall time), records an exception as the error of the solve
template <typename F>
void timed(Solve& s, F&& f)
{
    Timer timer;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        f();
    } catch (const std::exception& e) {
        s.error = e.what();
    }
    timer.stop();
    s.cpu  = timer.cpu_time();
    s.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void run_multilevel(Hierarchy& h, const DriverOptions& opts, Solve& s)
{
    FullApproximationScheme<Functional> fas(h.manifolds, h.point_transfers, h.transports, h.objectives, h.levels,
                                            h.descent, h.solver, opts.fas);
    fas.set_observer(s.rec);
    auto O_fine = fine_oracle(opts, *h.objectives[h.hi]);
    MassOracle<System> T_fine(*h.objectives[h.hi], opts.solver);
    Vector<double> x = h.start();
    std::ostringstream tables;  // tables of the coarse levels, not written
    with_coarse_grad(opts.coarse_grad, [&]<DensityGradient G>() {
        timed(s, [&] { fas_cycle<G>(fas, *O_fine, T_fine, x, h.levels.size(), tables); });
    });
    s.stats = fas.correction_stats();
}

void run_single(Hierarchy& h, const DriverOptions& opts, Solve& s)
{
    auto O = fine_oracle(opts, *h.objectives[h.hi]);
    GradientDescent gd(*O, O->get_residual(), *h.manifolds[h.hi], opts.descent);
    gd.set_observer(s.rec);
    Vector<double> x = h.start();
    std::ostringstream table;
    timed(s, [&] { gd.cycle(x, table); });
    // GradientDescent reports level 0
    for (auto& row : s.rec.rows) {
        row.level = h.hi;
    }
    s.stats.resize(h.hi, h.hi);
}

double extra_value(const CycleInfo& row, const std::string& name)
{
    for (const auto& [key, value] : row.extra) {
        if (key == name) {
            return value;
        }
    }
    return nan;
}

//! Result of a run, from the rows and the statistics
struct Summary
{
    std::string status;
    unsigned iterations = 0;
    double residual = nan, energy = nan, min_value = nan;
    CorrectionStats fine, mid;
    unsigned accepted = 0, rejected = 0, accepted_mid = 0, rejected_mid = 0, coarse_iter = 0;
    unsigned rho_n = 0, rho_nan = 0;
    double rho_min = nan, rho_median = nan;
};

Summary summarize(const Solve& s, const Hierarchy& h, const DriverOptions& opts)
{
    Summary r;
    std::vector<const CycleInfo*> fine;
    std::vector<double> rho;
    for (const auto& row : s.rec.rows) {
        const bool mid = row.level != h.hi;
        if (!mid) {
            fine.push_back(&row);
        }
        else if (row.iter > 0) {
            r.coarse_iter++;
        }
        if (row.coarse) {
            const bool accepted = row.step > 0;
            (mid ? (accepted ? r.accepted_mid : r.rejected_mid) : (accepted ? r.accepted : r.rejected))++;
            if (!mid && accepted && opts.fas.control == CorrectionControl::RATIO) {
                const double v = extra_value(row, "ratio");
                std::isfinite(v) ? rho.push_back(v) : void(r.rho_nan++);
            }
        }
    }
    for (unsigned l = h.lo; l <= h.hi; ++l) {
        auto& t = l == h.hi ? r.fine : r.mid;
        if (l == h.hi || l != h.lo) {  // the coarsest level has no coarser one
            t.attempted += s.stats[l].attempted;
            t.fallback  += s.stats[l].fallback;
            t.failed    += s.stats[l].failed;
            t.skipped   += s.stats[l].skipped;
        }
    }
    if (!rho.empty()) {
        std::ranges::sort(rho);
        r.rho_n      = rho.size();
        r.rho_min    = rho.front();
        r.rho_median = rho.size() % 2 ? rho[rho.size() / 2] : 0.5 * (rho[rho.size() / 2 - 1] + rho[rho.size() / 2]);
    }

    if (!fine.empty()) {
        const CycleInfo& last = *fine.back();
        r.iterations = last.iter;
        r.residual   = last.residual;
        r.energy     = last.energy;
        r.min_value  = last.min_value;
    }

    // Plateau: energy and residual unchanged over 20 consecutive rows of the finest level
    constexpr unsigned plateau = 20;
    bool flat = false;
    for (unsigned k = 0, run = 0; k + 1 < fine.size(); ++k) {
        const bool same = fine[k + 1]->energy == fine[k]->energy && fine[k + 1]->residual == fine[k]->residual;
        run  = same ? run + 1 : 0;
        flat = flat || run >= plateau;
    }
    if (!s.error.empty()) {
        r.status = "error";
    }
    else if (r.residual < opts.descent.tol_residual) {
        r.status = "converged";
    }
    else if (flat || (!fine.empty() && fine.back()->iter > 0 && fine.back()->step == 0.0)) {
        r.status = "stalled";
    }
    else if (r.iterations >= opts.descent.max_iter) {
        r.status = "max_iter";
    }
    else {
        r.status = "stopped";
    }
    return r;
}

std::string join(const std::vector<unsigned>& levels)
{
    std::string s;
    for (const unsigned l : levels) {
        s += (s.empty() ? "" : "-") + std::to_string(l);
    }
    return s;
}

std::string step_rule(const DescentOptions& d)
{
    return !d.line_search ? "fixed" : d.ls.exact ? "exact" : "armijo";
}

//! Number for the CSV; empty if not applicable (NaN)
std::string num(double v, const char* fmt_spec = "{:.6e}")
{
    return std::isnan(v) ? std::string() : fmt::format(fmt::runtime(fmt_spec), v);
}

const char* summary_header =
    "case,mesh,degree,levels,n_levels,level,n_dofs,metric,smoother,step_rule,coarse_retraction,point_transfer,"
    "coarse_grad,coarse_max_iter,floor,kappa,eps,coarse_every,control,backoff,ratio_eta,tol_inner_res,tol_residual,"
    "max_iter,beta,potential,radius,status,iterations,residual,energy,min_value,cpu,wall,attempted,accepted,rejected,"
    "fallback,failed,skipped,attempted_mid,accepted_mid,rejected_mid,fallback_mid,failed_mid,skipped_mid,"
    "coarse_iter,rho_n,rho_nan,rho_min,rho_median";

void write_summary(const DriverOptions& opts, const Hierarchy& h, const Solve& s, const Summary& r)
{
    const std::filesystem::path file = opts.out / "summary.csv";
    const bool header = !std::filesystem::exists(file);
    std::ofstream csv(file, std::ios::app);
    if (header) {
        csv << summary_header << '\n';
    }
    const bool ml = opts.multilevel();
    // Settings of the coarse levels: empty for the single-level case
    auto coarse = [&](const std::string& v) { return ml ? v : std::string(); };
    csv << fmt::format("{},{},{},{},{},{},{},{},{},{},", ml ? "ml" : "sl", enum_to_string(opts.gpe.mesh_kind),
                       opts.gpe.degree, join(h.levels), h.levels.size(), h.hi, h.builders[h.hi]->n_dofs(),
                       enum_to_string(opts.metric_smooth), opts.smoother, step_rule(opts.descent));
    csv << fmt::format("{},{},{},{},{},{},{},{},{},{},{},", coarse(opts.coarse_retraction),
                       coarse(opts.point_transfer), coarse(opts.coarse_grad),
                       coarse(std::to_string(opts.coarse_max_iter)), num(opts.floor, "{:.3g}"),
                       coarse(num(opts.fas.kappa, "{:.6g}")), coarse(num(opts.fas.eps, "{:.3g}")),
                       coarse(std::to_string(opts.fas.coarse_every)), coarse(enum_to_string(opts.fas.control)),
                       coarse(std::to_string(opts.fas.backoff)), coarse(num(opts.fas.ratio_eta, "{:.6g}")));
    csv << fmt::format("{},{},{},{},{},{},", num(opts.solver.tol_inner_res, "{:.3g}"),
                       num(opts.descent.tol_residual, "{:.3g}"), opts.descent.max_iter, opts.gpe.beta,
                       enum_to_string(opts.gpe.potential), opts.gpe.radius);
    csv << fmt::format("{},{},{},{},{},{},{},", r.status, r.iterations, num(r.residual),
                       num(r.energy, "{:.17g}"), num(r.min_value), num(s.cpu, "{:.6f}"), num(s.wall, "{:.6f}"));
    auto counts = [&](unsigned v) { return ml ? std::to_string(v) : std::string(); };
    csv << fmt::format("{},{},{},{},{},{},{},{},{},{},{},{},{},", counts(r.fine.attempted), counts(r.accepted),
                       counts(r.rejected), counts(r.fine.fallback), counts(r.fine.failed), counts(r.fine.skipped),
                       counts(r.mid.attempted), counts(r.accepted_mid), counts(r.rejected_mid),
                       counts(r.mid.fallback), counts(r.mid.failed), counts(r.mid.skipped), counts(r.coarse_iter));
    csv << fmt::format("{},{},{},{}\n", counts(r.rho_n), counts(r.rho_nan), num(r.rho_min), num(r.rho_median));
}

void write_iter(const DriverOptions& opts, const Solve& s)
{
    std::ofstream csv(opts.out / "iter.csv");
    csv << "iter,level,coarse,step,residual,energy,min_value,elapsed,grad_norm,grad_restr_norm,ratio\n";
    for (const auto& row : s.rec.rows) {
        csv << fmt::format("{},{},{},{},{},{},{},{},{},{},{}\n", row.iter, row.level, int(row.coarse),
                           num(row.step), num(row.residual), num(row.energy, "{:.17g}"), num(row.min_value),
                           num(row.elapsed), num(extra_value(row, "grad_norm")),
                           num(extra_value(row, "grad_restr_norm")), num(extra_value(row, "ratio")));
    }
}

void run(const DriverOptions& opts)
{
    Hierarchy h(opts);
    Solve s;
    opts.multilevel() ? run_multilevel(h, opts, s) : run_single(h, opts, s);
    const Summary r = summarize(s, h, opts);

    std::filesystem::create_directories(opts.out);
    write_summary(opts, h, s, r);
    write_iter(opts, s);
    if (!s.error.empty()) {
        std::cerr << "error: " << s.error << '\n';
    }
    std::cout << fmt::format("{} {} {}{} {} {} {}, beta {} {} R {}: {} after {} iterations, residual {:.3e}, "
                             "CPU {:.3f} s", opts.multilevel() ? "ml" : "sl", join(h.levels),
                             opts.gpe.mesh_kind == MeshKind::SIMPLEX ? "P" : "Q", opts.gpe.degree, opts.smoother,
                             enum_to_string(opts.metric_smooth), step_rule(opts.descent), opts.gpe.beta,
                             enum_to_string(opts.gpe.potential), opts.gpe.radius, r.status, r.iterations, r.residual,
                             s.cpu);
    if (opts.multilevel()) {
        std::cout << fmt::format(", corrections {} of {} accepted, {} fallbacks", r.accepted, r.fine.attempted,
                                 r.fine.fallback);
    }
    std::cout << '\n';
}

} // namespace


int main(int argc, char* argv[])
{
    DriverOptions opts{};
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(mg_cli_options());
        all.add(inner_cli_options());
        all.add(fas_cli_options());
        all.add(smoother_cli_options());
        all.add(positive_cli_options());
        po::options_description driver("Driver options");
        driver.add_options()("out", po::value<std::string>(), "output directory (summary.csv appended, iter.csv)");
        all.add(driver);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);
        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_options(vm, opts, "pos_sphere_ml");
        if (!vm.contains("out")) {
            throw std::invalid_argument("--out is required");
        }
        opts.out = vm["out"].as<std::string>();

        run(opts);
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
