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
#include <rmo/fe/interpolate.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/manifold_positive.h>
#include <rmo/gpe/metric.h>
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/oracle_coarse.h>
#include <rmo/gpe/transport.h>

#include <rmo/ropt/fas.h>
#include <rmo/ropt/solver.h>

#include <rmo/option.h>
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
constexpr int dim = 2;
using System     = GrossPitaevskiiLumpedSystem<dim>;
using MassMatrix = System::MassMatrix;
using Functional = GrossPitaevskiiFunctional<System>;
using InverseM   = Functional::InverseM;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct DriverOptions
{
    GPE_Options    gpe;
    DescentOptions descent;
    SolverOptions  solver;
    MG_Options     mg;
    FAS_Options    fas;
    MetricKind     metric_smooth = MetricKind::ENERGY_ADAPTIVE;
    std::string smoother, coarse_retraction, point_transfer, coarse_grad;
    unsigned coarse_max_iter = 4;
    double floor = 1e-12;
    std::filesystem::path out;

    [[nodiscard]] bool multilevel() const { return mg.v_levels.size() > 1; }
};

//! All rows of a run, every level
struct Recorder : IterationObserver<CycleInfo>
{
    void add(const CycleInfo& info) override { rows.push_back(info); }
    std::vector<CycleInfo> rows;
};

/**
 * Prototype of the restriction-based transfer, Version I (tex/drafts/positive_multilevel.tex, "Restriction-based
 * transfers"): point restriction by full weighting of the densities, r(x) = (D_H^{-1} I^T D_h x^2)^{1/2} (unit mass,
 * no normalization; the prolongation interpolates squares and is not used by FAS) ...
 */
class DensityFullWeighting : public ManifoldTransferBase
{
public:
    DensityFullWeighting(const LinearTransferBase& transfer, const MassMatrix& M_coarse, const MassMatrix& M_fine)
        : ManifoldTransferBase(transfer), M_coarse(M_coarse), M_fine(M_fine)
    {}

    void restriction(const Vector<double>& x, Vector<double>& y) const override
    {
        Vector<double> sq(x), Dsq(x.size());
        sq.scale(x);
        M_fine.vmult(Dsq, sq);
        y.reinit(transfer.n_coarse());
        transfer.Tfine(Dsq, y);
        const auto& d_H = M_coarse.get_vector();
        for (unsigned j = 0; j < y.size(); ++j) {
            y[j] = std::sqrt(y[j] / d_H[j]);
        }
    }

    void prolongation(const Vector<double>& y, Vector<double>& x) const override
    {
        Vector<double> sq(y);
        sq.scale(y);
        x.reinit(transfer.n_fine());
        transfer.to_fine_mesh(sq, x);
        for (auto& e : x) {
            e = std::sqrt(e);
        }
        ellipsoid::retract_by_norm(M_fine, x);
    }

private:
    const MassMatrix& M_coarse;
    const MassMatrix& M_fine;
};

/**
 * ... and the vector transfer P = Dr(x)^*, R = Dr(x) (mass metric): P v = x o I(v / y), R g = D_H^{-1} I^T D_h
 * (x o Pi_x g) / y. With y = r(x) both are tangent and adjoint (the projections only against rounding), and with the
 * density lifting the correction is x_new^2 = x^2 o (1 + alpha (I(z^2 / y^2) - 1)): no base point x_hat.
 */
class RestrictionTransport : public VectorTransportBase
{
public:
    RestrictionTransport(const LinearTransferBase& transfer, const MassMatrix& M_coarse, const MassMatrix& M_fine)
        : transfer(transfer), M_coarse(M_coarse), M_fine(M_fine)
    {}

    void vector_prolongation(const Vector<double>& x, const Vector<double>& y, const Vector<double>& v,
                             Vector<double>& dst) const override
    {
        Vector<double> rel(v.size()), t(transfer.n_fine());
        for (unsigned j = 0; j < v.size(); ++j) {
            rel[j] = v[j] / y[j];
        }
        transfer.to_fine_mesh(rel, t);
        t.scale(x);
        metric::mass::project_onto_tangent_space(x, M_fine, t, dst);
    }

    void vector_restriction(const Vector<double>& y, const Vector<double>& x, const Vector<double>& g,
                            Vector<double>& dst) const override
    {
        Vector<double> q(g.size()), Dq(g.size()), t(transfer.n_coarse());
        metric::mass::project_onto_tangent_space(x, M_fine, g, q);
        q.scale(x);
        M_fine.vmult(Dq, q);
        transfer.Tfine(Dq, t);
        const auto& d_H = M_coarse.get_vector();
        for (unsigned j = 0; j < t.size(); ++j) {
            t[j] /= d_H[j] * y[j];
        }
        metric::mass::project_onto_tangent_space(y, M_coarse, t, dst);
    }

private:
    const LinearTransferBase& transfer;
    const MassMatrix& M_coarse;
    const MassMatrix& M_fine;
};

/**
 * Comparison prototype: the density restriction with the plain transpose, W = I^T instead of D_H^{-1} I^T D_h:
 * r(x) = N_H(s), s = (I^T x^2)^{1/2}, n = |s|_{D_H}; R g = Dr(x)[Pi_x g] = n^{-1} Pi_y(I^T(x o Pi_x g) / s),
 * P v = Pi_x Dr(x)^* v = Pi_x(n^{-1} x o D_h^{-1} I(d_H o Pi_y v / s)) (mass-metric adjoint).
 */
class DensityTransposeRestriction : public ManifoldTransferBase
{
public:
    DensityTransposeRestriction(const LinearTransferBase& transfer, const MassMatrix& M_coarse, const MassMatrix& M_fine)
        : ManifoldTransferBase(transfer), M_coarse(M_coarse), M_fine(M_fine)
    {}

    void restriction(const Vector<double>& x, Vector<double>& y) const override
    {
        Vector<double> sq(x);
        sq.scale(x);
        y.reinit(transfer.n_coarse());
        transfer.Tfine(sq, y);
        for (auto& e : y) {
            e = std::sqrt(e);
        }
        ellipsoid::retract_by_norm(M_coarse, y);
    }

    void prolongation(const Vector<double>& y, Vector<double>& x) const override
    {
        Vector<double> sq(y);
        sq.scale(y);
        x.reinit(transfer.n_fine());
        transfer.to_fine_mesh(sq, x);
        for (auto& e : x) {
            e = std::sqrt(e);
        }
        ellipsoid::retract_by_norm(M_fine, x);
    }

private:
    const MassMatrix& M_coarse;
    const MassMatrix& M_fine;
};

class TransposeRestrictionTransport : public VectorTransportBase
{
public:
    TransposeRestrictionTransport(const LinearTransferBase& transfer, const MassMatrix& M_coarse,
                                  const MassMatrix& M_fine)
        : transfer(transfer), M_coarse(M_coarse), M_fine(M_fine)
    {}

    void vector_prolongation(const Vector<double>& x, const Vector<double>& y, const Vector<double>& v,
                             Vector<double>& dst) const override
    {
        const auto [s, n] = root(x);
        Vector<double> pv(v.size()), t(transfer.n_fine());
        metric::mass::project_onto_tangent_space(y, M_coarse, v, pv);
        const auto& d_H = M_coarse.get_vector();
        const auto& d_h = M_fine.get_vector();
        for (unsigned j = 0; j < pv.size(); ++j) {
            pv[j] *= d_H[j] / s[j];
        }
        transfer.to_fine_mesh(pv, t);
        for (unsigned i = 0; i < t.size(); ++i) {
            t[i] *= x[i] / (d_h[i] * n);
        }
        metric::mass::project_onto_tangent_space(x, M_fine, t, dst);
    }

    void vector_restriction(const Vector<double>& y, const Vector<double>& x, const Vector<double>& g,
                            Vector<double>& dst) const override
    {
        const auto [s, n] = root(x);
        Vector<double> q(g.size()), t(transfer.n_coarse());
        metric::mass::project_onto_tangent_space(x, M_fine, g, q);
        q.scale(x);
        transfer.Tfine(q, t);
        for (unsigned j = 0; j < t.size(); ++j) {
            t[j] /= s[j] * n;
        }
        metric::mass::project_onto_tangent_space(y, M_coarse, t, dst);
    }

private:
    std::pair<Vector<double>, double> root(const Vector<double>& x) const
    {
        Vector<double> sq(x), s(transfer.n_coarse()), Ms(transfer.n_coarse());
        sq.scale(x);
        transfer.Tfine(sq, s);
        for (auto& e : s) {
            e = std::sqrt(e);
        }
        M_coarse.vmult(Ms, s);
        return {s, std::sqrt(s * Ms)};
    }

    const LinearTransferBase& transfer;
    const MassMatrix& M_coarse;
    const MassMatrix& M_fine;
};

std::shared_ptr<ManifoldBase> make_manifold(const std::string& name, const MassMatrix& D, double floor)
{
    if (name == "floor") return std::make_shared<FlooredPosSphere<MassMatrix>>(D, floor, true);
    if (name == "plain") return std::make_shared<PosUnitMassSphere<MassMatrix>>(D);
    if (name == "e-cap-up" || name == "clip-e-cap-up") {
        return std::make_shared<ExpPosSphere<MassMatrix>>(D, ExpCap::GROWTH);
    }
    throw std::invalid_argument("unknown retraction " + name);
}

//! Levels, functionals, manifolds, transfers of the run (one level for the single-level case)
class Hierarchy
{
public:
    explicit Hierarchy(const DriverOptions& opts)
        : levels(opts.mg.v_levels)
        , lo(levels.front()), hi(levels.back())
        , builders(lo, hi), objectives(lo, hi), manifolds(lo, hi), transfers(lo, hi), point_transfers(lo, hi)
        , raw_transports(lo, hi), projections(lo, hi), transports(lo, hi), descent(lo, hi), solver(lo, hi)
    {
        auto potential_v = potential::get_potential<dim>(opts.gpe.potential, opts.gpe.potential_expr);
        for (const unsigned l : levels) {
            builders[l] = std::visit([&](auto&& V) { return std::make_unique<ModelBuilder<System>>(V, opts.gpe, l); },
                                     potential_v);
            objectives[l] = std::make_shared<Functional>(builders[l]->get_system(), opts.gpe.beta, opts.solver);
            manifolds[l] = make_manifold(l == hi ? opts.smoother : opts.coarse_retraction,
                                         builders[l]->get_system().get_M(), opts.floor);
            descent[l] = opts.descent;
            solver[l]  = opts.solver;
            if (l != hi) {
                descent[l].max_iter = opts.coarse_max_iter;
            }
        }
        for (unsigned i = 1; i < levels.size(); ++i) {
            const unsigned l = levels[i], lc = levels[i - 1];
            const auto& pc = builders[lc]->get_package();
            const auto& pf = builders[l]->get_package();
            const MassMatrix& Mc = builders[lc]->get_system().get_M();
            const MassMatrix& Mf = builders[l]->get_system().get_M();
            const InverseM& Mc_inv = objectives[lc]->get_M_inv();
            transfers[l] = std::make_shared<fe::LinearTransferMG<dim>>(pc.get_dofs(), pf.get_dofs(),
                                                                      pc.get_constraints(), pf.get_constraints());
            if (opts.point_transfer == "linear") {
                point_transfers[l] = std::make_shared<ManifoldTransfer<MassMatrix, InverseM>>(*transfers[l], Mc, Mf,
                                                                                               Mc_inv);
            }
            else if (opts.point_transfer == "restriction-transpose") {
                point_transfers[l] = std::make_shared<DensityTransposeRestriction>(*transfers[l], Mc, Mf);
                transports[l] = std::make_shared<TransposeRestrictionTransport>(*transfers[l], Mc, Mf);
                continue;
            }
            else if (opts.point_transfer == "restriction") {
                point_transfers[l] = std::make_shared<DensityFullWeighting>(*transfers[l], Mc, Mf);
                transports[l] = std::make_shared<RestrictionTransport>(*transfers[l], Mc, Mf);
                continue;
            }
            else {
                point_transfers[l] = std::make_shared<LogThresholdManifoldTransfer<MassMatrix, InverseM>>(
                    *transfers[l], Mc, Mf, Mc_inv);
            }
            raw_transports[l] = std::make_shared<AdjointDifferentialTransport<MassMatrix, InverseM>>(
                *transfers[l], *point_transfers[l], Mc, Mf, Mc_inv);
            projections[l] = std::make_shared<metric::mass::TangentProjection<MassMatrix>>(Mf);
            transports[l] = std::make_shared<TangentProjectedTransport>(*raw_transports[l], *projections[l]);
        }
    }

    //! Start as pos_sphere_study: constant, thresholded, normalized
    [[nodiscard]] Vector<double> start() const
    {
        Vector<double> x0 = builders[hi]->initial_value();
        builders[hi]->distribute(x0);
        ellipsoid_positive::threshold(x0);
        ellipsoid::retract_by_norm(builders[hi]->get_system().get_M(), x0);
        return x0;
    }

    std::vector<unsigned> levels;
    unsigned lo, hi;
    MGLevelObject<std::unique_ptr<ModelBuilder<System>>> builders;
    MGLevelObject<std::shared_ptr<Functional>> objectives;
    MGLevelObject<std::shared_ptr<ManifoldBase>> manifolds;
    MGLevelObject<std::shared_ptr<LinearTransferBase>> transfers;
    MGLevelObject<std::shared_ptr<ManifoldTransferBase>> point_transfers;
    MGLevelObject<std::shared_ptr<VectorTransportBase>> raw_transports;
    MGLevelObject<std::shared_ptr<TangentProjectionBase>> projections;
    MGLevelObject<std::shared_ptr<VectorTransportBase>> transports;
    MGLevelObject<DescentOptions> descent;
    MGLevelObject<SolverOptions> solver;
};

//! Descent oracle of the finest level: energy-adaptive (clipped for clip-*) or mass
std::unique_ptr<GrossPitaevskiiOracle<System>> fine_oracle(const DriverOptions& opts, Functional& f)
{
    if (opts.metric_smooth == MetricKind::MASS) {
        return std::make_unique<MassOracle<System>>(f, opts.solver);
    }
    if (opts.smoother.starts_with("clip-")) {
        return std::make_unique<ClippedEnergyOracle<System>>(f, opts.solver);
    }
    return std::make_unique<EnergyOracle<System>>(f, opts.solver);
}

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

template <DensityGradient G>
void run_fas(Hierarchy& h, const DriverOptions& opts, Solve& s)
{
    FullApproximationScheme<Functional> fas(h.manifolds, h.point_transfers, h.transports, h.objectives, h.levels,
                                            h.descent, h.solver, opts.fas);
    fas.set_observer(s.rec);
    auto O_fine = fine_oracle(opts, *h.objectives[h.hi]);
    MassOracle<System> T_fine(*h.objectives[h.hi], opts.solver);
    Vector<double> x = h.start();
    std::ostringstream tables;  // tables of the coarse levels, not written
    timed(s, [&] {
        fas.template cycle<MassOracle<System>, DensityCoarseOracle<System, DensityGradient::MASS>,
                           DensityCoarseOracle<System, G>, DensityCoarseResidual<System>>(
            *O_fine, T_fine, O_fine->get_residual(), x, h.levels.size() - 1, tables);
    });
    s.stats = fas.correction_stats();
}

void run_multilevel(Hierarchy& h, const DriverOptions& opts, Solve& s)
{
    if (opts.coarse_grad == "model") run_fas<DensityGradient::ENERGY_MODEL>(h, opts, s);
    else if (opts.coarse_grad == "model-shift") run_fas<DensityGradient::ENERGY_MODEL_SHIFT>(h, opts, s);
    else if (opts.coarse_grad == "fine") run_fas<DensityGradient::ENERGY_FINE>(h, opts, s);
    else if (opts.coarse_grad == "mass") run_fas<DensityGradient::MASS>(h, opts, s);
    else run_fas<DensityGradient::FROBENIUS>(h, opts, s);
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

template <typename T>
void check_value(const std::string& option, const T& value, std::initializer_list<T> allowed)
{
    if (std::ranges::find(allowed, value) == allowed.end()) {
        throw std::invalid_argument(fmt::format("--{}: unknown value {}", option, value));
    }
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
        po::options_description driver("Driver options");
        driver.add_options()
            ("smoother", po::value<std::string>()->default_value("floor"),
                "retraction of the finest level and of the correction (floor|plain|e-cap-up|clip-e-cap-up)")
            ("coarse-retraction", po::value<std::string>()->default_value("floor"),
                "retraction of the coarse levels, with the density lifting (floor|plain)")
            ("point-transfer", po::value<std::string>()->default_value("linear"), "point maps (linear|log-threshold), or restriction: the restriction-based transfer (prototype)")
            ("coarse-grad", po::value<std::string>()->default_value("model"),
                "gradient of the coarse model (model|model-shift|fine|mass|frobenius)")
            ("coarse-max-iter", po::value<unsigned>()->default_value(4), "iterations per coarse solve")
            ("floor", po::value<double>()->default_value(1e-12), "floor of FlooredPosSphere")
            ("out", po::value<std::string>(), "output directory (summary.csv appended, iter.csv)");
        all.add(driver);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);
        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, opts.gpe);
        AssertThrow(opts.gpe.mass_lumping && opts.gpe.dimension == dim,
                    ExcMessage("pos_sphere_ml requires --mass-lumping and --dimension 2"));
        apply_descent_options(vm, opts.descent);
        apply_mg_options(vm, opts.mg);
        std::ranges::sort(opts.mg.v_levels);
        apply_inner_options(vm, opts.solver);
        apply_fas_options(vm, opts.fas);
        apply_smoother_options(vm, opts.metric_smooth);
        opts.smoother          = vm["smoother"].as<std::string>();
        opts.coarse_retraction = vm["coarse-retraction"].as<std::string>();
        opts.point_transfer    = vm["point-transfer"].as<std::string>();
        opts.coarse_grad       = vm["coarse-grad"].as<std::string>();
        opts.coarse_max_iter   = vm["coarse-max-iter"].as<unsigned>();
        opts.floor             = vm["floor"].as<double>();
        if (!vm.contains("out")) {
            throw std::invalid_argument("--out is required");
        }
        opts.out = vm["out"].as<std::string>();

        using namespace std::string_literals;
        check_value("smoother", opts.smoother, {"floor"s, "plain"s, "e-cap-up"s, "clip-e-cap-up"s});
        check_value("coarse-retraction", opts.coarse_retraction, {"floor"s, "plain"s});
        check_value("point-transfer", opts.point_transfer, {"linear"s, "log-threshold"s, "restriction"s, "restriction-transpose"s});
        check_value("coarse-grad", opts.coarse_grad, {"model"s, "model-shift"s, "fine"s, "mass"s, "frobenius"s});
        if (opts.metric_smooth != MetricKind::ENERGY_ADAPTIVE && opts.metric_smooth != MetricKind::MASS) {
            throw std::invalid_argument("--metric-smooth: energy_adaptive or mass");
        }
        if (opts.smoother.starts_with("clip-") && opts.metric_smooth != MetricKind::ENERGY_ADAPTIVE) {
            throw std::invalid_argument("--smoother " + opts.smoother + " requires --metric-smooth energy_adaptive");
        }

        run(opts);
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
