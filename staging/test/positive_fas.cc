/**
 * @file
 * @brief Tests of FullApproximationScheme on the positive sphere with the density coarse model
 * (tex/drafts/positive_multilevel.tex), on lumped Q1 hierarchies (levels 4 -> 5 and 3 -> 4 -> 5, square potential,
 * beta 100, Neumann boundary): FlooredPosSphere on every level (the density lifting on the coarse levels), the linear
 * point maps (ManifoldTransfer), TangentProjectedTransport(AdjointDifferentialTransport), w in the mass metric,
 * coarse gradient ENERGY_MODEL, energy-adaptive smoother with Armijo steps:
 *
 * 1. 2 and 3 levels, radius 6 and 10: converged, positive iterates, energy of single level, accepted corrections,
 *    and per level accepted + rejected + fallback = attempted (CorrectionStats against the iteration table);
 * 2. the supporting inequality \f$ \langle \grad f_h(x), d \rangle \le q(\tilde z) - q(y) \f$ of one correction
 *    \f$ d = P \mathcal{L}^m_y(\tilde z) \f$ (outside FAS, which does not expose the model values);
 * 3. coarse max_iter = 0: \f$ \tilde z = y \f$, \f$ d = 0 \f$, every attempt is a fallback, the run equals single level.
 */
#include "check.h"

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

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/format.h>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;
using namespace dealii;

namespace
{
constexpr int dim = 2;
using System     = GrossPitaevskiiLumpedSystem<dim>;
using MassMatrix = System::MassMatrix;
using Functional = GrossPitaevskiiFunctional<System>;
using InverseM   = Functional::InverseM;
using Floored    = FlooredPosSphere<MassMatrix>;
using Tilt       = DensityCoarseOracle<System, DensityGradient::MASS>;
using Model      = DensityCoarseOracle<System, DensityGradient::ENERGY_MODEL>;

constexpr double beta = 100.0;

GPE_Options gpe_options(double radius)
{
    GPE_Options options{};
    options.dimension    = dim;
    options.degree       = 1;
    options.radius       = radius;
    options.beta         = beta;
    options.order        = Ordering::DEFAULT;
    options.bc           = BoundaryCondition::NEUMANN;
    options.mesh_kind    = MeshKind::QUADRILATERAL;
    options.potential    = Potential::SQUARE;
    options.mass_lumping = true;
    return options;
}

SolverOptions solver_options()
{
    SolverOptions options{};
    options.max_inner     = 5000;
    options.tol_inner     = 1e-12;
    options.tol_inner_res = 1e-6;
    options.solver        = SolverMethod::CG;
    options.precond       = Precondition::DIAGONAL;
    return options;
}

DescentOptions descent_options()
{
    DescentOptions options{};
    options.tol_residual = 1e-9;
    options.step_size    = 1.0;
    options.max_iter     = 1000;
    options.line_search  = true;
    options.ls.max_iter  = 40;
    options.ls.alpha     = 1.0;
    options.ls.beta      = 0.5;
    options.ls.sigma     = 1e-4;
    options.ls.min       = 1e-12;
    return options;
}

FAS_Options fas_options()
{
    FAS_Options options{};
    options.kappa        = 0.8;
    options.eps          = 2e-8;
    options.coarse_every = 2;
    return options;
}

//! Rows of the iteration table per level
struct Recorder : IterationObserver<CycleInfo>
{
    void add(const CycleInfo& info) override { rows[info.level].push_back(info); }
    std::map<unsigned, std::vector<CycleInfo>> rows;
};

//! Lumped Q1 hierarchy on the levels @p levels (ascending), as in staging/pos_sphere_ml.cc
struct Hierarchy
{
    Hierarchy(const std::vector<unsigned>& levels, double radius, unsigned coarse_max_iter)
        : levels(levels), lo(levels.front()), hi(levels.back())
        , builders(lo, hi), objectives(lo, hi), manifolds(lo, hi), transfers(lo, hi), point_transfers(lo, hi)
        , raw_transports(lo, hi), projections(lo, hi), transports(lo, hi), descent(lo, hi), solver(lo, hi)
    {
        for (const unsigned l : levels) {
            builders[l]   = std::make_unique<ModelBuilder<System>>(potential::Square<dim>{}, gpe_options(radius), l);
            objectives[l] = std::make_shared<Functional>(builders[l]->get_system(), beta, solver_options());
            manifolds[l]  = std::make_shared<Floored>(builders[l]->get_system().get_M(), 1e-12, true);
            descent[l]    = descent_options();
            solver[l]     = solver_options();
            if (l != hi) {
                descent[l].max_iter = coarse_max_iter;
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
            point_transfers[l] = std::make_shared<ManifoldTransfer<MassMatrix, InverseM>>(*transfers[l], Mc, Mf, Mc_inv);
            raw_transports[l]  = std::make_shared<AdjointDifferentialTransport<MassMatrix, InverseM>>(
                *transfers[l], *point_transfers[l], Mc, Mf, Mc_inv);
            projections[l] = std::make_shared<metric::mass::TangentProjection<MassMatrix>>(Mf);
            transports[l]  = std::make_shared<TangentProjectedTransport>(*raw_transports[l], *projections[l]);
        }
    }

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

struct Run
{
    std::map<unsigned, std::vector<CycleInfo>> rows;
    MGLevelObject<CorrectionStats> stats;
    Vector<double> x;
    [[nodiscard]] const std::vector<CycleInfo>& fine(unsigned level) const { return rows.at(level); }
};

//! FAS from the start of the hierarchy (stderr of FAS discarded)
Run run_fas(Hierarchy& h, const FAS_Options& options)
{
    FullApproximationScheme<Functional> fas(h.manifolds, h.point_transfers, h.transports, h.objectives, h.levels,
                                            h.descent, h.solver, options);
    Recorder rec;
    fas.set_observer(rec);
    EnergyOracle<System> O_fine(*h.objectives[h.hi], solver_options());
    MassOracle<System> T_fine(*h.objectives[h.hi], solver_options());
    Run run{.rows = {}, .stats = {}, .x = h.start()};
    std::ostringstream sink;
    auto* old = std::cerr.rdbuf(sink.rdbuf());
    fas.template cycle<MassOracle<System>, Tilt, Model, DensityCoarseResidual<System>>(
        O_fine, T_fine, O_fine.get_residual(), run.x, h.levels.size() - 1, sink);
    std::cerr.rdbuf(old);
    run.rows  = rec.rows;
    run.stats = fas.correction_stats();
    return run;
}

//! Single level on the finest level of the hierarchy, same oracle, manifold and options
std::vector<CycleInfo> run_single(Hierarchy& h)
{
    EnergyOracle<System> O(*h.objectives[h.hi], solver_options());
    Recorder rec;
    GradientDescent gd(O, O.get_residual(), *h.manifolds[h.hi], descent_options());
    gd.set_observer(rec);
    Vector<double> x = h.start();
    std::ostringstream sink;
    auto* old = std::cerr.rdbuf(sink.rdbuf());
    gd.cycle(x, sink);
    std::cerr.rdbuf(old);
    return rec.rows[0];
}

struct Counts
{
    unsigned accepted = 0, rejected = 0;
};

Counts corrections(const std::vector<CycleInfo>& rows)
{
    Counts c;
    for (const auto& row : rows) {
        if (row.coarse) {
            (row.step > 0 ? c.accepted : c.rejected)++;
        }
    }
    return c;
}

std::string name_of(const std::vector<unsigned>& levels, double radius)
{
    std::string s;
    for (const unsigned l : levels) {
        s += (s.empty() ? "" : "-") + std::to_string(l);
    }
    return fmt::format("{} r{}", s, radius);
}

//! Case 1: convergence, positivity, energy of single level, statistics against the table
void check_convergence(CheckReport& report, const std::vector<unsigned>& levels, double radius)
{
    Hierarchy h(levels, radius, 4);
    const Run run = run_fas(h, fas_options());
    const auto sl = run_single(h);
    const auto& rows = run.fine(h.hi);
    const std::string name = name_of(levels, radius);

    double min_value = std::numeric_limits<double>::infinity();
    for (const auto& row : rows) {
        min_value = std::min(min_value, row.min_value);
    }
    const double dE = std::abs(rows.back().energy - sl.back().energy) / std::abs(sl.back().energy);
    report.check(rows.back().residual < 1e-9 && min_value > 0 && dE < 1e-10, name + ": converged, positive, energy "
                 "of single level", fmt::format("residual {}, min {}, |E - E_sl| / |E| {}; iterations {} (single "
                 "level {})", sci(rows.back().residual), sci(min_value), sci(dE), rows.size() - 1, sl.size() - 1));

    bool consistent = true;
    std::string detail;
    for (const unsigned l : levels) {
        if (l == h.lo) {
            continue;
        }
        const Counts c = corrections(run.rows.count(l) ? run.rows.at(l) : std::vector<CycleInfo>{});
        const auto& s = run.stats[l];
        consistent = consistent && c.accepted + c.rejected + s.fallback == s.attempted;
        detail += fmt::format("level {}: attempted {}, accepted {}, rejected {}, fallback {}; ", l, s.attempted,
                              c.accepted, c.rejected, s.fallback);
    }
    const Counts c = corrections(rows);
    report.check(consistent && c.accepted > 0, name + ": accepted corrections, accepted + rejected + fallback = "
                 "attempted", detail);
}

//! Case 2: supporting inequality of one correction, at the start and after 5 fine steps
void check_supporting_inequality(CheckReport& report)
{
    Hierarchy h({4, 5}, 6.0, 4);
    Functional& f_h = *h.objectives[5];
    Functional& f_H = *h.objectives[4];
    EnergyOracle<System> O(f_h, solver_options());
    MassOracle<System> T_fine(f_h, solver_options()), T_coarse(f_H, solver_options());
    const ManifoldBase& coarse_manifold = *h.manifolds[4];
    CoarseOracleBase base(T_fine, T_coarse, coarse_manifold, *h.point_transfers[5], *h.transports[5]);

    for (const unsigned n_fine : {0U, 5U}) {
        Vector<double> x = h.start();
        if (n_fine > 0) {
            DescentOptions options = descent_options();
            options.max_iter = n_fine;
            GradientDescent gd(O, O.get_residual(), *h.manifolds[5], options);
            std::ostringstream sink;
            auto* old = std::cerr.rdbuf(sink.rdbuf());
            gd.cycle(x, sink);
            std::cerr.rdbuf(old);
        }
        O.update(x);
        base.update_model(x);
        const Vector<double> y = base.get_state().y;

        Model q(base, solver_options());
        q.update(y);
        const double q_y = q.value(y);
        Vector<double> z(y);
        DescentOptions coarse = descent_options();
        coarse.max_iter = 4;
        GradientDescent gd(q, q.get_residual(), coarse_manifold, coarse);
        std::ostringstream sink;
        auto* old = std::cerr.rdbuf(sink.rdbuf());
        gd.cycle(z, sink);
        std::cerr.rdbuf(old);
        q.update(z);
        const double q_z = q.value(z);

        Vector<double> v(z), d(x.size());
        coarse_manifold.retract_inv(v, y);
        h.transports[5]->vector_prolongation(x, y, v, d);
        O.update(x);
        const double dir_deriv = O.directional_derivative(x, d);
        const double tol = 10 * std::numeric_limits<double>::epsilon() * std::abs(T_coarse.value(y));
        report.check(q_z < q_y && dir_deriv <= q_z - q_y + tol,
                     fmt::format("supporting inequality after {} fine steps: Df_h(x)[d] <= q(z) - q(y) < 0", n_fine),
                     fmt::format("Df_h(x)[d] = {}, q(z) - q(y) = {}", sci(dir_deriv), sci(q_z - q_y)));
    }
}

//! Case 3: coarse max_iter = 0, every attempt a fallback
void check_empty_corrections(CheckReport& report)
{
    Hierarchy h({4, 5}, 6.0, 0);
    const auto sl = run_single(h);
    const Run none = run_fas(h, fas_options());

    auto same_iterates = [&](const std::vector<CycleInfo>& a, const std::vector<CycleInfo>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (unsigned k = 0; k < a.size(); ++k) {
            if (a[k].energy != b[k].energy || a[k].residual != b[k].residual) {
                return false;
            }
        }
        return true;
    };

    const auto& s = none.stats[5];
    report.check(s.attempted > 0 && s.fallback == s.attempted && corrections(none.fine(5)).accepted == 0 &&
                 same_iterates(none.fine(5), sl), "coarse max_iter = 0: every attempt a fallback, iterates of "
                 "single level", fmt::format("attempted {}, fallback {}, iterations {} (single level {})",
                                             s.attempted, s.fallback, none.fine(5).size() - 1, sl.size() - 1));
}

} // namespace


int main()
{
    return run_tests([](CheckReport& report) {
        for (const double radius : {6.0, 10.0}) {
            check_convergence(report, {4, 5}, radius);
            check_convergence(report, {3, 4, 5}, radius);
        }
        check_supporting_inequality(report);
        check_empty_corrections(report);
    });
}
