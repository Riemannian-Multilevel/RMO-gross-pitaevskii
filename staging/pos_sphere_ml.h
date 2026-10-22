/**
 * @file
 * @brief Shared part of the positive-sphere multilevel drivers pos_sphere_ml and main_coarse_lumping: the options of
 * the positive sphere (smoother, coarse retraction, point transfer, coarse gradient), the prototypes of the transfers,
 * the level hierarchy and the descent oracle of the finest level. Lumped mass, dimension 2.
 */
#ifndef RMO_STAGING_POS_SPHERE_ML_H
#define RMO_STAGING_POS_SPHERE_ML_H

#include <rmo/fe/interpolate.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/manifold_positive.h>
#include <rmo/gpe/metric.h>
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/oracle_coarse.h>
#include <rmo/gpe/transport.h>

#include <rmo/ropt/fas.h>

#include <rmo/option.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

namespace pos_sphere_ml
{
using namespace rmo;
using namespace rmo::gpe;
using namespace dealii;

constexpr int dim = 2;
using System     = GrossPitaevskiiLumpedSystem<dim>;
using MassMatrix = System::MassMatrix;
using Functional = GrossPitaevskiiFunctional<System>;
using InverseM   = Functional::InverseM;

//! Options of a run on the positive sphere (the option groups of main_coarse and the positive-sphere options)
struct Options
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

    [[nodiscard]] bool multilevel() const { return mg.v_levels.size() > 1; }
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

inline std::shared_ptr<ManifoldBase> make_manifold(const std::string& name, const MassMatrix& D, double floor)
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
    explicit Hierarchy(const Options& opts)
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
inline std::unique_ptr<GrossPitaevskiiOracle<System>> fine_oracle(const Options& opts, Functional& f)
{
    if (opts.metric_smooth == MetricKind::MASS) {
        return std::make_unique<MassOracle<System>>(f, opts.solver);
    }
    if (opts.smoother.starts_with("clip-")) {
        return std::make_unique<ClippedEnergyOracle<System>>(f, opts.solver);
    }
    return std::make_unique<EnergyOracle<System>>(f, opts.solver);
}


template <typename T>
void check_value(const std::string& option, const T& value, std::initializer_list<T> allowed)
{
    if (std::ranges::find(allowed, value) == allowed.end()) {
        throw std::invalid_argument(fmt::format("--{}: unknown value {}", option, value));
    }
}

//! Options of the positive sphere (with smoother_cli_options() for --metric-smooth)
inline po::options_description positive_cli_options()
{
    po::options_description opts("Positive sphere options");
    opts.add_options()
        ("smoother", po::value<std::string>()->default_value("floor"),
            "retraction of the finest level and of the correction (floor|plain|e-cap-up|clip-e-cap-up)")
        ("coarse-retraction", po::value<std::string>()->default_value("floor"),
            "retraction of the coarse levels, with the density lifting (floor|plain)")
        ("point-transfer", po::value<std::string>()->default_value("linear"), "point maps (linear|log-threshold), or restriction: the restriction-based transfer (prototype)")
        ("coarse-grad", po::value<std::string>()->default_value("model"),
            "gradient of the coarse model (model|model-shift|fine|mass|frobenius)")
        ("coarse-max-iter", po::value<unsigned>()->default_value(4), "iterations per coarse solve")
        ("floor", po::value<double>()->default_value(1e-12), "floor of FlooredPosSphere");
    return opts;
}

//! Reads the option groups of Options (gpe, descent, mg, inner, fas, smoother, positive) and checks them
inline void apply_options(const po::variables_map& vm, Options& opts, const std::string& program)
{
    apply_gpe_options(vm, opts.gpe);
    AssertThrow(opts.gpe.mass_lumping && opts.gpe.dimension == dim,
                ExcMessage(program + " requires --mass-lumping and --dimension 2"));
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
}

//! Calls @p f.template operator()<G>() with the DensityGradient G of --coarse-grad
template <typename F>
void with_coarse_grad(const std::string& name, F&& f)
{
    if (name == "model") f.template operator()<DensityGradient::ENERGY_MODEL>();
    else if (name == "model-shift") f.template operator()<DensityGradient::ENERGY_MODEL_SHIFT>();
    else if (name == "fine") f.template operator()<DensityGradient::ENERGY_FINE>();
    else if (name == "mass") f.template operator()<DensityGradient::MASS>();
    else f.template operator()<DensityGradient::FROBENIUS>();
}

//! The FAS cycle with the density coarse model, coarse gradient G; tables of the levels to @p os
template <DensityGradient G>
void fas_cycle(FullApproximationScheme<Functional>& fas, GrossPitaevskiiOracle<System>& O_fine,
               MassOracle<System>& T_fine, Vector<double>& x, unsigned n_levels, std::ostream& os)
{
    fas.template cycle<MassOracle<System>, DensityCoarseOracle<System, DensityGradient::MASS>,
                       DensityCoarseOracle<System, G>, DensityCoarseResidual<System>>(
        O_fine, T_fine, O_fine.get_residual(), x, n_levels - 1, os);
}

} // namespace pos_sphere_ml

#endif
