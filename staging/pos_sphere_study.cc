/**
 * @file
 * @brief Study of gradient descent on the positive unit-mass sphere (PosUnitMassSphere, density retraction) against
 * the unit-mass sphere (UnitMassSphere, norm retraction), see tex/pos_sphere_domain.tex.
 *
 * Takes the options of main (problem, descent, inner solver, --metric-smooth) on a lumped system, and runs
 * GradientDescent from the same start on the unit-mass sphere and, for every fraction in --thetas, on the positive
 * sphere with --boundary-fraction theta. Writes to --out:
 *
 * - `fields.csv`: per DoF the support point, the lumped mass d_i, the start x_0, the descent direction eta_0 at x_0,
 *   the step at which the density of the DoF reaches zero along eta_0 (inf if it grows), the final iterate of the
 *   unit-mass sphere and its eigenvalue residual D^{-1}(A x - lambda D x), and the first iterate of every case
 *   (x1_<case>);
 * - `iter_<case>.csv`: per iterate the residual, energy, minimum, the deviation |x^T D x - 1| from the sphere, the
 *   step taken to the next iterate, the step limit tau_max of the density retraction along the descent direction,
 *   whether the step was capped, the floor (continuation), the Z-matrix lower bound of tau_max (tau_bound()), and
 *   the starting step (--ls-initial) and reference value (--ls-reference) of the line search from the iterate, and
 *   the stationarity of the constrained problem (free residual, smallest multiplier, see stationarity());
 * - `summary.csv`: one line per case (appended), with the iterations, the final residual and energy, and whether the
 *   run failed (retraction undefined).
 *
 * The step limit is recomputed from the iterates (same gradient as in the solver, which is deterministic).
 *
 * Experimental variants of the positive sphere for large domains (--variants), not part of the library:
 * - `plain`: PosUnitMassSphere with option A (cases positive_theta<theta>);
 * - `clip`: energy-adaptive direction with w = A(x)^{-1} D x clipped at zero, i.e. eta_i >= -x_i as in exact
 *   arithmetic, with option A (positive_clip_theta<theta>);
 * - `floor`: density retraction clamped at the floor --floor, x_i' = sqrt(max(x_i^2 + 2 h x_i v_i, floor^2)), then
 *   normalized; option A only for the nodes above 10 floor (positive_floor_theta<theta>);
 * - `floor-nocap`: the clamped retraction without option A (positive_floor-nocap);
 * - `clip-floor`: both (positive_clip-floor_theta<theta>);
 * - `floor-cont`: the clamped retraction without option A, with a floor that decreases with the residual,
 *   floor_k = max(1e-150, min(floor_{k-1}, --floor-factor * r_k)), a continuation towards floor 0
 *   (positive_floor-cont).
 *
 * summary.csv also reports the discretization (mesh, degree) and its Z-matrix property: the number of
 * positive off-diagonal entries of A_0 (those of the stiffness matrix, for a lumped potential term).
 */
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/manifold.h>

#include <rmo/ropt/solver.h>

#include <rmo/option.h>
#include <rmo/util/util.h>

#include <deal.II/dofs/dof_tools.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>
#include <utility>
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

struct StudyOptions
{
    GPE_Options    gpe;
    DescentOptions descent;
    SolverOptions  solver;
    MG_Options     mg;
    MetricKind     metric_smooth;
    std::vector<double> thetas;
    std::vector<struct Variant> variants;
    double floor;
    double floor_factor;
    std::filesystem::path out;
};


//! Energy-adaptive oracle with w = A(x)^{-1} D x clipped at zero: grad_i <= x_i, i.e. eta_i >= -x_i, which holds in
//! exact arithmetic since A(x)^{-1} > 0; restores tau_max >= 1/2 for inexact inner solves.
class ClippedEnergyOracle : public EnergyOracle<System>
{
public:
    using EnergyOracle<System>::EnergyOracle;
    using EnergyOracle<System>::gradient;

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double residual) const override
    {
        const GradInfo info = EnergyOracle<System>::gradient(x, output, residual);
        for (unsigned i = 0; i < x.size(); ++i) {
            output[i] = std::min(output[i], x[i]);
        }
        return info;
    }
};


//! Positive sphere whose density retraction clamps every component at @p floor (then normalizes), so that it is
//! defined for every step; the step limit (option A) only counts the nodes above 10 floor, or none (@p cap false).
class FlooredPosSphere : public ManifoldBase
{
public:
    FlooredPosSphere(const MassMatrix& D, double floor, bool cap) : D(D), floor(floor), cap(cap) {}

    void set_floor(double floor_new) const { floor = floor_new; }
    [[nodiscard]] double get_floor() const { return floor; }

    void retract(const Vector<double>& z, Vector<double>& x, double factor) const override
    {
        for (unsigned i = 0; i < x.size(); ++i) {
            x[i] = std::sqrt(std::max(x[i] * x[i] + 2 * factor * x[i] * z[i], floor * floor));
        }
        ellipsoid::retract_by_norm(D, x);
    }

    void retract(const Vector<double>& z, const Vector<double>& x, Vector<double>& output, double factor) const override
    {
        output = x;
        retract(z, output, factor);
    }

    [[nodiscard]] double max_step(const Vector<double>& x, const Vector<double>& v) const override
    {
        double h_max = std::numeric_limits<double>::infinity();
        if (cap) {
            for (unsigned i = 0; i < x.size(); ++i) {
                if (v[i] < 0 && x[i] > 10 * floor) {
                    h_max = std::min(h_max, x[i] / (-2 * v[i]));
                }
            }
        }
        return h_max;
    }

    void retract_inv(Vector<double>& v, const Vector<double>& x) const override
    {
        ellipsoid_positive::retract_inv_by_density(v, x);
    }

    void retract_diff(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                      Vector<double>&) const override { throw ExcNotImplemented(); }
    void retract_inv_diff(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                          Vector<double>&) const override { throw ExcNotImplemented(); }
    void retract_inv_diff_adjoint(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                                  Vector<double>&) const override { throw ExcNotImplemented(); }

private:
    const MassMatrix& D;
    mutable double floor;
    bool cap;
};

//! Collects the rows reported by the solver; @p on_add (optional) runs after each row, before the next step.
struct Recorder : IterationObserver<CycleInfo>
{
    void add(const CycleInfo& info) override
    {
        rows.push_back(info);
        if (on_add) {
            on_add(info);
        }
    }
    std::vector<CycleInfo> rows;
    std::function<void(const CycleInfo&)> on_add;
};

//! Variant of the positive sphere (--variants), see the file documentation
struct Variant
{
    std::string name;
    bool clip;          // ClippedEnergyOracle
    bool floor;         // FlooredPosSphere
    bool cap;           // option A (fraction to the boundary)
    bool continuation;  // floor decreasing with the residual

    //! Name of the case: positive_theta<theta> (plain), positive_<name> (no cap), positive_<name>_theta<theta>
    [[nodiscard]] std::string case_name(double theta) const
    {
        if (!cap) {
            return "positive_" + name;
        }
        return name == "plain" ? fmt::format("positive_theta{}", theta) : fmt::format("positive_{}_theta{}", name, theta);
    }
};

const std::vector<Variant> all_variants = {
    {.name = "plain",       .clip = false, .floor = false, .cap = true,  .continuation = false},
    {.name = "clip",        .clip = true,  .floor = false, .cap = true,  .continuation = false},
    {.name = "floor",       .clip = false, .floor = true,  .cap = true,  .continuation = false},
    {.name = "floor-nocap", .clip = false, .floor = true,  .cap = false, .continuation = false},
    {.name = "clip-floor",  .clip = true,  .floor = true,  .cap = true,  .continuation = false},
    {.name = "floor-cont",  .clip = false, .floor = true,  .cap = false, .continuation = true},
};

Variant parse_variant(const std::string& name)
{
    const auto it = std::ranges::find(all_variants, name, &Variant::name);
    if (it == all_variants.end()) {
        throw std::invalid_argument("--variants: unknown variant " + name);
    }
    return *it;
}

//! Descent direction -grad f(x) of the oracle at x (updates the oracle to x).
Vector<double> descent_direction(GrossPitaevskiiOracle<System>& oracle, const Vector<double>& x)
{
    Vector<double> eta(x.size());
    oracle.update(x);
    oracle.gradient(x, eta);
    eta *= -1.0;
    return eta;
}

std::string num(double a)
{
    if (std::isnan(a)) {
        return "nan";
    }
    return std::isfinite(a) ? fmt::format("{:.10e}", a) : (a > 0 ? "inf" : "-inf");
}

/**
 * Stationarity of x for the positive sphere as a constrained problem (x >= 0): with r = A(x) x - lambda D x,
 * lambda = x^T A(x) x, the residual over the free nodes (x_i > 1e-8 max x), sqrt(sum r_i^2 / d_i), and the smallest
 * multiplier nu_i = r_i / d_i over the active nodes, which a KKT point has nonnegative.
 */
struct Stationarity
{
    double free_residual;
    double nu_min;  // +inf if no node is active
    unsigned n_active;
};

Stationarity stationarity(GrossPitaevskiiOracle<System>& oracle, const MassMatrix& D, const Vector<double>& x)
{
    oracle.update(x);
    Vector<double> Ax(x.size()), Dx(x.size());
    oracle.get_A().vmult(Ax, x);
    D.vmult(Dx, x);
    const double lambda = x * Ax;
    Ax.add(-lambda, Dx);  // r

    const auto& d = D.get_vector();
    const double active_tol = 1e-8 * x.linfty_norm();
    Stationarity result{0, std::numeric_limits<double>::infinity(), 0};
    for (unsigned i = 0; i < x.size(); ++i) {
        if (x[i] > active_tol) {
            result.free_residual += Ax[i] * Ax[i] / d[i];
        } else {
            result.nu_min = std::min(result.nu_min, Ax[i] / d[i]);
            ++result.n_active;
        }
    }
    result.free_residual = std::sqrt(result.free_residual);
    return result;
}

/**
 * Lower bound of the step limit tau_max along -grad f in exact arithmetic for a Z-matrix A(x) (off-diagonal entries
 * <= 0, x > 0): mass metric 1 / (2 max_i (A_ii / d_i - lambda)), Frobenius 1 / (2 max_i (A_ii - c d_i)) with
 * c = x^T D A x / x^T D^2 x; energy-adaptive 1/2 (A(x)^{-1} > 0).
 */
double tau_bound(GrossPitaevskiiOracle<System>& oracle, const MassMatrix& D, const Vector<double>& x, MetricKind metric)
{
    if (metric == MetricKind::ENERGY_ADAPTIVE) {
        return 0.5;
    }
    oracle.update(x);
    const auto& A = oracle.get_A();
    const Vector<double> diag = A.diagonal();
    Vector<double> Ax(x.size()), Dx(x.size());
    A.vmult(Ax, x);
    D.vmult(Dx, x);
    const auto& d = D.get_vector();

    double max_rate = 0;
    if (metric == MetricKind::MASS) {
        const double lambda = x * Ax;
        for (unsigned i = 0; i < x.size(); ++i) {
            max_rate = std::max(max_rate, diag[i] / d[i] - lambda);
        }
    } else {
        const double c = (Dx * Ax) / (Dx * Dx);
        for (unsigned i = 0; i < x.size(); ++i) {
            max_rate = std::max(max_rate, diag[i] - c * d[i]);
        }
    }
    return 1 / (2 * max_rate);
}

//! D^{-1} (A(x) x - lambda D x), lambda = x^T A(x) x: the eigenvalue residual per DoF
Vector<double> residual_field(GrossPitaevskiiOracle<System>& oracle, const MassMatrix& D, const Vector<double>& x)
{
    oracle.update(x);
    Vector<double> r(x.size()), Dx(x.size());
    oracle.get_A().vmult(r, x);
    D.vmult(Dx, x);
    r.add(-(x * r), Dx);
    const auto& d = D.get_vector();
    for (unsigned i = 0; i < x.size(); ++i) {
        r[i] /= d[i];
    }
    return r;
}

struct CaseResult
{
    std::string name;
    double theta;  // NaN: unit-mass sphere
    unsigned iterations;
    bool failed;
    bool converged;
    double residual, energy, min_value, elapsed;
    unsigned n_capped;
    Stationarity kkt;
};

//! Runs GradientDescent on @p manifold from @p x0 and writes iter_<name>.csv; @p x is the final (or last) iterate,
//! @p x1 the first.
CaseResult run_case(const std::string& name, double theta, const ManifoldBase& manifold, const MassMatrix& D,
                    GrossPitaevskiiOracle<System>& oracle, const Vector<double>& x0, DescentOptions options,
                    const std::filesystem::path& out, Vector<double>& x, Vector<double>& x1, MetricKind metric,
                    const FlooredPosSphere* continuation = nullptr, double floor_factor = 0)
{
    options.boundary_fraction = std::isnan(theta) ? 0.99 : theta;
    GradientDescent solver(oracle, oracle.get_residual(), manifold, options);
    Recorder recorder;
    std::vector<double> floors;  // floor used for the step from each iterate (continuation)
    if (continuation != nullptr) {
        continuation->set_floor(std::numeric_limits<double>::infinity());
        recorder.on_add = [&](const CycleInfo& info) {
            continuation->set_floor(std::max(1e-150, std::min(continuation->get_floor(), floor_factor * info.residual)));
            floors.push_back(continuation->get_floor());
        };
    }
    solver.set_observer(recorder);

    x = x0;
    bool failed = false;
    try {
        solver.cycle(x);
    }
    catch (const ExceptionBase&) {
        failed = true;
    }

    // Step limit of the density retraction along the descent direction at each iterate
    const auto& hist = solver.history();
    const double intended = options.line_search ? options.ls.alpha : options.step_size;
    const bool positive = dynamic_cast<const UnitMassSphere<MassMatrix>*>(&manifold) == nullptr;

    std::ofstream csv(out / fmt::format("iter_{}.csv", name));
    csv << "k,residual,energy,min_value,norm_dev,step,tau_max,cap,capped,floor,tau_bound,initial_step,reference,"
           "free_residual,nu_min\n";
    unsigned n_capped = 0;
    for (unsigned k = 0; k < recorder.rows.size(); ++k) {
        const CycleInfo& row = recorder.rows[k];
        double tau_max = std::numeric_limits<double>::infinity();
        double norm_dev = std::numeric_limits<double>::quiet_NaN();
        double bound = std::numeric_limits<double>::quiet_NaN();
        Stationarity kkt{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), 0};
        if (k < hist.size()) {
            bound = tau_bound(oracle, D, hist[k], metric);
            kkt = stationarity(oracle, D, hist[k]);
            const Vector<double> eta = descent_direction(oracle, hist[k]);
            tau_max = positive ? manifold.max_step(hist[k], eta) : ellipsoid_positive::max_step(hist[k], eta);
            Vector<double> Dx(hist[k].size());
            D.vmult(Dx, hist[k]);
            norm_dev = std::abs(hist[k] * Dx - 1);
        }
        const double cap = positive ? options.boundary_fraction * tau_max : std::numeric_limits<double>::infinity();
        const bool has_next = k + 1 < recorder.rows.size();
        const double step = has_next ? recorder.rows[k + 1].step : std::numeric_limits<double>::quiet_NaN();
        // Starting step and reference value of the step from x_k (nonmonotone line search, Barzilai-Borwein)
        double initial_step = intended;
        double reference = std::numeric_limits<double>::quiet_NaN();
        if (has_next) {
            for (const auto& [key, value] : recorder.rows[k + 1].extra) {
                if (key == "initial_step") initial_step = value;
                if (key == "reference") reference = value;
            }
        }
        const bool capped = has_next && cap < initial_step;
        n_capped += capped;
        csv << k << ',' << num(row.residual) << ',' << num(row.energy) << ',' << num(row.min_value) << ','
            << num(norm_dev) << ',' << num(step) << ',' << num(tau_max) << ',' << num(cap) << ',' << capped << ','
            << num(k < floors.size() ? floors[k] : std::numeric_limits<double>::quiet_NaN()) << ',' << num(bound)
            << ',' << num(has_next ? initial_step : std::numeric_limits<double>::quiet_NaN()) << ',' << num(reference)
            << ',' << num(kkt.free_residual) << ',' << num(kkt.nu_min) << '\n';
    }

    if (!hist.empty()) {
        x = hist.back();
    }
    x1 = hist.size() > 1 ? hist[1] : x0;
    AssertThrow(!recorder.rows.empty(), ExcMessage("no iterate recorded for case " + name));
    const CycleInfo& last = recorder.rows.back();
    return {name, theta, last.iter, failed, last.residual < options.tol_residual, last.residual, last.energy,
            last.min_value, last.elapsed, n_capped, stationarity(oracle, D, x)};
}

void write_fields(const ModelBuilder<System>& context, const Vector<double>& x0, const Vector<double>& eta0,
                  const Vector<double>& x_unit, const Vector<double>& res_unit,
                  const std::vector<std::pair<std::string, Vector<double>>>& first, const std::filesystem::path& out)
{
    const auto& package = context.get_package();
    std::vector<Point<dim>> points(context.n_dofs());
    DoFTools::map_dofs_to_support_points(package.get_mapping(), package.get_dofs(), points);
    const auto& d = context.get_M().get_vector();

    std::ofstream csv(out / "fields.csv");
    csv << "px,py,d,x0,eta0,limit0,x_unit,res_unit";
    for (const auto& [name, x1] : first) {
        csv << ",x1_" << name;
    }
    csv << '\n';
    for (unsigned i = 0; i < x0.size(); ++i) {
        const double limit = eta0[i] < 0 ? x0[i] / (-2 * eta0[i]) : std::numeric_limits<double>::infinity();
        csv << num(points[i][0]) << ',' << num(points[i][1]) << ',' << num(d[i]) << ',' << num(x0[i]) << ','
            << num(eta0[i]) << ',' << num(limit) << ',' << num(x_unit[i]) << ',' << num(res_unit[i]);
        for (const auto& [name, x1] : first) {
            csv << ',' << num(x1[i]);
        }
        csv << '\n';
    }
}

void study(const StudyOptions& opts)
{
    const unsigned level = opts.mg.v_levels.back();
    auto potential_v = potential::get_potential<dim>(opts.gpe.potential, opts.gpe.potential_expr);
    auto context = std::visit([&](auto&& V) { return ModelBuilder<System>(V, opts.gpe, level); }, potential_v);

    // Start as in main: constant, thresholded, normalized
    Vector<double> x0 = context.initial_value();
    context.distribute(x0);
    ellipsoid_positive::threshold(x0);
    ellipsoid::retract_by_norm(context.get_M(), x0);

    auto gp = context.get_eval(opts.gpe.beta, opts.solver);
    auto oracle = make_oracle(gp, opts.metric_smooth, opts.solver);
    const UnitMassSphere<MassMatrix> unit(context.get_M());
    const PosUnitMassSphere<MassMatrix> positive(context.get_M());

    std::filesystem::create_directories(opts.out);
    std::vector<CaseResult> results;

    Vector<double> x_unit(x0.size()), x(x0.size()), x1(x0.size());
    std::vector<std::pair<std::string, Vector<double>>> first;
    results.push_back(run_case("unit", std::numeric_limits<double>::quiet_NaN(), unit, context.get_M(), *oracle, x0,
                               opts.descent, opts.out, x_unit, x1, opts.metric_smooth));
    first.emplace_back("unit", x1);

    // Variants of the positive sphere, see the file documentation
    ClippedEnergyOracle clipped(gp, opts.solver);
    const FlooredPosSphere floored(context.get_M(), opts.floor, true);
    const FlooredPosSphere floored_nocap(context.get_M(), opts.floor, false);
    const FlooredPosSphere floored_cont(context.get_M(), opts.floor, false);
    for (const Variant& variant : opts.variants) {
        GrossPitaevskiiOracle<System>& variant_oracle = variant.clip ? clipped : *oracle;
        const ManifoldBase& manifold = !variant.floor        ? static_cast<const ManifoldBase&>(positive)
                                     : variant.continuation  ? static_cast<const ManifoldBase&>(floored_cont)
                                     : variant.cap           ? static_cast<const ManifoldBase&>(floored)
                                                             : static_cast<const ManifoldBase&>(floored_nocap);
        // Without the cap, theta has no effect
        const std::vector<double> thetas = variant.cap ? opts.thetas : std::vector<double>{0.99};
        for (const double theta : thetas) {
            const std::string name = variant.case_name(theta);
            results.push_back(run_case(name, theta, manifold, context.get_M(), variant_oracle, x0, opts.descent,
                                       opts.out, x, x1, opts.metric_smooth,
                                       variant.continuation ? &floored_cont : nullptr, opts.floor_factor));
            first.emplace_back(name, x1);
        }
    }
    const Vector<double> eta0 = descent_direction(*oracle, x0);
    write_fields(context, x0, eta0, x_unit, residual_field(*oracle, context.get_M(), x_unit), first, opts.out);

    // Z-matrix property: positive off-diagonal entries of A_0 (those of S, the potential term is lumped)
    unsigned n_pos_offdiag = 0;
    double max_pos_ratio = 0;
    const SparseMatrix<double>& A0 = context.get_A0();
    for (auto it = A0.begin(); it != A0.end(); ++it) {
        if (it->row() != it->column() && it->value() > 1e-12 * A0.diag_element(it->row())) {
            ++n_pos_offdiag;
            max_pos_ratio = std::max(max_pos_ratio, it->value() / A0.diag_element(it->row()));
        }
    }
    const std::filesystem::path summary = opts.out / "summary.csv";
    const bool header = !std::filesystem::exists(summary);
    std::ofstream csv(summary, std::ios::app);
    if (header) {
        csv << "mesh,degree,n_pos_offdiag,max_pos_ratio,"
               "level,n_dofs,radius,beta,metric,line_search,step_size,tol_inner_res,floor,case,theta,iterations,"
               "converged,failed,residual,energy,energy_diff,min_value,n_capped,tau_max0,elapsed,free_residual,nu_min,n_active,ls_reference,ls_initial\n";
    }
    const double E_unit = results.front().energy;
    const double tau_max0 = ellipsoid_positive::max_step(x0, eta0);
    std::cout << fmt::format("{} p{} level {} ({} DoFs), radius {}, beta {}, {}, positive off-diagonals {}, "
                             "tau_max(x0, eta0) = {:.3e}\n", enum_to_string(opts.gpe.mesh_kind), opts.gpe.degree,
                             level, x0.size(), opts.gpe.radius, opts.gpe.beta, enum_to_string(opts.metric_smooth),
                             n_pos_offdiag, tau_max0);
    for (const auto& r : results) {
        csv << enum_to_string(opts.gpe.mesh_kind) << ',' << opts.gpe.degree << ',' << n_pos_offdiag << ','
            << num(max_pos_ratio) << ','
            << level << ',' << x0.size() << ',' << opts.gpe.radius << ',' << opts.gpe.beta << ','
            << enum_to_string(opts.metric_smooth) << ',' << opts.descent.line_search << ','
            << opts.descent.step_size << ',' << opts.solver.tol_inner_res << ',' << num(opts.floor) << ','
            << r.name << ',' << num(r.theta)
            << ',' << r.iterations << ',' << r.converged << ',' << r.failed << ',' << num(r.residual) << ','
            << num(r.energy) << ',' << num(std::abs(r.energy - E_unit)) << ',' << num(r.min_value) << ','
            << r.n_capped << ',' << num(tau_max0) << ',' << num(r.elapsed) << ',' << num(r.kkt.free_residual) << ','
            << num(r.kkt.nu_min) << ',' << r.kkt.n_active << ',' << enum_to_string(opts.descent.ls.reference) << ','
            << enum_to_string(opts.descent.ls.initial) << '\n';
        std::cout << fmt::format("  {:24s} {:5d} iterations  {:9s}  residual {:.2e}  E {:.12f}  min {:.2e}  "
                                 "capped {:3d}  {:.2f} s  free res {:.1e}  nu_min {:.1e} ({} active)\n", r.name, r.iterations,
                                 r.failed ? "FAILED" : (r.converged ? "converged" : "stopped"), r.residual,
                                 r.energy, r.min_value, r.n_capped, r.elapsed, r.kkt.free_residual, r.kkt.nu_min,
                                 r.kkt.n_active);
    }
}

} // namespace


int main(int argc, char* argv[])
{
    StudyOptions opts{};
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(mg_cli_options());
        all.add(inner_cli_options());
        all.add(gpe::smoother_cli_options());
        po::options_description study_opts("Study options");
        study_opts.add_options()
            ("thetas", po::value<std::string>()->default_value("0.99"),
                "comma-separated fractions theta for the positive sphere")
            ("variants", po::value<std::string>()->default_value("plain"),
                "comma-separated variants of the positive sphere (plain|clip|floor|floor-nocap|clip-floor|floor-cont)")
            ("floor", po::value<double>()->default_value(1e-12),
                "floor of the variants floor, floor-nocap and clip-floor")
            ("floor-factor", po::value<double>()->default_value(1e-3),
                "floor of the variant floor-cont relative to the residual")
            ("out", po::value<std::string>()->default_value("pos_sphere_study"),
                "output directory");
        all.add(study_opts);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);
        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }

        apply_gpe_options(vm, opts.gpe);
        AssertThrow(opts.gpe.mass_lumping && opts.gpe.dimension == dim,
                    ExcMessage("pos_sphere_study requires --mass-lumping and --dimension 2"));
        apply_descent_options(vm, opts.descent);
        apply_mg_options(vm, opts.mg);
        apply_inner_options(vm, opts.solver);
        gpe::apply_smoother_options(vm, opts.metric_smooth);

        std::stringstream thetas(vm["thetas"].as<std::string>());
        for (std::string item; std::getline(thetas, item, ',');) {
            opts.thetas.push_back(std::stod(item));
        }
        std::stringstream variants(vm["variants"].as<std::string>());
        for (std::string item; std::getline(variants, item, ',');) {
            opts.variants.push_back(parse_variant(item));
            if (opts.variants.back().clip && opts.metric_smooth != MetricKind::ENERGY_ADAPTIVE) {
                throw std::invalid_argument("--variants " + item + " requires --metric-smooth energy_adaptive");
            }
        }
        opts.floor = vm["floor"].as<double>();
        opts.floor_factor = vm["floor-factor"].as<double>();
        opts.out = vm["out"].as<std::string>();

        study(opts);
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
