/**
 * @file
 * @brief Benchmark of the restriction of gradients with the point maps of the positive sphere
 * (tex/drafts/positive_multilevel.tex, md/positive_multilevel_benchmark.md, steps 3b/3c), on realistic points.
 *
 * For one problem (the options of pos_sphere_study; --levels gives the fine level, the coarse level is one below):
 * a ground state iteration on the fine level by GradientDescent with the descent options given -- on the positive
 * sphere (default; energy-adaptive with clip, e-retraction with the growth cap) or with --sphere unit on the unit
 * sphere (energy-adaptive, normalization retraction) -- evaluated at the iterates k = 10, 25, 50, 100, 200, the iterate
 * with the smallest component (positive sphere), and the final iterate. At each iterate x: the restriction y = r(x)
 * (the injection, the same for all point maps), and the fine gradients g of the oracles in the mass, Frobenius and
 * energy-adaptive metrics, and the mass gradient with the Rayleigh quotient lambda = x^T A x / x^T D x (`mass_rq`).
 *
 * The normal component of a gradient is |x^T D g| / |g|_D (D the lumped fine masses; x^T D x = 1 up to rounding);
 * the restrictions below use the projected gradient g - (x^T D g / x^T D x) x. For each g and transfer variant, the
 * restriction R g = Pi_y Dp(y)^* g, on the positive sphere also its relative form (R g)_j / y_j (the quantity w/y of
 * the density coarse model):
 * - `linear`: ManifoldTransfer (unit-sphere point maps; both spheres; no reference);
 * - `log`: LogManifoldTransfer (the adjoint forms x_hat_i d_i u_i, which underflows for x below ~1e-154);
 * - `banded`: the log maps with the adjoint by exponent bands (frexp/ldexp, Tfine once per band, exact rescaling by
 *   powers of two), evaluated directly in relative form; the reference for `log` and `threshold`;
 * - `threshold` (eps 1e-12, 1e-50, 1e-100, 1e-140, 1e-150): LogThresholdManifoldTransfer, the log maps on inputs
 *   clamped at eps, with the differentials of the clamped maps and their exact transpose.
 *
 * Writes (appends) one row per gradient and variant to --out (CSV): the sphere, tol_inner_res, the problem, the
 * iterate (`point`), its residual, min x, min y; the number of floored coarse nodes; the maximum entrywise relative
 * error of (R g)/y against `banded` (over all nodes, and over the nodes with y_j >= 10 eps), the error relative to the
 * size of w/y (max |(R g)/y| and |grad f_H(y)/y|), the number of entries 0 where the reference is not, the number of
 * non-finite entries of the variant and of the reference (counted separately, not part of the maxima); the Galerkin
 * defect |<P v, g>_h - <v, R g>_H| / (|P v| |g|) for a tangent coarse v, with the computed (`galerkin_raw`) and the
 * projected gradient (`galerkin`); the normal component of g; the number of bands of `banded`; the CPU time of one
 * restriction.
 *
 * With the environment variable RMO_DUMP_DIR, writes for every iterate x, the masses, A_0 (triplets), beta and the
 * computed gradients (17 significant digits, which reproduce the doubles exactly) to <dir>/<problem>_<iterate>_*.txt,
 * for the exact-arithmetic check of the gradients (staging scripts of the session, not part of the repository).
 */
#include <rmo/fe/interpolate.h>
#include <rmo/gpe/manifold_positive.h>
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/transport.h>
#include <rmo/ropt/solver.h>

#include <rmo/option.h>

#include <deal.II/base/timer.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
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
using Operator   = Functional::Operator;
using InverseM   = Functional::InverseM;

//! (R g)_j / y_j from an absolute restriction, with R g = Pi_y (...) already applied
Vector<double> divide(const Vector<double>& a, const Vector<double>& y)
{
    Vector<double> r(a);
    for (unsigned j = 0; j < r.size(); ++j) {
        r[j] /= y[j];
    }
    return r;
}

/**
 * Relative restriction (Pi_y Dp(y)^* u)_j / y_j for the log point maps, by exponent bands: the terms
 * a_i = x_hat_i d_i (Pi_x_hat u)_i are kept as mantissa m_i and exponent e_i (frexp of the factors; x_hat from its
 * logarithm, not thresholded), I^T is applied once per band of e_i of width 800 to ldexp(m_i, e_i - E) (normal
 * doubles, exact scaling), and each coarse entry is rescaled by 2^E / (d_j y_j^2). Returns the number of bands.
 */
unsigned banded_relative_restriction(const LinearTransferBase& transfer, const Vector<double>& d_h,
                                     const Vector<double>& d_H, const Vector<double>& y, const Vector<double>& u,
                                     Vector<double>& dst)
{
    const unsigned n_h = d_h.size(), n_H = d_H.size();
    // log x_hat = I log y - log n, n^2 = sum_i d_i exp(2 (I log y)_i)
    Vector<double> log_y(n_H), lx(n_h);
    for (unsigned j = 0; j < n_H; ++j) {
        log_y[j] = std::log(y[j]);
    }
    transfer.to_fine_mesh(log_y, lx);
    const double lx_max = *std::max_element(lx.begin(), lx.end());
    double sum = 0;
    for (unsigned i = 0; i < n_h; ++i) {
        sum += d_h[i] * std::exp(2 * (lx[i] - lx_max));
    }
    const double log_n = lx_max + 0.5 * std::log(sum);
    Vector<double> x_hat(n_h);
    double c = 0;
    for (unsigned i = 0; i < n_h; ++i) {
        lx[i] -= log_n;
        x_hat[i] = std::exp(lx[i]);
        c += d_h[i] * x_hat[i] * u[i];
    }
    // a_i = x_hat_i d_i (u_i - c x_hat_i) as m_i 2^e_i
    std::vector<double> m(n_h);
    std::vector<int> e(n_h);
    bool any = false;
    int e_min = std::numeric_limits<int>::max(), e_max = std::numeric_limits<int>::min();
    for (unsigned i = 0; i < n_h; ++i) {
        const double pu = u[i] - c * x_hat[i];
        if (pu == 0) {
            m[i] = 0;
            e[i] = 0;
            continue;
        }
        const double l2 = lx[i] / std::log(2.0);
        const int ex = static_cast<int>(std::floor(l2)) + 1;
        const double mx = std::exp2(l2 - ex);
        int ed, eu;
        const double md = std::frexp(d_h[i], &ed), mu = std::frexp(pu, &eu);
        m[i] = mx * md * mu;
        e[i] = ex + ed + eu;
        e_min = std::min(e_min, e[i]);
        e_max = std::max(e_max, e[i]);
        any = true;
    }
    // Coarse scale d_j y_j^2 as mantissa and exponent
    std::vector<double> mk(n_H);
    std::vector<int> ek(n_H);
    for (unsigned j = 0; j < n_H; ++j) {
        int ed, ey;
        const double md = std::frexp(d_H[j], &ed), my = std::frexp(y[j], &ey);
        mk[j] = md * my * my;
        ek[j] = ed + 2 * ey;
    }
    dst.reinit(n_H);
    unsigned n_bands = 0;
    if (any) {
        constexpr int width = 800;
        Vector<double> scaled(n_h), t(n_H);
        for (int E = e_max; E > e_min - width; E -= width) {
            scaled = 0;
            bool in_band = false;
            for (unsigned i = 0; i < n_h; ++i) {
                if (m[i] != 0 && e[i] <= E && e[i] > E - width) {
                    scaled[i] = std::ldexp(m[i], e[i] - E);
                    in_band = true;
                }
            }
            if (!in_band) {
                continue;
            }
            n_bands++;
            transfer.Tfine(scaled, t);
            for (unsigned j = 0; j < n_H; ++j) {
                if (t[j] != 0) {
                    dst[j] += std::ldexp(t[j] / mk[j], E - ek[j]);
                }
            }
        }
    }
    // Relative projection onto T_y: (Pi_y w)_j / y_j = w_j / y_j - y^T D_H w, y^T D_H w = sum_j d_j y_j^2 (w_j / y_j)
    double ydw = 0;
    for (unsigned j = 0; j < n_H; ++j) {
        ydw += d_H[j] * y[j] * y[j] * dst[j];
    }
    dst.add(-ydw);
    return n_bands;
}

double d_inner(const Vector<double>& d, const Vector<double>& a, const Vector<double>& b)
{
    double s = 0;
    for (unsigned i = 0; i < a.size(); ++i) {
        s += d[i] * a[i] * b[i];
    }
    return s;
}

std::string dump_prefix(const std::string& point_label, const GPE_Options& o, unsigned level)
{
    return fmt::format("{}{}_l{}_{}_r{}_b{}_{}", o.mesh_kind == MeshKind::QUADRILATERAL ? "q" : "p", o.degree, level,
                       o.potential == Potential::SQUARE ? "square" : "optical_lattice", o.radius, o.beta, point_label);
}

//! With RMO_DUMP_DIR set: writes @p v (17 digits, one per line) to <dir>/<prefix>_<name>.txt
void dump_vector(const std::string& point_label, const std::string& name, const Vector<double>& v,
                 const GPE_Options& o, unsigned level)
{
    if (const char* dir = std::getenv("RMO_DUMP_DIR")) {
        std::ofstream f(std::filesystem::path(dir) / (dump_prefix(point_label, o, level) + "_" + name + ".txt"));
        f << std::setprecision(17);
        for (const double a : v) {
            f << a << '\n';
        }
    }
}

template <typename Builder>
void evaluate(const GPE_Options& gpe_opts, unsigned level, const SolverOptions& solver, Builder& fine,
              Builder& coarse, Functional& func_f, Functional& func_c, const Vector<double>& d_h,
              const Vector<double>& d_H, const std::string& point_label, const Vector<double>& x, bool unit_sphere,
              const std::filesystem::path& out);

void study(const GPE_Options& gpe_opts, const DescentOptions& descent, const SolverOptions& solver, unsigned level,
           bool unit_sphere, const std::filesystem::path& out)
{
    auto potential_v = potential::get_potential<dim>(gpe_opts.potential, gpe_opts.potential_expr);
    auto fine = std::visit([&](auto&& V) { return ModelBuilder<System>(V, gpe_opts, level); }, potential_v);
    auto coarse = std::visit([&](auto&& V) { return ModelBuilder<System>(V, gpe_opts, level - 1); }, potential_v);
    auto func_f = fine.get_eval(gpe_opts.beta, solver);
    auto func_c = coarse.get_eval(gpe_opts.beta, solver);
    const Vector<double> d_h = fine.get_M().get_vector(), d_H = coarse.get_M().get_vector();

    // Ground state on the fine level: positive sphere (EA with clip, e-retraction with the growth cap) or unit sphere
    // (EA, normalization retraction)
    Vector<double> x = fine.initial_value();
    fine.distribute(x);
    ellipsoid_positive::threshold(x);
    ellipsoid::retract_by_norm(fine.get_M(), x);
    std::vector<std::pair<std::string, Vector<double>>> points;  // iterates at which the transfers are evaluated
    {
        ClippedEnergyOracle<System> clipped(func_f, solver);
        EnergyOracle<System> plain(func_f, solver);
        GrossPitaevskiiOracle<System>& oracle = unit_sphere ? static_cast<GrossPitaevskiiOracle<System>&>(plain) : clipped;
        const ExpPosSphere<MassMatrix> positive(fine.get_M(), ExpCap::GROWTH);
        const UnitMassSphere<MassMatrix> unit(fine.get_M());
        const ManifoldBase& manifold = unit_sphere ? static_cast<const ManifoldBase&>(unit) : positive;
        DescentOptions options = descent;
        options.boundary_fraction = 0.99;
        GradientDescent gd(oracle, oracle.get_residual(), manifold, options);
        try {
            gd.cycle(x);
        } catch (const ExceptionBase&) {
            std::cerr << "warning: ground state iteration failed, using the last iterate\n";
        }
        // Iterates 10, 25, 50, 100, 200 (approach to convergence), the iterate with the smallest component (transient
        // tails, positive sphere), and the final iterate
        const auto& hist = gd.history();
        for (const unsigned k : {10u, 25u, 50u, 100u, 200u}) {
            if (k + 1 < hist.size()) {
                points.emplace_back(fmt::format("k{}", k), hist[k]);
            }
        }
        if (!unit_sphere && !hist.empty()) {
            unsigned k_deep = 0;
            double deep = std::numeric_limits<double>::infinity();
            for (unsigned k = 0; k < hist.size(); ++k) {
                const double m = *std::min_element(hist[k].begin(), hist[k].end());
                if (m < deep) {
                    deep = m;
                    k_deep = k;
                }
            }
            if (k_deep + 1 < hist.size()) {
                points.emplace_back(fmt::format("deepest_k{}", k_deep), hist[k_deep]);
            }
        }
        points.emplace_back(fmt::format("final_k{}", hist.empty() ? 0 : hist.size() - 1), x);
    }
    for (const auto& [label, point] : points) {
        evaluate(gpe_opts, level, solver, fine, coarse, func_f, func_c, d_h, d_H, label, point, unit_sphere, out);
    }
}

//! Pi_y Dp(y)^* g (absolute), the restriction of AdjointDifferentialTransport for the point transfer @p pt
Vector<double> absolute_restriction(const ManifoldTransferBase& pt, const Vector<double>& d_H, const Vector<double>& y,
                                    const Vector<double>& g)
{
    Vector<double> w(d_H.size());
    pt.diff_prolongation_adjoint(y, g, w);
    w.add(-d_inner(d_H, y, w), y);
    return w;
}

template <typename Builder>
void evaluate(const GPE_Options& gpe_opts, unsigned level, const SolverOptions& solver, Builder& fine,
              Builder& coarse, Functional& func_f, Functional& func_c, const Vector<double>& d_h,
              const Vector<double>& d_H, const std::string& point_label, const Vector<double>& x, bool unit_sphere,
              const std::filesystem::path& out)
{
    // The residual uses A(x): update the functional to x first (M_pp is shared state)
    MassOracle<System> T_res(func_f, solver);
    T_res.update(x);
    const double res_x = T_res.get_residual().residual(x);

    const fe::LinearTransferMG<dim> transfer(coarse.get_dofs(), fine.get_dofs(), coarse.get_package().get_constraints(),
                                             fine.get_package().get_constraints());
    const ManifoldTransfer<Operator, InverseM> linear_maps(transfer, func_c.get_M(), func_f.get_M(), func_c.get_M_inv());
    const LogManifoldTransfer<Operator, InverseM> log_maps(transfer, func_c.get_M(), func_f.get_M(),
                                                           func_c.get_M_inv());
    Vector<double> y(d_H.size());
    linear_maps.restriction(x, y);  // the injection: the same point for the log maps
    // The log, banded and thresholded maps (and the relative quantities) only on the positive sphere
    const bool positive = !unit_sphere && *std::min_element(x.begin(), x.end()) > 0 &&
                          *std::min_element(y.begin(), y.end()) > 0;

    // Coarse M-gradient at y (for the size of w/y)
    MassOracle<System> T_coarse(func_c, solver);
    T_coarse.update(y);
    Vector<double> y_grad(d_H.size());
    T_coarse.gradient(y, y_grad);
    const Vector<double> y_grad_rel = positive ? divide(y_grad, y) : Vector<double>(d_H.size());

    // Fine gradients: mass (kernels::grad_mass, lambda = x^T A x), mass with the Rayleigh quotient
    // lambda = x^T A x / x^T D x (tangent for every x in exact arithmetic), Frobenius, energy-adaptive
    MassOracle<System> T_mass(func_f, solver);
    FrobeniusOracle<System> T_frob(func_f, solver);
    EnergyOracle<System> T_ea(func_f, solver);
    std::vector<std::string> names;
    std::vector<Vector<double>> g_raw, g_proj;
    std::vector<double> normal_part;
    // With RMO_DUMP_DIR: x, the masses, A_0 and beta of this point for an exact-arithmetic check of the gradients
    dump_vector(point_label, "x", x, gpe_opts, level);
    dump_vector(point_label, "d", d_h, gpe_opts, level);
    if (const char* dir = std::getenv("RMO_DUMP_DIR")) {
        std::ofstream a(std::filesystem::path(dir) / (dump_prefix(point_label, gpe_opts, level) + "_A0.txt"));
        a << std::setprecision(17);
        const SparseMatrix<double>& A0 = fine.get_A0();
        for (auto it = A0.begin(); it != A0.end(); ++it) {
            a << it->row() << ' ' << it->column() << ' ' << it->value() << '\n';
        }
        std::ofstream b(std::filesystem::path(dir) / (dump_prefix(point_label, gpe_opts, level) + "_beta.txt"));
        b << std::setprecision(17) << gpe_opts.beta << '\n';
    }
    const auto add_gradient = [&](const std::string& name, const Vector<double>& g) {
        dump_vector(point_label, "g_" + name, g, gpe_opts, level);
        names.push_back(name);
        g_raw.push_back(g);
        normal_part.push_back(std::abs(d_inner(d_h, x, g)) / std::sqrt(d_inner(d_h, g, g)));
        Vector<double> gp(g);
        gp.add(-d_inner(d_h, x, g) / d_inner(d_h, x, x), x);
        g_proj.push_back(gp);
    };
    for (auto [name, oracle] : std::vector<std::pair<std::string, GrossPitaevskiiOracle<System>*>>{
             {"mass", &T_mass}, {"frobenius", &T_frob}, {"energy_adaptive", &T_ea}}) {
        oracle->update(x);
        Vector<double> g(d_h.size());
        oracle->gradient(x, g);
        add_gradient(name, g);
        if (name == "mass") {
            Vector<double> Ax(d_h.size());
            func_f.get_A().vmult(Ax, x);
            const double lambda = (x * Ax) / d_inner(d_h, x, x);
            Vector<double> g_rq(d_h.size());
            for (unsigned i = 0; i < g_rq.size(); ++i) {
                g_rq[i] = (Ax[i] - lambda * d_h[i] * x[i]) / d_h[i];
            }
            add_gradient("mass_rq", g_rq);
        }
    }

    const bool new_file = !std::filesystem::exists(out);
    std::ofstream csv(out, std::ios::app);
    if (new_file) {
        csv << "sphere,tol_inner_res,mesh,degree,level,n_coarse,beta,potential,radius,point,residual_x,min_x,min_y,"
               "metric,variant,eps,n_floored,err_rel,err_rel_unfloored,err_w,n_zero,n_nonfinite,n_nonfinite_ref,"
               "galerkin_raw,galerkin,normal_part_g,n_bands,time_us\n";
    }
    const std::string potential_name = gpe_opts.potential == Potential::SQUARE ? "square" : "optical_lattice";
    const double min_x = *std::min_element(x.begin(), x.end()), min_y = *std::min_element(y.begin(), y.end());
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (unsigned ig = 0; ig < names.size(); ++ig) {
        const Vector<double>& g = g_proj[ig];
        // Reference (positive sphere, log maps): banded relative restriction
        Vector<double> ref(d_H.size());
        dealii::Timer timer;
        unsigned n_bands = 0;
        double t_banded = nan, w_scale = 0;
        if (positive) {
            n_bands = banded_relative_restriction(transfer, d_h, d_H, y, g, ref);
            t_banded = 1e6 * timer.cpu_time();
            for (unsigned j = 0; j < d_H.size(); ++j) {
                w_scale = std::max(w_scale, std::max(std::abs(ref[j]), std::abs(y_grad_rel[j])));
            }
        }

        // Variants: `linear` (both spheres, no reference), and on the positive sphere `log`, `banded`, `threshold`
        struct Variant { std::string name; double eps; const ManifoldTransferBase* pt; bool compare; };
        using Threshold = LogThresholdManifoldTransfer<Operator, InverseM>;
        std::vector<std::unique_ptr<Threshold>> thresholded;
        std::vector<Variant> variants{{"linear", 0, &linear_maps, false}};
        if (positive) {
            variants.push_back({"log", 0, &log_maps, true});
            variants.push_back({"banded", 0, nullptr, true});
            for (const double eps : {1e-12, 1e-50, 1e-100, Threshold::default_eps, 1e-150}) {
                thresholded.push_back(std::make_unique<Threshold>(transfer, func_c.get_M(), func_f.get_M(),
                                                                  func_c.get_M_inv(), eps));
                variants.push_back({"threshold", eps, thresholded.back().get(), true});
            }
        }

        for (const auto& v : variants) {
            Vector<double> rel(d_H.size()), y_v(y);
            double time_us = t_banded, galerkin = nan, galerkin_raw = nan;
            if (v.pt != nullptr) {
                v.pt->restriction(x, y_v);
                timer.restart();
                const Vector<double> rg = absolute_restriction(*v.pt, d_H, y_v, g);
                time_us = 1e6 * timer.cpu_time();
                if (positive) {
                    rel = divide(rg, y_v);
                }
                // Galerkin defect <P v, g>_h vs <v, R g>_H for a tangent coarse direction, with the projected and
                // the computed (raw) gradient; P v = Pi_x Dp(y) v is tangent, so <P v, g_raw> = <P v, g>
                Vector<double> vc(d_H.size()), pv(d_h.size());
                for (unsigned j = 0; j < vc.size(); ++j) {
                    vc[j] = y_v[j] * std::cos(0.37 * j);
                }
                vc.add(-d_inner(d_H, y_v, vc), y_v);
                v.pt->diff_prolongation(y_v, vc, pv);
                pv.add(-d_inner(d_h, x, pv) / d_inner(d_h, x, x), x);
                const double scale = std::sqrt(d_inner(d_h, pv, pv) * d_inner(d_h, g, g));
                const double lhs = d_inner(d_h, pv, g_raw[ig]);
                galerkin = std::abs(d_inner(d_h, pv, g) - d_inner(d_H, vc, rg)) / scale;
                const Vector<double> rg_raw = absolute_restriction(*v.pt, d_H, y_v, g_raw[ig]);
                galerkin_raw = std::abs(lhs - d_inner(d_H, vc, rg_raw)) / scale;
            } else {
                rel = ref;
            }
            double err = nan, err_unfl = nan, err_w = nan;
            unsigned n_zero = 0, n_floored = 0, n_nonfinite = 0, n_nonfinite_ref = 0;
            if (v.compare) {
                err = err_unfl = err_w = 0;
                for (unsigned j = 0; j < d_H.size(); ++j) {
                    const bool floored_node = v.eps > 0 && y[j] < 10 * v.eps;
                    n_floored += floored_node ? 1 : 0;
                    // Non-finite entries are counted (std::max would drop NaN silently)
                    if (!std::isfinite(ref[j])) {
                        n_nonfinite_ref++;
                        continue;
                    }
                    if (!std::isfinite(rel[j])) {
                        n_nonfinite++;
                        continue;
                    }
                    if (ref[j] == 0) {
                        continue;
                    }
                    const double e_j = std::abs(rel[j] - ref[j]) / std::abs(ref[j]);
                    err = std::max(err, e_j);
                    if (!floored_node) {
                        err_unfl = std::max(err_unfl, e_j);
                    }
                    err_w = std::max(err_w, std::abs(rel[j] - ref[j]) / w_scale);
                    n_zero += rel[j] == 0 ? 1 : 0;
                }
            }
            csv << fmt::format("{},{:.0e},{},{},{},{},{},{},{},{},{:.3e},{:.3e},{:.3e},{},{},{:.0e},{},{:.3e},{:.3e},"
                               "{:.3e},{},{},{},{:.3e},{:.3e},{:.1e},{},{:.1f}\n",
                               unit_sphere ? "unit" : "positive", solver.tol_inner_res,
                               gpe_opts.mesh_kind == MeshKind::QUADRILATERAL ? "Q" : "P", gpe_opts.degree, level,
                               d_H.size(), gpe_opts.beta, potential_name, gpe_opts.radius, point_label, res_x, min_x,
                               min_y, names[ig], v.name, v.eps, n_floored, err, err_unfl, err_w, n_zero, n_nonfinite,
                               n_nonfinite_ref, galerkin_raw, galerkin, normal_part[ig], n_bands, time_us);
        }
    }
    std::cout << fmt::format("{} {}{} level {} radius {} {} {}: residual {:.1e}, min x {:.1e}, min y {:.1e}\n",
                             unit_sphere ? "unit" : "positive", gpe_opts.mesh_kind == MeshKind::QUADRILATERAL ? "Q" : "P",
                             gpe_opts.degree, level, gpe_opts.radius, potential_name, point_label, res_x, min_x, min_y);
}

} // namespace


int main(int argc, char* argv[])
{
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(mg_cli_options());
        all.add(inner_cli_options());
        all.add_options()("out", po::value<std::string>()->default_value("transfer_study.csv"), "output file (appended)")
            ("sphere", po::value<std::string>()->default_value("positive"),
                "positive (log, banded, floored and linear point maps) or unit (linear point maps)");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);
        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }
        GPE_Options gpe_opts{};
        DescentOptions descent{};
        MG_Options mg{};
        SolverOptions solver{};
        apply_gpe_options(vm, gpe_opts);
        AssertThrow(gpe_opts.mass_lumping && gpe_opts.dimension == dim,
                    ExcMessage("transfer_study requires --mass-lumping and --dimension 2"));
        apply_descent_options(vm, descent);
        apply_mg_options(vm, mg);
        apply_inner_options(vm, solver);
        const std::string sphere = vm["sphere"].as<std::string>();
        AssertThrow(sphere == "positive" || sphere == "unit", ExcMessage("--sphere: positive or unit"));
        study(gpe_opts, descent, solver, mg.v_levels.back(), sphere == "unit", vm["out"].as<std::string>());
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
