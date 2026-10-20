/**
 * @file
 * @brief Checks of the transfers of the positive unit-mass sphere (tex/pos_sphere_variants.tex, "Transfers for the
 * positive sphere") on nested finite element spaces, with \f$ \tilde P \f$ the embedding of LinearTransferMG
 * (to_fine_mesh(), transpose Tfine()), lumped masses \f$ d_h, d_H \f$ and Neumann boundary, for a positive point
 * \f$ x \f$ with Gaussian tails (radius 10; the point maps also at radius 22, tails below \f$ 10^{-154} \f$, where
 * squares and products of components underflow):
 *
 * - partition of unity \f$ \tilde P \onebb = \onebb \f$ and mass consistency \f$ \tilde P^\top d_h = d_H \f$;
 * - point restriction by full weighting of the densities, \f$ y = (\tilde P^\top(d_h x^2) / d_H)^{1/2} \f$: on the
 *   coarse sphere; the restriction of a constant is constant;
 * - relative prolongation \f$ P v = x \circ \tilde P(v / y) \f$: tangent at \f$ x \f$ (and not for the injected
 *   point of ManifoldTransfer); step limit \f$ \tau_{\max}(x, Pv) \ge \tau_{\max}(y, v) \f$;
 * - restriction of gradients \f$ R g = y^{-1} \circ D_H^{-1} \tilde P^\top D_h (x \circ g) \f$: tangent at \f$ y \f$,
 *   and the Galerkin condition \f$ \langle P v, g \rangle_{D_h} = \langle v, R g \rangle_{D_H} \f$;
 * - coarse correction \f$ \Phi(z) = x \circ \tilde P(z^2 / y^2)^{1/2} \f$: \f$ \Phi(y) = x \f$, on the fine sphere,
 *   equal to the density retraction \f$ R_x(P L_y(z)) \f$, and positive where the density retraction cancels.
 *
 * The relative transfers need \f$ \tilde P \ge 0 \f$. The embedding of Q2 has negative entries (reported); for Q2 the
 * checks use the Q1 embedding between the Q2 nodes instead, which is nonnegative but not mass consistent.
 *
 * Reported (INFO): the relative difference of the relative and the linear prolongation \f$ \Pi_x \tilde P v \f$, and
 * the largest relative change \f$ |v_i| / x_i \f$ of both, which sets the step limit.
 */
#include "check.h"

#include <rmo/fe/interpolate.h>
#include <rmo/gpe/gpe.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/transport.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/full_matrix.h>

#include <cmath>
#include <limits>
#include <map>
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

struct Discretization
{
    std::string name;
    MeshKind mesh;
    int degree;
    bool nonnegative;  // the embedding of the coarse into the fine space has nonnegative entries
};

GPE_Options gpe_options(const Discretization& disc, double radius = 10.0)
{
    GPE_Options options{};
    options.dimension    = dim;
    options.degree       = disc.degree;
    options.radius       = radius;
    options.beta         = 100.0;
    options.order        = Ordering::DEFAULT;
    options.bc           = BoundaryCondition::NEUMANN;
    options.mesh_kind    = disc.mesh;
    options.potential    = Potential::SQUARE;
    options.mass_lumping = true;
    return options;
}

struct Level
{
    Level(const Discretization& disc, unsigned n_levels, double radius = 10.0)
        : package(gpe_options(disc, radius), n_levels)
        , system(package.system<System>(V))
    {}

    [[nodiscard]] const MassMatrix& D() const { return system.get_M(); }
    [[nodiscard]] const Vector<double>& d() const { return system.get_M().get_vector(); }

    const potential::Square<dim> V{};
    GrossPitaevskiiPackage<dim> package;
    System system;
};

double d_inner(const Vector<double>& d, const Vector<double>& a, const Vector<double>& b)
{
    double s = 0;
    for (unsigned i = 0; i < a.size(); ++i) {
        s += d[i] * a[i] * b[i];
    }
    return s;
}

//! Function values at the support points of the level
template <typename Function>
Vector<double> at_support_points(const Level& level, Function&& f)
{
    std::vector<Point<dim>> points(level.package.n_dofs());
    DoFTools::map_dofs_to_support_points(level.package.get_mapping(), level.package.get_dofs(), points);
    Vector<double> v(points.size());
    for (unsigned i = 0; i < points.size(); ++i) {
        v[i] = f(points[i]);
    }
    return v;
}

//! Columns P e_j of a linear transfer, as a dense matrix
FullMatrix<double> transfer_matrix(const LinearTransferBase& transfer)
{
    FullMatrix<double> P(transfer.n_fine(), transfer.n_coarse());
    Vector<double> e(transfer.n_coarse()), col(transfer.n_fine());
    for (unsigned j = 0; j < transfer.n_coarse(); ++j) {
        e = 0.0;
        e[j] = 1.0;
        transfer.to_fine_mesh(e, col);
        for (unsigned i = 0; i < transfer.n_fine(); ++i) {
            P(i, j) = col[i];
        }
    }
    return P;
}

//! Index of every support point of @p from among the support points of @p to (same point set)
std::vector<unsigned> match_points(const Level& from, const Level& to)
{
    const auto points = [](const Level& l) {
        std::vector<Point<dim>> pts(l.package.n_dofs());
        DoFTools::map_dofs_to_support_points(l.package.get_mapping(), l.package.get_dofs(), pts);
        return pts;
    };
    const auto p_from = points(from), p_to = points(to);
    AssertDimension(p_from.size(), p_to.size());
    std::map<std::pair<long, long>, unsigned> index;
    const auto key = [](const Point<dim>& p) { return std::make_pair(std::lround(p[0] * 1e8), std::lround(p[1] * 1e8)); };
    for (unsigned i = 0; i < p_to.size(); ++i) {
        index[key(p_to[i])] = i;
    }
    std::vector<unsigned> map(p_from.size());
    for (unsigned i = 0; i < p_from.size(); ++i) {
        map[i] = index.at(key(p_from[i]));
    }
    return map;
}

/**
 * Nonnegative transfer for Q2 on refinement n -> n+1: the Q1 embedding between the Q2 nodes, which are the Q1
 * nodes of refinement n+1 (coarse) and n+2 (fine), mapped to the Q2 DoFs by their support points.
 */
FullMatrix<double> q1_on_q2_nodes(const Discretization& disc, const Level& fine, const Level& coarse,
                                  unsigned n_levels)
{
    Discretization q1 = disc;
    q1.degree = 1;
    const Level q1_fine(q1, n_levels + 1), q1_coarse(q1, n_levels);
    const fe::LinearTransferMG<dim> t(q1_coarse.package.get_dofs(), q1_fine.package.get_dofs(),
                                      q1_coarse.package.get_constraints(), q1_fine.package.get_constraints());
    const FullMatrix<double> P1 = transfer_matrix(t);
    const auto map_f = match_points(q1_fine, fine), map_c = match_points(q1_coarse, coarse);
    FullMatrix<double> P(P1.m(), P1.n());
    for (unsigned i = 0; i < P1.m(); ++i) {
        for (unsigned j = 0; j < P1.n(); ++j) {
            P(map_f[i], map_c[j]) = P1(i, j);
        }
    }
    return P;
}

void check_transfers(CheckReport& report, const Discretization& disc, unsigned n_levels, bool q1_nodes = false)
{
    const Level fine(disc, n_levels), coarse(disc, n_levels - 1);
    const fe::LinearTransferMG<dim> embedding(coarse.package.get_dofs(), fine.package.get_dofs(),
                                              coarse.package.get_constraints(), fine.package.get_constraints());
    const FullMatrix<double> P = q1_nodes ? q1_on_q2_nodes(disc, fine, coarse, n_levels) : transfer_matrix(embedding);
    struct DenseTransfer
    {
        const FullMatrix<double>& P;
        void to_fine_mesh(const Vector<double>& src, Vector<double>& dst) const { dst.reinit(P.m()); P.vmult(dst, src); }
        void Tfine(const Vector<double>& src, Vector<double>& dst) const { dst.reinit(P.n()); P.Tvmult(dst, src); }
    } transfer{P};
    const auto& d_h = fine.d();
    const auto& d_H = coarse.d();
    const unsigned n_h = d_h.size(), n_H = d_H.size();
    const std::string name = disc.name + (q1_nodes ? " (Q1 on the nodes)" : "");
    const auto tag = [&](const std::string& s) { return fmt::format("{}: {}", name, s); };
    report.info(tag("levels"), fmt::format("{} -> {} DoFs", n_H, n_h));

    // The relative transfers need a nonnegative P; the embedding of Q2 has negative entries
    double min_entry = 0;
    for (unsigned i = 0; i < P.m(); ++i) {
        for (unsigned j = 0; j < P.n(); ++j) {
            min_entry = std::min(min_entry, P(i, j));
        }
    }
    if (!disc.nonnegative && !q1_nodes) {
        report.info(tag("smallest entry of P (negative: densities restrict to negative values)"), sci(min_entry));
        return;
    }
    report.check(min_entry >= 0, tag("P is nonnegative"), sci(min_entry));

    // Partition of unity and mass consistency
    Vector<double> ones_H(n_H), P1(n_h), PTd(n_H);
    ones_H = 1.0;
    transfer.to_fine_mesh(ones_H, P1);
    P1.add(-1.0, Vector<double>(n_h) = 1.0);
    report.check(P1.linfty_norm() < 1e-13, tag("partition of unity P 1 = 1"), sci(P1.linfty_norm()));
    transfer.Tfine(d_h, PTd);
    PTd -= d_H;
    const double mass_err = PTd.linfty_norm() / d_H.linfty_norm();
    if (!q1_nodes) {
        report.check(mass_err < 1e-12, tag("mass consistency P^T d_h = d_H"), sci(mass_err));
    } else {
        report.info(tag("mass consistency P^T d_h = d_H, relative error"), sci(mass_err));
    }

    // Point with Gaussian tails, and its density restriction
    auto x = at_support_points(fine, [](const Point<dim>& p) {
        return std::exp(-0.5 * p.norm_square()) * (1 + 0.3 * std::sin(p[0]) * std::cos(0.7 * p[1]));
    });
    x /= std::sqrt(d_inner(d_h, x, x));
    const auto restrict_point = [&](const Vector<double>& xf) {
        Vector<double> p(n_h), rho(n_H), y(n_H);
        for (unsigned i = 0; i < n_h; ++i) {
            p[i] = d_h[i] * xf[i] * xf[i];
        }
        transfer.Tfine(p, rho);
        for (unsigned j = 0; j < n_H; ++j) {
            y[j] = std::sqrt(rho[j] / d_H[j]);
        }
        return y;
    };
    const Vector<double> y = restrict_point(x);
    report.check(std::abs(d_inner(d_H, y, y) - 1) < 1e-13, tag("restricted point on the coarse sphere"),
                 sci(std::abs(d_inner(d_H, y, y) - 1)));
    report.info(tag("min x, min y"), sci(*std::ranges::min_element(x)) + ", " + sci(*std::ranges::min_element(y)));

    Vector<double> c(n_h);
    c = 1.0;
    c /= std::sqrt(d_inner(d_h, c, c));
    const Vector<double> yc = restrict_point(c);
    const double spread = (yc.linfty_norm() - *std::ranges::min_element(yc)) / yc.linfty_norm();
    if (!q1_nodes) {
        report.check(spread < 1e-12, tag("restriction of a constant is constant"), sci(spread));
    } else {
        report.info(tag("restriction of a constant: relative spread"), sci(spread));
    }

    // Relative prolongation of a tangent vector v at y, and restriction of a tangent vector g at x
    const auto prolong = [&](const Vector<double>& v) {
        Vector<double> r(n_H), Pr(n_h);
        for (unsigned j = 0; j < n_H; ++j) {
            r[j] = v[j] / y[j];
        }
        transfer.to_fine_mesh(r, Pr);
        for (unsigned i = 0; i < n_h; ++i) {
            Pr[i] *= x[i];
        }
        return Pr;
    };
    const auto restrict_gradient = [&](const Vector<double>& g) {
        Vector<double> u(n_h), Rg(n_H);
        for (unsigned i = 0; i < n_h; ++i) {
            u[i] = d_h[i] * x[i] * g[i];
        }
        transfer.Tfine(u, Rg);
        for (unsigned j = 0; j < n_H; ++j) {
            Rg[j] /= d_H[j] * y[j];
        }
        return Rg;
    };
    const auto tangent = [&](const Vector<double>& base, const Vector<double>& dm, Vector<double> v) {
        v.add(-d_inner(dm, base, v), base);
        return v;
    };

    // Smooth relative perturbation (v = y u) and a rough one
    auto u_smooth = at_support_points(coarse, [](const Point<dim>& p) { return std::sin(0.4 * p[0] + 0.2 * p[1]); });
    Vector<double> v_smooth(n_H), v_rough(n_H);
    for (unsigned j = 0; j < n_H; ++j) {
        v_smooth[j] = y[j] * u_smooth[j];
        v_rough[j] = y[j] * std::cos(1.7 * j);
    }
    v_smooth = tangent(y, d_H, v_smooth);
    v_rough = tangent(y, d_H, v_rough);
    Vector<double> g(n_h);
    for (unsigned i = 0; i < n_h; ++i) {
        g[i] = std::cos(0.9 * i) * x[i] + 1e-3 * std::sin(0.3 * i);
    }
    g = tangent(x, d_h, g);

    double tangency = 0, galerkin = 0, limit_ratio = std::numeric_limits<double>::infinity();
    for (const Vector<double>* v : {&v_smooth, &v_rough}) {
        const Vector<double> Pv = prolong(*v);
        tangency = std::max(tangency, std::abs(d_inner(d_h, x, Pv)) / std::sqrt(d_inner(d_h, Pv, Pv)));
        const double lhs = d_inner(d_h, Pv, g), rhs = d_inner(d_H, *v, restrict_gradient(g));
        galerkin = std::max(galerkin, std::abs(lhs - rhs) / std::abs(lhs));
        limit_ratio = std::min(limit_ratio,
                               ellipsoid_positive::max_step(x, Pv) / ellipsoid_positive::max_step(y, *v));
    }
    report.check(tangency < 1e-13, tag("relative prolongation is tangent at x"), sci(tangency));
    report.check(galerkin < 1e-12, tag("Galerkin <P v, g>_h = <v, R g>_H"), sci(galerkin));
    const Vector<double> Rg = restrict_gradient(g);
    const double Rg_tangency = std::abs(d_inner(d_H, y, Rg)) / std::sqrt(d_inner(d_H, Rg, Rg));
    report.check(Rg_tangency < 1e-13, tag("restricted gradient is tangent at y"), sci(Rg_tangency));
    report.check(limit_ratio >= 1 - 1e-12, tag("step limit tau_max(x, P v) >= tau_max(y, v)"),
                 "min ratio " + sci(limit_ratio));

    // With the injected (normalized) point of ManifoldTransfer instead, P v is not tangent
    if (!q1_nodes) {
        Vector<double> y_inj(n_H), r(n_H), Pr(n_h);
        embedding.to_coarse_mesh(x, y_inj);
        y_inj /= std::sqrt(d_inner(d_H, y_inj, y_inj));
        Vector<double> v = tangent(y_inj, d_H, v_smooth);
        for (unsigned j = 0; j < n_H; ++j) {
            r[j] = v[j] / y_inj[j];
        }
        transfer.to_fine_mesh(r, Pr);
        for (unsigned i = 0; i < n_h; ++i) {
            Pr[i] *= x[i];
        }
        report.info(tag("tangency of x P(v / y) for the injected point"),
                    sci(std::abs(d_inner(d_h, x, Pr)) / std::sqrt(d_inner(d_h, Pr, Pr))));
    }

    // Relative against linear prolongation, Pi_x P v (MassProjectionTransport), in the D_h-norm
    {
        const Vector<double> Pv = prolong(v_smooth);
        Vector<double> Lv(n_h);
        transfer.to_fine_mesh(v_smooth, Lv);
        Lv = tangent(x, d_h, Lv);
        Vector<double> diff(Pv);
        diff -= Lv;
        report.info(tag("|P_rel v - P_lin v| / |P_lin v| (D_h), smooth v"),
                    sci(std::sqrt(d_inner(d_h, diff, diff) / d_inner(d_h, Lv, Lv))));
        // Largest relative change |v_i| / x_i of a component, which sets the step limit
        double rel_lin = 0, rel_rel = 0;
        for (unsigned i = 0; i < n_h; ++i) {
            rel_lin = std::max(rel_lin, std::abs(Lv[i]) / x[i]);
            rel_rel = std::max(rel_rel, std::abs(Pv[i]) / x[i]);
        }
        report.info(tag("max_i |P v|_i / x_i, smooth v: linear, relative"), sci(rel_lin) + ", " + sci(rel_rel));
    }

    // Coarse correction Phi(z) = x sqrt(P(z^2 / y^2)), for z on the density curve from y to a coarse point z_hat
    // whose tails are 1e-12 of those of y
    auto z_hat = at_support_points(coarse, [](const Point<dim>& p) {
        return std::exp(-0.5 * p.norm_square()) * std::exp(-2 * std::log(10.0) * std::min(6.0, 0.06 * p.norm_square()));
    });
    z_hat /= std::sqrt(d_inner(d_H, z_hat, z_hat));
    const auto correction = [&](const Vector<double>& z) {
        Vector<double> t(n_H), Pt(n_h), out(x);
        for (unsigned j = 0; j < n_H; ++j) {
            t[j] = (z[j] / y[j]) * (z[j] / y[j]);
        }
        transfer.to_fine_mesh(t, Pt);
        for (unsigned i = 0; i < n_h; ++i) {
            out[i] *= std::sqrt(Pt[i]);
        }
        return out;
    };
    {
        Vector<double> fixed = correction(y);
        fixed -= x;
        report.check(fixed.linfty_norm() <= 1e-15 * x.linfty_norm(), tag("Phi(y) = x"), sci(fixed.linfty_norm()));
    }
    double sphere = 0, agreement = 0;
    for (const double t : {0.25, 0.5, 0.9}) {
        Vector<double> lift(z_hat), z(y);
        ellipsoid_positive::retract_inv_by_density(lift, y);
        ellipsoid_positive::retract_by_density(lift, z, t);
        const Vector<double> phi = correction(z);
        sphere = std::max(sphere, std::abs(d_inner(d_h, phi, phi) - 1));
        // Density retraction of the relative prolongation of the lifting, alpha = 1
        Vector<double> v(z), x_add(x);
        ellipsoid_positive::retract_inv_by_density(v, y);
        ellipsoid_positive::retract_by_density(prolong(v), x_add, 1.0);
        double rel = 0;
        for (unsigned i = 0; i < n_h; ++i) {
            rel = std::max(rel, std::abs(x_add[i] - phi[i]) / phi[i]);
        }
        agreement = std::max(agreement, rel);
    }
    report.check(sphere < 1e-13, tag("Phi(z) on the fine sphere"), sci(sphere));
    report.check(agreement < 1e-8, tag("Phi(z) = R_x(P L_y(z)) (density curve, t <= 0.9)"), sci(agreement));

    // At z = z_hat (t = 1) the tail densities shrink by up to 1e-12: Phi stays accurate, the additive form loses
    // all digits where the ratio is below the rounding level
    {
        const Vector<double> phi = correction(z_hat);
        Vector<double> v(z_hat), x_add(x);
        ellipsoid_positive::retract_inv_by_density(v, y);
        const Vector<double> Pv = prolong(v);
        double worst = 0, min_ratio = std::numeric_limits<double>::infinity();
        for (unsigned i = 0; i < n_h; ++i) {
            const double additive = x[i] * x[i] + 2 * x[i] * Pv[i];  // density retraction at alpha = 1
            min_ratio = std::min(min_ratio, phi[i] / x[i]);
            worst = std::max(worst, std::abs(std::sqrt(std::max(additive, 0.0)) - phi[i]) / phi[i]);
        }
        report.check(*std::ranges::min_element(phi) > 0, tag("Phi(z_hat) > 0"), "min phi_i/x_i " + sci(min_ratio));
        report.info(tag("relative error of the additive density retraction at alpha = 1"), sci(worst));
        const double limit = ellipsoid_positive::max_step(x, Pv);
        report.check(limit >= 1 - 1e-12, tag("step limit tau_max(x, P L_y(z)) >= 1"), fmt::format("{:.15f}", limit));

        // Option A on the coarse correction: alpha = theta tau_max >= theta; the additive form is accurate there
        const double theta = 0.99, alpha = std::min(1.0, theta * limit);
        Vector<double> x_capped(x);
        ellipsoid_positive::retract_by_density(Pv, x_capped, alpha);
        double capped_err = 0;
        for (unsigned i = 0; i < n_h; ++i) {
            const double t = phi[i] * phi[i] / (x[i] * x[i]);  // P(z^2 / y^2)
            const double exact = x[i] * std::sqrt(1 - alpha + alpha * t);
            capped_err = std::max(capped_err, std::abs(x_capped[i] - exact) / exact);
        }
        report.check(capped_err < 1e-12, tag("additive density retraction at alpha = theta tau_max is accurate"),
                     fmt::format("alpha {:.4f}, relative error {}", alpha, sci(capped_err)));
    }
}


//! Diagonal inverse of a lumped mass matrix, with the interface the transports expect (vmult, set_tol)
struct DiagonalMassInverse
{
    const Vector<double>& d;
    void vmult(Vector<double>& dst, const Vector<double>& src) const
    {
        dst.reinit(src.size(), true);
        for (unsigned i = 0; i < src.size(); ++i) {
            dst[i] = src[i] / d[i];
        }
    }
    void set_tol(double) const {}
};

/**
 * Point maps of the positive sphere (tex/drafts/positive_multilevel.tex, (U1)-(U3), (E1)-(E3)): the unit-sphere
 * ManifoldTransfer (linear interpolation, needs a nonnegative embedding) and LogManifoldTransfer (interpolation of
 * logarithms), with AdjointDifferentialTransport (adjoint from ManifoldTransferBase::diff_prolongation_adjoint()).
 */
void check_point_transfers(CheckReport& report, const Discretization& disc, unsigned n_levels, double radius = 10.0)
{
    const Level fine(disc, n_levels, radius), coarse(disc, n_levels - 1, radius);
    const fe::LinearTransferMG<dim> embedding(coarse.package.get_dofs(), fine.package.get_dofs(),
                                              coarse.package.get_constraints(), fine.package.get_constraints());
    const auto& d_h = fine.d();
    const auto& d_H = coarse.d();
    const auto tag = [&](const std::string& s) {
        return fmt::format("{} point maps{}: {}", disc.name, radius == 10.0 ? "" : fmt::format(" (radius {})", radius), s);
    };
    const DiagonalMassInverse M_inv_H{d_H};

    report.check(is_nonnegative(embedding) == disc.nonnegative, tag("is_nonnegative(I_H^h)"),
                 is_nonnegative(embedding) ? "nonnegative" : "negative entries");

    auto x = at_support_points(fine, [](const Point<dim>& p) {
        return std::exp(-0.5 * p.norm_square()) * (1 + 0.3 * std::sin(p[0]) * std::cos(0.7 * p[1]));
    });
    x /= std::sqrt(d_inner(d_h, x, x));

    const ManifoldTransfer<MassMatrix, DiagonalMassInverse> lin(embedding, coarse.D(), fine.D(), M_inv_H);
    const LogManifoldTransfer<MassMatrix, DiagonalMassInverse> log(embedding, coarse.D(), fine.D(), M_inv_H);
    std::vector<std::pair<const ManifoldTransferBase*, std::string>> maps{{&log, "log"}};
    if (disc.nonnegative) {
        maps.insert(maps.begin(), {&lin, "linear"});
    }

    const auto rel_max = [](const Vector<double>& a, const Vector<double>& b) {
        double m = 0;
        for (unsigned i = 0; i < a.size(); ++i) {
            m = std::max(m, std::abs(a[i] / b[i] - 1));
        }
        return m;
    };
    const auto tangent_at = [](const Vector<double>& dd, const Vector<double>& z, double freq) {
        Vector<double> v(z.size());
        for (unsigned i = 0; i < z.size(); ++i) {
            v[i] = z[i] * std::cos(freq * i);
        }
        v.add(-d_inner(dd, z, v), z);
        return v;
    };

    for (const auto& [pt, name] : maps) {
        const auto t = [&](const std::string& s) { return tag(name + ": " + s); };
        Vector<double> y(d_H.size()), xp(d_h.size());
        pt->restriction(x, y);
        pt->prolongation(y, xp);
        const double min_y = *std::ranges::min_element(y), min_xp = *std::ranges::min_element(xp);
        const double norm_err = std::max(std::abs(d_inner(d_H, y, y) - 1), std::abs(d_inner(d_h, xp, xp) - 1));
        report.check(min_y > 0 && min_xp > 0 && norm_err < 1e-13, t("r(x), p(y) positive and on the spheres"),
                     fmt::format("min {}, {}; |norm - 1| {}", sci(min_y), sci(min_xp), sci(norm_err)));

        // Differentials against central differences, componentwise relative to the value of the map at the base point:
        // the relative derivatives (Dp v)_i / p_i are O(1) at any magnitude, and the rounding of p(y +- h v) is
        // eps / h relative to p_i. (An error relative to the largest entry fails on Q1 at radius 22, where the bulk
        // of y is one coarse node: |Dr u| / |r| ~ 3e-7, below the rounding of r in the difference quotient.)
        const Vector<double> v = tangent_at(d_H, y, 0.37), u = tangent_at(d_h, x, 0.53);
        const auto fd_error = [&](auto&& map, const Vector<double>& base, const Vector<double>& dir,
                                  const Vector<double>& diff, double h) {
            Vector<double> bp(base), bm(base), fp, fm, f0;
            bp.add(h, dir);
            bm.add(-h, dir);
            map(bp, fp);
            map(bm, fm);
            map(base, f0);
            double err = 0, scale = 0;
            for (unsigned i = 0; i < f0.size(); ++i) {
                err = std::max(err, std::abs((fp[i] - fm[i]) / (2 * h) - diff[i]) / f0[i]);
                scale = std::max(scale, std::abs(diff[i]) / f0[i]);
            }
            return err / scale;
        };
        Vector<double> dp(d_h.size()), dr(d_H.size());
        pt->diff_prolongation(y, v, dp);
        pt->diff_restriction(x, u, dr);
        const auto p_map = [&](const Vector<double>& a, Vector<double>& b) { b.reinit(d_h.size()); pt->prolongation(a, b); };
        const auto r_map = [&](const Vector<double>& a, Vector<double>& b) { b.reinit(d_H.size()); pt->restriction(a, b); };
        const double ep1 = fd_error(p_map, y, v, dp, 1e-3), ep2 = fd_error(p_map, y, v, dp, 1e-4);
        const double er1 = fd_error(r_map, x, u, dr, 1e-3), er2 = fd_error(r_map, x, u, dr, 1e-4);
        // Rounding floor of the difference quotient at h = 1e-4: a map evaluated in logarithms has the relative
        // rounding error eps |log f_i| per component (|log f_i| ~ 480 at radius 22), the linear maps eps
        const auto floor_at = [](const Vector<double>& f) {
            double m = 1;
            for (const double a : f) {
                m = std::max(m, std::abs(std::log(a)));
            }
            return 10 * std::numeric_limits<double>::epsilon() * m / 1e-4;
        };
        const double fp = floor_at(xp), fr = floor_at(y);
        report.check(ep2 < 1e-6 && (ep2 < fp || ep1 / ep2 > 50), t("Dp(y) (central differences)"),
                     fmt::format("relative error {}, ratio {}, rounding floor {}", sci(ep2), sci(ep1 / ep2), sci(fp)));
        report.check(er2 < 1e-6 && (er2 < fr || er1 / er2 > 50), t("Dr(x) (central differences)"),
                     fmt::format("relative error {}, ratio {}, rounding floor {}", sci(er2), sci(er1 / er2), sci(fr)));

        // Transport: tangency and Galerkin; Dp(y) itself is tangent at p(y), not at x
        const AdjointDifferentialTransport<MassMatrix, DiagonalMassInverse> transport(embedding, *pt, coarse.D(),
                                                                                      fine.D(), M_inv_H);
        const Vector<double> g = tangent_at(d_h, x, 1.1);
        Vector<double> Pv(d_h.size()), Rg(d_H.size());
        transport.vector_prolongation(x, y, v, Pv);
        transport.vector_restriction(y, x, g, Rg);
        const double tang_P = std::abs(d_inner(d_h, x, Pv)) / std::sqrt(d_inner(d_h, Pv, Pv));
        const double tang_R = std::abs(d_inner(d_H, y, Rg)) / std::sqrt(d_inner(d_H, Rg, Rg));
        const double lhs = d_inner(d_h, Pv, g), rhs = d_inner(d_H, v, Rg);
        const double galerkin = std::abs(lhs - rhs) / std::abs(lhs);
        report.check(tang_P < 1e-13 && tang_R < 1e-13, t("P v, R g tangent"), sci(tang_P) + ", " + sci(tang_R));
        report.check(galerkin < 1e-12, t("Galerkin <P v, g>_h = <v, R g>_H"), "relative error " + sci(galerkin));
        report.info(t("tangency of Dp(y) v at x (without Pi_x)"),
                    sci(std::abs(d_inner(d_h, x, dp)) / std::sqrt(d_inner(d_h, dp, dp))));

        // Tails: p(r(x)) / x and max |P v| / x
        double pv_rel = 0;
        for (unsigned i = 0; i < d_h.size(); ++i) {
            pv_rel = std::max(pv_rel, std::abs(Pv[i]) / x[i]);
        }
        double lo = std::numeric_limits<double>::infinity(), hi = 0;
        for (unsigned i = 0; i < d_h.size(); ++i) {
            lo = std::min(lo, xp[i] / x[i]);
            hi = std::max(hi, xp[i] / x[i]);
        }
        report.info(t("p(r(x)) / x in [min, max]; max |P v| / x (max |v/y| = 1)"),
                    fmt::format("[{}, {}]; {}", sci(lo), sci(hi), sci(pv_rel)));

        // Entrywise accuracy of the adjoint Dp(y)^* g: the map is linear, and the scaling by F = 2^1000 is exact, so
        // Dp(y)^*(F g) / F is a reference without underflow of the products x_hat_i d_i g_i (for x_i >= 1e-300)
        Vector<double> adj(d_H.size()), adj_ref(d_H.size()), g_scaled(g);
        pt->diff_prolongation_adjoint(y, g, adj);
        const double F = std::ldexp(1.0, 1000);
        g_scaled *= F;
        pt->diff_prolongation_adjoint(y, g_scaled, adj_ref);
        adj_ref /= F;
        double adj_err = 0;
        unsigned n_zero = 0, n_ref = 0;
        for (unsigned j = 0; j < d_H.size(); ++j) {
            if (adj_ref[j] != 0) {
                n_ref++;
                adj_err = std::max(adj_err, std::abs(adj[j] - adj_ref[j]) / std::abs(adj_ref[j]));
                n_zero += adj[j] == 0 ? 1 : 0;
            }
        }
        // Known limitation (tex/drafts/positive_multilevel.tex, adjoint in log space): the products underflow for
        // x_i below about 1e-154, so this is reported, not checked
        report.info(t("adjoint Dp(y)^* g entrywise: max relative error; entries 0 instead of nonzero"),
                    fmt::format("{}; {} of {}", sci(adj_err), n_zero, n_ref));
    }

    // Injection: the log restriction is the unit-sphere restriction
    if (disc.nonnegative) {
        Vector<double> y_lin(d_H.size()), y_log(d_H.size());
        lin.restriction(x, y_lin);
        log.restriction(x, y_log);
        report.check(rel_max(y_log, y_lin) < 1e-13, tag("r^e = r (injection)"), sci(rel_max(y_log, y_lin)));
    }
}

} // namespace


int main()
{
    return run_tests([](CheckReport& report) {
        const std::vector<Discretization> discs = {
            {"Q1", MeshKind::QUADRILATERAL, 1, true},
            {"Q2", MeshKind::QUADRILATERAL, 2, false},
            {"P1", MeshKind::SIMPLEX, 1, true},
        };
        for (const auto& disc : discs) {
            check_transfers(report, disc, disc.degree == 1 ? 5 : 4);
            if (!disc.nonnegative) {
                check_transfers(report, disc, 4, true);
            }
            check_point_transfers(report, disc, disc.degree == 1 ? 5 : 4);
            // Tails below 1e-154 (min x ~ 1e-210), where squares and products of components underflow
            check_point_transfers(report, disc, disc.degree == 1 ? 5 : 4, 22.0);
        }
    });
}
