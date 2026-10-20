/**
 * @file
 * @brief Tests of the positive unit-mass sphere \f$ \mathcal M_h^+ = \{ x > 0 : x^\top D x = 1 \} \f$ with the
 * density retraction (ellipsoid_positive, PosUnitMassSphere), against tex/BlackboardS.tex (density retraction and
 * lifting, exact supporting identity), for a lumped Q1 discretization with Neumann boundary:
 *
 * - density map: \f$ \onebb^\top \pi(x) = 1 \f$ on the sphere, density_map_diff() by finite differences;
 * - retraction: \f$ R_x(0) = x \f$, stays on the sphere, \f$ \mathrm D R_x(0) = \mathrm{id} \f$, undefined for
 *   steps beyond \f$ h_{\max} = \min_{v_i < 0} x_i / (2|v_i|) \f$ (and leaves \f$ x \f$ unchanged);
 * - lifting: tangent, inverse of the retraction in both directions, affine density curve
 *   \f$ \pi(R_x(t L_x(\hat x))) = (1-t) \pi(x) + t \pi(\hat x) \f$;
 * - differential of the lifting (PosUnitMassSphere, FlooredPosSphere): \f$ \mathrm D L_x(\zeta)[u] = (\zeta/x) \circ u \f$
 *   against central differences (exact up to rounding, the lifting is quadratic in \f$ \zeta \f$), the identity at
 *   \f$ \zeta = x \f$, and the D-adjoint \f$ \langle w, \mathrm D L\, u \rangle_D = \langle \mathrm D L^* w, u \rangle_D \f$;
 * - e-retraction (ellipsoid_positive::retract_by_e()) and its inverse: \f$ R_x(0) = x \f$, positive and on the
 *   sphere for every step, \f$ \mathrm D R_x(0) = \mathrm{id} \f$, inverse in both directions, geometric curve
 *   \f$ R_x(t L_x(\hat x)) \propto x^{1-t} \hat x^t \f$ (the e-geodesic of the densities);
 * - the energy of GrossPitaevskiiFunctional satisfies the exact supporting identity, and is convex along the curve;
 * - reported, not checked: the energy along the e-geodesic, which need not be convex (it is not from a peaked
 *   point to the constant);
 * - without fraction-to-the-boundary steps, a full energy-adaptive step is undefined from points on the density
 *   curve towards the constant start (a component has to shrink by more than a factor ~2);
 * - energy-adaptive gradient descent on PosUnitMassSphere (GradientDescent with fraction-to-the-boundary steps
 *   \f$ h = \min(1, \theta h_{\max}) \f$, DescentOptions::boundary_fraction; full steps or Armijo line search)
 *   converges to the ground state of the norm retraction from these points and from multiplicative perturbations of
 *   the ground state;
 * - exact line search (exact_line_search(), tex/exact_line_search.tex): the velocities of the normalization, density
 *   and e retractions against central differences; the closed-form ray step (OracleBase::ray_step()) without cap and
 *   with caps above and below its minimizer, against grids; the root finder of phi' on the density curve (interior
 *   root), with a binding cap (theta t_max), and on the e-path with and without the growth cap.
 */
#include "check.h"

#include <rmo/gpe/gpe.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/manifold_positive.h>
#include <rmo/gpe/oracle.h>
#include <rmo/ropt/solver.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;
using namespace dealii;

namespace
{
constexpr int    dim    = 2;
constexpr double radius = 4.0;
constexpr double beta   = 100.0;

using System     = GrossPitaevskiiLumpedSystem<dim>;
using Functional = GrossPitaevskiiFunctional<System>;
using MassMatrix = System::MassMatrix;

GPE_Options gpe_options()
{
    GPE_Options options{};
    options.dimension = dim;
    options.degree    = 1;
    options.radius    = radius;
    options.beta      = beta;
    options.order     = Ordering::DEFAULT;
    options.bc        = BoundaryCondition::NEUMANN;
    options.mesh_kind = MeshKind::QUADRILATERAL;
    options.potential = Potential::SQUARE;
    options.mass_lumping = true;
    return options;
}

SolverOptions solver_options()
{
    SolverOptions options{};
    options.max_inner     = 2000;
    options.tol_inner     = 1e-12;
    options.tol_inner_res = 1e-6;
    options.solver        = SolverMethod::CG;
    options.precond       = Precondition::DIAGONAL;
    return options;
}

//! Lumped system of the square potential on a Neumann package.
struct Problem
{
    explicit Problem(unsigned n_levels)
        : package(gpe_options(), n_levels)
        , system(package.system<System>(V))
    {}

    [[nodiscard]] unsigned n_dofs() const { return package.n_dofs(); }
    [[nodiscard]] const MassMatrix& D() const { return system.get_M(); }

    const potential::Square<dim> V{};
    GrossPitaevskiiPackage<dim> package;
    System system;
};

template <typename Function>
Vector<double> make_vector(unsigned n, Function&& f)
{
    Vector<double> v(n);
    for (unsigned i = 0; i < n; ++i) {
        v[i] = f(i);
    }
    return v;
}

double d_inner(const MassMatrix& D, const Vector<double>& a, const Vector<double>& b)
{
    Vector<double> Db(b.size());
    D.vmult(Db, b);
    return a * Db;
}

void normalize(const MassMatrix& D, Vector<double>& x) { x /= std::sqrt(d_inner(D, x, x)); }

//! Projection onto \f$ T_x \mathcal M_h^+ \f$ in the D-metric, for x on the sphere.
void project_tangent(const MassMatrix& D, const Vector<double>& x, Vector<double>& v) { v.add(-d_inner(D, x, v), x); }

//! Point on the sphere at x, retracted from t L_x(x_hat): the density curve from x (t = 0) to x_hat (t = 1).
Vector<double> density_curve(const Vector<double>& x, const Vector<double>& x_hat, double t)
{
    Vector<double> lift(x_hat), y(x);
    ellipsoid_positive::retract_inv_by_density(lift, x);
    ellipsoid_positive::retract_by_density(lift, y, t);
    return y;
}

//! Energy at x; updates the functional to x.
double energy(Functional& func, const Vector<double>& x)
{
    func.update(x);
    return func.value(x);
}

//! Two positive points on the sphere and a tangent direction at the first.
struct Points
{
    explicit Points(const MassMatrix& D, unsigned n)
        : x(make_vector(n, [](unsigned i) { return std::sin(0.37 * i) + 1.2; }))
        , x_hat(make_vector(n, [](unsigned i) { return std::cos(0.11 * i) + 1.5; }))
        , v(make_vector(n, [](unsigned i) { return std::cos(1.3 * i); }))
    {
        normalize(D, x);
        normalize(D, x_hat);
        project_tangent(D, x, v);
        v /= v.linfty_norm();
    }

    Vector<double> x, x_hat, v;
};


void check_density_map(CheckReport& report, const Problem& p)
{
    const Points pts(p.D(), p.n_dofs());
    Vector<double> density(p.n_dofs()), dp(p.n_dofs());

    ellipsoid_positive::density_map(p.D(), pts.x, density);
    double sum = 0;
    for (const double a : density) {
        sum += a;
    }
    report.check(std::abs(sum - 1) < 1e-13 && *std::ranges::min_element(density) > 0,
                 "density map: positive, sums to 1 on the sphere", "|1^T p - 1| = " + sci(std::abs(sum - 1)));

    // Central difference of the density map, exact up to rounding (quadratic map)
    const double h = 1e-4;
    Vector<double> x_h(pts.x), p_plus(p.n_dofs()), p_minus(p.n_dofs());
    x_h.add(h, pts.v);
    ellipsoid_positive::density_map(p.D(), x_h, p_plus);
    x_h.add(-2 * h, pts.v);
    ellipsoid_positive::density_map(p.D(), x_h, p_minus);
    p_plus -= p_minus;
    p_plus /= 2 * h;

    ellipsoid_positive::density_map_diff(p.D(), pts.x, pts.v, dp);
    p_plus -= dp;
    report.check(p_plus.linfty_norm() < 1e-10 * dp.linfty_norm(), "density map: J_x v = 2 D diag(x) v",
                 "max diff " + sci(p_plus.linfty_norm()));
}

void check_retraction(CheckReport& report, const Problem& p)
{
    const MassMatrix& D = p.D();
    const Points pts(D, p.n_dofs());
    const PosUnitMassSphere<MassMatrix> sphere(D);
    const double h_max = ellipsoid_positive::max_step(pts.x, pts.v);
    report.info("retraction: h_max of the test direction", sci(h_max));

    Vector<double> y(pts.x);
    sphere.retract(pts.v, y, 0.0);
    y -= pts.x;
    // Up to the rounding of the normalization in PosUnitMassSphere::retract()
    report.check(y.linfty_norm() < 1e-15 * pts.x.linfty_norm(), "retraction: R_x(0) = x",
                 "max diff " + sci(y.linfty_norm()));

    double max_norm_error = 0;
    for (const double t : {0.1, 0.5, 0.9, 0.999}) {
        sphere.retract(pts.v, pts.x, y, t * h_max);
        max_norm_error = std::max(max_norm_error, std::abs(d_inner(D, y, y) - 1));
    }
    report.check(max_norm_error < 1e-13, "retraction: stays on the sphere for h < h_max",
                 "max |y^T D y - 1| = " + sci(max_norm_error));

    // D R_x(0)[v] = v: central differences with error O(h^2)
    auto diff_error = [&](double h) {
        Vector<double> y_plus(pts.x), y_minus(pts.x);
        sphere.retract(pts.v, y_plus, h);
        sphere.retract(pts.v, y_minus, -h);
        y_plus -= y_minus;
        y_plus /= 2 * h;
        y_plus -= pts.v;
        return y_plus.linfty_norm();
    };
    const double error_1 = diff_error(1e-2 * h_max), error_2 = diff_error(1e-3 * h_max);
    report.check(error_2 < 1e-6 && error_1 / error_2 > 50, "retraction: D R_x(0) = id (finite differences)",
                 "error " + sci(error_2) + ", ratio " + sci(error_1 / error_2));

    y = pts.x;
    bool thrown = false;
    try {
        sphere.retract(pts.v, y, 1.01 * h_max);
    }
    catch (const ExceptionBase&) {
        thrown = true;
    }
    y -= pts.x;
    report.check(thrown && y.linfty_norm() == 0, "retraction: undefined beyond h_max, x unchanged");
}

void check_lifting(CheckReport& report, const Problem& p)
{
    const MassMatrix& D = p.D();
    const Points pts(D, p.n_dofs());
    const PosUnitMassSphere<MassMatrix> sphere(D);

    Vector<double> lift(pts.x_hat);
    sphere.retract_inv(lift, pts.x);
    const double tangent = std::abs(d_inner(D, pts.x, lift)) / lift.l2_norm();
    report.check(tangent < 1e-13, "lifting: L_x(x_hat) is tangent", "|x^T D L| / |L| = " + sci(tangent));

    Vector<double> y(pts.x);
    sphere.retract(lift, y, 1.0);
    y -= pts.x_hat;
    report.check(y.linfty_norm() < 1e-14 * pts.x_hat.linfty_norm(), "lifting: R_x(L_x(x_hat)) = x_hat",
                 "max diff " + sci(y.linfty_norm()));

    Vector<double> step(pts.v);
    step *= 0.5 * ellipsoid_positive::max_step(pts.x, pts.v);
    sphere.retract(step, pts.x, y, 1.0);
    sphere.retract_inv(y, pts.x);
    y -= step;
    report.check(y.linfty_norm() < 1e-13 * step.linfty_norm(), "lifting: L_x(R_x(v)) = v",
                 "max diff " + sci(y.linfty_norm()));

    // Affine density curve, eq. (GP_density_affine)
    Vector<double> p_x(p.n_dofs()), p_hat(p.n_dofs()), p_t(p.n_dofs());
    ellipsoid_positive::density_map(D, pts.x, p_x);
    ellipsoid_positive::density_map(D, pts.x_hat, p_hat);
    double max_error = 0;
    for (const double t : {0.25, 0.5, 0.9}) {
        ellipsoid_positive::density_map(D, density_curve(pts.x, pts.x_hat, t), p_t);
        p_t.add(-(1 - t), p_x);
        p_t.add(-t, p_hat);
        max_error = std::max(max_error, p_t.linfty_norm());
    }
    report.check(max_error < 1e-15, "lifting: pi(R_x(t L_x(x_hat))) = (1-t) pi(x) + t pi(x_hat)",
                 "max diff " + sci(max_error));
}

//! Differential of the density lifting and its D-adjoint, for PosUnitMassSphere and FlooredPosSphere.
void check_lifting_diff(CheckReport& report, const Problem& p)
{
    const MassMatrix& D = p.D();
    const unsigned n = p.n_dofs();
    const Points pts(D, n);
    const Vector<double> u = make_vector(n, [](unsigned i) { return std::sin(0.71 * i + 0.3); });
    const Vector<double> w = make_vector(n, [](unsigned i) { return std::cos(0.53 * i - 0.2); });

    const PosUnitMassSphere<MassMatrix> pos_sphere(D);
    const FlooredPosSphere<MassMatrix> floored(D, 1e-12, true);
    const std::vector<std::pair<const ManifoldBase*, std::string>> manifolds{
        {&pos_sphere, "PosUnitMassSphere"}, {&floored, "FlooredPosSphere"}};

    for (const auto& [m, name] : manifolds) {
        // Central difference of zeta -> L_x(zeta), with h small against min_i zeta_i / |u_i|
        double h = std::numeric_limits<double>::infinity();
        for (unsigned i = 0; i < n; ++i) {
            if (u[i] != 0) {
                h = std::min(h, pts.x_hat[i] / std::abs(u[i]));
            }
        }
        h *= 1e-3;
        Vector<double> plus(pts.x_hat), minus(pts.x_hat), dl(n);
        plus.add(h, u);
        minus.add(-h, u);
        m->retract_inv(plus, pts.x);
        m->retract_inv(minus, pts.x);
        plus -= minus;
        plus /= 2 * h;
        m->retract_inv_diff(pts.x, pts.x_hat, u, dl);
        plus -= dl;
        const double fd_error = plus.linfty_norm() / dl.linfty_norm();
        report.check(fd_error < 1e-9, name + ": D L_x(zeta)[u] = (zeta/x) u (central differences)",
                     "relative error " + sci(fd_error));

        m->retract_inv_diff(pts.x, pts.x, u, dl);
        dl -= u;
        report.check(dl.linfty_norm() < 1e-15 * u.linfty_norm(), name + ": D L_x(x) = id",
                     "max diff " + sci(dl.linfty_norm()));

        Vector<double> dl_u(n), dl_adj_w(n);
        m->retract_inv_diff(pts.x, pts.x_hat, u, dl_u);
        m->retract_inv_diff_adjoint(pts.x, pts.x_hat, w, dl_adj_w);
        const double lhs = d_inner(D, w, dl_u), rhs = d_inner(D, dl_adj_w, u);
        const double adj_error = std::abs(lhs - rhs) / std::abs(lhs);
        report.check(adj_error < 1e-13, name + ": <w, D L u>_D = <D L^* w, u>_D", "relative error " + sci(adj_error));
    }
}

//! e-retraction and its inverse: retraction for every step, geometric curve x(t) ~ x^(1-t) x_hat^t.
void check_e_retraction(CheckReport& report, const Problem& p)
{
    const MassMatrix& D = p.D();
    const Points pts(D, p.n_dofs());
    const double h_max = ellipsoid_positive::max_step(pts.x, pts.v);

    Vector<double> y(pts.x);
    ellipsoid_positive::retract_by_e(D, pts.v, y, 0.0);
    y -= pts.x;
    report.check(y.linfty_norm() < 1e-15 * pts.x.linfty_norm(), "e-retraction: R_x(0) = x",
                 "max diff " + sci(y.linfty_norm()));

    // Beyond the step limit of the density retraction, and far enough that components underflow
    double max_norm_error = 0, min_value = std::numeric_limits<double>::infinity();
    for (const double h : {0.5 * h_max, 10 * h_max, 1e4 * h_max}) {
        y = pts.x;
        ellipsoid_positive::retract_by_e(D, pts.v, y, h);
        max_norm_error = std::max(max_norm_error, std::abs(d_inner(D, y, y) - 1));
        min_value = std::min(min_value, *std::ranges::min_element(y));
    }
    report.check(max_norm_error < 1e-13 && min_value > 0, "e-retraction: positive and on the sphere for every step",
                 "max |y^T D y - 1| = " + sci(max_norm_error) + ", min " + sci(min_value));

    auto diff_error = [&](double h) {
        Vector<double> y_plus(pts.x), y_minus(pts.x);
        ellipsoid_positive::retract_by_e(D, pts.v, y_plus, h);
        ellipsoid_positive::retract_by_e(D, pts.v, y_minus, -h);
        y_plus -= y_minus;
        y_plus /= 2 * h;
        y_plus -= pts.v;
        return y_plus.linfty_norm();
    };
    const double error_1 = diff_error(1e-2 * h_max), error_2 = diff_error(1e-3 * h_max);
    report.check(error_2 < 1e-6 && error_1 / error_2 > 50, "e-retraction: D R_x(0) = id (finite differences)",
                 "error " + sci(error_2) + ", ratio " + sci(error_1 / error_2));

    Vector<double> lift(pts.x_hat);
    ellipsoid_positive::retract_inv_by_e(D, lift, pts.x);
    const double tangent = std::abs(d_inner(D, pts.x, lift)) / lift.l2_norm();
    report.check(tangent < 1e-13, "e-lifting: L_x(x_hat) is tangent", "|x^T D L| / |L| = " + sci(tangent));

    y = pts.x;
    ellipsoid_positive::retract_by_e(D, lift, y, 1.0);
    y -= pts.x_hat;
    report.check(y.linfty_norm() < 1e-14 * pts.x_hat.linfty_norm(), "e-lifting: R_x(L_x(x_hat)) = x_hat",
                 "max diff " + sci(y.linfty_norm()));

    Vector<double> step(pts.v);
    step *= 3 * h_max;
    y = pts.x;
    ellipsoid_positive::retract_by_e(D, step, y, 1.0);
    ellipsoid_positive::retract_inv_by_e(D, y, pts.x);
    y -= step;
    report.check(y.linfty_norm() < 1e-13 * step.linfty_norm(), "e-lifting: L_x(R_x(v)) = v",
                 "max diff " + sci(y.linfty_norm()));

    // e-geodesic: pi(x(t)) ~ pi(x)^(1-t) pi(x_hat)^t, i.e. x(t) ~ x^(1-t) x_hat^t
    double max_error = 0;
    for (const double t : {0.25, 0.5, 0.9}) {
        y = pts.x;
        ellipsoid_positive::retract_by_e(D, lift, y, t);
        Vector<double> geo = make_vector(p.n_dofs(), [&](unsigned i) {
            return std::pow(pts.x[i], 1 - t) * std::pow(pts.x_hat[i], t);
        });
        normalize(D, geo);
        y -= geo;
        max_error = std::max(max_error, y.linfty_norm());
    }
    report.check(max_error < 1e-14, "e-lifting: R_x(t L_x(x_hat)) ~ x^(1-t) x_hat^t", "max diff " + sci(max_error));
}

//! Exact supporting identity (GP_exact_support) and R-convexity, with the energy of GrossPitaevskiiFunctional.
void check_supporting_identity(CheckReport& report, Problem& p)
{
    const MassMatrix& D = p.D();
    const Points pts(D, p.n_dofs());
    const Vector<double>& x = pts.x;
    const Vector<double>& x_hat = pts.x_hat;
    Functional func(p.system, beta, solver_options());

    Vector<double> lift(x_hat);
    ellipsoid_positive::retract_inv_by_density(lift, x);

    const double E_hat = energy(func, x_hat);
    const double E     = energy(func, x);
    const double lhs   = E_hat - E - func.directional_derivative(x, lift);

    // Off-diagonal entries of A_0 = S + M_{V,L} are those of S (lumped potential term)
    double picone = 0;
    const SparseMatrix<double>& A0 = p.system.get_A0();
    for (auto it = A0.begin(); it != A0.end(); ++it) {
        const auto i = it->row(), j = it->column();
        if (j > i) {
            const double q = x_hat[i] / x[i] - x_hat[j] / x[j];
            picone += -it->value() * x[i] * x[j] * q * q;
        }
    }
    double quartic = 0;
    const auto& d = D.get_vector();
    for (unsigned i = 0; i < x.size(); ++i) {
        const double q = x_hat[i] * x_hat[i] - x[i] * x[i];
        quartic += d[i] * q * q;
    }
    const double rhs = 0.5 * picone + 0.25 * beta * quartic;

    report.check(std::abs(lhs - rhs) < 1e-11 * (std::abs(E) + std::abs(E_hat)) && lhs >= 0,
                 "energy: exact supporting identity f(x_hat) - f(x) - Df(x)[L_x(x_hat)] = Picone + quartic",
                 "lhs " + sci(lhs) + ", rhs " + sci(rhs) + ", diff " + sci(std::abs(lhs - rhs)));

    // R-convexity: second differences of phi(t) = f(R_x(t L_x(x_hat))) on [0, 1]
    const unsigned n_t = 40;
    std::vector<double> phi(n_t + 1);
    for (unsigned k = 0; k <= n_t; ++k) {
        phi[k] = energy(func, density_curve(x, x_hat, static_cast<double>(k) / n_t));
    }
    double min_second_diff = std::numeric_limits<double>::infinity();
    for (unsigned k = 1; k < n_t; ++k) {
        min_second_diff = std::min(min_second_diff, phi[k - 1] - 2 * phi[k] + phi[k + 1]);
    }
    report.check(min_second_diff > -1e-12 * std::abs(E), "energy: convex along the density curve",
                 "min second difference " + sci(min_second_diff));

    // e-geodesic (no convexity known): supporting inequality and second differences, from the test points and from a
    // pair with fast-decaying tails, x ~ exp(-|i - c|^2 / w^2) to the constant
    const unsigned n = x.size();
    const auto peak = make_vector(n, [&](unsigned i) {
        const double s = (static_cast<double>(i) - n / 2.0) / (n / 8.0);
        return std::exp(-s * s);
    });
    const auto flat = make_vector(n, [](unsigned) { return 1.0; });
    for (const auto& [name, a, b] : {std::tuple{"test points", x, x_hat}, std::tuple{"peak to constant", peak, flat}}) {
        Vector<double> x_e(a), x_hat_e(b);
        normalize(D, x_e);
        normalize(D, x_hat_e);
        Vector<double> lift_e(x_hat_e);
        ellipsoid_positive::retract_inv_by_e(D, lift_e, x_e);
        const double E_e = energy(func, x_e);
        const double support = energy(func, x_hat_e) - E_e - func.directional_derivative(x_e, lift_e);
        for (unsigned k = 0; k <= n_t; ++k) {
            Vector<double> y(x_e);
            ellipsoid_positive::retract_by_e(D, lift_e, y, static_cast<double>(k) / n_t);
            phi[k] = energy(func, y);
        }
        double min_diff = std::numeric_limits<double>::infinity();
        for (unsigned k = 1; k < n_t; ++k) {
            min_diff = std::min(min_diff, (phi[k - 1] - 2 * phi[k] + phi[k + 1]) * n_t * n_t);
        }
        report.info(std::string("energy along the e-geodesic, ") + name,
                    "f(x_hat) - f(x) - Df(x)[L^e_x(x_hat)] = " + sci(support) + ", min phi'' = " + sci(min_diff));
    }
}

DescentOptions descent_options(double step_size, bool line_search, unsigned max_iter)
{
    DescentOptions options{};
    options.tol_residual = 1e-9;
    options.step_size    = step_size;
    options.max_iter     = max_iter;
    options.line_search  = line_search;
    options.ls           = {.max_iter = 20, .alpha = 1.0, .beta = 0.5, .sigma = 1e-4, .min = 1e-8};
    return options;
}

//! Full steps or Armijo line search, with fraction-to-the-boundary steps of fraction @p theta (option A)
DescentOptions capped_descent_options(double theta, bool line_search, unsigned max_iter)
{
    DescentOptions options = descent_options(1.0, line_search, max_iter);
    options.boundary_fraction = theta;
    return options;
}

//! Energy-adaptive descent: ground state with the norm retraction, then the density retraction (GradientDescent with
//! fraction-to-the-boundary steps, option A) from multiplicative perturbations of it, and from points on the density
//! curve between the ground state and the constant start.
void check_descent(CheckReport& report, Problem& p)
{
    const MassMatrix& D = p.D();
    const unsigned n = p.n_dofs();
    Functional func(p.system, beta, solver_options());
    EnergyOracle<System> oracle(func, solver_options());

    Vector<double> x0(n);
    x0 = 1.0;
    normalize(D, x0);

    // Reference: norm retraction, full energy-adaptive steps
    const UnitMassSphere<MassMatrix> sphere(D);
    Vector<double> x_star(x0);
    GradientDescent reference(oracle, oracle.get_residual(), sphere, descent_options(1.0, false, 500));
    reference.cycle(x_star);
    const double E_star = energy(func, x_star);
    const double res_star = oracle.get_residual().residual(x_star);
    report.check(res_star < 1e-9 && *std::ranges::min_element(x_star) > 0,
                 "descent: norm retraction converges and stays positive",
                 std::to_string(reference.history().size() - 1) + " iterations, E* = "
                 + std::to_string(E_star) + ", min x* = " + sci(*std::ranges::min_element(x_star)));

    // Start points: x* (1 + eps w) normalized, and the density curve from x* to the constant start (t = 1)
    std::vector<std::pair<std::string, Vector<double>>> starts;
    for (const double eps : {0.01, 0.1, 0.5}) {
        Vector<double> z(x_star);
        for (unsigned i = 0; i < n; ++i) {
            z[i] *= 1 + eps * std::sin(0.37 * i);
        }
        normalize(D, z);
        starts.emplace_back("x* (1 + " + std::to_string(eps).substr(0, 4) + " w)", z);
    }
    for (const double t : {0.01, 0.1, 1.0}) {
        starts.emplace_back("density curve t = " + std::to_string(t).substr(0, 4), density_curve(x_star, x0, t));
    }

    // Without option A: a full step is undefined from the density curve, where a component has to shrink by more
    // than a factor ~2, and defined from the local perturbations
    const PosUnitMassSphere<MassMatrix> pos_sphere(D);
    for (const auto& [start_name, start] : starts) {
        Vector<double> eta(n);
        oracle.update(start);
        oracle.gradient(start, eta);
        eta *= -1.0;
        const double h_max = pos_sphere.max_step(start, eta);
        if (start_name.starts_with("density curve")) {
            report.check(h_max < 1, "descent: full step undefined without option A from " + start_name,
                         "h_max " + sci(h_max));
        }
        else {
            report.info("descent: h_max of the first step from " + start_name, sci(h_max));
        }
    }

    // Option A in GradientDescent: converges from every start, full steps or Armijo line search
    for (const double theta : {0.9, 0.99}) {
        for (const bool line_search : {false, true}) {
            for (const auto& [start_name, start] : starts) {
                Vector<double> z = start;
                GradientDescent solver(oracle, oracle.get_residual(), pos_sphere,
                                       capped_descent_options(theta, line_search, 500));
                solver.cycle(z);
                const double E   = energy(func, z);
                const double res = oracle.get_residual().residual(z);
                report.check(res < 1e-9 && std::abs(E - E_star) < 1e-10 * E_star && *std::ranges::min_element(z) > 0,
                             "descent (option A, theta = " + std::to_string(theta).substr(0, 4)
                                 + (line_search ? ", line search" : ", step 1") + ") from " + start_name,
                             std::to_string(solver.history().size() - 1) + " iterations, |E - E*| = "
                                 + sci(std::abs(E - E_star)));
            }
        }
    }
}

//! Velocities of the retractions (central differences) and the exact line search (tex/exact_line_search.tex): on the
//! density path phi'(t*) = 0 with t* < t_max, on the normalization ray the closed-form step of ray_step(); both at
//! least as good as a grid search. Mass metric, from a perturbed constant point.
void check_exact_line_search(CheckReport& report, Problem& p)
{
    const MassMatrix& D = p.D();
    const unsigned n = p.n_dofs();
    Functional func(p.system, beta, solver_options());
    MassOracle<System> oracle(func, solver_options());

    Vector<double> x = make_vector(n, [](unsigned i) { return 1.0 + 0.5 * std::sin(0.37 * i); });
    normalize(D, x);
    Vector<double> eta(n);
    oracle.update(x);
    oracle.gradient(x, eta);
    eta *= -1.0;

    const UnitMassSphere<MassMatrix> sphere(D);
    const PosUnitMassSphere<MassMatrix> pos_sphere(D);
    const double t_max = pos_sphere.max_step(x, eta);

    // Velocity against a central difference of the retraction, at t = 0.3 t_max
    const auto velocity_error = [&](const auto& velocity, const auto& retract) {
        const double t = 0.3 * t_max, h = 1e-6 * t_max;
        Vector<double> y(n), y_dot(n), y_plus(x), y_minus(x);
        velocity(t, y, y_dot);
        retract(t + h, y_plus);
        retract(t - h, y_minus);
        y_plus -= y_minus;
        y_plus /= 2 * h;
        y_plus -= y_dot;
        return y_plus.linfty_norm() / y_dot.linfty_norm();
    };
    const double err_norm = velocity_error(
        [&](double t, Vector<double>& y, Vector<double>& y_dot) { sphere.retract_velocity(x, eta, t, y, y_dot); },
        [&](double t, Vector<double>& y) { sphere.retract(eta, y, t); });
    const double err_density = velocity_error(
        [&](double t, Vector<double>& y, Vector<double>& y_dot) { pos_sphere.retract_velocity(x, eta, t, y, y_dot); },
        [&](double t, Vector<double>& y) { pos_sphere.retract(eta, y, t); });
    const double err_e = velocity_error(
        [&](double t, Vector<double>& y, Vector<double>& y_dot) {
            ellipsoid_positive::retract_velocity_by_e(D, x, eta, t, y, y_dot);
        },
        [&](double t, Vector<double>& y) { ellipsoid_positive::retract_by_e(D, eta, y, t); });
    report.check(std::max({err_norm, err_density, err_e}) < 1e-6, "exact: retraction velocities (central differences)",
                 "norm " + sci(err_norm) + ", density " + sci(err_density) + ", e " + sci(err_e));

    DescentOptions options = capped_descent_options(0.999999, true, 1);
    options.ls.exact = true;
    options.ls.max_iter = 60;

    // phi on a manifold, its central difference, and its minimum on a uniform grid of [0, t_end]
    const auto phi = [&](const ManifoldBase& m, double t) {
        Vector<double> y(x);
        m.retract(eta, y, t);
        return energy(func, y);
    };
    const auto grid_min = [&](const ManifoldBase& m, double t_end) {
        double best = energy(func, x);  // retract_by_norm() rejects the step 0
        for (unsigned j = 1; j <= 2000; ++j) {
            best = std::min(best, phi(m, t_end * j / 2000.0));
        }
        return best;
    };
    const double dphi0 = oracle.directional_derivative(x, eta);

    // Density path: interior zero of phi' (Theorem 3.7). From this rough point the mass gradient has its zero of
    // phi' within 1e-6 t_max of the boundary (the barrier grows only like (t_max - t)^(-1/2)); the check uses the
    // density curve towards a smooth point instead, which has t_max > 1.
    {
        const Points pts(D, n);
        Vector<double> v(pts.x_hat);
        ellipsoid_positive::retract_inv_by_density(v, x);
        const double v_max = pos_sphere.max_step(x, v);
        const auto phi_v = [&](double t) {
            Vector<double> y(x);
            pos_sphere.retract(v, y, t);
            return energy(func, y);
        };
        Vector<double> z(x);
        oracle.update(x);
        const double dphi0_v = oracle.directional_derivative(x, v);
        const double t_density = exact_line_search(oracle, pos_sphere, z, v, options, 1.0);
        const double h = 1e-4 * std::min(t_density, v_max - t_density);
        const double dphi_density = (phi_v(t_density + h) - phi_v(t_density - h)) / (2 * h);
        double grid = phi_v(0.0);
        for (unsigned j = 1; j < 2000; ++j) {
            grid = std::min(grid, phi_v(v_max * j / 2000.0));
        }
        const double gap_density = phi_v(t_density) - grid;
        report.check(dphi0_v < 0 && t_density > 0 && t_density < v_max
                         && std::abs(dphi_density) < 1e-6 * std::abs(dphi0_v) && gap_density <= 1e-12,
                     "exact: density path, interior stationary point at least as good as a grid",
                     "t*/t_max " + sci(t_density / v_max) + ", |phi'(t*)|/|phi'(0)| "
                         + sci(std::abs(dphi_density / dphi0_v)) + ", phi(t*) - grid min " + sci(gap_density));
    }
    Vector<double> z(x);

    // Normalization ray: closed form (ray_step) against central differences and a grid
    z = x;
    oracle.update(x);
    const double t_ray = exact_line_search(oracle, sphere, z, eta, options, 1.0);
    const double h_ray = 1e-7 * t_ray;
    const double dphi_ray = (phi(sphere, t_ray + h_ray) - phi(sphere, t_ray - h_ray)) / (2 * h_ray);
    const double gap_ray = phi(sphere, t_ray) - grid_min(sphere, 10 * t_ray);
    report.check(t_ray > 0 && std::abs(dphi_ray) < 1e-6 * std::abs(dphi0) && gap_ray <= 1e-12,
                 "exact: normalization ray, closed-form step stationary and at least as good as a grid",
                 "t* " + sci(t_ray) + ", |phi'(t*)|/|phi'(0)| " + sci(std::abs(dphi_ray / dphi0))
                     + ", phi(t*) - grid min " + sci(gap_ray));
}

//! Positive sphere with the normalization retraction and the positivity cap (as NormPosSphere of the study), or the
//! e-retraction with or without the growth cap (as ExpPosSphere), for the cap checks of the exact line search.
class TestPosSphere : public ManifoldBase
{
public:
    enum class Kind : std::uint8_t { NORM_CAP, E, E_CAP_UP };
    TestPosSphere(const MassMatrix& D, Kind kind) : D(D), kind(kind) {}

    void retract(const Vector<double>& z, Vector<double>& x, double factor) const override
    {
        if (kind == Kind::NORM_CAP) {
            ellipsoid::retract_by_norm(D, z, x, factor);
        }
        else {
            ellipsoid_positive::retract_by_e(D, z, x, factor);
        }
    }
    void retract(const Vector<double>& z, const Vector<double>& x, Vector<double>& out, double factor) const override
    {
        out = x;
        retract(z, out, factor);
    }
    [[nodiscard]] double max_step(const Vector<double>& x, const Vector<double>& v) const override
    {
        if (kind == Kind::NORM_CAP) {
            return 2 * ellipsoid_positive::max_step(x, v);
        }
        if (kind == Kind::E) {
            return std::numeric_limits<double>::infinity();
        }
        double rate = 0;
        for (unsigned i = 0; i < x.size(); ++i) {
            rate = std::max(rate, v[i] / x[i]);
        }
        return 1 / rate;
    }
    void retract_velocity(const Vector<double>& x, const Vector<double>& v, double t, Vector<double>& y,
                          Vector<double>& y_dot) const override
    {
        if (kind == Kind::NORM_CAP) {
            ellipsoid::retract_velocity_by_norm(D, x, v, t, y, y_dot);
        }
        else {
            ellipsoid_positive::retract_velocity_by_e(D, x, v, t, y, y_dot);
        }
    }
    [[nodiscard]] double constraint_inner(const Vector<double>& y, const Vector<double>& z) const override
    {
        return ellipsoid::mass_inner(D, y, z);
    }
    [[nodiscard]] bool retracts_by_normalization() const override { return kind == Kind::NORM_CAP; }
    void retract_inv(Vector<double>&, const Vector<double>&) const override { throw ExcNotImplemented(); }
    void retract_diff(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                      Vector<double>&) const override { throw ExcNotImplemented(); }
    void retract_inv_diff(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                          Vector<double>&) const override { throw ExcNotImplemented(); }
    void retract_inv_diff_adjoint(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                                  Vector<double>&) const override { throw ExcNotImplemented(); }

private:
    const MassMatrix& D;
    Kind kind;
};

//! Caps of the exact step (tex/exact_line_search.tex, Sections 4 and 7): the closed-form ray step without cap, with a
//! cap above and below its minimizer, against grids; the root finder of phi' with a binding cap on the density path
//! (phi still decreasing at theta t_max), and on the e-path with and without the growth cap (descent, and a root of
//! phi' unless at the cap). Mass metric, from a perturbed constant point.
void check_exact_caps(CheckReport& report, Problem& p)
{
    const MassMatrix& D = p.D();
    const unsigned n = p.n_dofs();
    Functional func(p.system, beta, solver_options());
    MassOracle<System> oracle(func, solver_options());

    Vector<double> x = make_vector(n, [](unsigned i) { return 1.0 + 0.5 * std::sin(0.37 * i); });
    normalize(D, x);
    Vector<double> eta(n);
    oracle.update(x);
    oracle.gradient(x, eta);
    eta *= -1.0;
    const double E0 = energy(func, x);

    const auto phi = [&](const ManifoldBase& m, double t) {
        if (t == 0.0) {
            return E0;
        }
        Vector<double> y(x);
        m.retract(eta, y, t);
        return energy(func, y);
    };
    const auto grid_min = [&](const ManifoldBase& m, double t_end) {
        double best = E0;
        for (unsigned j = 1; j <= 2000; ++j) {
            best = std::min(best, phi(m, t_end * j / 2000.0));
        }
        return best;
    };

    // Closed form without cap and with caps above and below its minimizer
    const UnitMassSphere<MassMatrix> sphere(D);
    oracle.update(x);
    const auto t_free = oracle.ray_step(x, eta);
    report.check(t_free.has_value() && *t_free > 0
                     && phi(sphere, *t_free) <= grid_min(sphere, 20 * *t_free) + 1e-12,
                 "caps: closed-form ray step without cap is the minimizer on a grid of [0, 20 t*]",
                 "t* " + sci(t_free.value_or(0)));
    double worst_gap = 0, worst_t = 0;
    for (const double c : {0.25, 0.5, 2.0, 8.0}) {
        const double cap = c * *t_free;
        const double t = oracle.ray_step(x, eta, cap).value_or(-1);
        double best = std::min(E0, phi(sphere, cap));
        for (unsigned j = 1; j < 2000; ++j) {
            best = std::min(best, phi(sphere, cap * j / 2000.0));
        }
        worst_gap = std::max(worst_gap, phi(sphere, t) - best);
        const double expected = c >= 1 ? *t_free : cap;  // phi decreases on [0, t*]: the cap binds below t*
        worst_t = std::max(worst_t, std::abs(t - expected) / expected);
    }
    report.check(worst_gap <= 1e-12 && worst_t <= 1e-10,
                 "caps: closed-form ray step with caps 0.25, 0.5, 2, 8 t*: the cap below t*, t* above, grid minimum",
                 "max phi(t) - grid min " + sci(worst_gap) + ", max |t - expected| / expected " + sci(worst_t));

    // Normalization on the positive sphere (norm-cap): closed form, capped at theta min x_i / |v_i|
    DescentOptions options = capped_descent_options(0.99, true, 1);
    options.ls.exact = true;
    options.ls.max_iter = 40;
    const TestPosSphere norm_cap(D, TestPosSphere::Kind::NORM_CAP);
    const double cap_norm = 0.99 * norm_cap.max_step(x, eta);
    Vector<double> z(x);
    oracle.update(x);
    const double t_norm = exact_line_search(oracle, norm_cap, z, eta, options, 1.0);
    report.check(t_norm > 0 && t_norm <= cap_norm * (1 + 1e-14)
                     && phi(norm_cap, t_norm) <= grid_min(norm_cap, cap_norm) + 1e-12
                     && *std::ranges::min_element(z) > 0,
                 "caps: norm-cap, closed form within the positivity cap, positive",
                 "t/cap " + sci(t_norm / cap_norm) + ", t*/cap " + sci(*t_free / cap_norm));

    // Density path, mass gradient from this point: the zero of phi' lies within 1e-6 t_max of the boundary, so the cap
    // theta t_max binds and phi decreases on [0, theta t_max]
    const PosUnitMassSphere<MassMatrix> pos_sphere(D);
    const double cap_density = 0.99 * pos_sphere.max_step(x, eta);
    z = x;
    oracle.update(x);
    const double t_density = exact_line_search(oracle, pos_sphere, z, eta, options, 1.0);
    report.check(std::abs(t_density - cap_density) <= 1e-14 * cap_density
                     && phi(pos_sphere, t_density) <= grid_min(pos_sphere, cap_density) + 1e-12,
                 "caps: density path, binding cap: the step is theta t_max, the minimum on [0, theta t_max]",
                 "t/(theta t_max) " + sci(t_density / cap_density));

    // e-path with the growth cap and without cap: sufficient decrease, and a stationary point unless at the cap
    for (const auto kind : {TestPosSphere::Kind::E_CAP_UP, TestPosSphere::Kind::E}) {
        const TestPosSphere e_sphere(D, kind);
        const double cap = 0.99 * e_sphere.max_step(x, eta);
        z = x;
        oracle.update(x);
        const double dphi0 = oracle.directional_derivative(x, eta);
        const double t = exact_line_search(oracle, e_sphere, z, eta, options, 1.0);
        const double h = 1e-6 * t;
        const double dphi = (phi(e_sphere, t + h) - phi(e_sphere, t - h)) / (2 * h);
        const bool at_cap = std::isfinite(cap) && std::abs(t - cap) <= 1e-14 * cap;
        const bool descent = phi(e_sphere, t) < E0;
        report.check(t > 0 && descent && (at_cap || std::abs(dphi) <= 1e-6 * std::abs(dphi0)),
                     std::string("caps: e-path ") + (kind == TestPosSphere::Kind::E ? "without cap" : "with growth cap")
                         + ", descent and a root of phi' (or at the cap)",
                     "t " + sci(t) + (std::isfinite(cap) ? ", t/cap " + sci(t / cap) : std::string(", no cap"))
                         + ", |phi'(t)|/|phi'(0)| " + sci(std::abs(dphi / dphi0)));
    }
}

} // namespace


int main()
{
    return run_tests([](CheckReport& report) {
        Problem p(5);
        report.info("problem", "Q1 lumped, Neumann, square potential, radius 4, beta 100, "
                               + std::to_string(p.n_dofs()) + " DoFs");
        check_density_map(report, p);
        check_retraction(report, p);
        check_lifting(report, p);
        check_lifting_diff(report, p);
        check_e_retraction(report, p);
        check_supporting_identity(report, p);
        check_descent(report, p);
        check_exact_line_search(report, p);
        check_exact_caps(report, p);
    });
}
