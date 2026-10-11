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
 * - the energy of GrossPitaevskiiFunctional satisfies the exact supporting identity, and is convex along the curve;
 * - without fraction-to-the-boundary steps, a full energy-adaptive step is undefined from points on the density
 *   curve towards the constant start (a component has to shrink by more than a factor ~2);
 * - energy-adaptive gradient descent on PosUnitMassSphere (GradientDescent with fraction-to-the-boundary steps
 *   \f$ h = \min(1, \theta h_{\max}) \f$, DescentOptions::boundary_fraction; full steps or Armijo line search)
 *   converges to the ground state of the norm retraction from these points and from multiplicative perturbations of
 *   the ground state.
 */
#include "check.h"

#include <rmo/gpe/gpe.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/oracle.h>
#include <rmo/ropt/solver.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
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
        check_supporting_identity(report, p);
        check_descent(report, p);
    });
}
