//
// Sanity check for the continuous-cuts ambient objective and Fisher-Rao oracle
// implemented in include/rmo/cc/cc.h and include/rmo/cc/oracle.h.
//
#include <rmo/cc/cc.h>
#include <rmo/cc/grid.h>
#include <rmo/cc/metric.h>
#include <rmo/cc/oracle.h>
#include <rmo/util/random.h>

#include <cmath>
#include <iostream>
#include <string>

using namespace rmo;
using namespace rmo::cc;

namespace
{
constexpr double tol_exact = 1e-9;
constexpr double tol_fd    = 1e-6;
constexpr double fd_h      = 1e-6;

int n_failed = 0;

void check(bool ok, const std::string& name)
{
    if (ok) {
        std::cout << "  PASS  " << name << "\n";
    } else {
        std::cerr << "  FAIL  " << name << "\n";
        n_failed++;
    }
}

double max_abs_diff(const Vector<double>& a, const Vector<double>& b)
{
    AssertDimension(a.size(), b.size());
    double d = 0.0;
    for (unsigned int i = 0; i < a.size(); i++) {
        d = std::max(d, std::abs(a[i] - b[i]));
    }
    return d;
}

Vector<double> raster(std::initializer_list<double> values)
{
    Vector<double> v(values.size());
    unsigned int i = 0;
    for (double x : values) {
        v[i++] = x;
    }
    return v;
}

Vector<double> random_vector(unsigned int n, double stddev = 1.0)
{
    Vector<double> v(n);
    normrnd(0.0, stddev, v);
    return v;
}

// Random point of (0,1)^n, kept away from the boundary.
Vector<double> random_phi(unsigned int n)
{
    Vector<double> phi(n);
    unifrnd(0.2, 0.8, phi);
    return phi;
}

// -------------------------------------------------------------------------

void check_directional_derivative_finite_difference(ContinuousCutsFunctional& func)
{
    const unsigned int n = func.n_dofs();
    const Vector<double> phi = random_vector(n, 0.5);
    const Vector<double> z   = random_vector(n, 1.0);

    func.update(phi);
    const double dd = func.directional_derivative(phi, z);

    Vector<double> phi_plus(phi), phi_minus(phi);
    phi_plus.add(fd_h, z);
    phi_minus.add(-fd_h, z);

    func.update(phi_plus);
    const double v_plus = func.value(phi_plus);
    func.update(phi_minus);
    const double v_minus = func.value(phi_minus);

    const double fd = (v_plus - v_minus) / (2.0 * fd_h);

    check(std::abs(dd - fd) < tol_fd * std::max(1.0, std::abs(dd)),
        "directional_derivative matches central difference of value");
}

void check_gradient_matches_directional_derivative(ContinuousCutsFunctional& func)
{
    const unsigned int n = func.n_dofs();
    const Vector<double> phi = random_vector(n, 0.5);
    const Vector<double> z   = random_vector(n, 1.0);

    func.update(phi);
    Vector<double> g(n);
    func.gradient(phi, g);

    const double dd  = func.directional_derivative(phi, z);
    const double g_z = g * z;

    check(std::abs(dd - g_z) < tol_exact * std::max(1.0, std::abs(dd)),
        "gradient dot z equals directional_derivative");
}

void check_oracle_gradient_is_metric_weighted_ambient_gradient(ContinuousCutsFunctional& func)
{
    const unsigned int n = func.n_dofs();
    const Vector<double> phi = random_phi(n);

    FisherRaoOracle oracle(func);
    oracle.update(phi);

    Vector<double> riemannian(n);
    oracle.gradient(phi, riemannian);

    Vector<double> ambient(n);
    func.gradient(phi, ambient);

    Vector<double> expected(n);
    for (unsigned int i = 0; i < n; i++) {
        expected[i] = metric::fisher_rao::weight(phi[i]) * ambient[i];
    }

    check(max_abs_diff(riemannian, expected) < tol_exact,
        "FisherRaoOracle::gradient == diag(phi(1-phi)) * ambient gradient");
}

void check_oracle_norm_and_inner(ContinuousCutsFunctional& func)
{
    const unsigned int n = func.n_dofs();
    const Vector<double> phi = random_phi(n);

    FisherRaoOracle oracle(func);
    oracle.update(phi);

    const Vector<double> u = random_vector(n);
    const Vector<double> v = random_vector(n);

    check(std::abs(oracle.norm(v) - metric::fisher_rao::norm(v, phi)) < tol_exact,
        "FisherRaoOracle::norm matches metric::fisher_rao::norm");
    check(std::abs(oracle.inner(u, v) - metric::fisher_rao::inner(u, v, phi)) < tol_exact,
        "FisherRaoOracle::inner matches metric::fisher_rao::inner");

    // apply_metric applies G(phi); inner(u,v) = u^T G(phi) v = u . apply_metric(v)
    Vector<double> Gv(n);
    oracle.apply_metric(v, Gv);
    check(std::abs(oracle.inner(u, v) - u * Gv) < tol_exact,
        "inner(u,v) == u . apply_metric(v)");
}

void check_residual_matches_gradient_norm(ContinuousCutsFunctional& func)
{
    const unsigned int n = func.n_dofs();
    const Vector<double> phi = random_phi(n);

    FisherRaoOracle oracle(func);
    oracle.update(phi);

    Vector<double> g(n);
    oracle.gradient(phi, g);

    check(std::abs(oracle.residual(phi) - oracle.norm(g)) < tol_exact,
        "residual(phi) == norm(gradient(phi))");
}

// Cross-check against RMO-continuous-cuts/src/bernoulli_multilevel/objective.py:
// value() against create_cc_objective(rho,alpha,eps)(coarse), and gradient()
// against its torch.autograd gradient; see the plan for the exact derivation.
void check_against_numpy_reference()
{
    const PixelGrid grid(3, 3);
    const ForwardDifference D(grid);

    const Vector<double> rho_raster = raster({0.1, -0.2, 0.3, 0.0, 0.4, -0.1, 0.2, 0.1, -0.3});
    const Vector<double> phi_raster = raster({0.2, 0.5, 0.7, 0.3, 0.6, 0.1, 0.8, 0.4, 0.9});

    // The reference's isotropic_tv() adds eps (not eps^2) under the square root, unlike
    // eq. (42)'s epsilon^2; pass sqrt(1e-3) so our epsilon^2 matches its eps=1e-3.
    ContinuousCutsFunctional func(D, grid.to_dof_order(rho_raster), /*alpha=*/0.7, /*epsilon=*/std::sqrt(1e-3));

    const Vector<double> phi_dof = grid.to_dof_order(phi_raster);
    func.update(phi_dof);

    check(std::abs(func.value(phi_dof) - 3.0916522252) < 1e-8, "value(phi) matches create_cc_objective(rho,alpha,eps)(coarse)");

    Vector<double> g_dof(grid.n_dofs());
    func.gradient(phi_dof, g_dof);
    const Vector<double> g = grid.to_raster_order(g_dof);

    const Vector<double> g_expected = raster({-0.7810434857, -0.4691130155, 1.6189602184,
                                               -0.7387219719, 1.9779261735, -2.1472995616,
                                                1.4971869779, -1.5559532861, 1.0980579511});
    check(max_abs_diff(g, g_expected) < 1e-6, "gradient(phi) matches torch.autograd gradient of create_cc_objective");
}

} // namespace


int main()
{
    try {
        const PixelGrid grid(4, 5);
        const ForwardDifference D(grid);

        Vector<double> rho = random_vector(grid.n_dofs(), 1.0);
        ContinuousCutsFunctional func(D, rho, /*alpha=*/0.7, /*epsilon=*/1e-3);

        std::cout << "ContinuousCutsFunctional (ambient objective)\n";
        check_directional_derivative_finite_difference(func);
        check_gradient_matches_directional_derivative(func);

        std::cout << "\nFisherRaoOracle\n";
        check_oracle_gradient_is_metric_weighted_ambient_gradient(func);
        check_oracle_norm_and_inner(func);
        check_residual_matches_gradient_norm(func);
        check_against_numpy_reference();
    }
    catch (std::exception& e) {
        std::cerr << "\nFAIL (exception): " << e.what() << "\n";
        return 1;
    }

    if (n_failed > 0) {
        std::cerr << "\n" << n_failed << " check(s) failed.\n";
        return 1;
    }
    std::cout << "\nAll checks passed.\n";
    return 0;
}
