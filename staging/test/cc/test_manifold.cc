//
// Sanity check for the product Bernoulli manifold geometry (Table 9) and the
// pixel-grid forward-difference operator (eq. (42), Sec. 6.3.1) implemented in
// include/rmo/cc/manifold.h and include/rmo/cc/grid.h.
//
// The Bernoulli manifold's operations are closed-form and elementwise, so
// exact algebraic identities (retraction/lifting are mutual inverses, the two
// differentials are mutually inverse linear maps) already catch sign/index
// mistakes; a single fixed-step central-difference comparison per
// differential then confirms the closed forms match the functions they
// linearize.
//
#include <rmo/cc/grid.h>
#include <rmo/cc/manifold.h>
#include <rmo/util/random.h>

#include <cmath>
#include <iostream>
#include <string>

using namespace rmo;
using namespace rmo::cc;

namespace
{
constexpr double tol_exact = 1e-9;   // algebraic identities (composition of closed forms)
constexpr double tol_fd    = 1e-6;   // central-difference comparisons (O(h^2) truncation + cancellation)
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

// Random point of B = (0,1)^n, kept away from the boundary so that finite
// differences with step fd_h stay well inside the domain.
Vector<double> raster(std::initializer_list<double> values)
{
    Vector<double> v(values.size());
    unsigned int i = 0;
    for (double x : values) {
        v[i++] = x;
    }
    return v;
}

Vector<double> random_phi(unsigned int n)
{
    Vector<double> phi(n);
    unifrnd(0.2, 0.8, phi);
    return phi;
}

Vector<double> random_tangent(unsigned int n, double stddev = 0.5)
{
    Vector<double> v(n);
    normrnd(0.0, stddev, v);
    return v;
}

// -------------------------------------------------------------------------
// Bernoulli manifold checks
// -------------------------------------------------------------------------

void check_retraction_inverses(unsigned int n)
{
    const Vector<double> phi = random_phi(n);
    const Vector<double> v   = random_tangent(n);

    // retract_inv(retract(v, phi), phi) == v
    Vector<double> z;
    bernoulli::retract(v, phi, z);
    Vector<double> v_rec(z);
    bernoulli::retract_inv(v_rec, phi);
    check(max_abs_diff(v_rec, v) < tol_exact, "retract_inv(retract(v)) == v");

    // retract(retract_inv(z, phi), phi) == z, for a fresh random point z
    const Vector<double> z0 = random_phi(n);
    Vector<double> u(z0);
    bernoulli::retract_inv(u, phi);
    Vector<double> z_rec;
    bernoulli::retract(u, phi, z_rec);
    check(max_abs_diff(z_rec, z0) < tol_exact, "retract(retract_inv(z)) == z");
}

void check_differentials_are_mutually_inverse(unsigned int n)
{
    const Vector<double> phi = random_phi(n);
    const Vector<double> v   = random_tangent(n);
    const Vector<double> w   = random_tangent(n, 1.0);

    Vector<double> z;
    bernoulli::retract(v, phi, z);   // z = R_phi(v)

    // D L_phi(z) [ D R_phi(v)[w] ] == w
    Vector<double> DRw;
    bernoulli::retract_diff(phi, v, w, DRw);

    Vector<double> DL_DRw;
    bernoulli::retract_inv_diff(phi, z, DRw, DL_DRw);

    check(max_abs_diff(DL_DRw, w) < tol_exact, "D L_phi(z)[D R_phi(v)[w]] == w");
}

void check_retract_diff_finite_difference(unsigned int n)
{
    const Vector<double> phi = random_phi(n);
    const Vector<double> v   = random_tangent(n);
    const Vector<double> w   = random_tangent(n, 1.0);

    Vector<double> analytic;
    bernoulli::retract_diff(phi, v, w, analytic);

    // Central difference of v -> R_phi(v) in direction w.
    Vector<double> v_plus(v), v_minus(v);
    v_plus.add(fd_h, w);
    v_minus.add(-fd_h, w);

    Vector<double> R_plus, R_minus;
    bernoulli::retract(v_plus, phi, R_plus);
    bernoulli::retract(v_minus, phi, R_minus);

    Vector<double> fd(R_plus);
    fd -= R_minus;
    fd /= (2.0 * fd_h);

    check(max_abs_diff(analytic, fd) < tol_fd, "retract_diff matches central difference of retract");
}

void check_retract_inv_diff_finite_difference(unsigned int n)
{
    const Vector<double> phi = random_phi(n);
    const Vector<double> z   = random_phi(n);
    const Vector<double> u   = random_tangent(n, 0.05); // small: z +/- h*u must stay in (0,1)

    Vector<double> analytic;
    bernoulli::retract_inv_diff(phi, z, u, analytic);

    Vector<double> z_plus(z), z_minus(z);
    z_plus.add(fd_h, u);
    z_minus.add(-fd_h, u);

    Vector<double> L_plus(z_plus), L_minus(z_minus);
    bernoulli::retract_inv(L_plus, phi);
    bernoulli::retract_inv(L_minus, phi);

    Vector<double> fd(L_plus);
    fd -= L_minus;
    fd /= (2.0 * fd_h);

    check(max_abs_diff(analytic, fd) < tol_fd, "retract_inv_diff matches central difference of retract_inv");
}

void check_adjoint_self_consistency(unsigned int n)
{
    const Vector<double> phi = random_phi(n);
    const Vector<double> z   = random_phi(n);
    const Vector<double> u   = random_tangent(n);
    const Vector<double> w   = random_tangent(n);

    // D L_phi(z) is diagonal, hence Euclidean-self-adjoint:
    // <D L_phi(z)[u], w> == <u, D L_phi(z)^*[w]> == <u, D L_phi(z)[w]>
    Vector<double> DLu, DLw;
    bernoulli::retract_inv_diff(phi, z, u, DLu);
    bernoulli::retract_inv_diff_adjoint(phi, z, w, DLw);

    const double lhs = DLu * w;
    const double rhs = u * DLw;

    check(std::abs(lhs - rhs) < tol_exact * std::max(1.0, std::abs(lhs)), "retract_inv_diff_adjoint is the Euclidean adjoint");
}

// Cross-check against RMO-continuous-cuts/src/bernoulli_multilevel/manifold.py
// (bernoulli_multilevel.manifold.exp/lifting) on fixed inputs; see the plan
// (doc/plan_continuous_cuts.tex) for the exact derivation.
void check_against_numpy_reference()
{
    const Vector<double> phi = raster({0.3, 0.6, 0.2, 0.8});
    const Vector<double> v   = raster({0.1, -0.2, 0.05, 0.3});

    Vector<double> z;
    bernoulli::retract(v, phi, z);
    const Vector<double> z_expected = raster({0.4082734656, 0.3946354944, 0.2546821715, 0.9630768456});
    check(max_abs_diff(z, z_expected) < 1e-9, "retract(v,phi) matches numpy exp(phi,v)");

    Vector<double> v_rec(z);
    bernoulli::retract_inv(v_rec, phi);
    check(max_abs_diff(v_rec, v) < 1e-9, "retract_inv(retract(v,phi),phi) matches numpy lifting(phi,exp(phi,v)) == v");
}

void check_manifold_base_matches_free_functions(unsigned int n)
{
    const BernoulliManifold manifold;
    const Vector<double> phi = random_phi(n);
    const Vector<double> v   = random_tangent(n);

    Vector<double> z_free, z_iface;
    bernoulli::retract(v, phi, z_free);
    manifold.retract(v, phi, z_iface);

    check(max_abs_diff(z_free, z_iface) < tol_exact, "BernoulliManifold::retract matches bernoulli::retract");
}

// -------------------------------------------------------------------------
// PixelGrid / ForwardDifference checks
// -------------------------------------------------------------------------

void check_pixel_grid_round_trip(unsigned int rows, unsigned int cols)
{
    const PixelGrid grid(rows, cols);
    const unsigned int n = grid.n_dofs();

    Vector<double> raster(n);
    for (unsigned int i = 0; i < n; i++) {
        raster[i] = static_cast<double>(i);   // distinct values so a mismatched permutation is caught
    }

    const Vector<double> dof_order    = grid.to_dof_order(raster);
    const Vector<double> raster_again = grid.to_raster_order(dof_order);

    check(max_abs_diff(raster, raster_again) < tol_exact, "PixelGrid raster <-> DoF round trip is the identity");
}

void check_forward_difference(unsigned int rows, unsigned int cols)
{
    const PixelGrid grid(rows, cols);
    const ForwardDifference fd(grid);
    const unsigned int n = grid.n_dofs();

    // phi(row, col) = col + 10*row: constant unit horizontal, constant 10x vertical difference.
    Vector<double> phi(n);
    for (unsigned int row = 0; row < rows; row++) {
        for (unsigned int col = 0; col < cols; col++) {
            phi[grid.dof_index(row, col)] = static_cast<double>(col) + 10.0 * static_cast<double>(row);
        }
    }

    Vector<double> Dphi(fd.n_rows());
    fd.matrix().vmult(Dphi, phi);

    bool ok = true;
    for (unsigned int row = 0; row < rows && ok; row++) {
        for (unsigned int col = 0; col < cols && ok; col++) {
            const unsigned int i = grid.dof_index(row, col);

            const double d1_expected = (col + 1 < cols) ? 1.0 : 0.0;
            const double d2_expected = (row + 1 < rows) ? 10.0 : 0.0;

            if (std::abs(Dphi[i] - d1_expected) > tol_exact || std::abs(Dphi[n + i] - d2_expected) > tol_exact) {
                ok = false;
            }
        }
    }
    check(ok, "ForwardDifference reproduces the known stencil on a linear ramp, incl. zero boundary rows");
}

} // namespace


int main()
{
    try {
        std::cout << "Bernoulli manifold (Table 9)\n";
        for (unsigned int n : {1u, 5u, 50u}) {
            check_retraction_inverses(n);
            check_differentials_are_mutually_inverse(n);
            check_retract_diff_finite_difference(n);
            check_retract_inv_diff_finite_difference(n);
            check_adjoint_self_consistency(n);
            check_manifold_base_matches_free_functions(n);
        }
        check_against_numpy_reference();

        std::cout << "\nPixelGrid / ForwardDifference (Sec. 6.3.1)\n";
        check_pixel_grid_round_trip(4, 5);
        check_forward_difference(4, 5);
        check_pixel_grid_round_trip(2, 2);
        check_forward_difference(2, 2);
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
