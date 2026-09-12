//
// Sanity check for include/rmo/cc/oracle_coarse.h, exercised through the full
// CoarseOracleBase machinery (point restriction, vector restriction, model
// update) built from interpolate.h/transport.h. Run under both point-restriction
// choices in BernoulliPointTransfer (injection: Table 10 Option 1; full-weighting:
// Option 4), since they share the same vector transport R, P.
//
#include <rmo/cc/cc.h>
#include <rmo/cc/grid.h>
#include <rmo/cc/interpolate.h>
#include <rmo/cc/manifold.h>
#include <rmo/cc/metric.h>
#include <rmo/cc/oracle.h>
#include <rmo/cc/oracle_coarse.h>
#include <rmo/cc/transport.h>
#include <rmo/util/random.h>

#include <cmath>
#include <iostream>
#include <string>

using namespace rmo;
using namespace rmo::cc;

namespace
{
constexpr double tol_exact = 1e-8;
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

std::string restriction_label(BernoulliPointTransfer::Restriction kind)
{
    return kind == BernoulliPointTransfer::Restriction::INJECTION ? "injection" : "full-weighting";
}

Vector<double> random_phi(unsigned int n)
{
    Vector<double> phi(n);
    unifrnd(0.2, 0.8, phi);
    return phi;
}

Vector<double> random_vector(unsigned int n, double stddev = 1.0)
{
    Vector<double> v(n);
    normrnd(0.0, stddev, v);
    return v;
}

// Runs the full Prop. 3.2 coherence suite for one choice of point restriction. Only the
// point restriction r changes between INJECTION and FULL_WEIGHTING (Table 10's Options 1
// and 4 share the same vector transport R, P -- see transport.h); this exercises whether
// CoarseOracleBase's coherence property (which only depends on R, P, not r) survives the
// change, and whether the FULL_WEIGHTING branch (untested here before) is wired correctly.
void run_checks(BernoulliPointTransfer::Restriction kind)
{
    const std::string label = " [" + restriction_label(kind) + "]";

    const PixelGrid coarse_grid(3, 3);
    const PixelGrid fine_grid(5, 5);

    const ForwardDifference D_coarse(coarse_grid);
    const ForwardDifference D_fine(fine_grid);

    ContinuousCutsFunctional f_coarse(D_coarse, random_vector(coarse_grid.n_dofs()), 0.7, 1e-3);
    ContinuousCutsFunctional f_fine(D_fine, random_vector(fine_grid.n_dofs()), 0.7, 1e-3);

    FisherRaoOracle T_coarse(f_coarse);
    FisherRaoOracle T_fine(f_fine);

    const BernoulliManifold manifold;
    const BernoulliGridTransfer grid_transfer(coarse_grid, fine_grid);
    const BernoulliPointTransfer point_transfer(grid_transfer, kind);
    const GeometricBilinearTransport vector_transport(grid_transfer);

    CoarseOracleBase qk_base(T_fine, T_coarse, manifold, point_transfer, vector_transport);

    const Vector<double> x_fine = random_phi(fine_grid.n_dofs());
    T_fine.update(x_fine);   // CoarseOracleBase::update_model assumes T_fine already matches x_fine
    qk_base.update_model(x_fine);
    const auto& state = qk_base.get_state();

    ContinuousCutsCoarseOracle qk(qk_base);

    std::cout << "ContinuousCutsCoarseOracle" << label << "\n";

    // Proposition 3.2 (first-order coherence): q_k(psi_k) == E_H(psi_k), i.e. the
    // correction term vanishes exactly at the base point.
    {
        qk.update(state.y);
        check(std::abs(qk.value(state.y) - T_coarse.value(state.y)) < tol_exact,
            "value(psi_k) == E_H(psi_k) (correction term vanishes at the base point)" + label);
    }

    // Proposition 3.2: grad q_k(psi_k) == R^{psi_k}_x(grad E_h(x)) == state.x_grad_restr.
    {
        qk.update(state.y);
        Vector<double> g(coarse_grid.n_dofs());
        qk.gradient(state.y, g);

        double diff = 0.0;
        for (unsigned int i = 0; i < g.size(); i++) {
            diff = std::max(diff, std::abs(g[i] - state.x_grad_restr[i]));
        }
        check(diff < tol_exact, "grad q_k(psi_k) == restricted fine gradient (first-order coherence)" + label);
    }

    // <gradient(z), v>_z == directional_derivative(z,v) (gradient() is the Riemannian
    // gradient, so the defining identity uses the Fisher-Rao inner product, not the
    // plain Euclidean dot product), at a trial point z != psi_k.
    {
        const Vector<double> z = random_phi(coarse_grid.n_dofs());
        const Vector<double> v = random_vector(coarse_grid.n_dofs());

        qk.update(z);
        Vector<double> g(coarse_grid.n_dofs());
        qk.gradient(z, g);
        const double g_v = qk.inner(g, v);
        const double dd  = qk.directional_derivative(z, v);

        check(std::abs(g_v - dd) < tol_exact * std::max(1.0, std::abs(dd)),
            "<gradient(z), v>_z == directional_derivative(z, v)" + label);
    }

    // directional_derivative(z,v) matches a central difference of value(.) along v.
    {
        const Vector<double> z = random_phi(coarse_grid.n_dofs());
        const Vector<double> v = random_vector(coarse_grid.n_dofs());

        qk.update(z);
        const double dd = qk.directional_derivative(z, v);

        Vector<double> z_plus(z), z_minus(z);
        z_plus.add(fd_h, v);
        z_minus.add(-fd_h, v);

        qk.update(z_plus);
        const double v_plus = qk.value(z_plus);
        qk.update(z_minus);
        const double v_minus = qk.value(z_minus);

        const double fd = (v_plus - v_minus) / (2.0 * fd_h);
        check(std::abs(dd - fd) < tol_fd * std::max(1.0, std::abs(dd)),
            "directional_derivative matches central difference of value" + label);
    }

    // residual(z) == norm(gradient(z)) at z.
    {
        const Vector<double> z = random_phi(coarse_grid.n_dofs());
        qk.update(z);
        Vector<double> g(coarse_grid.n_dofs());
        qk.gradient(z, g);

        check(std::abs(qk.residual(z) - qk.norm(g)) < tol_exact, "residual(z) == norm(gradient(z))" + label);
    }
}

} // namespace


int main()
{
    try {
        run_checks(BernoulliPointTransfer::Restriction::INJECTION);
        run_checks(BernoulliPointTransfer::Restriction::FULL_WEIGHTING);
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
