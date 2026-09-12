//
// Sanity check for include/rmo/cc/interpolate.h and include/rmo/cc/transport.h.
//
// The expected values below were produced by calling
// RMO-continuous-cuts/src/bernoulli_multilevel/{primitives,operators}.py
// directly (r_injection, prolong_bilinear, restrict_adjoint_bilinear,
// BASE_TRIPLES["Option 1"]) on a fixed 3x3 coarse / 5x5 fine grid; see the
// plan (doc/plan_continuous_cuts.tex) for the exact derivation.
//
#include <rmo/cc/interpolate.h>
#include <rmo/cc/manifold.h>
#include <rmo/cc/metric.h>
#include <rmo/cc/transport.h>
#include <rmo/util/random.h>

#include <cmath>
#include <iostream>
#include <string>

using namespace rmo;
using namespace rmo::cc;

namespace
{
constexpr double tol = 1e-9;
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

} // namespace


int main()
{
    try {
        const PixelGrid coarse(3, 3);
        const PixelGrid fine(5, 5);
        const BernoulliGridTransfer transfer(coarse, fine);

        std::cout << "BernoulliGridTransfer\n";

        // injection: coarse(I,J) = fine(2I,2J)
        {
            const Vector<double> fine_r = raster({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
            const Vector<double> expected = raster({1, 3, 5, 11, 13, 15, 21, 23, 25});

            Vector<double> dof_coarse(coarse.n_dofs());
            transfer.to_coarse_mesh(fine.to_dof_order(fine_r), dof_coarse);
            const Vector<double> result = coarse.to_raster_order(dof_coarse);

            check(max_abs_diff(result, expected) < tol, "injection matches reference r_injection");
        }

        const Vector<double> coarse_r = raster({0.2, 0.5, 0.7, 0.3, 0.6, 0.1, 0.8, 0.4, 0.9});

        // bilinear interpolation
        {
            const Vector<double> expected = raster({
                0.2000000000, 0.3500000000, 0.5000000000, 0.6000000000, 0.7000000000,
                0.2500000000, 0.4000000000, 0.5500000000, 0.4750000000, 0.4000000000,
                0.3000000000, 0.4500000000, 0.6000000000, 0.3500000000, 0.1000000000,
                0.5500000000, 0.5250000000, 0.5000000000, 0.5000000000, 0.5000000000,
                0.8000000000, 0.6000000000, 0.4000000000, 0.6500000000, 0.9000000000});

            Vector<double> dof_fine(fine.n_dofs());
            transfer.to_fine_mesh(coarse.to_dof_order(coarse_r), dof_fine);
            const Vector<double> result = fine.to_raster_order(dof_fine);

            check(max_abs_diff(result, expected) < tol, "to_fine_mesh matches reference prolong_bilinear");
        }

        const Vector<double> fine_v_r = raster({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});

        // Tfine (exact transpose of bilinear = 4x full-weighting)
        {
            const Vector<double> expected = raster({6.75, 14.0, 14.25, 34.0, 52.0, 44.0, 44.25, 64.0, 51.75});

            Vector<double> dof_coarse(coarse.n_dofs());
            transfer.Tfine(fine.to_dof_order(fine_v_r), dof_coarse);
            const Vector<double> result = coarse.to_raster_order(dof_coarse);

            check(max_abs_diff(result, expected) < tol, "Tfine matches reference restrict_adjoint_bilinear");
        }

        // adjoint identity <Tfine(u), w> == <u, to_fine_mesh(w)>, cross-checked against the reference's 227.5
        {
            const Vector<double> w_r = raster({1, 0, 2, 0, 1, 0, 2, 0, 1});
            const Vector<double> u_dof = fine.to_dof_order(fine_v_r);
            const Vector<double> w_dof = coarse.to_dof_order(w_r);

            Vector<double> Ttu(coarse.n_dofs());
            transfer.Tfine(u_dof, Ttu);
            Vector<double> Bw(fine.n_dofs());
            transfer.to_fine_mesh(w_dof, Bw);

            const double lhs = Ttu * w_dof;
            const double rhs = u_dof * Bw;

            check(std::abs(lhs - 227.5) < tol && std::abs(rhs - 227.5) < tol,
                "adjoint identity <Tfine(u),w> == <u,to_fine_mesh(w)> == 227.5 (reference)");
        }

        std::cout << "\nBernoulliPointTransfer\n";
        {
            Vector<double> phi_r(fine.n_dofs());
            for (unsigned int k = 0; k < fine.n_dofs(); k++) {
                phi_r[k] = 0.1 + 0.8 * static_cast<double>(k) / 24.0; // matches the reference's x_fine ramp
            }
            const Vector<double> phi_dof = fine.to_dof_order(phi_r);

            BernoulliPointTransfer r_inj(transfer, BernoulliPointTransfer::Restriction::INJECTION);
            Vector<double> psi_inj(coarse.n_dofs()), psi_direct(coarse.n_dofs());
            r_inj.restriction(phi_dof, psi_inj);
            transfer.to_coarse_mesh(phi_dof, psi_direct);
            check(max_abs_diff(psi_inj, psi_direct) < tol,
                "injection point restriction commutes with logit/sigmoid (matches to_coarse_mesh directly)");

            BernoulliPointTransfer r_fw(transfer, BernoulliPointTransfer::Restriction::FULL_WEIGHTING);
            Vector<double> psi_fw(coarse.n_dofs());
            r_fw.restriction(phi_dof, psi_fw);

            Vector<double> logit_phi, logit_psi_expected(coarse.n_dofs()), psi_fw_expected;
            bernoulli::logit(phi_dof, logit_phi);
            transfer.Tfine(logit_phi, logit_psi_expected);
            logit_psi_expected *= 0.25;
            bernoulli::sigmoid(logit_psi_expected, psi_fw_expected);
            check(max_abs_diff(psi_fw, psi_fw_expected) < tol,
                "full-weighting point restriction matches sigmoid(0.25*Tfine(logit(phi)))");

            Vector<double> phi_p(fine.n_dofs()), logit_psi, logit_phi_expected(fine.n_dofs()), phi_p_expected;
            r_inj.prolongation(psi_inj, phi_p);
            bernoulli::logit(psi_inj, logit_psi);
            transfer.to_fine_mesh(logit_psi, logit_phi_expected);
            bernoulli::sigmoid(logit_phi_expected, phi_p_expected);
            check(max_abs_diff(phi_p, phi_p_expected) < tol,
                "point prolongation matches sigmoid(to_fine_mesh(logit(psi)))");
        }

        std::cout << "\nGeometricBilinearTransport (Table 10, Options 1 & 4)\n";
        {
            Vector<double> x_fine_r(fine.n_dofs());
            for (unsigned int k = 0; k < fine.n_dofs(); k++) {
                x_fine_r[k] = 0.1 + 0.8 * static_cast<double>(k) / 24.0;
            }
            const Vector<double> x_fine_dof = fine.to_dof_order(x_fine_r);
            const Vector<double> psi_coarse_dof = coarse.to_dof_order(coarse_r);
            const Vector<double> v_coarse_dof = coarse.to_dof_order(raster({1, 0, 2, 0, 1, 0, 2, 0, 1}));

            GeometricBilinearTransport gbt(transfer);

            Vector<double> P(fine.n_dofs());
            gbt.vector_prolongation(x_fine_dof, psi_coarse_dof, v_coarse_dof, P);
            const Vector<double> P_expected = fine.to_dof_order(raster({
                0.5625000000, 0.3611111111, 0.0000000000, 0.7619047619, 1.7037037037,
                0.6111111111, 0.5468750000, 0.4629629630, 0.7948082011, 1.1428571429,
                0.0000000000, 0.5185185185, 1.0416666667, 0.5185185185, 0.0000000000,
                1.5000000000, 0.9675925926, 0.4629629630, 0.8020833333, 1.0864197531,
                2.2361111111, 1.0000000000, 0.0000000000, 0.6419753086, 1.0000000000}));

            check(max_abs_diff(P, P_expected) < 1e-6,
                "vector_prolongation matches reference apply_G_inv(phi, bilinear(apply_G(psi,v)))");

            const Vector<double> v_fine_dof = fine.to_dof_order(fine_v_r);
            Vector<double> R(coarse.n_dofs()), R_direct(coarse.n_dofs());
            gbt.vector_restriction(psi_coarse_dof, x_fine_dof, v_fine_dof, R);
            transfer.Tfine(v_fine_dof, R_direct);
            check(max_abs_diff(R, R_direct) < tol, "vector_restriction is exactly Tfine (no metric weighting)");
        }
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
