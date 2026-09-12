//
// Sanity check for oracle_coarse.h's CoarseOracleBase + MassCoarseOracle, exercised
// through an actual two-mesh-level Gross-Pitaevskii setup (LinearTransferMG,
// ManifoldTransfer, MassProjectionTransport) -- analogous to
// test/cc/test_oracle_coarse.cc, and the first genuine (not single-mesh-synthetic)
// exercise of this transfer chain in the test suite.
//
#include <rmo/fe/interpolate.h>
#include <rmo/gpe/gpe.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/oracle_coarse.h>
#include <rmo/gpe/transport.h>

#include <rmo/util/random.h>

#include <iostream>
#include <stdexcept>
#include <string>

using namespace rmo;
using namespace rmo::gpe;

namespace
{
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
} // namespace


int main()
{
    try {
        GPE_Options options{};
        options.dimension = 2;
        options.degree    = 1;
        options.radius    = 10;
        options.beta      = 50;   // nonzero: beta=0 hits LinearCombination's zero-weight assert (see test_arpack)
        options.order     = Ordering::DEFAULT;
        options.bc        = BoundaryCondition::NEUMANN;
        options.mesh_kind = MeshKind::QUADRILATERAL;

        SolverOptions options_slv{};
        options_slv.solver        = SolverMethod::CG;
        options_slv.precond       = Precondition::NONE;
        options_slv.max_inner     = 2000;
        options_slv.tol_inner     = 1e-10;
        options_slv.tol_inner_res = 1e-2;

        ModelBuilder<2> builder_coarse(potential::Square<2>(), options, /*n_levels=*/2);
        ModelBuilder<2> builder_fine  (potential::Square<2>(), options, /*n_levels=*/3);

        GrossPitaevskiiFunctional<2> obj_coarse(builder_coarse.get_system(), options.beta, options_slv);
        GrossPitaevskiiFunctional<2> obj_fine  (builder_fine.get_system(),   options.beta, options_slv);

        const UnitMassSphere<OperatorType> coarse_manifold(obj_coarse.get_M());

        const fe::LinearTransferMG<2> transfer(
            builder_coarse.get_package().get_dofs(), builder_fine.get_package().get_dofs(),
            builder_coarse.get_package().get_constraints(), builder_fine.get_package().get_constraints());

        const ManifoldTransfer<OperatorType> point_transfer(transfer, obj_coarse.get_M(), obj_fine.get_M());
        const MassProjectionTransport<OperatorType> vector_transport(transfer, obj_coarse.get_M(), obj_fine.get_M());

        MassOracle<2> T_fine(obj_fine, options_slv);
        MassOracle<2> T_coarse(obj_coarse, options_slv);

        CoarseOracleBase qk_base(T_fine, T_coarse, coarse_manifold, point_transfer, vector_transport);

        Vector<double> x_fine(obj_fine.n_dofs());
        ellipsoid::random_point(x_fine, obj_fine.get_M());
        builder_fine.distribute(x_fine);

        T_fine.update(x_fine);   // CoarseOracleBase::update_model assumes T_fine already matches x_fine
        qk_base.update_model(x_fine);
        const auto& state = qk_base.get_state();

        MassCoarseOracle<2> qk(qk_base, options_slv);

        std::cout << "GPE coarse oracle (mass metric)\n";

        // Proposition 3.2 (first-order coherence): q(psi_k) == f_H(psi_k), i.e. the
        // correction term vanishes exactly at the base point.
        {
            qk.update(state.y);
            check(std::abs(qk.value(state.y) - T_coarse.value(state.y)) < 1e-8,
                "value(psi_k) == f_H(psi_k) (correction term vanishes at the base point)");
        }

        // Proposition 3.2: grad q(psi_k) == R^{psi_k}_x(grad f_h(x)) == state.x_grad_restr.
        {
            qk.update(state.y);
            Vector<double> g(obj_coarse.n_dofs());
            qk.gradient(state.y, g);

            double diff = 0.0;
            for (unsigned int i = 0; i < g.size(); i++) {
                diff = std::max(diff, std::abs(g[i] - state.x_grad_restr[i]));
            }
            check(diff < 1e-6, "grad q(psi_k) == restricted fine gradient (first-order coherence)");
        }

        // <gradient(z), v>_M == directional_derivative(z, v) at a trial point z != psi_k,
        // with v tangent *at z* (gradient() is the M-Riemannian gradient; the defining
        // identity uses the mass inner product, not the plain Euclidean dot product).
        {
            Vector<double> z(state.y.size()), v(state.y.size());
            ellipsoid::random_point(z, obj_coarse.get_M());
            metric::mass::random_tangent_vector(z, obj_coarse.get_M(), v);

            qk.update(z);
            Vector<double> g(z.size());
            qk.gradient(z, g);
            const double g_v = qk.inner(g, v);
            const double dd  = qk.directional_derivative(z, v);

            check(std::abs(g_v - dd) < 1e-6 * std::max(1.0, std::abs(dd)),
                "<gradient(z), v>_M == directional_derivative(z, v)");
        }
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    if (n_failed > 0) {
        std::cerr << "\n" << n_failed << " check(s) failed.\n";
        return 1;
    }
    std::cout << "\nAll checks passed.\n";
    return 0;
}
