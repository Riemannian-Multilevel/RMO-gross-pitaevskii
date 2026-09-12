/**
 * @file
 * @brief First-order coherence of the coarse model of the Gross-Pitaevskii problem, on two mesh levels.
 *
 * With the restricted point \f$ y = r(x) \f$ as base point, the coarse model \f$ \Psi \f$ (MassCoarseOracle) satisfies
 * \f$ \Psi(y) = f_H(y) \f$ and \f$ \grad \Psi(y) = \mathcal{R} \grad f_h(x) \f$: the correction term does not change
 * the value, and corrects the gradient to the restricted fine gradient. The test also checks
 * \f$ \langle \grad \Psi(z), v \rangle_M = \mathrm{D}\Psi(z)[v] \f$ at a random point \f$ z \f$.
 *
 * Unlike staging/test/gradient.cc, which uses a single mesh and a synthetic correction, this test runs the whole transfer
 * chain: LinearTransferMG, ManifoldTransfer and MassProjectionTransport.
 */
#include "check.h"

#include <rmo/fe/interpolate.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/model.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/oracle_coarse.h>
#include <rmo/gpe/transport.h>

#include <cmath>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;

int main()
{
    using System = GrossPitaevskiiSystem<2>;

    GPE_Options options{};
    options.dimension = 2;
    options.degree    = 1;
    options.radius    = 10;
    options.beta      = 50;
    options.order     = Ordering::DEFAULT;
    options.bc        = BoundaryCondition::NEUMANN;
    options.mesh_kind = MeshKind::QUADRILATERAL;

    SolverOptions options_slv{};
    options_slv.solver        = SolverMethod::CG;
    options_slv.precond       = Precondition::NONE;
    options_slv.max_inner     = 2000;
    options_slv.tol_inner     = 1e-10;
    options_slv.tol_inner_res = 1e-2;

    return run_tests([&](CheckReport& report) {
        ModelBuilder<System> builder_coarse(potential::Square<2>(), options, 2);
        ModelBuilder<System> builder_fine(potential::Square<2>(), options, 3);
        GrossPitaevskiiFunctional<System> func_coarse(builder_coarse.get_system(), options.beta, options_slv);
        GrossPitaevskiiFunctional<System> func_fine(builder_fine.get_system(), options.beta, options_slv);

        const UnitMassSphere<OperatorType> coarse_manifold(func_coarse.get_M());
        const fe::LinearTransferMG<2> transfer(builder_coarse.get_dofs(), builder_fine.get_dofs(),
                                               builder_coarse.get_system().get_constraints(),
                                               builder_fine.get_system().get_constraints());
        const ManifoldTransfer<OperatorType> point_transfer(transfer, func_coarse.get_M(), func_fine.get_M());
        const MassProjectionTransport<OperatorType> vector_transport(transfer, func_coarse.get_M(), func_fine.get_M());

        MassOracle<System> T_fine(func_fine, options_slv);
        MassOracle<System> T_coarse(func_coarse, options_slv);
        CoarseOracleBase model(T_fine, T_coarse, coarse_manifold, point_transfer, vector_transport);

        Vector<double> x(func_fine.n_dofs());
        ellipsoid::random_point(x, func_fine.get_M());
        builder_fine.distribute(x);
        T_fine.update(x);  // update_model() expects T_fine at x
        model.update_model(x);
        const auto& state = model.get_state();

        MassCoarseOracle<System> coarse(model, options_slv);
        coarse.update(state.y);
        const double value_diff = std::abs(coarse.value(state.y) - T_coarse.value(state.y));
        report.check(value_diff < 1e-8, "Psi(y) = f_H(y)", "diff " + sci(value_diff));

        Vector<double> g(func_coarse.n_dofs());
        coarse.gradient(state.y, g);
        g -= state.x_grad_restr;
        report.check(g.linfty_norm() < 1e-6, "grad Psi(y) = R grad f_h(x)", "max diff " + sci(g.linfty_norm()));

        Vector<double> z(state.y.size()), v(state.y.size()), g_z(state.y.size());
        ellipsoid::random_point(z, func_coarse.get_M());
        metric::mass::random_tangent_vector(z, func_coarse.get_M(), v);
        coarse.update(z);
        coarse.gradient(z, g_z);
        const double g_v = coarse.metric().inner(g_z, v);
        const double dd  = coarse.directional_derivative(z, v);
        report.check(std::abs(g_v - dd) < 1e-6 * std::max(1.0, std::abs(dd)), "<grad Psi(z), v>_M = D Psi(z)[v]",
                     "diff " + sci(std::abs(g_v - dd)));
    });
}
