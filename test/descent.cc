//
// Checks that the energy-adaptive gradient g of the F-metric coarse model is a descent direction,
// D Psi(x)[-g] < 0, at a random point x near the base point phi.
//
#include "check.h"

#include <rmo/gpe/kernels.h>
#include <rmo/gpe/model.h>
#include <rmo/util/random.h>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;

namespace
{

//! Parameters of a coarse model and a point x on the sphere where it is evaluated.
struct CoarseModelSample
{
    Vector<double> phi;  ///< base point
    Vector<double> w;    ///< correction, F-tangent at phi
    Vector<double> x;    ///< R_phi(v) for a random unit tangent vector v at phi
};

template <typename MatrixType>
CoarseModelSample random_sample(const MatrixType& M, unsigned n_dofs)
{
    CoarseModelSample s{Vector<double>(n_dofs), Vector<double>(n_dofs), Vector<double>(n_dofs)};
    ellipsoid::random_point(s.phi, M);

    Vector<double> w(n_dofs);
    normrnd(0.0, 1.0, w);
    metric::frobenius::project_onto_tangent_space(s.phi, M, w, s.w);

    Vector<double> v(n_dofs);
    metric::frobenius::random_tangent_vector(s.phi, M, v);
    v /= v.l2_norm();
    s.x = s.phi;
    ellipsoid::retract_by_norm(M, v, s.x);  // x <- (phi + v) / |phi + v|_M
    return s;
}

template <int dim>
void check_energy_adaptive_descent(CheckReport& report, const GrossPitaevskiiSystem<dim>& system, double beta,
                                   const SolverOptions& options)
{
    const auto A = system.get_operator_A(beta);
    const auto M = system.get_operator_M();
    const InverseOpType A_inv(A, options);
    const auto [phi, w, x] = random_sample(M, system.n_dofs());

    Vector<double> g(x.size());
    kernels::coarse_frobenius_grad_energy_adaptive(M, A_inv, A, x, phi, w, g);
    g *= -1.0;

    const double slope = kernels::coarse_frobenius_dir_deriv(x, phi, w, g, M, A);
    report.check(slope < 0.0, "energy-adaptive gradient of the F-metric coarse model is a descent direction",
                 "D Psi(x)[-g] = " + sci(slope));
}

} // namespace


int main()
{
    // The condition holds at every point, so one small mesh suffices
    GPE_Options options{};
    options.dimension = 2;
    options.degree    = 1;
    options.radius    = 10;
    options.beta      = 100;
    options.order     = Ordering::CUTHILL_MCKEE;
    options.bc        = BoundaryCondition::DIRICHLET;
    options.mesh_kind = MeshKind::QUADRILATERAL;
    constexpr unsigned n_levels = 6;

    SolverOptions options_slv{};
    options_slv.solver    = SolverMethod::CG;
    options_slv.precond   = Precondition::NONE;
    options_slv.max_inner = 2000;
    options_slv.tol_inner = 1e-12;

    return run_tests([&](CheckReport& report) {
        const ModelBuilder<GrossPitaevskiiSystem<2>> builder(potential::Square<2>(), options, n_levels);
        check_energy_adaptive_descent(report, builder.get_system(), options.beta, options_slv);
    });
}
