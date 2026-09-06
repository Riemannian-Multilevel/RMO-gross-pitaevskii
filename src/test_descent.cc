//
// Created by Ferdinand Vanmaele on 04.04.26.
//

#include <rmo/gpe/gpe.h>
#include <rmo/gpe/model.h>
#include <rmo/ropt/manifold.h>
#include <rmo/util/random.h>

#include <iostream>
#include <stdexcept>

using namespace rmo;
using namespace rmo::gpe;

template <int dim>
void check_adaptive_descent_condition(const GrossPitaevskiiSystem<dim>& problem,
                                      double beta, SolverOptions options_slv)
{
    const unsigned n_dofs = problem.n_dofs();
    auto A = problem.get_operator_A(beta);
    auto M = problem.get_operator_M();
    InverseOpType A_inv(A, options_slv);

    // 1. Generate random base point and tilt vector
    Vector<double> phi(n_dofs), w(n_dofs);
    ellipsoid::random_point(phi, M);
    normrnd(0.0, 1.0, w);
    Vector<double> w_proj(n_dofs);
    ellipsoid::frobenius::project_onto_tangent_space(phi, M, w, w_proj);

    // 2. Generate random evaluation point safely near phi
    Vector<double> x(n_dofs), v(n_dofs);
    ellipsoid::frobenius::random_tangent_vector(phi, M, v);
    v /= v.l2_norm();
    ellipsoid::retract_by_norm(M, phi, v, x); // x is now on the manifold

    // 3. Compute the adaptive gradient
    Vector<double> g_adapt(n_dofs);
    coarse::frobenius::energy_adaptive_gradient(M, A_inv, A, x, phi, w_proj, g_adapt);

    // 4. Verify it is a valid descent direction: Df(x)[-g_adapt] < 0
    Vector<double> neg_g_adapt(g_adapt);
    neg_g_adapt *= -1.0;

    double slope = coarse::frobenius::directional_derivative(x, phi, w_proj, neg_g_adapt, M, A);

    std::cerr << "Adaptive Gradient Slope: " << slope << "\n";
    if (slope >= 0.0) {
        throw std::runtime_error("FAIL: Energy-adaptive gradient is not a descent direction!");
    } else {
        std::cerr << "PASS: Energy-adaptive gradient points downhill.\n";
    }
}

int main()
{
    // Discretization: the condition is metric-level, so one small level suffices
    GPE_Options options{};
    options.dimension = 2;
    options.degree    = 1;
    options.radius    = 10;
    options.beta      = 100;
    options.order     = Ordering::CUTHILL_MCKEE;
    options.bc        = BoundaryCondition::DIRICHLET;
    options.mesh_kind = MeshKind::QUADRILATERAL;

    SolverOptions options_slv{};
    options_slv.solver    = SolverMethod::CG;
    options_slv.precond   = Precondition::NONE;
    options_slv.max_inner = 2000;
    options_slv.tol_inner = 1e-12;

    constexpr unsigned n_levels = 6;

    try {
        ModelBuilder<2> builder(potential::Square<2>(), options, n_levels);
        check_adaptive_descent_condition(builder.get_system(), options.beta, options_slv);
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
