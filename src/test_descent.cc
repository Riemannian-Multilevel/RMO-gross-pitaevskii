//
// Created by Ferdinand Vanmaele on 04.04.26.
//

#include "test_gradient.h"

using namespace gpe;

template <int dim>
void check_adaptive_descent_condition(const GrossPitaevskiiSystem<dim>& problem,
                                      double beta, SolverOptions options_slv)
{
    const unsigned n_dofs = problem.n_dofs();
    auto A = problem.get_operator_A(beta);
    auto M = problem.get_operator_M();
    PreconditionInverse<decltype(A), decltype(M)> A_inv(A, options_slv);

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
    return 0;
}
