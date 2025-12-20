#ifndef GPE_ENERGY_H
#define GPE_ENERGY_H

#include "lac.h"
#include <cmath>

namespace gpe
{

struct GdControl
{
    double mass;
    double lambda;
    double residual;
    double rg_norm;
};

// TODO: move this to options.h?
struct GdOptions
{
    double tol_inner;     // relative tolerance for inner solver
    double tol_lambda;    // tolerance for rayleigh quotients
    double tol_residual;  // tolerance for M-residual
    double step_size;     // fixed step-size used in iteration steps
    int max_iter;         // maximum GD iterations
    int max_inner;        // maximum sparse solver iterations
    SolverMethod solver;  // method for solving sparse linear equations
};

template <typename Matrix>
void energy_residual(GdControl& control, const Vector<double>& x,
                     const Matrix& A, const Matrix& M)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    control.mass = x * Mx;      // should be ~ 1 (energy constraint)

    Vector<double> Ax(x.size());
    A.vmult(Ax, x);
    control.lambda = x * Ax;   // Rayleigh quotient (x'Ax / x'Mx)

    Vector<double> r(Ax);
    r.add(-control.lambda, Mx);        // r = A x - lambda M x
    //double res = r.l2_norm(); // or M-norm, see below

    Vector<double> Mr(r.size());
    M.vmult(Mr, r);
    control.residual = std::sqrt(r * Mr);
}

template <typename Matrix>
[[maybe_unused]] double
energy(const Vector<double>& x, const Matrix& A_0, const Matrix& Mpp)
{
    Vector<double> Bx(x.size());
    A_0.vmult(Bx, x);
    Bx *= 0.5;

    Vector<double> Mpp_x(x.size());
    Mpp.vmult(Mpp_x, x);

    Bx.add(0.25, Mpp_x);
    return x * Bx;
}

inline Vector<double>
energy_gradient(const SparseMatrix<double>& A, const Vector<double>& x,
    const GdOptions& options, const dealii::AffineConstraints<double>& constraints,
    unsigned int& last_step)
{
    dealii::PreconditionIdentity precondition{};

    // y <- A^-1 x
    auto [y,y_iter] = solve_sparse(A, x, options.solver,
        precondition, options.max_inner, options.tol_inner);

    // Apply boundary condition
    constraints.distribute(y);

    // z <- A^{-1}x / (x' A^{-1}x)
    Vector<double> z(y);
    double denom1 = x * y; // x' A^-1 x
    Assert(denom1 > 0, dealii::ExcInternalError("x' A^{-1} x <= 0"));
    z /= denom1;

    // g <- x - z
    Vector<double> g(x);
    g.add(-1.0, z);

    last_step = y_iter;
    return g;
}

// Retraction by normalization
inline void
energy_retract(const SparseMatrix<double>& M, const Vector<double>& g, Vector<double>& x, double step_size)
{
    // x <- x - h g
    x.add(-step_size, g);

    // x <- x / ||x||_M
    Vector<double> Mx(x.size());  // TODO: unnecessary allocation
    M.vmult(Mx, x);
    x /= std::sqrt(x * Mx);
}

// TODO: gradient norm termination (when gradient is computed)
template <typename Matrix>
bool energy_terminate_iteration(const Matrix& M, const Matrix& A,
                                const Vector<double>& x, GdControl& ctrl,
                                double tol_lambda, double tol_residual)
{
    // Compute criteria
    const GdControl ctrl_prev(ctrl);
    energy_residual(ctrl, x, A, M);

    // Check criteria
    const double lmb_diff   = std::abs(ctrl.lambda - ctrl_prev.lambda);
    const double lmb_factor = 1.0 + std::abs(ctrl.lambda);  // avoid numerical issues near lmb ~ 0

    if (lmb_diff < tol_lambda * lmb_factor && ctrl.residual < tol_residual) {
        return true;
    }
    return false;
}

//! Riemannian gradient descent for the GPE energy minimization
//! @tparam dim Problem dimension
//! @param A_0 Sum of (potential) weighed mass matrix and stiffness matrix
//! @param M Mass matrix
//! @param Mpp Weighed mass matrix, updated in every iteration step
//! @param x0 Starting value
//! @param beta Non-linearity factor for GPE
//! @param constraints Object for applying FE constraints to the solution
//! @param update_mpp
//! @param options Termination criteria
//! @param check_every Number of iterations after which to check termination criteria
//! @return
template <int dim, typename Function>
Vector<double>
gp_energy_rgd(const SparseMatrix<double>& A_0, const SparseMatrix<double>& M, SparseMatrix<double>& Mpp,
              Function&& update_mpp, const Vector<double>& x0, double beta,
              const dealii::AffineConstraints<double>& constraints,
              const GdOptions& options, int check_every = 5, std::ostream& os = std::cerr)
{
    Assert(options.step_size > 0, dealii::ExcInternalError("Step size must be positive"));

    Vector x(x0);
    GdControl control{};
    unsigned int last_step = 0;  // number of iterations in inner solver taken
    int iter;

    // TODO: unnecessary copies g, z
    //       dealii::Function with value() and grad_value()
    os << "iter,lac_iter,mass,lambda,residual,energy" << std::endl;

    SparseMatrix<double> A;
    A.reinit(A_0.get_sparsity_pattern());

    // Begin RGD iteration
    for (iter = 0; iter < options.max_iter; iter++) {
        // Apply constraints to incumbent solution
        constraints.distribute(x);

        // A = A_0 + beta * M_xx
        // TODO: matrix copy in every iteration - assemble A in single call, or use matrix-free operator
        A.copy_from(A_0);
        update_mpp(Mpp, x);
        A.add(beta, Mpp);

        // Termination criteria
        if (iter % check_every == 0) {
            // Update control structure
            if (energy_terminate_iteration(M, A, x,  control, options.tol_lambda, options.tol_residual)) {
                break;
            }
            const double E = energy(x, A_0, Mpp);

            // Values for previous iteration (including starting step)
            os << iter << "," << last_step << "," << control.mass << "," << control.lambda << ","
               << control.residual << "," << E << std::endl;
        }
        // Riemannian gradient: g <- x - A^{-1}x / (x' A^{-1}x)
        auto g = energy_gradient(A, x, options, constraints, last_step);

        // TODO
        // Nash: g(x) <- g(x) + v
        // v <- interpolate(g_fine(x_fine)) - g(interpolate(x_fine))
        // Nash objective: f + < v, x - Ry> - not required for fixed step size GD

        // Retraction: x <- (x - h g) / ||x - h g||_M
        energy_retract(M, g, x, options.step_size);
    }

    // Print values for last iteration
    if (iter == options.max_iter - 1) {
        constraints.distribute(x);
        const double E = energy(x, A_0, Mpp);

        os << iter << "," << last_step << "," << control.mass << "," << control.lambda << ","
           << control.residual << "," << E << std::endl;
    }
    //constraints.distribute(x);
    return x;
}

} // namespace gpe
#endif //GPE_ENERGY_H