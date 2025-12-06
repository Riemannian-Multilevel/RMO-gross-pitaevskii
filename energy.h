#ifndef GPE_ENERGY_H
#define GPE_ENERGY_H

#include "lac.h"
#include <cmath>

namespace gpe
{

struct GdControl
{
    double mass;
    double lmb;
    double residual;
    double rg_norm;
};

template <typename Matrix>
void energy_residual(GdControl& control, const Vector<double>& x, const Vector<double>& g,
                     const Matrix& A, const Matrix& M)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    control.mass = x * Mx;      // should be ~ 1 (energy constraint)

    Vector<double> Ax(x.size());
    A.vmult(Ax, x);
    control.lmb = x * Ax;   // Rayleigh quotient (x'Ax / x'Mx)

    Vector<double> r(Ax);
    r.add(-control.lmb, Mx);        // r = A x - lambda M x
    //double res = r.l2_norm(); // or M-norm, see below

    Vector<double> Mr(r.size());
    M.vmult(Mr, r);
    control.residual = std::sqrt(r * Mr);

    Vector<double> Mg(g.size());
    M.vmult(Mg, g);
    control.rg_norm = std::sqrt(g * Mg);
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

struct GdOptions
{
    double tol_inner;     // relative tolerance for inner solver
    double tol_lambda;    // tolerance for rayleigh quotients
    double tol_residual;  // tolerance for M-residual
    double step_size;     // fixed step-size used in iteration steps
    int max_iter;         // maximum GD iterations
    int max_inner;        // maximum sparse solver iterations
};

template <typename Matrix>
bool energy_terminate_iteration(const Matrix& M, const Matrix& A,
                                const Vector<double>& x, const Vector<double>& g,
                                GdControl& ctrl, GdOptions options)
{
    // Compute criteria
    GdControl ctrl_prev(ctrl);
    energy_residual(ctrl, x, g, A, M);

    // Check criteria
    const double lmb_diff   = std::abs(ctrl.lmb - ctrl_prev.lmb);
    const double lmb_factor = 1.0 + std::abs(ctrl.lmb);  // avoid numerical issues near lmb ~ 0

    if (lmb_diff < options.tol_lambda * lmb_factor && ctrl.residual < options.tol_residual) {
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
//! @param constraints Object for applying FE constraints to the solution
//! @param beta Non-linearity factor for GPE
//! @param solver Used sparse solver (gmres|minres|cg)
//! @param update_mpp
//! @param options Termination criteria
//! @param check_every Number of iterations after which to check termination criteria
//! @return
template <int dim, typename Function>
Vector<double>
// TODO: SolverOptions for inner solve
gp_energy_rgd(const SparseMatrix<double>& A_0, const SparseMatrix<double>& M, SparseMatrix<double>& Mpp,
              Function&& update_mpp, const Vector<double>& x0,
              const dealii::AffineConstraints<double>& constraints,
              double beta, SolverMethod solver,
              const GdOptions& options, int check_every = 5)
{
    Assert(options.step_size > 0, dealii::ExcInternalError("Step size must be positive"));

    // Begin RGD iteration
    Vector x(x0);
    GdControl control{};
    dealii::PreconditionIdentity precondition{};

    // TODO: unnecessary copies g, z
    //       dealii::Function with value() and grad_value()
    for (int it = 0; it < options.max_iter; it++) {
        // A = A_0 + beta * M_phiphi
        SparseMatrix<double> A = sp_copy(A_0);

        // Apply constraints to incumbent solution
        constraints.distribute(x);

        // Update mass matrix
        update_mpp(Mpp, x);
        A.add(beta, Mpp);

        // Solve linear system (boundary constraints assumed applied to A_0, M, Mpp)
        Vector<double> y = solve_sparse(A, x, solver,
            precondition, options.max_inner, options.tol_inner);

        // z <- A^{-1}x / (x' A^{-1}x)
        Vector<double> z(y);
        double denom1 = x * y; // x' A^-1 x
        Assert(denom1 > 0, dealii::ExcInternalError("x' A^{-1} x <= 0"));
        z /= denom1;

        // g <- x - z
        Vector<double> g(x);
        g.add(-1.0, z);

        // x <- x - h g
        x.add(-options.step_size, g);

        // x <- x / ||x||_M
        Vector<double> Mx(x.size());
        M.vmult(Mx, x);
        x /= std::sqrt(x * Mx);

        // Check termination criteria every N steps
        if (it % check_every == 0 || it == options.max_iter - 1) {
            // Update control structure
            if (energy_terminate_iteration(M, A, x, g, control, options)) {
                break;
            }
            // Print newly computed values
            const double E = energy(x, A_0, Mpp);
            std::cerr << "Mass = " << control.mass << ", lambda = " << control.lmb
                      << ", residual = " << control.residual << ", energy = " << E
                      << std::endl;
        }
    }
    return x;
}

} // namespace gpe
#endif //GPE_ENERGY_H