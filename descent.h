#ifndef GPE_ENERGY_H
#define GPE_ENERGY_H

#include "lac.h"
#include "option_types.h"

#include <cmath>
#include <deal.II/base/convergence_table.h>

namespace gpe
{
using dealii::ConvergenceTable::RateMode::reduction_rate;
using dealii::ConvergenceTable::RateMode::reduction_rate_log2;

struct GdControl
{
    double mass;
    double lambda;
    double residual;
    double rg_norm;
};

// TODO: avoid allocating vectors in every function/call
// (struct for temporary vectors passed by reference)
namespace energy
{

template <typename Matrix>
void residual(GdControl& control, const Vector<double>& x,
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
function_value(const Vector<double>& x, const Matrix& A_0, const Matrix& Mpp)
{
    Vector<double> Bx(x.size());
    A_0.vmult(Bx, x);
    Bx *= 0.5;

    Vector<double> Mpp_x(x.size());
    Mpp.vmult(Mpp_x, x);

    Bx.add(0.25, Mpp_x);
    return x * Bx;
}

// case x != v
inline Vector<double>
project_onto_tangent_space(const Vector<double>& Ainv_Mx, const Vector<double>& x,
    const SparseMatrix<double>& M, const Vector<double>& v)
{
    AssertDimension(x.size(), v.size());
    AssertDimension(x.size(), Ainv_Mx.size());

    Vector<double> Proj_v(v);
    Vector<double> My(x.size());
    M.vmult(My, Ainv_Mx);  // M A_x^{-1} M x

    Vector<double> Mv(v.size());
    M.vmult(Mv, v);

    double denom = x * My;
    AssertThrow(denom > 0, dealii::ExcInternalError("x' M A^{-1} M x <= 0"));

    Proj_v.add(-(x*Mv)/denom, Ainv_Mx);
    return Proj_v;
}

// case x == v
inline Vector<double>
project_onto_tangent_space(const Vector<double>& Ainv_Mx, const Vector<double>& x,
    const SparseMatrix<double>& M)
{
    AssertDimension(x.size(), Ainv_Mx.size());

    Vector<double> Proj_v(x);
    Vector<double> My(x.size());
    M.vmult(My, Ainv_Mx);  // M A_x^{-1} M x

    double denom = x * My;
    AssertThrow(denom > 0, dealii::ExcInternalError("x' M A^{-1} M x <= 0"));

    Proj_v.add(-1.0/denom, Ainv_Mx);
    return Proj_v;
}

// Riemannian gradient in S^{n-1} with energy metric
inline Vector<double>
gradient(const SparseMatrix<double>& A, const SparseMatrix<double>& M, const Vector<double>& x,
    const GdOptions& options, const dealii::AffineConstraints<double>& constraints, unsigned int& last_step)
{
    // TODO: use simple preconditioner for SPD matrices (e.g Jacobi)
    dealii::PreconditionIdentity precondition{};

    // y <- A^{-1} Mx
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    Vector<double> y(x.size());
    auto solve_control = solve_sparse(A, Mx, y, options.solver,
        precondition, options.max_inner, options.tol_inner);
    last_step = solve_control.last_step();

    // Apply boundary condition
    constraints.distribute(y);

    return project_onto_tangent_space(y, x, M); // \Pi_x(x): R^n -> T_x S^{n-1}
}

// Retraction by normalization
inline void
retract_by_norm(const SparseMatrix<double>& M, const Vector<double>& g, Vector<double>& x, double step_size)
{
    // x <- x - h g
    x.add(-step_size, g);

    // x <- x / ||x||_M
    Vector<double> Mx(x.size());  // TODO: unnecessary allocation
    M.vmult(Mx, x);
    x /= std::sqrt(x * Mx);
}

} // namespace energy

// TODO: gradient norm termination (when gradient is computed)
inline bool
terminate_iteration(const GdControl& ctrl, const GdControl& ctrl_prev,
    const double tol_lambda, const double tol_residual)
{
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
//! @param os Output stream for diagnostics
//! @return
template <int dim, typename Function>
Vector<double>
gp_energy_rgd(const SparseMatrix<double>& A_0, const SparseMatrix<double>& M, SparseMatrix<double>& Mpp,
              Function&& update_mpp, const Vector<double>& x0, double beta,
              const dealii::AffineConstraints<double>& constraints,
              const GdOptions& options, std::ostream& os)
{
    Assert(options.step_size > 0, dealii::ExcInternalError("Step size must be positive"));

    Vector x(x0);
    GdControl ctrl{}, ctrl_prev{};
    dealii::ConvergenceTable convergence_table;
    SparseMatrix<double> A;
    A.reinit(A_0.get_sparsity_pattern());

    bool break_on_next = false;
    unsigned int iter;
    unsigned int lac_iter = 0;  // number of iterations in inner solver taken
    // TODO: turn debug printing into logger/verbosity flag in options
    std::cerr << "Iteration: ";

    // Begin RGD iteration
    // TODO: unnecessary copies g, z
    for (iter = 0; iter < options.max_iter; iter++) {
        // Apply constraints to incumbent solution
        // TODO: merge to update_mpp (or separate object)
        constraints.distribute(x);

        // A = A_0 + beta * M_xx
        // TODO: matrix copy in every iteration - assemble A in single call, or use matrix-free operator
        A.copy_from(A_0);
        update_mpp(Mpp, x);
        A.add(beta, Mpp);

        // Termination criteria
        // TODO: check_every, ConvergenceTable == true -> check_every = 1
        std::cerr << iter << "..";
        ctrl_prev = ctrl;
        energy::residual(ctrl, x, A, M);

        // Values for previous iteration (including starting step)
        convergence_table.add_value("iter", iter);
        convergence_table.add_value("lac_iter", lac_iter);
        convergence_table.add_value("mass", ctrl.mass);
        convergence_table.add_value("lambda", ctrl.lambda);
        convergence_table.add_value("residual", ctrl.residual);
        convergence_table.add_value("energy", energy::function_value(x, A_0, Mpp));

        if (break_on_next) {
            break;
        }
        if (terminate_iteration(ctrl, ctrl_prev, options.tol_lambda,  options.tol_residual)) {
            // trick so that convergence_table is updated for last step
            // n iterations + starting solution -> n+1 table entries
            break_on_next = true;
            continue;
        }
        // Riemannian gradient: g <- x - A^{-1}x / (x' A^{-1}x)
        // TODO: variable function with gradient() / value() / update() methods
        auto g = energy::gradient(A, M, x, options, constraints, lac_iter);

        // Retraction: x <- (x - h g) / ||x - h g||_M
        energy::retract_by_norm(M, g, x, options.step_size);
    }
    std::cerr << std::endl << std::endl;
    convergence_table.set_precision("mass", 4);
    convergence_table.set_precision("lambda", 4);
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 4);

    convergence_table.set_scientific("lambda", true);
    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("energy", true);

    convergence_table.evaluate_convergence_rates("residual", reduction_rate);
    convergence_table.evaluate_convergence_rates("residual", reduction_rate_log2);
    convergence_table.write_text(os, dealii::TableHandler::TextOutputFormat::table_with_headers);

    //constraints.distribute(x);
    return x;
}

} // namespace gpe
#endif //GPE_ENERGY_H