#ifndef GPE_LAC_H
#define GPE_LAC_H
#define SOLVER_MIN_TOL 1e-10

#include "option_types.h"

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>  // for general matrices
#include <deal.II/lac/solver_cg.h>     // for symmetric positive definite matrices
#include <deal.II/lac/solver_minres.h>

namespace gpe
{
using dealii::Vector;
using dealii::SparseMatrix;
using dealii::FullMatrix;
using dealii::SparsityPattern;
using dealii::DynamicSparsityPattern;
using dealii::Vector;
using dealii::Point;

struct LevelMatrix
{
    SparseMatrix<double> A0, M, Mpp;
    SparsityPattern sparsity_pattern;
    unsigned int level;

    void reinit(DynamicSparsityPattern&& sp)
    {
        sparsity_pattern.copy_from(sp);
        A0.reinit(sparsity_pattern);
        M.reinit(sparsity_pattern);
        Mpp.reinit(sparsity_pattern);
    }
};

// TODO: return SolverInfo (converged/did_not_converge/error)
template <typename PreconditionerType>
[[maybe_unused]] dealii::SolverControl
solve_sparse(const SparseMatrix<double>& system_matrix, const Vector<double>& system_rhs,
    Vector<double>& solution, const SolverMethod method = SolverMethod::GMRES,
    const PreconditionerType& preconditioner = dealii::PreconditionIdentity(),
    const unsigned max_iter = 1000, const double reltol = 1e-6)
{
    solution = 0.0;
    // TODO: use M-norm for tolerance?
    const double rhs_norm = system_rhs.l2_norm();
    // Avoid zero tolerance
    const double tol = std::max(reltol * rhs_norm, SOLVER_MIN_TOL);
    dealii::SolverControl solver_control(max_iter, tol);

    switch (method) {
    case SolverMethod::GMRES:
        {
            dealii::SolverGMRES solver(solver_control);
            solver.solve(system_matrix, solution, system_rhs, preconditioner);
        }
        break;
    case SolverMethod::MINRES:
        {
            dealii::SolverMinRes solver(solver_control);
            solver.solve(system_matrix, solution, system_rhs, preconditioner);
        }
        break;
    case SolverMethod::CG:
        {
            dealii::SolverCG solver(solver_control);
            solver.solve(system_matrix, solution, system_rhs, preconditioner);
        }
        break;
    default:
        throw std::invalid_argument("Unknown SolverMethod");
    }
    return solver_control;
}

} // namespace gpe

#endif //GPE_LAC_H