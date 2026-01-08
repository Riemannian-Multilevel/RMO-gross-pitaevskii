#ifndef GPE_LAC_H
#define GPE_LAC_H

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
using dealii::Vector;
using dealii::Point;

// Matrix used for boundary residuals in geometric multigrid
struct InterfaceMatrix
{
    SparsityPattern sp;
    SparseMatrix<double> IM;

    void reinit(const SparsityPattern& sparsity)
    {
        sp.copy_from(sparsity);
        IM.reinit(sp);
    }
};

enum class SolverMethod
{
    GMRES,
    MINRES,
    CG
};

inline std::string
to_string(SolverMethod method)
{
    switch (method) {
    case SolverMethod::GMRES:
        return "GMRES";
    case SolverMethod::MINRES:
        return "MINRES";
    case SolverMethod::CG:
        return "CG";
    default:
        throw std::invalid_argument("Unknown SolverMethod");
    }
}

// TODO: return SolverInfo (converged/did_not_converge/error)
template <typename PreconditionerType>
[[maybe_unused]] unsigned int
solve_sparse(const SparseMatrix<double>& system_matrix, const Vector<double>& system_rhs,
    Vector<double>& solution, const SolverMethod method = SolverMethod::GMRES,
    const PreconditionerType& preconditioner = dealii::PreconditionIdentity(),
    const unsigned max_iter = 1000, const double reltol = 1e-6)
{
    solution = 0.0;
    // TODO: use M-norm for tolerance?
    dealii::SolverControl solver_control(max_iter, reltol * system_rhs.l2_norm());

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
    return solver_control.last_step();
}

} // namespace gpe

#endif //GPE_LAC_H