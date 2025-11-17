#ifndef GPE_LAC_H
#define GPE_LAC_H

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>  // for general matrices
#include <deal.II/lac/solver_cg.h>     // for symmetric positive definite matrices
#include <deal.II/lac/solver_minres.h>

namespace gpe
{
using dealii::Vector;
using dealii::SparseMatrix;

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

// TODO: preconditioners
inline Vector<double>
solve_sparse(const SparseMatrix<double>& system_matrix, const Vector<double>& system_rhs,
    const SolverMethod method = SolverMethod::GMRES,
    const unsigned n_iter = 1000, const double rel_tol = 1e-6)
{
    Vector<double> solution(system_rhs.size());
    dealii::SolverControl solver_control(n_iter, rel_tol * system_rhs.l2_norm());

    switch (method) {
    case SolverMethod::GMRES:
        {
            dealii::SolverGMRES solver(solver_control);
            solver.solve(system_matrix, solution, system_rhs, dealii::PreconditionIdentity());
        }
        break;
    case SolverMethod::MINRES:
        {
            dealii::SolverMinRes solver(solver_control);
            solver.solve(system_matrix, solution, system_rhs, dealii::PreconditionIdentity());
        }
    case SolverMethod::CG:
        {
            dealii::SolverCG solver(solver_control);
            solver.solve(system_matrix, solution, system_rhs, dealii::PreconditionIdentity());
        }
        break;
    default:
        throw std::invalid_argument("Unknown SolverMethod");
    }

    std::cout << solver_control.last_step()
              << " " + to_string(method) + " iterations needed to obtain convergence." << std::endl;
    return solution;
}

} // namespace gpe

#endif //GPE_LAC_H