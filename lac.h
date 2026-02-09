#ifndef GPE_LAC_H
#define GPE_LAC_H
#define SOLVER_MIN_TOL 1e-10

#include "option_types.h"
#include <vector>
#include <utility>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>  // for general matrices
#include <deal.II/lac/solver_cg.h>     // for symmetric positive definite matrices
#include <deal.II/lac/solver_minres.h>

namespace gpe
{
// TODO: use aliases for Trilinos, PETSc, ... matrices
using dealii::Vector;
using dealii::SparseMatrix;
using dealii::FullMatrix;
using dealii::SparsityPattern;
using dealii::DynamicSparsityPattern;
using dealii::Vector;
using dealii::Point;

using dealii::types::global_dof_index;
using dealii::SolverControl;

// Simple alternative to dealii::LinearOperator, which is also performant for small (n < 1000) matrices
template <typename MatrixType, typename VectorType>
class LinearCombinationMatrix
{
public:
    LinearCombinationMatrix() = default;

    // Helper to add a component: A <- A + weight * matrix
    void add_component(double weight, const MatrixType &matrix)
    {
        components.push_back({weight, &matrix});
    }

    // Clear components (e.g. for a new time step)
    void clear()
    {
        components.clear();
    }

    // Avoid allocations of tmp_vector in vmult()
    void reinit(unsigned int size)
    {
        tmp_vector.reinit(size);
    }
    void reinit(const VectorType& vector)
    {
        tmp_vector.reinit(vector);
    }

    // Return number of rows (required by solvers)
    global_dof_index m() const
    {
        Assert(!components.empty(), dealii::ExcMessage("No matrices added"));
        return components[0].second->m();
    }

    // Return number of columns (required by solvers)
    global_dof_index n() const
    {
        Assert(!components.empty(), dealii::ExcMessage("No matrices added"));
        return components[0].second->n();
    }

    // y = A * x = sum(c_i * M_i * x)
    void vmult(VectorType &dst, const VectorType &src) const
    {
        dst = 0.0;
        vmult_add(dst, src);
    }

    // y = y + A * x
    void vmult_add(VectorType &dst, const VectorType &src) const
    {
        // Loop over all stored matrices
        for (const auto &pair : components)
        {
            const double weight = pair.first;
            const auto* matrix  = pair.second;

            // Optimization: Skip zero weights
            //if (std::abs(weight) < 1e-15) continue;

            // Step 1: Compute M_i * x into temporary vector
            // tmp_vector avoids allocating memory every iteration
            matrix->vmult(tmp_vector, src);

            // Step 2: Add to destination: dst += weight * tmp
            dst.add(weight, tmp_vector);
        }
    }

private:
    // Store pairs of (coefficient, pointer to matrix)
    std::vector<std::pair<double, const MatrixType*>> components;

    // Mutable temporary vector for intermediate calculations
    // 'mutable' allows modification inside const vmult()
    mutable VectorType tmp_vector;
};


// Represents the operation x = M^-1 * b
// TODO: compare step-22 for relative tolerances
template <typename MatrixType, typename VectorType, typename PreconditionerType>
class InverseMatrix
{
public:
    InverseMatrix(const MatrixType& matrix_,
                  const SolverMethod method_,
                  const PreconditionerType& precond_ = dealii::PreconditionIdentity(),
                  const unsigned max_iter_ = 1000,
                  const double reltol_ = 1e-6)
        : matrix(matrix_)
        , precond(precond_)
        , method(method_)
        , max_iter(max_iter_)
        , reltol(reltol_)
        , ctrl(max_iter, SOLVER_MIN_TOL)
    {}

    // "Multiplying" by the inverse is "solving" the system
    void vmult(VectorType &dst, const VectorType &rhs) const
    {
        // Reset entries of destination vector
        dst = 0.0;
        // TODO: use M-norm for tolerance?
        const double rhs_norm = rhs.l2_norm();
        // Avoid zero tolerance
        const double tol = std::max(reltol * rhs_norm, SOLVER_MIN_TOL);
        ctrl = SolverControl(max_iter, tol);

        switch (method) {
            case SolverMethod::GMRES:
                {
                    dealii::SolverGMRES solver(ctrl);
                    solver.solve(matrix, dst, rhs, precond);
                }
                break;
            case SolverMethod::MINRES:
                {
                    dealii::SolverMinRes solver(ctrl);
                    solver.solve(matrix, dst, rhs, precond);
                }
                break;
            case SolverMethod::CG:
                {
                    dealii::SolverCG solver(ctrl);
                    solver.solve(matrix, dst, rhs, precond);
                }
                break;
            default:
                throw std::invalid_argument("Unknown SolverMethod");
        }
    }

    const SolverControl& control() const { return ctrl; }

private:
    const MatrixType &matrix;
    const PreconditionerType &precond;

    const SolverMethod method;
    unsigned max_iter;
    double reltol;
    // Note: use thread-local storage for this class
    mutable SolverControl ctrl;  // for const vmult()
};

} // namespace gpe

#endif //GPE_LAC_H