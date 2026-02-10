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

/**
 * @brief A lightweight alternative to dealii::LinearOperator.
 *
 * This class represents a linear combination of matrices, effectively computing
 * \f$ A = \sum c_i M_i \f$. It is optimized for cases where the matrices are
 * relatively small (n < 1000) or when avoiding the full overhead of
 * dealii::LinearOperator is desired.
 *
 * @tparam MatrixType The type of the matrix (e.g., SparseMatrix or FullMatrix).
 * @tparam VectorType The type of the vector (e.g., Vector or BlockVector).
 */
template <typename MatrixType, typename VectorType>
class LinearCombination
{
public:
    /** @brief Default constructor. */
    LinearCombination() = default;

    /**
     * @brief Adds a matrix component to the linear combination.
     * Adds a contribution such that the operator becomes \f$ A \leftarrow A + w \cdot M \f$.
     *
     * @param weight The scalar coefficient for this matrix.
     * @param matrix A reference to the matrix to be added.
     */
    void add_component(double weight, const MatrixType &matrix)
    {
        m_components.push_back({weight, &matrix});
    }

    /** @brief Clears all stored matrix components. */
    void clear()
    {
        m_components.clear();
    }

    /**
     * @brief Reinitializes the internal temporary vector.
     * This should be called to ensure the internal scratch space matches
     * the size of the system, avoiding reallocations during @ref vmult.
     *
     * @param size The number of degrees of freedom.
     */
    void reinit(unsigned int size)
    {
        m_vector.reinit(size);
    }

    /**
     * @brief Reinitializes the internal temporary vector based on an existing vector.
     * @param vector A vector used as a template for size and structure.
     */
    void reinit(const VectorType& vector)
    {
        m_vector.reinit(vector);
    }

    /**
     * @brief Returns the number of rows in the operator.
     * @return global_dof_index Number of rows.
     */
    global_dof_index m() const
    {
        Assert(!m_components.empty(), dealii::ExcMessage("No matrices added"));
        return m_components[0].second->m();
    }

    /**
     * @brief Returns the number of columns in the operator.
     * @return global_dof_index Number of columns.
     */
    global_dof_index n() const
    {
        Assert(!m_components.empty(), dealii::ExcMessage("No matrices added"));
        return m_components[0].second->n();
    }

    /**
     * @brief Matrix-vector multiplication: \f$ y = Ax \f$.
     * @param dst The destination vector.
     * @param src The source vector.
     */
    void vmult(VectorType &dst, const VectorType &src) const
    {
        dst = 0.0;
        vmult_add(dst, src);
    }

    /**
     * @brief Matrix-vector addition: \f$ y = y + Ax \f$.
     * Loops over all stored components and adds their contributions to @p dst.
     * @param dst The destination vector to which the result is added.
     * @param src The source vector.
     */
    void vmult_add(VectorType &dst, const VectorType &src) const
    {
        for (const auto &pair : m_components)
        {
            const double weight = pair.first;
            const auto* matrix  = pair.second;

            matrix->vmult(m_vector, src);
            dst.add(weight, m_vector);
        }
    }

private:
    /** @brief Collection of weights and matrix pointers. */
    std::vector<std::pair<double, const MatrixType*>> m_components;

    /**
     * @brief Temporary scratch vector to prevent frequent allocations.
     * Marked mutable to allow use within const @ref vmult methods.
     */
    mutable VectorType m_vector;
};


/**
 * @brief Represents the inverse operation of a matrix using an iterative solver.
 * This class acts as a wrapper that transforms a system solve into a
 * matrix-vector multiplication interface. Calling @ref vmult(dst, rhs) effectively
 * computes \f$ dst = M^{-1} \cdot rhs \f$ by solving the system \f$ M \cdot dst = rhs \f$.
 *
 * @tparam MatrixType The type of the matrix.
 * @tparam VectorType The type of the vector.
 * @tparam PreconditionerType The type of the preconditioner.
 */
template <typename MatrixType, typename VectorType, typename PreconditionerType>
class InverseMatrix
{
public:
    /**
     * @brief Constructor.
     * @param matrix The matrix to be inverted (solved).
     * @param method The iterative solver method to use (CG, GMRES, etc.).
     * @param precond The preconditioner to apply.
     * @param max_iter Maximum number of iterations allowed.
     * @param reltol Relative tolerance for the solver.
     */
    InverseMatrix(const MatrixType& matrix,
                  const SolverMethod method,
                  const PreconditionerType& precond = dealii::PreconditionIdentity(),
                  const unsigned max_iter = 1000,
                  const double reltol = 1e-6)
        : m_matrix(matrix)
        , m_precond(precond)
        , m_method(method)
        , m_max_iter(max_iter)
        , m_reltol(reltol)
        , m_control(m_max_iter, SOLVER_MIN_TOL)
    {}

    /**
     * @brief Performs the system solve: \f$ dst = M^{-1} \cdot rhs \f$.
     * This method initializes the solver and control parameters based on the
     * norm of the @p rhs vector and the specified relative tolerance.
     *
     * @param dst The solution vector.
     * @param rhs The right-hand side vector.
     * @throws std::invalid_argument If an unsupported SolverMethod is provided.
     */
    void vmult(VectorType &dst, const VectorType &rhs) const
    {
        dst = 0.0;
        const double rhs_norm = rhs.l2_norm();
        const double tol = std::max(m_reltol * rhs_norm, SOLVER_MIN_TOL);
        m_control = SolverControl(m_max_iter, tol);

        switch (m_method) {
            case SolverMethod::GMRES:
                {
                    dealii::SolverGMRES<VectorType> solver(m_control);
                    solver.solve(m_matrix, dst, rhs, m_precond);
                }
                break;
            case SolverMethod::MINRES:
                {
                    dealii::SolverMinRes<VectorType> solver(m_control);
                    solver.solve(m_matrix, dst, rhs, m_precond);
                }
                break;
            case SolverMethod::CG:
                {
                    dealii::SolverCG<VectorType> solver(m_control);
                    solver.solve(m_matrix, dst, rhs, m_precond);
                }
                break;
            default:
                throw std::invalid_argument("Unknown SolverMethod");
        }
    }

    /** @brief Returns the solver control object used in the last solve. */
    const SolverControl& control() const { return m_control; }

private:
    const MatrixType &m_matrix;
    const PreconditionerType &m_precond;

    const SolverMethod m_method;
    unsigned int m_max_iter;
    double m_reltol;

    /**
     * @brief Solver control object.
     * Marked mutable to allow updating convergence state during @ref vmult.
     */
    mutable SolverControl m_control;
};

} // namespace gpe

#endif //GPE_LAC_H