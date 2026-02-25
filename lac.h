#ifndef GPE_LAC_H
#define GPE_LAC_H
#define SOLVER_MIN_TOL 1e-10

#include "option_types.h"
#include <vector>
#include <utility>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>

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
            const auto* matrix = pair.second;

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
 * @brief A wrapper class that represents the inverse of a linear operator.
 *
 * This class encapsulates a `deal.II` iterative Krylov solver (such as CG, MINRES,
 * or GMRES) and a preconditioner. It acts mathematically as the inverse operator
 * $A^{-1}$, meaning that calling `vmult(dst, src)` executes the iterative solver
 * to find `dst` such that $A \cdot dst = src$.
 *
 * ### Architectural Role
 * By wrapping the solver in this interface, it can be passed into other algorithms
 * (like Riemannian Gradient Descent) exactly as if it were a standard matrix.
 * It strictly separates the action of the operator from the construction of the
 * preconditioner.
 *
 * @tparam OperatorType The forward linear operator $A$. This is typically a
 * matrix-free abstraction (e.g., @ref LinearCombination) that computes matrix-vector
 * products on the fly. The only strict requirement is that it provides a
 * `vmult(VectorType&, const VectorType&)` method.
 *
 * @tparam VectorType The vector space for the domain and range
 * (e.g., `dealii::Vector<double>`).
 *
 * @tparam PreconditionerType The preconditioner applied to accelerate the Krylov
 * solver (e.g., `dealii::SparseILU`, `dealii::PreconditionJacobi`). Note that while
 * this preconditioner is often constructed from an explicitly assembled `MatrixType`
 * before being passed to this class, the `InverseMatrix` only requires it to be
 * invocable by the solver.
 */
template <typename OperatorType, typename VectorType, typename PreconditionerType>
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
    InverseMatrix(const OperatorType& matrix,
                  const SolverMethod method,
                  const PreconditionerType& precond,
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

        // TODO: continue on a partial solve with diagnostic, instead of throwing an exception
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

        // catch (dealii::SolverControl::NoConvergence &e) {
        //     // Log the warning but allow the descent algorithm to continue.
        //     // A partial solve might still provide a valid descent direction!
        //     std::cerr << "\n[Warning] Krylov solver did not converge!"
        //               << "\n  Method:    " << static_cast<int>(method)
        //               << "\n  Steps:     " << e.last_step
        //               << "\n  Residual:  " << e.last_residual
        //               << "\n  Tolerance: " << solver_control.tolerance()
        //               << std::endl;
        // }
    }

    /** @brief Returns the solver control object used in the last solve. */
    const SolverControl& control() const { return m_control; }

private:
    const OperatorType &m_matrix;
    const PreconditionerType &m_precond;

    const SolverMethod m_method;
    unsigned int m_max_iter;
    double m_reltol;

    // Must be mutable because the deal.II solve() routine updates its
    // internal state (iteration count, achieved residual) during a logically const vmult
    mutable SolverControl m_control;
};


/**
 * @brief A generic, type-erasing manager for linear solvers and preconditioners.
 *
 * This class isolates the deal.II Krylov solver template dispatching from the
 * physical problem definition. It handles both static preconditioners (built once)
 * and dynamic preconditioners (rebuilt when the non-linear density changes).
 *
 * ### Template Parameter Contracts
 * @tparam OperatorType Represents the linear operator $A$. This does **not** need
 * to be an explicitly assembled matrix. It can be any matrix-free class (such as
 * @ref LinearCombination) as long as it provides a `vmult(dst, src)` method for
 * the Krylov solver to compute matrix-vector products.
 *
 * @tparam MatrixType Represents an explicitly assembled sparse matrix
 * (e.g., `dealii::SparseMatrix<double>`). This strict requirement exists because
 * preconditioners like ILU, Jacobi, and SSOR must directly access explicit matrix
 * elements (like the diagonal or triangular components) during setup.
 *
 * @tparam VectorType The vector type used for the domain and range of the operator
 * (e.g., `dealii::Vector<double>`).
 */
// TODO pass on additional data to preconditioner, instead of fixing parameters
template <typename OperatorType, typename MatrixType, typename VectorType>
class PreconditionInverse
{
public:
    /**
     * @brief Constructs the generic preconditioned solver.
     * @param solver_method_ The Krylov method to use (e.g., CG, MINRES, GMRES).
     * @param max_iter_ Maximum number of inner iterations for the linear solver.
     * @param tol_ Target tolerance for the linear solver residual.
     * @param precond_type_ The preconditioning strategy.
     */
    PreconditionInverse(SolverMethod solver_method_,
                         unsigned int max_iter_,
                         double       tol_,
                         Precondition precond_type_)
        : solver_method(solver_method_)
        , max_iter(max_iter_)
        , tol(tol_)
        , precond_type(precond_type_)
    {}

    /**
     * @brief Builds preconditioners that only need to be set up once.
     * e.g., ILU or AMG on the stationary matrix A0.
     */
    void setup_static(const MatrixType& static_matrix)
    {
        if (precond_type == Precondition::SPARSE_ILU) {
            ilu_precond.initialize(static_matrix);
        }
        if (precond_type == Precondition::AMG) {
            throw std::invalid_argument("Precondition::AMG not implemented");
        }
    }

    /**
     * @brief Rebuilds dynamic preconditioners using the latest assembled matrix.
     */
    void update_dynamic(const MatrixType& dynamic_matrix)
    {
        if (precond_type == Precondition::JACOBI) {
            typename dealii::PreconditionJacobi<MatrixType>::AdditionalData data(0.6);
            jacobi_precond.initialize(dynamic_matrix, data);
        }
        else if (precond_type == Precondition::SSOR) {
            typename dealii::PreconditionSSOR<MatrixType>::AdditionalData data(1.2);
            ssor_precond.initialize(dynamic_matrix, data);
        }
    }

    /**
     * @brief Solves op * dst = src.
     */
    unsigned solve(const OperatorType& op, const VectorType& src, VectorType& dst) const
    {
        switch (precond_type) {
            case Precondition::SPARSE_ILU:
                return solve_with(op, dst, src, ilu_precond);
            case Precondition::JACOBI:
                return solve_with(op, dst, src, jacobi_precond);
            case Precondition::SSOR:
                return solve_with(op, dst, src, ssor_precond);
            case Precondition::NONE:
            default:
                return solve_with(op, dst, src, dealii::PreconditionIdentity());
        }
    }

private:
    template <typename PrecondType>
    unsigned solve_with(const OperatorType& op, VectorType& dst, const VectorType& src,
                        const PrecondType& precond) const
    {
        // InverseMatrix now takes the decoupled parameters perfectly
        const InverseMatrix<OperatorType, VectorType, PrecondType>
        inv(op, solver_method, precond, max_iter, tol);

        inv.vmult(dst, src);
        return inv.control().last_step();
    }

    SolverMethod solver_method;
    unsigned int max_iter;
    double       tol;
    Precondition precond_type;

    dealii::SparseILU<double> ilu_precond;
    dealii::PreconditionJacobi<MatrixType> jacobi_precond;
    dealii::PreconditionSSOR<MatrixType> ssor_precond;
};

} // namespace gpe

#endif //GPE_LAC_H