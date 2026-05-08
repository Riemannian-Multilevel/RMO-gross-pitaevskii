#ifndef GPE_LAC_H
#define GPE_LAC_H
#define SOLVER_MIN_TOL 1e-2

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
        Assert(weight != 0, dealii::ExcMessage("weight must be non-zero"));

        if (m_components.size()) {
            AssertDimension(matrix.n(), this->n());
            AssertDimension(matrix.m(), this->m());
        }
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
        return m_components.back().second->m();
    }

    /**
     * @brief Returns the number of columns in the operator.
     * @return global_dof_index Number of columns.
     */
    global_dof_index n() const
    {
        Assert(!m_components.empty(), dealii::ExcMessage("No matrices added"));
        return m_components.back().second->n();
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

    Vector<double> diagonal() const
    {
        Vector<double> diag(m());
        diag = 0.0;

        for (const auto& pair : m_components) {
            const double weight = pair.first;
            const auto* matrix = pair.second;

            for (unsigned row = 0; row < m(); ++row) {
                diag[row] += weight*matrix->diag_element(row);
            }
        }
        return diag;
    }

private:
    /** @brief Collection of weights and matrix pointers. */
    std::vector<std::pair<double, const MatrixType*>> m_components;

    /**
     * @brief Temporary scratch vector to prevent frequent allocations.
     * Marked mutable to allow use within const @ref vmult methods.
     */
    mutable VectorType m_vector;
    // TODO: not thread-safe
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
 * @tparam PrecondType The preconditioner applied to accelerate the Krylov
 * solver (e.g., `dealii::SparseILU`, `dealii::PreconditionJacobi`). Note that while
 * this preconditioner is often constructed from an explicitly assembled `MatrixType`
 * before being passed to this class, the `InverseMatrix` only requires it to be
 * invocable by the solver.
 */
template <typename OperatorType, typename PrecondType = dealii::PreconditionIdentity>
class InverseMatrix
{
public:
    /**
     * @brief Constructor.
     * @param matrix The matrix to be inverted (solved).
     * @param options Options for the iterative solver (preconditioner, iteration count, tolerance)
     */
    InverseMatrix(const OperatorType& matrix,
                  const SolverOptions options,
                  const PrecondType&  precond)
        : m_matrix(matrix)
        , m_precond(precond)
        , m_method(options.solver)
        , m_max_iter(options.max_inner)
        , m_reltol(options.tol_inner)
        , m_control(m_max_iter, SOLVER_MIN_TOL)
    {}

    // Set absolute tolerance for vmult() step
    void set_tol(double tol) const { m_tol = tol; }

    /**
     * @brief Performs the system solve: \f$ dst = M^{-1} \cdot rhs \f$.
     * This method initializes the solver and control parameters based on the
     * norm of the @p rhs vector and the specified relative tolerance.
     *
     * @param dst The solution vector.
     * @param rhs The right-hand side vector.
     * @throws std::invalid_argument If an unsupported SolverMethod is provided.
     */
    template <typename VectorType>
    void vmult(VectorType &dst, const VectorType &rhs) const
    {
        dst = 0.0;
        const double rhs_norm = rhs.l2_norm();
        const double tol = std::min(m_tol > 0 ? m_tol : m_reltol * rhs_norm, SOLVER_MIN_TOL);
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
    const OperatorType &m_matrix;  // TODO: m_oper?
    const PrecondType &m_precond;

    const SolverMethod m_method;
    unsigned int m_max_iter;
    double m_reltol;

    // Mutable: relative tolerance changes depending on rhs
    // in every (logically constant) vmult()
    mutable SolverControl m_control;
    mutable double m_tol = 0.0;
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
 * preconditioners like ILU must directly access explicit matrix elements
 * during the setup phase.
 */
// TODO pass on additional data to preconditioner, instead of fixed parameters
template <typename OperatorType, typename MatrixType>
class PreconditionInverse
{
public:
    using VectorType = Vector<double>;

    /**
     * @brief Constructs the generic preconditioned solver.
     * @param options Options for the iterative solver (method, preconditioner, iteration count, tolerance)
     */
    PreconditionInverse(const OperatorType& matrix, SolverOptions options)
        : m_op(matrix)
        , m_method(options.solver)
        , m_precond_type(options.precond)
        , m_options(options)
        , m_control(options.max_inner)
    {}

    /**
     * @brief Builds preconditioners that only need to be set up once.
     * e.g., ILU or AMG on the stationary matrix A0.
     */
    void update_static(const MatrixType& static_matrix)
    {
        if (m_precond_type == Precondition::SPARSE_ILU) {
            ilu_precond.initialize(static_matrix);
        }
        if (m_precond_type == Precondition::AMG) {
            throw std::invalid_argument("Precondition::AMG not implemented");
        }
    }

    /**
     * @brief Rebuilds dynamic preconditioners using the latest assembled matrix.
     * To minimize storage usage and avoid copies of large matrix objects, we
     * assume a diagonal preconditioner. The diagonal can arise from the
     * preconditioned matrix itself, or from mass lumping.
     */
    void update_dynamic(const Vector<double>& diag)
    {
        if (m_precond_type == Precondition::DIAGONAL) {
            // Create a vector to hold the inverse diagonal elements
            Vector<double> inv_diag(diag);
            const double relaxation = 0.6;

            for (unsigned int i = 0; i < inv_diag.size(); ++i) {
                // Safeguard against division by zero
                if (std::abs(inv_diag[i]) > 1e-14) {
                    inv_diag[i] = relaxation / inv_diag[i];
                } else {
                    inv_diag[i] = 1.0;
                }
            }
            jacobi_precond.reinit(inv_diag);
        }
    }

    /** @brief Solves op * dst = src. */
    void vmult(VectorType& dst, const VectorType& src) const
    {
        // Type erasure: select solve_with template argument, based on preconditioner argument
        switch (m_precond_type) {
            case Precondition::SPARSE_ILU:
                solve_with(m_op, dst, src, ilu_precond);
                break;
            case Precondition::DIAGONAL:
                solve_with(m_op, dst, src, jacobi_precond);
                break;
            case Precondition::NONE:
                solve_with(m_op, dst, src, dealii::PreconditionIdentity());
                break;
            default:
                throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
        }
    }

    /** @brief Returns the solver control object used in the last solve. */
    const SolverControl& control() const { return m_control; }

    void set_tol(double tol) const { m_tol = tol; }

private:
    template <typename VectorType, typename PrecondType>
    void solve_with(const OperatorType& matrix, VectorType& dst, const VectorType& src,
                    const PrecondType& precond) const
    {
        const InverseMatrix<OperatorType, PrecondType> inv(matrix, m_options, precond);
        if (m_tol > 0) {
            inv.set_tol(m_tol);
        }
        inv.vmult(dst, src);
        m_control = inv.control();
    }

    const OperatorType &m_op;
    SolverMethod  m_method;
    Precondition  m_precond_type;
    SolverOptions m_options;

    dealii::SparseILU<double> ilu_precond;
    dealii::DiagonalMatrix<VectorType> jacobi_precond;

    // Stores the state of the most recent solve_with() call
    mutable SolverControl m_control;
    mutable double m_tol = 0.0;
};

using OperatorType  = LinearCombination<SparseMatrix<double>, Vector<double>>;
using InverseOpType = PreconditionInverse<OperatorType, SparseMatrix<double>>;

} // namespace gpe

#endif //GPE_LAC_H