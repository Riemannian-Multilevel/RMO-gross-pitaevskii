#ifndef RMO_LAC_H
#define RMO_LAC_H
// Upper bound on the inner-solver tolerance (loosest allowed), applied with
// std::min: the actual tolerance is never larger than this, i.e. the inner
// solve is always at least this accurate.
#define SOLVER_MAX_TOL 1e-2

#include "option_types.h"
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <deal.II/base/observer_pointer.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>  // for general matrices
#include <deal.II/lac/solver_cg.h>     // for symmetric positive definite matrices
#include <deal.II/lac/solver_minres.h>

/**
 * @file
 * @brief Operators and inverse operators for the Krylov solvers.
 *
 * All operators provide deal.II's `vmult(dst, src)` and `Tvmult(dst, src)`: LinearCombination (weighted sum
 * of matrices) and TransposeMatrix. The inverse operators apply \f$ A^{-1} \f$ through the same interface:
 * InverseMatrix (one Krylov solve per application), PreconditionInverse (solver and preconditioner selected
 * by SolverOptions).
 *
 * Inner tolerance: `vmult(dst, rhs)` uses `SolverOptions::tol_inner` relative to the norm of rhs,
 * `vmult(dst, rhs, AbsTol{t})` the absolute tolerance t; both capped at SOLVER_MAX_TOL.
 */
namespace rmo
{
//! Absolute tolerance of an inner solve; must be positive.
struct AbsTol { double value; };

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
 * @brief Weighted sum of matrices \f$ A = \sum_i w_i M_i \f$, a lightweight alternative to
 * dealii::LinearOperator.
 *
 * The matrices may have different types.
 * They are stored by pointer: they must outlive the operator, and may be reassembled after it is created.
 * Each type provides `vmult`, `Tvmult`, `m()`, `n()` and `diag_element(i)`.
 *
 * @tparam VectorType Vector type of vmult() and Tvmult().
 * @tparam MatrixTypes Admissible matrix types, all distinct.
 */
template <typename VectorType, typename... MatrixTypes>
class LinearCombination
{
    static_assert(sizeof...(MatrixTypes) > 0, "LinearCombination requires at least one matrix type");

public:
    LinearCombination() = default;

    /** @brief \f$ A \leftarrow A + w M \f$; components with zero weight are skipped. */
    template <typename MatrixType>
        requires (std::is_same_v<MatrixType, MatrixTypes> || ...)
    void add_component(double weight, const MatrixType &matrix)
    {
        if (weight == 0.0) {
            return;
        }
        if (m_components.size()) {
            AssertDimension(matrix.n(), this->n());
            AssertDimension(matrix.m(), this->m());
        }
        m_components.emplace_back(weight, Observer<MatrixType>(&matrix));
    }

    /** @brief Removes all components. */
    void clear()
    {
        m_components.clear();
    }

    /** @brief Sizes the scratch vector of vmult() and Tvmult(); required before their first use. */
    void reinit(unsigned int size)
    {
        m_vector.reinit(size);
    }

    /** @brief Sizes the scratch vector like @p vector. */
    void reinit(const VectorType& vector)
    {
        m_vector.reinit(vector);
    }

    /** @brief Number of rows. */
    global_dof_index m() const
    {
        Assert(!m_components.empty(), dealii::ExcMessage("No matrices added"));
        return std::visit([](const auto& matrix) -> global_dof_index { return matrix->m(); },
                          m_components.back().second);
    }

    /** @brief Number of columns. */
    global_dof_index n() const
    {
        Assert(!m_components.empty(), dealii::ExcMessage("No matrices added"));
        return std::visit([](const auto& matrix) -> global_dof_index { return matrix->n(); },
                          m_components.back().second);
    }

    /** @brief \f$ dst = A \, src \f$ */
    void vmult(VectorType &dst, const VectorType &src) const
    {
        dst = 0.0;
        vmult_add(dst, src);
    }

    /** @brief \f$ dst = A^\top src \f$ */
    void Tvmult(VectorType &dst, const VectorType &src) const
    {
        dst = 0.0;
        Tvmult_add(dst, src);
    }

    /** @brief \f$ dst \mathrel{+}= A \, src \f$ */
    void vmult_add(VectorType &dst, const VectorType &src) const
    {
        for (const auto& [weight, matrix] : m_components)
        {
            std::visit([&](const auto& M) { M->vmult(m_vector, src); }, matrix);
            dst.add(weight, m_vector);
        }
    }

    /** @brief \f$ dst \mathrel{+}= A^\top src \f$ */
    void Tvmult_add(VectorType &dst, const VectorType &src) const
    {
        for (const auto& [weight, matrix] : m_components)
        {
            std::visit([&](const auto& M) { M->Tvmult(m_vector, src); }, matrix);
            dst.add(weight, m_vector);
        }
    }

    /** @brief Diagonal \f$ \sum_i w_i \operatorname{diag}(M_i) \f$. */
    Vector<double> diagonal() const
    {
        Vector<double> diag(m());
        diag = 0.0;

        for (const auto& [weight, matrix] : m_components) {
            std::visit([&, weight = weight](const auto& M) { add_diagonal(diag, weight, *M); }, matrix);
        }
        return diag;
    }

private:
    /** @brief Adds weight * diag(matrix) to @p diag. */
    template <typename MatrixType>
    static void add_diagonal(Vector<double>& diag, double weight, const MatrixType& matrix)
    {
        for (unsigned row = 0; row < diag.size(); ++row) {
            diag[row] += weight*matrix.diag_element(row);
        }
    }

    //! Observer of a component; debug builds report use of a destroyed component
    template <typename MatrixType>
    using Observer = dealii::ObserverPointer<const MatrixType, LinearCombination>;

    /** @brief Collection of weights and matrix observers. */
    std::vector<std::pair<double, std::variant<Observer<MatrixTypes>...>>> m_components;

    /** @brief Scratch vector of the (const) vmult() and Tvmult(), hence mutable. */
    mutable VectorType m_vector;
    // TODO: not thread-safe
};


/** @brief Transpose \f$ A^\top \f$ of an operator: vmult() applies its Tvmult() and vice versa. */
template <typename OperatorType, typename VectorType>
class TransposeMatrix
{
public:
    TransposeMatrix(const OperatorType& matrix)
        : m_matrix(matrix)
    {}

    void vmult_add(VectorType &dst, const VectorType &src) const
    {
        m_matrix.Tvmult_add(dst, src);
    }

    void vmult(VectorType &dst, const VectorType &src) const
    {
        m_matrix.Tvmult(dst, src);
    }

    void Tvmult_add(VectorType &dst, const VectorType &src) const
    {
        m_matrix.vmult_add(dst, src);
    }

    void Tvmult(VectorType &dst, const VectorType &src) const
    {
        m_matrix.vmult(dst, src);
    }

    global_dof_index m() const
    {
        return m_matrix.m();
    }

    global_dof_index n() const
    {
        return m_matrix.n();
    }

private:
    const OperatorType& m_matrix;
};


/**
 * @brief Inverse \f$ A^{-1} \f$ of an operator: vmult() runs one Krylov solve (CG, MINRES or GMRES).
 *
 * @tparam OperatorType Operator \f$ A \f$ with `vmult(dst, src)`, e.g. LinearCombination; stored by reference.
 * @tparam PrecondType Preconditioner accepted by the deal.II solvers; stored by reference.
 */
template <typename OperatorType, typename PrecondType = dealii::PreconditionIdentity>
class InverseMatrix
{
public:
    /** @brief Stores @p matrix and @p precond by reference; @p options gives the solver method, maximum number of
     *  iterations and relative tolerance. */
    InverseMatrix(const OperatorType& matrix,
                  const SolverOptions options,
                  const PrecondType&  precond)
        : m_matrix(matrix)
        , m_precond(precond)
        , m_method(options.solver)
        , m_max_iter(options.max_inner)
        , m_reltol(options.tol_inner)
        , m_control(m_max_iter, SOLVER_MAX_TOL)
    {}

    /**
     * @brief Solves \f$ matrix \cdot dst = rhs \f$, starting from zero.
     * @throws std::invalid_argument for an unknown SolverMethod.
     */
    template <typename VectorType>
    void vmult(const OperatorType& matrix, VectorType &dst, const VectorType &rhs, const AbsTol tol) const
    {
        AssertThrow(tol.value > 0, dealii::ExcMessage("AbsTol must be positive"));
        solve(matrix, dst, rhs, tol.value);
    }

    /** @brief As above, with the relative tolerance `SolverOptions::tol_inner`. */
    template <typename VectorType>
    void vmult(const OperatorType& matrix, VectorType &dst, const VectorType &rhs) const
    {
        solve(matrix, dst, rhs, m_reltol * rhs.l2_norm());
    }

    /** @brief \f$ dst = A^{-1} rhs \f$, with the relative tolerance `SolverOptions::tol_inner`. */
    template <typename VectorType>
    void vmult(VectorType& dst, const VectorType& rhs) const { vmult(m_matrix, dst, rhs); }

    /** @brief \f$ dst = A^{-1} rhs \f$, with the absolute tolerance @p tol. */
    template <typename VectorType>
    void vmult(VectorType& dst, const VectorType& rhs, const AbsTol tol) const { vmult(m_matrix, dst, rhs, tol); }

    /** @brief \f$ dst = A^{-\top} rhs \f$ */
    template <typename VectorType>
    void Tvmult(VectorType& dst, const VectorType& rhs) const
    {
        // solve() applies vmult(), so we need to use a vmult() -> Tvmult() wrapper
        vmult(TransposeMatrix(m_matrix), dst, rhs);
    }

    /** @brief Solver statistics of the last solve. */
    const SolverControl& control() const { return m_control; }


private:
    template <typename VectorType>
    void solve(const OperatorType& matrix, VectorType &dst, const VectorType &rhs, const double tol) const
    {
        dst = 0.0;
        m_control = SolverControl(m_max_iter, std::min(tol, SOLVER_MAX_TOL));

        // TODO: continue on a partial solve with diagnostic, instead of throwing an exception
        switch (m_method) {
            case SolverMethod::GMRES:
                {
                    dealii::SolverGMRES<VectorType> solver(m_control);
                    solver.solve(matrix, dst, rhs, m_precond);
                }
                break;
            case SolverMethod::MINRES:
                {
                    dealii::SolverMinRes<VectorType> solver(m_control);
                    solver.solve(matrix, dst, rhs, m_precond);
                }
                break;
            case SolverMethod::CG:
                {
                    dealii::SolverCG<VectorType> solver(m_control);
                    solver.solve(matrix, dst, rhs, m_precond);
                }
                break;
            default:
                throw std::invalid_argument("Unknown SolverMethod");
        }
    }

    const OperatorType &m_matrix;  // TODO: m_oper?
    const PrecondType &m_precond;

    const SolverMethod m_method;
    unsigned int m_max_iter;
    double m_reltol;

    // Mutable: relative tolerance changes depending on rhs
    // in every (logically constant) vmult()
    mutable SolverControl m_control;
};


/**
 * @brief Inverse \f$ A^{-1} \f$ with the Krylov solver and preconditioner selected by SolverOptions.
 *
 * Preconditioners: ILU of an assembled matrix (built once, update_static()), damped Jacobi (rebuilt with
 * the operator, update_dynamic()), or none.
 *
 * @tparam OperatorType Operator \f$ A \f$ with `vmult(dst, src)`, e.g. LinearCombination; stored by reference.
 * @tparam MatrixType Assembled matrix for the ILU preconditioner, e.g. `SparseMatrix<double>`.
 */
// TODO pass on additional data to preconditioner, instead of fixed parameters
template <typename OperatorType, typename MatrixType>
class PreconditionInverse
{
public:
    using VectorType = Vector<double>;

    /** @brief Stores @p matrix by reference; @p options gives the solver method, preconditioner, maximum number of
     *  iterations and relative tolerance. */
    PreconditionInverse(const OperatorType& matrix, SolverOptions options)
        : m_op(matrix)
        , m_method(options.solver)
        , m_precond_type(options.precond)
        , m_options(options)
        , m_control(options.max_inner)
    {}

    /**
     * @brief Builds the ILU preconditioner of @p static_matrix (Precondition::SPARSE_ILU only).
     * @throws std::invalid_argument for Precondition::AMG, which is not implemented.
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
     * @brief Rebuilds the Jacobi preconditioner \f$ 0.6 \operatorname{diag}(d)^{-1} \f$ from @p diag
     * (Precondition::DIAGONAL only); entries with \f$ |d_i| \le 10^{-14} \f$ are replaced by 1.
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

    /** @brief Solves \f$ matrix \cdot dst = src \f$ with the configured solver and preconditioner, and the relative
     *  tolerance `SolverOptions::tol_inner`. */
    void vmult(const OperatorType& matrix, VectorType& dst, const VectorType& src) const
    {
        dispatch(matrix, dst, src, std::nullopt);
    }

    /** @brief As above, with the absolute tolerance @p tol. */
    void vmult(const OperatorType& matrix, VectorType& dst, const VectorType& src, const AbsTol tol) const
    {
        dispatch(matrix, dst, src, tol);
    }

    /** @brief \f$ dst = A^{-1} src \f$, with the relative tolerance `SolverOptions::tol_inner`. */
    void vmult(VectorType& dst, const VectorType& src) const { vmult(m_op, dst, src); }

    /** @brief \f$ dst = A^{-1} src \f$, with the absolute tolerance @p tol. */
    void vmult(VectorType& dst, const VectorType& src, const AbsTol tol) const { vmult(m_op, dst, src, tol); }

    /** @brief \f$ dst = A^{-\top} src \f$ */
    void Tvmult(VectorType& dst, const VectorType& src) const
    {
        vmult(TransposeMatrix(m_op), dst, src);
    }

    /** @brief Solver statistics of the last solve. */
    const SolverControl& control() const { return m_control; }


private:
    //! @p tol: absent for the relative tolerance `SolverOptions::tol_inner`.
    void dispatch(const OperatorType& matrix, VectorType& dst, const VectorType& src,
                  const std::optional<AbsTol> tol) const
    {
        // Type erasure: select solve_with template argument, based on preconditioner argument
        switch (m_precond_type) {
            case Precondition::SPARSE_ILU:
                solve_with(matrix, dst, src, ilu_precond, tol);
                break;
            case Precondition::DIAGONAL:
                solve_with(matrix, dst, src, jacobi_precond, tol);
                break;
            case Precondition::NONE:
                solve_with(matrix, dst, src, dealii::PreconditionIdentity(), tol);
                break;
            default:
                throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
        }
    }

    template <typename VectorType, typename PrecondType>
    void solve_with(const OperatorType& matrix, VectorType& dst, const VectorType& src,
                    const PrecondType& precond, const std::optional<AbsTol> tol) const
    {
        const InverseMatrix<OperatorType, PrecondType> inv(matrix, m_options, precond);
        if (tol) {
            inv.vmult(dst, src, *tol);
        } else {
            inv.vmult(dst, src);
        }
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
};

//! An inverse with a fixed tolerance, for code that calls `vmult(dst, rhs)` (the kernels). Holds a reference.
template <typename Inverse>
struct WithTolerance
{
    const Inverse& inv;
    AbsTol tol;

    template <typename VectorType>
    void vmult(VectorType& dst, const VectorType& rhs) const { inv.vmult(dst, rhs, tol); }
};

template <typename Inverse>
[[nodiscard]] WithTolerance<Inverse> with_tol(const Inverse& inv, const AbsTol tol) { return {inv, tol}; }

// No view of a temporary inverse: it would dangle
template <typename Inverse>
WithTolerance<Inverse> with_tol(const Inverse&& inv, AbsTol tol) = delete;

//! Operator and inverse of the consistent (sparse) Gross-Pitaevskii system.
using OperatorType  = LinearCombination<Vector<double>, SparseMatrix<double>>;
using InverseOpType = PreconditionInverse<OperatorType, SparseMatrix<double>>;

} // namespace rmo

#endif //RMO_LAC_H