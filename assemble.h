#ifndef GPE_ASSEMBLE_H
#define GPE_ASSEMBLE_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_fe.h>

namespace gpe
{
using dealii::types::global_dof_index;
using dealii::numbers::invalid_unsigned_int;

/**
 * @brief Generic system assembly loop.
 *
 * This function implements the standard finite element assembly loop. It iterates over
 * all cells (either active cells on the finest grid or cells on a specific multigrid level),
 * initializes `FEValues`, calls the user-provided `assemble_cell` functor to compute local
 * integrals, and distributes the result into the global system matrix applying constraints.
 *
 * The `assemble_cell` functor must have the signature:
 * `void(const FEValues<dim>&, FullMatrix<double>&, const std::vector<global_dof_index>&)`
 *
 * @tparam dim The spatial dimension.
 * @tparam Assembly The type of the lambda/functor performing the local integration.
 * @param[out] system_matrix The sparse matrix to be filled. Existing entries are cleared.
 * @param[in] dof_handler The DoFHandler object.
 * @param[in] quadrature The quadrature rule to use for integration.
 * @param[in] mapping The mapping from reference to real cell.
 * @param[in] flags Update flags required by the assembly kernel (e.g., update_values, update_gradients).
 * @param[in] assemble_cell The functor that computes the local matrix on a single cell.
 * @param[in] constraints Constraints to apply during distribution (e.g., hanging nodes, BCs).
 * @param[in] level The multigrid level to assemble. If set to `invalid_unsigned_int` (default),
 * the function assembles on the active cells (global system).
 */
template <int dim, typename Assembly>
// BUG: DoFHandler::get_fe() - error: variable type 'FiniteElement<1, 1>' is an abstract class
void assemble_system(dealii::SparseMatrix<double>& system_matrix,
                     const dealii::DoFHandler<dim>& dof_handler,
                     const dealii::Quadrature<dim>& quadrature,
                     const dealii::Mapping<dim>& mapping,
                     dealii::UpdateFlags flags, Assembly&& assemble_cell,
                     const dealii::AffineConstraints<double>& constraints,
                     unsigned int level = invalid_unsigned_int)
{
    // Quadrature formula for the evaluation of the integrals on each cell
    const auto& element = dof_handler.get_fe();

    // Class which handles finite element, quadrature, and mapping objects
    dealii::FEValues<dim> fe_values(mapping, element, quadrature, flags);

    // Compute contributions of each cell in a local dense matrix, to avoid
    // updating a large sparse matrix in every step
    const unsigned int dofs_per_cell = element.n_dofs_per_cell();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<global_dof_index> local_dof_indices(dofs_per_cell);

    // Clear existing matrix entries
    // XXX: optional parameter if (reinit) ... (default true)
    system_matrix = 0;  // system_matrix.reinit(system_matrix.get_sparsity_pattern())

    // Generic lambda: works for active-cell range and mg-level range
    auto assemble_over_cells = [&](const auto &cell_range)
    {
        for (const auto &cell : cell_range)  // cell is a DoFHandler<dim>::(active|level)_cell_iterator
        {
            fe_values.reinit(cell); // convertible to Triangulation::cell_iterator
            cell_matrix = 0;
            cell->get_active_or_mg_dof_indices(local_dof_indices);

            // Pass on populated DoF indices to assemble matrix
            assemble_cell(fe_values, cell_matrix, local_dof_indices);

            // Apply boundary conditions (Dirichlet and hanging nodes, if any)
            // when distributing local (cell) matrix entries
            constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
        }
    };

    // Iterate over cells / degrees of freedom
    if (level == invalid_unsigned_int) {
        AssertDimension(system_matrix.m(), dof_handler.n_dofs());
        AssertDimension(system_matrix.n(), dof_handler.n_dofs());

        // Iterate over active cells
        // assemble_system_impl(dof_handler.active_cell_iterators(),
        //     fe_values, system_matrix, cell_matrix, local_dof_indices, assemble_cell, constraints);
        assemble_over_cells(dof_handler.active_cell_iterators());
    }
    else {
        AssertDimension(system_matrix.m(), dof_handler.n_dofs(level));
        AssertDimension(system_matrix.n(), dof_handler.n_dofs(level));

        // Iterate over multigrid cells on given level
        // assemble_system_impl(dof_handler.mg_cell_iterators_on_level(level),
        //     fe_values, system_matrix, cell_matrix, local_dof_indices, assemble_cell, constraints);
        assemble_over_cells(dof_handler.mg_cell_iterators_on_level(level));
    }
}

/**
 * @brief Assembles the standard Mass matrix.
 *
 * Computes entries:
 * \f[
 * M_{ij} = \int_{\Omega} \phi_i(x) \phi_j(x) \, dx
 * \f]
 *
 * @tparam dim The spatial dimension.
 * @param[out] system_matrix The matrix to store the result.
 * @param[in] dof_handler The DoFHandler.
 * @param[in] quadrature The quadrature formula.
 * @param[in] mapping The geometric mapping.
 * @param[in] constraints Affine constraints.
 * @param[in] level MG level (optional).
 */
// TODO: cache shape_value(i, q) and shape_grad(i, q) in local arrays for Q2 or higher elements
template <int dim>
void assemble_mass(dealii::SparseMatrix<double>& system_matrix,
                   const dealii::DoFHandler<dim>& dof_handler,
                   const dealii::Quadrature<dim>& quadrature,
                   const dealii::Mapping<dim>& mapping,
                   const dealii::AffineConstraints<double>& constraints,
                   unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_JxW_values);

    auto f_mass = [](const dealii::FEValues<dim>& fe_values,
        dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                        fe_values.shape_value(j, q_index) * // phi_j(x_q)
                        fe_values.JxW(q_index)); // dx
                }
            }
        }
    };
    assemble_system(system_matrix, dof_handler, quadrature, mapping, flags,
        f_mass, constraints, level);
}

/**
 * @brief Assembles the Stiffness matrix (Laplacian).
 *
 * Computes entries:
 * \f[
 * S_{ij} = \int_{\Omega} \nabla \phi_i(x) \cdot \nabla \phi_j(x) \, dx
 * \f]
 *
 * @tparam dim The spatial dimension.
 * @param[out] system_matrix The matrix to store the result.
 * @param[in] dof_handler The DoFHandler.
 * @param[in] quadrature The quadrature formula.
 * @param[in] mapping The geometric mapping.
 * @param[in] constraints Affine constraints.
 * @param[in] level MG level (optional).
 */
template <int dim>
void assemble_stiffness(dealii::SparseMatrix<double>& system_matrix,
                        const dealii::DoFHandler<dim>& dof_handler,
                        const dealii::Quadrature<dim>& quadrature,
                        const dealii::Mapping<dim>& mapping,
                        const dealii::AffineConstraints<double>& constraints,
                        unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    auto f_stiffness = [](const dealii::FEValues<dim>& fe_values,
        dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                        fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                        fe_values.JxW(q_index)); // dx
                }
            }
        }
    };
    assemble_system(system_matrix, dof_handler, quadrature, mapping, flags,
        f_stiffness, constraints, level);
}

/**
 * @brief Assembles a Mass matrix weighted by a scalar potential \f$ V(x) \f$.
 *
 * Computes entries:
 * \f[
 * (M_V)_{ij} = \int_{\Omega} V(x) \phi_i(x) \phi_j(x) \, dx
 * \f]
 *
 * @tparam dim The spatial dimension.
 * @tparam Function A functor or function object type that can be called as `double V(Point<dim>)`.
 * @param[out] system_matrix The matrix to store the result.
 * @param[in] V The potential function \f$ V(x) \f$.
 * @param[in] dof_handler The DoFHandler.
 * @param[in] quadrature The quadrature formula.
 * @param[in] mapping The geometric mapping.
 * @param[in] constraints Affine constraints.
 * @param[in] level MG level (optional).
 */
template <int dim, typename Function>
void assemble_mass_weighted(dealii::SparseMatrix<double>& system_matrix,
                           Function&& V,
                           const dealii::DoFHandler<dim>& dof_handler,
                           const dealii::Quadrature<dim>& quadrature,
                           const dealii::Mapping<dim>& mapping,
                           const dealii::AffineConstraints<double>& constraints,
                           unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    auto f_mass_weighted = [&V](const dealii::FEValues<dim>& fe_values,
        dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            dealii::Point<dim> x = fe_values.quadrature_point(q_index);

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += V(x) * (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                        fe_values.shape_value(j, q_index) * // phi_j(x_q)
                        fe_values.JxW(q_index)); // dx
                }
            }
        }
    };
    assemble_system(system_matrix, dof_handler, quadrature, mapping, flags,
        f_mass_weighted, constraints, level);
}

/**
 * @brief Assembles the linear Hamiltonian operator \f$ A_0 \f$.
 *
 * This corresponds to the linear part of the Gross-Pitaevskii operator (Kinetic energy + Potential energy).
 * Computes entries:
 * \f[
 * (A_0)_{ij} = \int_{\Omega} \nabla \phi_i \cdot \nabla \phi_j + V(x) \phi_i \phi_j \, dx
 * \f]
 *
 * @tparam dim The spatial dimension.
 * @tparam Function Type of the potential function V.
 * @param[out] system_matrix The matrix to store the result.
 * @param[in] V The potential function \f$ V(x) \f$.
 * @param[in] dof_handler The DoFHandler.
 * @param[in] quadrature The quadrature formula.
 * @param[in] mapping The geometric mapping.
 * @param[in] constraints Affine constraints.
 * @param[in] level MG level (optional).
 */
template <int dim, typename Function>
void assemble_A0(dealii::SparseMatrix<double>& system_matrix,
                 Function&& V,
                 const dealii::DoFHandler<dim>& dof_handler,
                 const dealii::Quadrature<dim>& quadrature,
                 const dealii::Mapping<dim>& mapping,
                 const dealii::AffineConstraints<double>& constraints,
                 unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_gradients
        | dealii::update_JxW_values | dealii::update_quadrature_points);

    auto f_A0 = [&V](const dealii::FEValues<dim>& fe_values,
        dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            dealii::Point<dim> x = fe_values.quadrature_point(q_index);

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += V(x) * (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                        fe_values.shape_value(j, q_index) * // phi_j(x_q)
                        fe_values.JxW(q_index)); // dx
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                        fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                        fe_values.JxW(q_index)); // dx
                }
            }
        }
    };
    assemble_system(system_matrix, dof_handler, quadrature, mapping, flags,
        f_A0, constraints, level);
}

/**
 * @brief Assembles the nonlinear interaction term \f$ M_{\phi\phi} \f$ based on a current state \f$ u \f$.
 *
 * This matrix represents the cubic nonlinearity in the GP equation linearized around \f$ u \f$.
 * Computes entries:
 * \f[
 * (M_{\phi\phi})_{ij} = \int_{\Omega} |u_h(x)|^2 \phi_i(x) \phi_j(x) \, dx
 * \f]
 * where \f$ u_h(x) = \sum u_k \phi_k(x) \f$ is the finite element solution defined by vector @p u.
 *
 * @tparam dim The spatial dimension.
 * @param[out] matrix The matrix to store the result.
 * @param[in] u The current solution vector defining the nonlinearity density \f$ |u|^2 \f$.
 * @param[in] dof_handler The DoFHandler.
 * @param[in] quadrature The quadrature formula.
 * @param[in] mapping The geometric mapping.
 * @param[in] constraints Affine constraints.
 * @param[in] level MG level (optional).
 */
// TODO: optimizations (symmetry, caching, matrix-free operator)
template <int dim>
void assemble_mass_phiphi(dealii::SparseMatrix<double>& matrix,
                          const dealii::Vector<double>& u,
                          const dealii::DoFHandler<dim>& dof_handler,
                          const dealii::Quadrature<dim>& quadrature,
                          const dealii::Mapping<dim>& mapping,
                          const dealii::AffineConstraints<double>& constraints,
                          unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_JxW_values);

    auto f_mass_phiphi = [&u](const dealii::FEValues<dim>& fe_values,
        dealii::FullMatrix<double>& cell_matrix, const auto& local_dof_indices)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            double u_x = 0.0;
            for (const unsigned int i : fe_values.dof_indices()) {
                u_x += u(local_dof_indices[i]) * fe_values.shape_value(i, q_index);
            }
            u_x = std::abs(u_x);
            u_x *= u_x; // u_x2

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += u_x * (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                        fe_values.shape_value(j, q_index) * // phi_j(x_q)
                        fe_values.JxW(q_index)); // dx
                }
            }
        }
    };
    assemble_system(matrix, dof_handler, quadrature, mapping, flags,
        f_mass_phiphi, constraints, level);
}

} // namespace gpe

#endif //GPE_ASSEMBLE_H