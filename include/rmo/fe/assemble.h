#ifndef RMO_FE_ASSEMBLE_H
#define RMO_FE_ASSEMBLE_H

#include <rmo/lac_traits.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_fe.h>

#include <vector>

/**
 * @file
 * @brief Matrix assembly for the Gross-Pitaevskii discretization.
 *
 * The `assemble_*` functions share their arguments: the output matrix (sized beforehand, entries
 * overwritten), the DoFHandler, the quadrature rule and mapping, the constraints applied when
 * distributing cell matrices, and an optional multigrid level (default: active cells).
 *
 * The `*_lumped` variants assemble diagonal matrices into a `DiagonalMatrix` or a `SparseMatrix`.
 * They require a lumpable element (\f$ \int_K \phi_i > 0 \f$, e.g. `FE_Q`, `FE_SimplexP(1)`,
 * `FE_SimplexP_Bubbles(2)`) and no hanging-node constraints.
 */
namespace rmo::fe
{
using dealii::types::global_dof_index;
using dealii::numbers::invalid_unsigned_int;

/**
 * @brief Generic assembly loop over the active cells, or the cells of multigrid @p level.
 *
 * On each cell, `assemble_cell(fe_values, cell_matrix, local_dof_indices)` computes the cell matrix,
 * which is then distributed with @p constraints. The caller constructs @p fe_values (quadrature,
 * mapping, update flags); a kernel that needs a second quadrature rule can reinitialize its own
 * `FEValues` with `fe_values.get_cell()`.
 *
 * @p system_matrix must have the size of the DoF space (on @p level), and is zeroed first if
 * @p reinit is set. A `DiagonalMatrix` keeps only the diagonal entries.
 */
template <int dim, typename GlobalMatrix, typename Assembly>
void assemble_system(GlobalMatrix& system_matrix,
                     const dealii::DoFHandler<dim>& dof_handler,
                     dealii::FEValues<dim>& fe_values,
                     Assembly&& assemble_cell,
                     const dealii::AffineConstraints<double>& constraints,
                     unsigned int level = invalid_unsigned_int,
                     const bool reinit = true)
{
    const unsigned int dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
    AssertThrow(fe_values.dofs_per_cell == dofs_per_cell,
        dealii::ExcDimensionMismatch(fe_values.dofs_per_cell, dofs_per_cell));

    const auto n_rows = (level == invalid_unsigned_int) ? dof_handler.n_dofs() : dof_handler.n_dofs(level);
    AssertThrow(system_matrix.m() == n_rows, dealii::ExcDimensionMismatch(system_matrix.m(), n_rows));
    AssertThrow(system_matrix.n() == n_rows, dealii::ExcDimensionMismatch(system_matrix.n(), n_rows));

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<global_dof_index> local_dof_indices(dofs_per_cell);

    if (reinit) {
        if constexpr (rmo::is_diagonal_matrix_v<GlobalMatrix>)
            system_matrix.get_vector() = 0;
        else
            system_matrix = 0;
    }

    auto assemble_over_cells = [&](const auto &cell_range)
    {
        for (const auto &cell : cell_range)
        {
            fe_values.reinit(cell);
            cell_matrix = 0;
            cell->get_active_or_mg_dof_indices(local_dof_indices);

            assemble_cell(static_cast<const dealii::FEValues<dim>&>(fe_values), cell_matrix, local_dof_indices);

            constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
        }
    };

    if (level == invalid_unsigned_int) {
        assemble_over_cells(dof_handler.active_cell_iterators());
    }
    else {
        assemble_over_cells(dof_handler.mg_cell_iterators_on_level(level));
    }
}

/**
 * @brief Mass matrix \f$ M_{ij} = \int_\Omega \phi_i \phi_j \, dx \f$.
 */
template <int dim, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_mass(GlobalMatrix& system_matrix,
                   const dealii::DoFHandler<dim>& dof_handler,
                   const dealii::Quadrature<dim>& quadrature,
                   const dealii::Mapping<dim>& mapping,
                   const dealii::AffineConstraints<double>& constraints,
                   unsigned int level = invalid_unsigned_int)
{
    dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_JxW_values);

    auto f_mass = [](const dealii::FEValues<dim>& fe_values, dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            const auto JxW = fe_values.JxW(q_index);

            for (const unsigned int i : fe_values.dof_indices()) {
                const auto value_i = fe_values.shape_value(i, q_index);

                for (const unsigned int j : fe_values.dof_indices()) {
                    const auto value_j = fe_values.shape_value(j, q_index);
                    cell_matrix(i, j) += value_i * value_j * JxW;
                }
            }
        }
    };
    assemble_system(system_matrix, dof_handler, fe_values, f_mass, constraints, level, true);
}

/**
 * @brief Lumped mass matrix \f$ (M_L)_{ii} = \int_\Omega \phi_i \, dx \f$, the row sums of \f$ M \f$.
 *
 * @p quadrature must integrate the shape functions exactly; Gauss and nodal rules give the same result.
 */
template <int dim, typename GlobalMatrix = dealii::DiagonalMatrix<dealii::Vector<double>>>
void assemble_mass_lumped(GlobalMatrix& system_matrix,
                          const dealii::DoFHandler<dim>& dof_handler,
                          const dealii::Quadrature<dim>& quadrature,
                          const dealii::Mapping<dim>& mapping,
                          const dealii::AffineConstraints<double>& constraints,
                          unsigned int level = invalid_unsigned_int)
{
    dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_JxW_values);

    auto f_mass = [](const dealii::FEValues<dim>& fe_values, dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            const auto JxW = fe_values.JxW(q_index);

            for (const unsigned int j : fe_values.dof_indices()) {
                cell_matrix(j, j) += fe_values.shape_value(j, q_index) * JxW;
            }
        }
    };
    assemble_system(system_matrix, dof_handler, fe_values, f_mass, constraints, level, true);
}

/**
 * @brief Stiffness matrix \f$ S_{ij} = \int_\Omega \nabla\phi_i \cdot \nabla\phi_j \, dx \f$.
 */
template <int dim, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_stiffness(GlobalMatrix& system_matrix,
                        const dealii::DoFHandler<dim>& dof_handler,
                        const dealii::Quadrature<dim>& quadrature,
                        const dealii::Mapping<dim>& mapping,
                        const dealii::AffineConstraints<double>& constraints,
                        unsigned int level = invalid_unsigned_int)
{
    dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

    auto f_stiffness = [](const dealii::FEValues<dim>& fe_values, dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            const auto JxW = fe_values.JxW(q_index);

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j, q_index) * JxW;
                }
            }
        }
    };
    assemble_system(system_matrix, dof_handler, fe_values, f_stiffness, constraints, level, true);
}

/**
 * @brief Potential-weighted mass matrix \f$ (M_V)_{ij} = \int_\Omega V \phi_i \phi_j \, dx \f$.
 *
 * @p V is callable as `double(const Point<dim>&)`.
 */
template <int dim, typename Function, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_mass_weighted(GlobalMatrix& system_matrix,
                           Function&& V,
                           const dealii::DoFHandler<dim>& dof_handler,
                           const dealii::Quadrature<dim>& quadrature,
                           const dealii::Mapping<dim>& mapping,
                           const dealii::AffineConstraints<double>& constraints,
                           unsigned int level = invalid_unsigned_int)
{
    dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    auto f_mass_weighted = [&V](const dealii::FEValues<dim>& fe_values, dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            const double V_JxW = V(fe_values.quadrature_point(q_index)) * fe_values.JxW(q_index);

            for (const unsigned int i : fe_values.dof_indices()) {
                const auto value_i = fe_values.shape_value(i, q_index);

                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += value_i * fe_values.shape_value(j, q_index) * V_JxW;
                }
            }
        }
    };
    assemble_system(system_matrix, dof_handler, fe_values, f_mass_weighted, constraints, level, true);
}

/**
 * @brief Linear part of the Gross-Pitaevskii operator, \f$ A_0 = S + M_V \f$.
 */
template <int dim, typename Function, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_A0(GlobalMatrix& system_matrix,
                 Function&& V,
                 const dealii::DoFHandler<dim>& dof_handler,
                 const dealii::Quadrature<dim>& quadrature,
                 const dealii::Mapping<dim>& mapping,
                 const dealii::AffineConstraints<double>& constraints,
                 unsigned int level = invalid_unsigned_int)
{
    dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    auto f_A0 = [&V](const dealii::FEValues<dim>& fe_values, dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            const auto   JxW = fe_values.JxW(q_index);
            const double V_x = V(fe_values.quadrature_point(q_index));

            for (const unsigned int i : fe_values.dof_indices()) {
                const auto  value_i = fe_values.shape_value(i, q_index);
                const auto& grad_i  = fe_values.shape_grad(i, q_index);

                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += (grad_i * fe_values.shape_grad(j, q_index)
                        + V_x * value_i * fe_values.shape_value(j, q_index)) * JxW;
                }
            }
        }
    };
    assemble_system(system_matrix, dof_handler, fe_values, f_A0, constraints, level, true);
}

/**
 * @brief Linear part of the Gross-Pitaevskii operator with lumped potential, \f$ A_0 = S + M_{V,L} \f$.
 *
 * @p quadrature is used for \f$ S \f$ and must integrate gradients accurately (e.g. `QGauss(p + 1)`);
 * @p quadrature_mass is used for \f$ M_{V,L} \f$ and should be the nodal rule.
 */
template <int dim, typename Function, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_A0_lumped(GlobalMatrix& system_matrix,
                        Function&& V,
                        const dealii::DoFHandler<dim>& dof_handler,
                        const dealii::Quadrature<dim>& quadrature,
                        const dealii::Mapping<dim>& mapping,
                        const dealii::AffineConstraints<double>& constraints,
                        unsigned int level = invalid_unsigned_int)
{
    dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    std::vector<double> V_support(dof_handler.get_fe().n_dofs_per_cell());

    auto f_A0 = [&V, &mapping, &V_support](const dealii::FEValues<dim>& fe_values,
                                            dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int j : fe_values.dof_indices()) {
            const auto a_j = mapping.transform_unit_to_real_cell(fe_values.get_cell(),
                                                                 fe_values.get_fe().unit_support_point(j));
            V_support[j] = V(a_j);
        }

        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            const auto JxW = fe_values.JxW(q_index);

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j, q_index) * JxW;
                }
                cell_matrix(i, i) += V_support[i] * fe_values.shape_value(i, q_index) * JxW;
            }
        }
    };
    assemble_system(system_matrix, dof_handler, fe_values, f_A0, constraints, level, true);
}

/**
 * @brief Nonlinear term \f$ (M_{\phi\phi})_{ij} = \int_\Omega u_h^2 \phi_i \phi_j \, dx \f$ for the state @p u.
 */
// TODO: optimizations (symmetry, caching, matrix-free operator)
template <int dim, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_mass_phiphi(GlobalMatrix& matrix,
                          const dealii::Vector<double>& u,
                          const dealii::DoFHandler<dim>& dof_handler,
                          const dealii::Quadrature<dim>& quadrature,
                          const dealii::Mapping<dim>& mapping,
                          const dealii::AffineConstraints<double>& constraints,
                          unsigned int level = invalid_unsigned_int)
{
    dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_JxW_values);

    auto f_mass_phiphi = [&u](const dealii::FEValues<dim>& fe_values, dealii::FullMatrix<double>& cell_matrix,
        const auto& local_dof_indices)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            double u_x = 0.0;
            for (const unsigned int i : fe_values.dof_indices()) {
                u_x += u(local_dof_indices[i]) * fe_values.shape_value(i, q_index);
            }
            const double u2_JxW = u_x * u_x * fe_values.JxW(q_index);

            for (const unsigned int i : fe_values.dof_indices()) {
                const auto value_i = fe_values.shape_value(i, q_index);

                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += value_i * fe_values.shape_value(j, q_index) * u2_JxW;
                }
            }
        }
    };
    assemble_system(matrix, dof_handler, fe_values, f_mass_phiphi, constraints, level, true);
}

/**
 * @brief Lumped nonlinear term \f$ (M_{\phi\phi,L})_{ii} = u_i^2 \, (M_L)_{ii} \f$ for the state @p u.
 *
 * \f$ M_{\phi\phi,L}(u)\,u \f$ is the exact gradient of the lumped energy
 * \f$ \tfrac14 \sum_i (M_L)_{ii} u_i^4 \f$, so value and gradient of the functional are consistent.
 * Any quadrature that integrates \f$ \phi_i \f$ exactly gives the same result.
 */
template <int dim, typename GlobalMatrix = dealii::DiagonalMatrix<dealii::Vector<double>>>
void assemble_mass_phiphi_lumped(GlobalMatrix& matrix,
                                 const dealii::Vector<double>& u,
                                 const dealii::DoFHandler<dim>& dof_handler,
                                 const dealii::Quadrature<dim>& quadrature,
                                 const dealii::Mapping<dim>& mapping,
                                 const dealii::AffineConstraints<double>& constraints,
                                 unsigned int level = invalid_unsigned_int)
{
    dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_JxW_values);

    auto f_mass_phiphi = [&u](const dealii::FEValues<dim>& fe_values, dealii::FullMatrix<double>& cell_matrix,
        const auto& local_dof_indices)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            const auto JxW = fe_values.JxW(q_index);

            for (const unsigned int j : fe_values.dof_indices()) {
                const double u_j = u(local_dof_indices[j]);
                cell_matrix(j, j) += u_j * u_j * fe_values.shape_value(j, q_index) * JxW;
            }
        }
    };
    assemble_system(matrix, dof_handler, fe_values, f_mass_phiphi, constraints, level, true);
}

} // namespace rmo::fe
#endif //RMO_FE_ASSEMBLE_H
