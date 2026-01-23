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

//! Mass matrix assembly, implementation based on tutorial/step-3
//! @tparam dim Problem dimension
//! @tparam Assembly
//! @param system_matrix Matrix to be populated with entries
//! @param dof_handler DOF object, contains triangulation and finite element
//! @param quadrature Quadrature formula used
//! @param flags Required update flags, typically set in Assembly object
//! @param assemble_cell Function object which iterates over local cells
//! @param constraints Affine constraints applied to matrix rows and columns
//! @param mapping Polynomial mapping between reference cell and real cell
template <int dim, typename Assembly>
// BUG: DoFHandler::get_fe() - error: variable type 'FiniteElement<1, 1>' is an abstract class
void assemble_system(dealii::SparseMatrix<double>& system_matrix,
                     const dealii::DoFHandler<dim>& dof_handler,
                     const dealii::Quadrature<dim>& quadrature,
                     const dealii::Mapping<dim>& mapping,
                     dealii::UpdateFlags flags, Assembly&& assemble_cell,
                     const dealii::AffineConstraints<double>& constraints)
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

    // Iterate over cells / degrees of freedom
    AssertDimension(system_matrix.m(), dof_handler.n_dofs());
    AssertDimension(system_matrix.n(), dof_handler.n_dofs());

    // Generic lambda: works for active-cell range
    for (const auto &cell : dof_handler.active_cell_iterators())  // cell is a DoFHandler<dim>::(active|level)_cell_iterator
    {
        fe_values.reinit(cell); // convertible to Triangulation::cell_iterator
        cell_matrix = 0;
        //cell->get_active_or_mg_dof_indices(local_dof_indices);
        cell->get_dof_indices(local_dof_indices);

        // Pass on populated DoF indices to assemble matrix
        assemble_cell(fe_values, cell_matrix, local_dof_indices);

        // Apply boundary conditions (Dirichlet and hanging nodes, if any)
        // when distributing local (cell) matrix entries
        constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
    }
}

// TODO: cache shape_value(i, q) and shape_grad(i, q) in local arrays for Q2 or higher elements
template <int dim>
void assemble_mass(dealii::SparseMatrix<double>& system_matrix,
                   const dealii::DoFHandler<dim>& dof_handler,
                   const dealii::Quadrature<dim>& quadrature,
                   const dealii::Mapping<dim>& mapping,
                   const dealii::AffineConstraints<double>& constraints)
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
    assemble_system(system_matrix, dof_handler, quadrature, mapping, flags, f_mass, constraints);
}

template <int dim>
void assemble_stiffness(dealii::SparseMatrix<double>& system_matrix,
                        const dealii::DoFHandler<dim>& dof_handler,
                        const dealii::Quadrature<dim>& quadrature,
                        const dealii::Mapping<dim>& mapping,
                        const dealii::AffineConstraints<double>& constraints)
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
    assemble_system(system_matrix, dof_handler, quadrature, mapping, flags, f_stiffness, constraints);
}

template <int dim, typename Function>
void assemble_mass_weighted(dealii::SparseMatrix<double>& system_matrix,
                           Function&& V,
                           const dealii::DoFHandler<dim>& dof_handler,
                           const dealii::Quadrature<dim>& quadrature,
                           const dealii::Mapping<dim>& mapping,
                           const dealii::AffineConstraints<double>& constraints)
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
    assemble_system(system_matrix, dof_handler, quadrature, mapping, flags, f_mass_weighted, constraints);
}

// A0 = stiffness + mass_weighted
template <int dim, typename Function>
void assemble_A0(dealii::SparseMatrix<double>& system_matrix,
                 Function&& V,
                 const dealii::DoFHandler<dim>& dof_handler,
                 const dealii::Quadrature<dim>& quadrature,
                 const dealii::Mapping<dim>& mapping,
                 const dealii::AffineConstraints<double>& constraints)
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
    assemble_system(system_matrix, dof_handler, quadrature, mapping, flags, f_A0, constraints);
}

// TODO: optimizations (symmetry, caching, matrix-free operator)
template <int dim>
void assemble_mass_phiphi(dealii::SparseMatrix<double>& matrix,
                          const dealii::Vector<double>& u,
                          const dealii::DoFHandler<dim>& dof_handler,
                          const dealii::Quadrature<dim>& quadrature,
                          const dealii::Mapping<dim>& mapping,
                          const dealii::AffineConstraints<double>& constraints)
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
    assemble_system(matrix, dof_handler, quadrature, mapping, flags, f_mass_phiphi, constraints);
}

} // namespace gpe

#endif //GPE_ASSEMBLE_H