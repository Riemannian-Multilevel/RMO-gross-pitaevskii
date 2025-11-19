#ifndef GPE_ASSEMBLY_H
#define GPE_ASSEMBLY_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/quadrature_lib.h>

namespace gpe
{
using dealii::types::global_dof_index;
using dealii::numbers::invalid_unsigned_int;

template <int dim, typename CellRange, typename Assembly>
void assemble_system_impl(const CellRange& cells, dealii::FEValues<dim> &fe_values,
    dealii::SparseMatrix<double> &system_matrix,
    dealii::FullMatrix<double> &cell_matrix,
    std::vector<global_dof_index> &local_dof_indices,
    Assembly&& assemble_cell)
{
    for (const auto& cell : cells) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        // Same code for active cells, or cells on a level in a multigrid hierarchy
        cell->get_active_or_mg_dof_indices(local_dof_indices);

        // Pass on populated DoF indices to assemble M_phiphi
        assemble_cell(fe_values, cell_matrix, local_dof_indices);

        for (const unsigned int i : fe_values.dof_indices()) {
            for (const unsigned int j : fe_values.dof_indices()) {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
            }
        }
    }
}
// TODO: specification of boundary values (Dirichlet or Neumann on part of the boundary)
//! Mass matrix assembly, implementation based on tutorial/step-3
//! @tparam dim Problem dimension
//! @tparam Assembly
//! @param system_matrix Matrix to be populated with entries
//! @param dof_handler DOF object, contains triangulation and finite element
//! @param flags Required update flags, typically set in Assembly object
//! @param assemble_cell Function object which iterates over local cells
//! @param level Multigrid level (0 for active cells)
template <int dim, typename Assembly>
// BUG: DoFHandler::get_fe() - error: variable type 'FiniteElement<1, 1>' is an abstract class
void assemble_system(dealii::SparseMatrix<double>& system_matrix,
                     const dealii::DoFHandler<dim>& dof_handler,
                     dealii::UpdateFlags flags, Assembly&& assemble_cell,
                     unsigned int level = invalid_unsigned_int)
{
    const auto& element = dof_handler.get_fe();
    // Quadrature formula for the evaluation of the integrals on each cel
    const dealii::QGauss<dim> quadrature_formula(element.degree + 1);
    // Class which handles finite element, quadrature, and mapping objects
    dealii::FEValues<dim> fe_values(element, quadrature_formula, flags);

    // Compute contributions of each cell in a local dense matrix, to avoid
    // updating a large sparse matrix in every step
    // Consistent between active and multigrid approaches
    const unsigned int dofs_per_cell = element.n_dofs_per_cell();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<global_dof_index> local_dof_indices(dofs_per_cell);

    // Clear existing matrix entries
    system_matrix = 0;  // system_matrix.reinit(system_matrix.get_sparsity_pattern())

    // Iterate over cells / degrees of freedom
    if (level == invalid_unsigned_int) {
        AssertDimension(system_matrix.m(), dof_handler.n_dofs());
        AssertDimension(system_matrix.n(), dof_handler.n_dofs());

        // Iterate over active cells
        assemble_system_impl(dof_handler.active_cell_iterators(),
            fe_values, system_matrix, cell_matrix, local_dof_indices, assemble_cell);
    } else {
        AssertDimension(system_matrix.m(), dof_handler.n_dofs(level));
        AssertDimension(system_matrix.n(), dof_handler.n_dofs(level));

        // Iterate over multigrid cells on given level
        assemble_system_impl(dof_handler.mg_cell_iterators_on_level(level),
            fe_values, system_matrix, cell_matrix, local_dof_indices, assemble_cell);
    }
}

// TODO: optional assembly for right-hand side (separate function?)
template <int dim>
void assemble_mass(dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler, unsigned int level = invalid_unsigned_int)
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
    assemble_system(system_matrix, dof_handler, flags, f_mass, level);
}

template <int dim>
void assemble_stiffness(dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler, unsigned int level = invalid_unsigned_int)
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
    assemble_system(system_matrix, dof_handler, flags, f_stiffness, level);
}

template <int dim, typename Function>
void assemble_mass_weighed(dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler,
    Function&& V, unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    auto f_mass_weighed = [V](const dealii::FEValues<dim>& fe_values,
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
    assemble_system(system_matrix, dof_handler, flags, f_mass_weighed, level);
}

template <int dim>
void assemble_mass_phiphi(dealii::SparseMatrix<double>& matrix,
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Vector<double>& u, unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_JxW_values);

    auto f_mass_phiphi = [u](const dealii::FEValues<dim>& fe_values,
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
    assemble_system(matrix, dof_handler, flags, f_mass_phiphi, level);
}

} // namespace gpe

#endif //GPE_ASSEMBLY_H