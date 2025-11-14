#ifndef GPE_ASSEMBLY_H
#define GPE_ASSEMBLY_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/quadrature_lib.h>

// TODO: specification of boundary values (Dirichlet or Neumann on part of the boundary)
//! Mass matrix assembly, implementation based on tutorial/step-3
//! @tparam dim Problem dimension
//! @tparam Assembly
//! @param system_matrix Matrix to be populated with entries
//! @param dof_handler DOF object, contains triangulation and finite element
//! @param flags Required update flags, typically set in Assembly object
//! @param assemble Function object which iterates over local cells
//! @param mg_level Multigrid level (>0) to iterate cells over, instead of the level with active cells
template <int dim, typename Assembly>
void assemble_matrix(dealii::SparseMatrix<double>& system_matrix, const dealii::DoFHandler<dim>& dof_handler,
    dealii::UpdateFlags flags, Assembly&& assemble, const int mg_level = 0)
{
    auto element = dof_handler.get_fe();
    const dealii::QGauss<dim> quadrature_formula(element.degree + 1);
    dealii::FEValues<dim> fe_values(element, quadrature_formula, flags);

    const unsigned int dofs_per_cell = element.n_dofs_per_cell();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Iterator to iterate over all active cells, or all cells in a multigrid hierarchy
    auto dof_iterator_range = (mg_level > 0)
        ? dof_handler.mg_cell_iterators_on_level(mg_level)
        : dof_handler.active_cell_iterators();

    for (const auto& cell : dof_iterator_range) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell->get_dof_indices(local_dof_indices);

        // Pass on populated DoF indices to assemble M_phiphi
        assemble(cell_matrix, fe_values, local_dof_indices);

        for (const unsigned int i : fe_values.dof_indices()) {
            for (const unsigned int j : fe_values.dof_indices()) {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
            }
        }
    }
}

template <int dim>
void assemble_mass(dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler, const int mg_level = 0)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_JxW_values);

    auto f_mass = [](const dealii::FEValues<dim>& fe_values,
        dealii::FullMatrix<double>& cell_matrix, auto&&...)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                        fe_values.shape_values(j, q_index) * // phi_j(x_q)
                        fe_values.JxW(q_index)); // dx
                }
            }
        }
    };
    assemble_matrix(system_matrix, dof_handler, flags, f_mass, mg_level);
}

template <int dim>
void assemble_stiffness(dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler, const int mg_level = 0)
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
    assemble_matrix(system_matrix, dof_handler, flags, f_stiffness, mg_level);
}

template <int dim, typename Function>
void assemble_mass_weighed(dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler, Function&& V, const int mg_level = 0)
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
                        fe_values.shape_values(j, q_index) * // phi_j(x_q)
                        fe_values.JxW(q_index)); // dx
                }
            }
        }
    };
    assemble_matrix(system_matrix, dof_handler, flags, f_mass_weighed, mg_level);
}

template <int dim>
void assemble_mass_phiphi(dealii::SparseMatrix<double>& matrix,
    const dealii::DoFHandler<dim>& dof_handler, const dealii::Vector<double>& u, const int mg_level = 0)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    auto f_mass_phiphi = [u](const dealii::FEValues<dim>& fe_values,
        dealii::FullMatrix<double>& cell_matrix, const auto& local_dof_indices)
    {
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            dealii::Point<dim> x = fe_values.quadrature_point(q_index);
            double u_x = 0.0;

            for (const unsigned int i : fe_values.dof_indices()) {
                u_x += u(local_dof_indices[i]) * fe_values.shape_value(i, q_index);
            }
            u_x = std::abs(u_x);
            u_x *= u_x; // u_x2

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) += u_x * (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                        fe_values.shape_values(j, q_index) * // phi_j(x_q)
                        fe_values.JxW(q_index)); // dx
                }
            }
        }
    };
    assemble_matrix(matrix, dof_handler, flags, f_mass_phiphi, mg_level);
}

#endif //GPE_ASSEMBLY_H