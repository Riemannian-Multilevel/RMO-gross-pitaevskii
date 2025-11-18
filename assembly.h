#ifndef GPE_ASSEMBLY_H
#define GPE_ASSEMBLY_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/quadrature_lib.h>

// TODO: specification of boundary values (Dirichlet or Neumann on part of the boundary)
//       iterate over cells in a level of a multigrid hierarchy (possible refactoring)
//! Mass matrix assembly, implementation based on tutorial/step-3
//! @tparam dim Problem dimension
//! @tparam Assembly
//! @param system_matrix Matrix to be populated with entries
//! @param dof_handler DOF object, contains triangulation and finite element
//! @param element Finite element corresponding to DOF object
//! @param flags Required update flags, typically set in Assembly object
//! @param assemble Function object which iterates over local cells
template <int dim, typename Assembly>
// BUG: DoFHandler::get_fe() - error: variable type 'FiniteElement<1, 1>' is an abstract class
void assemble_matrix(dealii::SparseMatrix<double>& system_matrix, const dealii::DoFHandler<dim>& dof_handler,
    const dealii::FE_Q<dim>& element, dealii::UpdateFlags flags, Assembly&& assemble)
{
    const dealii::QGauss<dim> quadrature_formula(element.degree + 1);
    dealii::FEValues<dim> fe_values(element, quadrature_formula, flags);

    const unsigned int dofs_per_cell = element.n_dofs_per_cell();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Iterate over all active cells
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell->get_dof_indices(local_dof_indices);

        // Pass on populated DoF indices to assemble M_phiphi
        assemble(fe_values, cell_matrix, local_dof_indices);

        for (const unsigned int i : fe_values.dof_indices()) {
            for (const unsigned int j : fe_values.dof_indices()) {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
            }
        }
    }
}

template <int dim>
void assemble_mass(dealii::SparseMatrix<double>& system_matrix, const dealii::DoFHandler<dim>& dof_handler,
    const dealii::FE_Q<dim>& element)
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
    assemble_matrix(system_matrix, dof_handler, element, flags, f_mass);
}

template <int dim>
void assemble_stiffness(dealii::SparseMatrix<double>& system_matrix, const dealii::DoFHandler<dim>& dof_handler,
    const dealii::FE_Q<dim>& element)
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
    assemble_matrix(system_matrix, dof_handler, element, flags, f_stiffness);
}

template <int dim, typename Function>
void assemble_mass_weighed(dealii::SparseMatrix<double>& system_matrix, const dealii::DoFHandler<dim>& dof_handler,
    const dealii::FE_Q<dim>& element, Function&& V)
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
    assemble_matrix(system_matrix, dof_handler, element, flags, f_mass_weighed);
}

template <int dim>
void assemble_mass_phiphi(dealii::SparseMatrix<double>& matrix, const dealii::DoFHandler<dim>& dof_handler,
    const dealii::FE_Q<dim>& element, const dealii::Vector<double>& u)
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
    assemble_matrix(matrix, dof_handler, element, flags, f_mass_phiphi);
}

#endif //GPE_ASSEMBLY_H