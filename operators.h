#ifndef GPE_OPERATORS_H
#define GPE_OPERATORS_H

#include "assemble.h"

namespace gpe
{

// TODO: optional assembly for right-hand side (separate function?)
//       add execution policy
template <int dim, typename ExecutionPolicy>
void assemble_mass(ExecutionPolicy&& policy,
    dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler,
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
    assemble_system(policy, system_matrix, dof_handler, flags, f_mass, constraints, level);
}

template <int dim, typename ExecutionPolicy>
void assemble_stiffness(ExecutionPolicy&& policy,
    dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler,
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
    assemble_system(policy, system_matrix, dof_handler, flags, f_stiffness, constraints, level);
}

template <int dim, typename Function, typename ExecutionPolicy>
void assemble_mass_weighed(ExecutionPolicy&& policy,
    dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler,
    Function&& V, const dealii::AffineConstraints<double>& constraints,
    unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    auto f_mass_weighed = [&V](const dealii::FEValues<dim>& fe_values,
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
    assemble_system(policy, system_matrix, dof_handler, flags, f_mass_weighed, constraints, level);
}

// A0 = stiffness + mass_weighed
template <int dim, typename Function, typename ExecutionPolicy>
void assemble_A0(ExecutionPolicy&& policy,
    dealii::SparseMatrix<double>& system_matrix,
    const dealii::DoFHandler<dim>& dof_handler,
    Function&& V, const dealii::AffineConstraints<double>& constraints,
    unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_gradients
        | dealii::update_JxW_values | dealii::update_quadrature_points);

    auto f_A0 = [V](const dealii::FEValues<dim>& fe_values,
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
    assemble_system(policy, system_matrix, dof_handler, flags, f_A0, constraints, level);
}

// TODO: optimizations (symmetry, caching, matrix-free operator)
template <int dim, typename ExecutionPolicy>
void assemble_mass_phiphi(ExecutionPolicy&& policy,
    dealii::SparseMatrix<double>& matrix,
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Vector<double>& u, const dealii::AffineConstraints<double>& constraints,
    unsigned int level = invalid_unsigned_int)
{
    dealii::UpdateFlags flags = (dealii::update_values | dealii::update_JxW_values);

    auto f_mass_phiphi = [&u](const dealii::FEValues<dim>& fe_values,
        dealii::FullMatrix<double>& cell_matrix, const auto& local_dof_indices)
    {
        std::vector<double> u_local(fe_values.dofs_per_cell);
        for (const unsigned int i : fe_values.dof_indices()) {
            u_local[i] = u(local_dof_indices[i]);
        }
        
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            double u_x = 0.0;
            for (const unsigned int i : fe_values.dof_indices()) {
                u_x += u_local[i] * fe_values.shape_value(i, q_index);
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
    assemble_system(policy, matrix, dof_handler, flags, f_mass_phiphi, constraints, level);
}

}
#endif //GPE_OPERATORS_H