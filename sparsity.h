//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef GPE_SPARSITY_H
#define GPE_SPARSITY_H

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/fe/mapping_q1.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_tools.h>

namespace gpe
{
//!
//! @tparam dim
//! @param dof_handler
//! @param filename
template <int dim>
void write_dof_locations(const dealii::DoFHandler<dim>& dof_handler, const std::string& filename)
{
    // Mapping from reference element to linear elements, using shape functions of degree 1
    // Combining this with FE_Q of degree 1 yields an isoparametric element
    const std::map<dealii::types::global_dof_index, dealii::Point<dim>> dof_location_map =
        dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<dim>(), dof_handler);

    std::ofstream dof_location_file(filename);
    dealii::DoFTools::write_gnuplot_dof_support_point_info(dof_location_file, dof_location_map);
}

//! Visualize the geometric distribution of degrees of freedom per level
//!
//! @tparam dim
//! @param dof_handler
//! @param level
//! @param filename
template <int dim>
void write_level_vertex_points(const dealii::DoFHandler<dim> &dof_handler,
                               const unsigned int level,
                               const std::string &filename)
{
    const auto &fe = dof_handler.get_fe();
    AssertThrow(fe.dofs_per_vertex == 1 && fe.degree == 1,
                dealii::ExcMessage("This helper assumes FE_Q(1) with one DoF per vertex."));

    std::map<dealii::types::global_dof_index, dealii::Point<dim>> id_to_point;

    for (auto cell = dof_handler.begin(level); cell != dof_handler.end(level); ++cell)
    {
        for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            const auto vid = cell->vertex_index(v);              // stable across levels
            const auto &x  = cell->vertex(v);
            id_to_point[static_cast<dealii::types::global_dof_index>(vid)] = x;
        }
    }

    std::ofstream out(filename);
    dealii::DoFTools::write_gnuplot_dof_support_point_info(out, id_to_point);
}

template <int dim>
dealii::DynamicSparsityPattern
make_sparsity_pattern(const dealii::DoFHandler<dim>& dof_handler,
                      const dealii::AffineConstraints<double>& constraints,
                      bool keep_constrained_dofs = true)
{
    // Create sparsity pattern based on dof numbering
    const unsigned int n = dof_handler.n_dofs();
    dealii::DynamicSparsityPattern dsp(n, n);

    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, keep_constrained_dofs);
    return dsp;
}

template <int dim>
dealii::DynamicSparsityPattern
make_sparsity_pattern_mg(const dealii::DoFHandler<dim>& dof_handler,
                         const dealii::MGConstrainedDoFs& mg_constrained_dofs,
                         unsigned int level,
                         bool keep_constrained_dofs = true)
{
    // Create sparsity pattern based on dof numbering
    const unsigned int n = dof_handler.n_dofs(level);
    dealii::DynamicSparsityPattern dsp(n, n);

    dealii::MGTools::make_sparsity_pattern(dof_handler, dsp,
        level, mg_constrained_dofs.get_level_constraints(level), keep_constrained_dofs);
    return dsp;
}

template <int dim>
dealii::DynamicSparsityPattern
make_interface_sparsity_pattern(const dealii::DoFHandler<dim>& dof_handler,
                                const dealii::MGConstrainedDoFs& mg_constrained_dofs,
                                unsigned int level)
{
    const unsigned int n = dof_handler.n_dofs(level);
    dealii::DynamicSparsityPattern dsp(n, n);

    dealii::MGTools::make_interface_sparsity_pattern(dof_handler, mg_constrained_dofs, dsp, level);
    return dsp;
}

} // namespace gpe

#endif //GPE_SPARSITY_H