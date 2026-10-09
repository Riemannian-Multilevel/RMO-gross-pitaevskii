//
// Created by Ferdinand Vanmaele on 09.10.26.
//

#ifndef RMO_FE_UTIL_H
#define RMO_FE_UTIL_H

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_tools.h>

/**
 * @file
 * @brief Sparsity patterns of the active and multigrid level matrices, and Gnuplot output of DoF and vertex
 * locations for debugging.
 */
namespace rmo::fe
{

/** @brief Writes the support points of all DoFs, mapped with @p mapping, to @p filename (Gnuplot format). */
template <int dim>
void write_dof_locations(const dealii::DoFHandler<dim>& dof_handler,
                         const std::string& filename,
                         const dealii::Mapping<dim>& mapping = dealii::MappingQ1<dim>())
{
    // Mapping from reference element to linear elements, using shape functions of degree 1
    // Combining this with FE_Q of degree 1 yields an isoparametric element
    const std::map<dealii::types::global_dof_index, dealii::Point<dim>> dof_location_map =
        dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler);

    std::ofstream dof_location_file(filename);
    dealii::DoFTools::write_gnuplot_dof_support_point_info(dof_location_file, dof_location_map);
}

/**
 * @brief Writes the vertices of the cells on mesh @p level to @p filename.
 * Requires DoFs at the vertices only, i.e. `FE_Q(1)`; throws otherwise.
 */
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

/**
 * @brief Sparsity pattern of the matrices on the active mesh, condensed with @p constraints;
 * @p keep_constrained_dofs keeps the entries of constrained rows and columns.
 */
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

/**
 * @brief Sparsity pattern of the matrices on multigrid @p level, condensed with its level constraints;
 * @p keep_constrained_dofs keeps the entries of constrained rows and columns.
 */
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

/** @brief Sparsity pattern of the interface (edge) matrices of multigrid @p level, for local refinement. */
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

} // namespace rmo::fe

#endif //RMO_FE_UTIL_H
