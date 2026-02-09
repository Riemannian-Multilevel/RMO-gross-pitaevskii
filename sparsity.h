//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef GPE_SPARSITY_H
#define GPE_SPARSITY_H

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_tools.h>

namespace gpe
{
/**
 * @brief Writes the physical locations of all degrees of freedom to a file.
 *
 * This function computes the support points for all DoFs using the specified mapping
 * and writes them to a file in a format suitable for Gnuplot. This is useful for
 * visual debugging of finite element node distributions, especially for high-order
 * or isoparametric elements.
 *
 * @tparam dim The spatial dimension.
 * @param[in] dof_handler The DoFHandler managing the degrees of freedom.
 * @param[in] filename The output filename.
 * @param[in] mapping The mapping to use for computing real-world coordinates from
 * reference cell points (defaults to Q1 mapping).
 */
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
 * @brief Visualizes the geometric distribution of vertices on a specific mesh level.
 *
 * Iterates over all cells on the given refinement level and extracts the vertex locations.
 * This is primarily used to visualize the grid hierarchy in Multigrid contexts.
 *
 * @note This function explicitly requires that the Finite Element matches the vertices
 * (i.e., linear Lagrange elements, FE_Q(1)). It will throw an exception if used with
 * higher-order elements.
 *
 * @tparam dim The spatial dimension.
 * @param[in] dof_handler The DoFHandler.
 * @param[in] level The mesh refinement level to visualize.
 * @param[in] filename The output filename.
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
 * @brief Constructs a dynamic sparsity pattern for the active system matrix.
 *
 * Helper function that initializes a `DynamicSparsityPattern` with the correct size
 * and coupling information derived from the DoFHandler and constraints.
 *
 * @tparam dim The spatial dimension.
 * @param[in] dof_handler The DoFHandler.
 * @param[in] constraints Constraints object (used to determine entries for hanging nodes).
 * @param[in] keep_constrained_dofs If true, entries are kept in the sparsity pattern
 * even for constrained DoFs. This is usually required if you plan to write a constant on the
 * diagonal and 0.0 off-diagonal for constrained rows.
 * @return An initialized DynamicSparsityPattern.
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
 * @brief Constructs a sparsity pattern for a specific Multigrid level matrix.
 *
 * Used to build the level matrices (smoothers) in geometric multigrid. It uses
 * `MGTools` to account for the specific connectivity on a single level of the hierarchy.
 *
 * @tparam dim The spatial dimension.
 * @param[in] dof_handler The DoFHandler.
 * @param[in] mg_constrained_dofs The MG constraint handler (provides level-specific constraints).
 * @param[in] level The multigrid level index.
 * @param[in] keep_constrained_dofs Whether to keep entries for constrained DoFs.
 * @return An initialized DynamicSparsityPattern for the specified level.
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

/**
 * @brief Constructs a sparsity pattern for the interface between multigrid levels.
 *
 * This is required for flux-correction or edge matrices in complex Multigrid schemes.
 * It determines the sparsity pattern required to couple DoFs at the interface of
 * refined and unrefined cells.
 *
 * @tparam dim The spatial dimension.
 * @param[in] dof_handler The DoFHandler.
 * @param[in] mg_constrained_dofs The MG constraint handler.
 * @param[in] level The multigrid level index.
 * @return An initialized DynamicSparsityPattern for the interface.
 */
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