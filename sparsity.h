//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef GPE_SPARSITY_H
#define GPE_SPARSITY_H

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/sparse_matrix.h>

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

} // namespace gpe

#endif //GPE_SPARSITY_H