#ifndef GPE_DOFS_HH
#define GPE_DOFS_HH

// step 2 -- dof libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_renumbering.h>

// step 16 -- multigrid libraries
#include <deal.II/multigrid/mg_tools.h>

#include <fstream>

namespace gpe
{
// Every ordering should be compatible to (geometric) multigrid
enum class Ordering
{
    DEFAULT,
    RANDOM,
    CUTHILL_MCKEE,
    KING,
    MIN_DEG
};

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
    AssertIndexRange(level, dof_handler.get_triangulation().n_global_levels());
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

//! Create a sparsity pattern from a DoF object
//!
//! @tparam dim Dimension of the domain
//! @param dof_handler
//! @return
template <int dim>
dealii::SparsityPattern
make_sparsity_pattern(const dealii::DoFHandler<dim>& dof_handler)
{
    // Since the bandwidth of the system matrix is unknown beforehand, or
    // with a pessimistic upper bound, we use `DynamicSparsityPattern` to allocate
    // positions and then copy it to CSR format (`SparsityPattern`).
    dealii::DynamicSparsityPattern dynamic_sparsity_pattern(
        dof_handler.n_dofs(), dof_handler.n_dofs());

    dealii::DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);

    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);

    return sparsity_pattern;
}

//!
//! @tparam dim Dimension of the domain
//! @param dof_handler
//! @param level Level in multigrid hierarchy
//! @return
template <int dim>
dealii::SparsityPattern
make_sparsity_pattern_mg(const dealii::DoFHandler<dim>& dof_handler, const unsigned int level)
{
    AssertIndexRange(level, dof_handler.get_triangulation().n_global_levels());
    dealii::DynamicSparsityPattern dynamic_sparsity_pattern(
        dof_handler.n_dofs(level), dof_handler.n_dofs(level));

    // Write the sparsity structure of the matrix belonging to the specified level
    dealii::MGTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern, level);

    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);

    return sparsity_pattern;
}

//! Renumber degrees of freedom for improved conditioning of system matrix.
//!
//! For some methods, the algorithm can be applied to a specified level in a multigrid hierarchy.
//! @tparam dim Dimension of the domain
//! @param dof_handler
//! @param order A valid member of gpe::Ordering
//! @param level Ordering for a specified level in multigrid
template <int dim>
void renumber_dofs(dealii::DoFHandler<dim>& dof_handler, Ordering order = Ordering::CUTHILL_MCKEE, int level = -1)
{
    // TODO: further orderings for multilevel, check level index range
    switch (order) {
    case Ordering::DEFAULT:
        break;

    case Ordering::RANDOM:
        level == -1
            ? dealii::DoFRenumbering::random(dof_handler)
            : dealii::DoFRenumbering::random(dof_handler, level);
        break;

    case Ordering::CUTHILL_MCKEE:
        // TODO: use_constraints for global reordering
        level == -1
            ? dealii::DoFRenumbering::Cuthill_McKee(dof_handler, false, false)
            : dealii::DoFRenumbering::Cuthill_McKee(dof_handler, static_cast<unsigned int>(level), false);
        break;

    case Ordering::KING:
        level == -1
            ? dealii::DoFRenumbering::boost::king_ordering(dof_handler)
            : throw std::logic_error("KING ordering not implemented for multilevel");
        break;

    case Ordering::MIN_DEG:
        level == -1
            ? dealii::DoFRenumbering::boost::minimum_degree(dof_handler)
            : throw std::logic_error("MIN_DEG ordering not implemented for multilevel");
        break;

    default:
        throw std::invalid_argument("unknown ordering");
    }
}

template <int dim>
void distribute_dofs(dealii::DoFHandler<dim>& dof_handler, const dealii::FE_Q<dim>& element,
                     Ordering order = Ordering::DEFAULT)
{
    // Distribute degrees of freedom according to (default or other) ordering,
    // such that a basis of V_h can be enumerated in a deterministic way
    dof_handler.distribute_dofs(element);

    // Reorder degrees of freedom for improved conditioning of system matrix
    // (default: order vertices, faces, ... by refinement level)
    if (order != Ordering::DEFAULT) {
        renumber_dofs<dim>(dof_handler, order);
    }
}

template <int dim>
void distribute_dofs_mg(dealii::DoFHandler<dim>& dof_handler, const dealii::FE_Q<dim>& element,
                        Ordering order = Ordering::DEFAULT,
                        const std::vector<int>& levels = {})
{
    // Distribute degrees of freedom according to (default or other) ordering,
    // such that a basis of V_h can be enumerated in a deterministic way
    dof_handler.distribute_dofs(element);

    // Distribute level degrees of freedom on each level for geometric multigrid
    dof_handler.distribute_mg_dofs();

    // Reorder degrees of freedom for improved conditioning of system matrix
    // (default: order vertices, faces, ... by refinement level)
    if (order != Ordering::DEFAULT) {
        if (levels.empty()) {
            renumber_dofs<dim>(dof_handler, order);
            return;
        }
        // TODO: assumes unique indices in levels[]
        for (int i : levels) {
            renumber_dofs<dim>(dof_handler, order, i);
        }
    }
}

} // namespace gpe

#endif //GPE_DOFS_HH
