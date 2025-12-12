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
#include <deal.II/numerics/vector_tools.h>

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

inline Ordering
select_order(const std::string& order_str)
{
    if (order_str == "DEFAULT") {
        return Ordering::DEFAULT;
    }
    if (order_str == "RANDOM") {
        return Ordering::RANDOM;
    }
    if (order_str == "CUTHILL_MCKEE") {
        return Ordering::CUTHILL_MCKEE;
    }
    if (order_str == "KING") {
        return Ordering::KING;
    }
    if (order_str == "MIN_DEG") {
        return Ordering::MIN_DEG;
    }
    throw std::runtime_error(order_str + ": invalid ordering");
}

enum class BoundaryCondition
{
    NEUMANN,
    DIRICHLET,
    ROBIN
};

inline BoundaryCondition
select_boundary_condition(const std::string& boundary_str)
{
    if (boundary_str == "NEUMANN") {
        return BoundaryCondition::NEUMANN;
    }
    if (boundary_str == "DIRICHLET") {
        return BoundaryCondition::DIRICHLET;
    }
    if (boundary_str == "ROBIN") {
        return BoundaryCondition::ROBIN;
    }
    throw std::runtime_error(boundary_str + ": invalid boundary condition");
}

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

template <typename... Args>
dealii::SparsityPattern
copy_pattern(const dealii::DynamicSparsityPattern& dsp)
{
    dealii::SparsityPattern sp;
    sp.copy_from(dsp);
    return sp;
}

//! Create a sparsity pattern from a DoF object
//!
//! @tparam dim Dimension of the domain
//! @param dof_handler
//! @param level
//! @return
template <int dim>
dealii::SparsityPattern
make_sparsity_pattern(const dealii::DoFHandler<dim>& dof_handler,
    const dealii::AffineConstraints<double>& constraints = {})
{
    const unsigned int n = dof_handler.n_dofs();
    dealii::DynamicSparsityPattern dynamic_sparsity_pattern(n, n);

    dealii::DoFTools::make_sparsity_pattern(dof_handler,
        dynamic_sparsity_pattern, constraints, false);

    return copy_pattern(dynamic_sparsity_pattern);
}

template <int dim>
dealii::SparsityPattern
make_sparsity_pattern_mg(const dealii::DoFHandler<dim>& dof_handler,
    unsigned int level, const dealii::AffineConstraints<double>& constraints = {})
{
    const unsigned int n = dof_handler.n_dofs(level);
    dealii::DynamicSparsityPattern dynamic_sparsity_pattern(n, n);

    dealii::MGTools::make_sparsity_pattern(dof_handler,
        dynamic_sparsity_pattern, level, constraints, false);

    return copy_pattern(dynamic_sparsity_pattern);
}

//! Renumber degrees of freedom for improved conditioning of system matrix.
//!
//! For some methods, the algorithm can be applied to a specified level in a multigrid hierarchy.
//! @tparam dim Dimension of the domain
//! @param dof_handler
//! @param order A valid member of gpe::Ordering
//! @param level Ordering for a specified level in multigrid
template <int dim>
void renumber_dofs(dealii::DoFHandler<dim>& dof_handler,
    const Ordering order = Ordering::CUTHILL_MCKEE,
    unsigned int level = dealii::numbers::invalid_unsigned_int)
{
    if (level != dealii::numbers::invalid_unsigned_int) {
        AssertIndexRange(level, dof_handler.get_triangulation().n_levels());
    }
    // TODO: further orderings for multilevel
    switch (order) {
        case Ordering::DEFAULT:
            break;

        case Ordering::RANDOM:
            level == dealii::numbers::invalid_unsigned_int
                ? dealii::DoFRenumbering::random(dof_handler)
                : dealii::DoFRenumbering::random(dof_handler, level);
            break;

        case Ordering::CUTHILL_MCKEE:
            // TODO: use_constraints for global reordering
            level == dealii::numbers::invalid_unsigned_int
                ? dealii::DoFRenumbering::Cuthill_McKee(dof_handler, false, false)
                : dealii::DoFRenumbering::Cuthill_McKee(dof_handler, static_cast<unsigned int>(level), false);
            break;

        case Ordering::KING:
            level == dealii::numbers::invalid_unsigned_int
                ? dealii::DoFRenumbering::boost::king_ordering(dof_handler)
                : throw std::logic_error("KING ordering not implemented for multilevel");
            break;

        case Ordering::MIN_DEG:
            level == dealii::numbers::invalid_unsigned_int
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
void distribute_mg_dofs(dealii::DoFHandler<dim>& dof_handler, const dealii::FE_Q<dim>& element,
                        Ordering order = Ordering::DEFAULT,
                        const std::vector<bool>& levels = {})
{
    if (levels.size()) {
        AssertDimension(levels.size(), dof_handler.get_triangulation().n_levels());
    }
    // Distribute degrees of freedom according to (default or other) ordering,
    // such that a basis of V_h can be enumerated in a deterministic way
    dof_handler.distribute_dofs(element);

    // Distribute level degrees of freedom on each level for geometric multigrid
    dof_handler.distribute_mg_dofs();

    // Reorder degrees of freedom for improved conditioning of system matrix
    // (default: order vertices, faces, ... by refinement level)
    if (order != Ordering::DEFAULT) {
        // A level is either reordered, or not (bool vector)
        for (unsigned i = 0; i < levels.size(); i++) {
            if (levels[i])
                renumber_dofs<dim>(dof_handler, order, i);
        }
        if (levels.empty()) {
            renumber_dofs<dim>(dof_handler, order);
        }
    }
}

//!
//! @tparam dim Domain dimension
//! @param dof_handler
//! @param bc Boundary condition to be applied
//! @param dirichlet_ids Boundary components for Dirichlet conditions
//! @return
template <int dim>
dealii::AffineConstraints<double>
make_boundary(dealii::DoFHandler<dim>& dof_handler, BoundaryCondition bc,
    const std::set<dealii::types::boundary_id>& dirichlet_ids = {0})
{
    dealii::AffineConstraints<double> constraints;
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    dealii::Functions::ZeroFunction<dim> boundary_function(dof_handler.get_fe().n_components());

    switch (bc) {
        case BoundaryCondition::NEUMANN:
            // Natural boundary conditions
            break;

        case BoundaryCondition::DIRICHLET:
            // Dirichlet boundary (zero-valued)
            for (const auto id: dirichlet_ids) {
                dealii::VectorTools::interpolate_boundary_values(dof_handler, id, boundary_function, constraints);
            }
            break;
        default:
            throw std::invalid_argument("Unknown boundary condition");
    }
    return constraints;
}

//!
//! @tparam dim Domain dimension
//! @param dof_handler
//! @param bc Boundary condition to be applied
//! @param dirichlet_ids Boundary components for Dirichlet conditions
//! @return
template <int dim>
dealii::MGConstrainedDoFs
make_boundary_mg(dealii::DoFHandler<dim>& dof_handler, const BoundaryCondition bc,
    const std::set<dealii::types::boundary_id>& dirichlet_ids = {0})
{
    dealii::MGConstrainedDoFs mg_constrained_dofs;
    mg_constrained_dofs.initialize(dof_handler);

    switch (bc) {
        case BoundaryCondition::NEUMANN:
            // Natural boundary conditions, hanging nodes only
            break;

        case BoundaryCondition::DIRICHLET:
            // Dirichlet boundary (zero-valued)
            // TODO: makes sense from a residual perspective; general case?
            mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_ids);
            break;

        default:
            throw std::invalid_argument("Unknown boundary condition");
    }
    return mg_constrained_dofs;
}

} // namespace gpe

#endif //GPE_DOFS_HH
