#ifndef RMO_FE_SPACE_H
#define RMO_FE_SPACE_H

#include <rmo/option_types.h>

// step 2 -- dof libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools.h>


/**
 * @file
 * @brief Finite element spaces: DoF distribution, renumbering and constraints (hanging nodes, homogeneous
 * Dirichlet values on boundary id 0), for the active mesh (FeSpace) and for geometric multigrid
 * (FeSpaceMG).
 *
 * DoF renumbering reduces the bandwidth of the system matrices: Ordering::CUTHILL_MCKEE, KING and MIN_DEG,
 * RANDOM (for testing), or DEFAULT (no renumbering).
 */
namespace rmo::fe
{
using dealii::numbers::invalid_unsigned_int;

/**
 * @brief Renumbers the DoFs of the active mesh with @p order; with @p use_constraints, the connectivity graph
 * respects the constraints, and @p reversed_numbering reverses the ordering (for Cuthill-McKee, this usually
 * gives a smaller profile).
 *
 * @throws std::invalid_argument for an unknown ordering.
 */
template <int dim>
void renumber_dofs(dealii::DoFHandler<dim>& dof_handler,
                   const Ordering order     = Ordering::CUTHILL_MCKEE,
                   bool use_constraints     = false,
                   bool reversed_numbering  = false)
{
    switch (order) {
        case Ordering::DEFAULT:
            break;
        case Ordering::RANDOM:
            dealii::DoFRenumbering::random(dof_handler);
            break;
        case Ordering::CUTHILL_MCKEE:
            dealii::DoFRenumbering::Cuthill_McKee(dof_handler, reversed_numbering, use_constraints);
            break;
        case Ordering::KING:
            dealii::DoFRenumbering::boost::king_ordering(dof_handler, reversed_numbering, use_constraints);
            break;
        case Ordering::MIN_DEG:
            dealii::DoFRenumbering::boost::minimum_degree(dof_handler, reversed_numbering, use_constraints);
            break;
        default:
            throw std::invalid_argument("unknown ordering");
    }
}

/**
 * @brief Renumbers the DoFs of multigrid @p level with @p order, see renumber_dofs().
 * @throws std::invalid_argument for KING and MIN_DEG, which have no level-wise variant in deal.II.
 */
template <int dim>
void renumber_dofs_mg(dealii::DoFHandler<dim>& dof_handler, unsigned int level,
                      const Ordering order = Ordering::CUTHILL_MCKEE,
                      bool reversed_numbering = false)
{
    switch (order) {
        case Ordering::DEFAULT:
            break;
        case Ordering::RANDOM:
            dealii::DoFRenumbering::random(dof_handler, level);
            break;
        case Ordering::CUTHILL_MCKEE:
            dealii::DoFRenumbering::Cuthill_McKee(dof_handler, level, reversed_numbering);
            break;
        case Ordering::KING:
        case Ordering::MIN_DEG:
            // The boost-graph renumberings have no level-wise variants in deal.II.
            throw std::invalid_argument("king/min_deg orderings unavailable for multigrid level renumbering");
        default:
            throw std::invalid_argument("unknown ordering");
    }
}

/** @brief Finite element space on the active mesh: construct, then call setup_dofs() and setup_constraints(). */
template <int dim>
class FeSpace
{
public:
    FeSpace(const dealii::Triangulation<dim>& triangulation)
        : dof_handler(triangulation)
    {}

    FeSpace() {}
    FeSpace(const FeSpace&) = delete;
    FeSpace& operator=(const FeSpace&) = delete;

    /** @brief Distributes the DoFs of @p element (copied) and renumbers them with @p order. */
    void setup_dofs(const Ordering order, const dealii::FiniteElement<dim>& element)
    {
        // Distribute degrees of freedom according to (default or other) ordering,
        // such that a basis of V_h can be enumerated in a deterministic way
        // This function stores a copy of the finite element given as argument
        dof_handler.distribute_dofs(element);

        // Reorder degrees of freedom for improved conditioning of system matrix
        // (default: order vertices, faces, ... by refinement level)
        if (order != Ordering::DEFAULT) {
            renumber_dofs<dim>(dof_handler, order);
        }
    }

    /** @brief Hanging-node constraints and, for @p bounds = DIRICHLET, homogeneous boundary values. */
    void setup_constraints(const BoundaryCondition bounds)
    {
        dealii::Functions::ZeroFunction<dim> boundary_function(dof_handler.get_fe().n_components());

        // Define hanging nodes (optional for global refinement)
        constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        // Set boundary condition for linear system (after dof distribution)
        if (bounds == BoundaryCondition::DIRICHLET) {
            dealii::VectorTools::interpolate_boundary_values(dof_handler,
                0, boundary_function, constraints);
        }
        constraints.close();
    }

    /** @return Const reference to the underlying DoFHandler. */
    const dealii::DoFHandler<dim>& get_dofs() const {
        return dof_handler;
    }

    /** @return Const reference to the FiniteElement used. */
    const dealii::FiniteElement<dim>& get_fe() const {
        return dof_handler.get_fe();
    }

    /** @return The global number of degrees of freedom. */
    unsigned int n_dofs() const {
        return dof_handler.n_dofs();
    }

    /** @return Constraints of the active mesh (hanging nodes and boundary values). */
    const dealii::AffineConstraints<double>& get_constraints() const{
        return constraints;
    }

private:
    dealii::DoFHandler<dim> dof_handler;
    dealii::AffineConstraints<double> constraints;
};


/** @brief FeSpace with level DoFs and level constraints (`dealii::MGConstrainedDoFs`) for geometric multigrid. */
template <int dim>
class FeSpaceMG
{
public:
    FeSpaceMG(const dealii::Triangulation<dim>& triangulation)
        : dof_handler(triangulation)
    {}

    FeSpaceMG() {}
    FeSpaceMG(const FeSpaceMG&) = delete;
    FeSpaceMG& operator=(const FeSpaceMG&) = delete;

    /** @brief Distributes the active and level DoFs of @p element and renumbers every level with @p order. */
    void setup_dofs(const Ordering order, const dealii::FiniteElement<dim>& element)
    {
        const unsigned int n_levels = dof_handler.get_triangulation().n_levels();

        // Distribute degrees of freedom according to (default or other) ordering,
        // such that a basis of V_h can be enumerated in a deterministic way
        dof_handler.distribute_dofs(element);

        // Distribute level degrees of freedom on each level for geometric multigrid
        dof_handler.distribute_mg_dofs();

        // Reorder degrees of freedom for improved conditioning of system matrix
        // (default: order vertices, faces, ... by refinement level)
        if (order != Ordering::DEFAULT) {
            for (unsigned i = 0; i < n_levels; i++) {
                renumber_dofs_mg<dim>(dof_handler, i, order);
            }
        }
    }

    /** @brief Constraints of the active mesh as in FeSpace, and the level constraints (refinement edges;
     *  boundary for DIRICHLET). */
    void setup_constraints(const BoundaryCondition bounds)
    {
        dealii::Functions::ZeroFunction<dim> boundary_function(dof_handler.get_fe().n_components());

        // Define hanging nodes (optional for global refinement)
        constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        // MG constraints
        mg_constraints.clear();
        mg_constraints.initialize(dof_handler);

        // Set boundary condition for linear system (after dof distribution)
        if (bounds == BoundaryCondition::DIRICHLET) {
            dealii::VectorTools::interpolate_boundary_values(dof_handler,
                0, boundary_function, constraints);

            // Apply Zero BCs to MG levels (essential for defect correction in MG)
            mg_constraints.make_zero_boundary_constraints(dof_handler, {0});
        }
        constraints.close();
    }

    /** @return Constraints of multigrid @p level. */
    const dealii::AffineConstraints<double>& get_level_constraints(unsigned int level) const {
        return mg_constraints.get_level_constraints(level);
    }

    /** @return Const reference to the Multigrid constraint handler. */
    const dealii::MGConstrainedDoFs& get_mg_dofs() const {
        return mg_constraints;
    }

    /** @return Const reference to the underlying DoFHandler. */
    const dealii::DoFHandler<dim>& get_dofs() const {
        return dof_handler;
    }

    /** @return Const reference to the FiniteElement used. */
    const dealii::FiniteElement<dim>& get_fe() const {
        return dof_handler.get_fe();
    }

    /** @return The global number of degrees of freedom. */
    unsigned int n_dofs() const {
        return dof_handler.n_dofs();
    }

    /** @return Constraints of the active mesh (hanging nodes and boundary values). */
    const dealii::AffineConstraints<double>& get_constraints() const{
        return constraints;
    }

private:
    dealii::DoFHandler<dim> dof_handler;
    dealii::MGConstrainedDoFs mg_constraints;
    dealii::AffineConstraints<double> constraints;
};

} // namespace rmo::fe
#endif //RMO_FE_SPACE_H
