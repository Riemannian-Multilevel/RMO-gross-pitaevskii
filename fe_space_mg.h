//
// Created by Ferdinand Vanmaele on 28.01.26.
//
#ifndef GPE_FE_SPACE_MG_H
#define GPE_FE_SPACE_MG_H

#include "option_types.h"

// step 2 -- dof libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools.h>

namespace gpe
{

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
        default:
            throw std::invalid_argument("unknown ordering");
    }
}


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
                renumber_dofs_mg<dim>(dof_handler, order, i);
            }
        }
    }

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

            mg_constraints.make_zero_boundary_constraints(dof_handler, {0});
        }
        constraints.close();
    }

    // multigrid accessors
    const dealii::AffineConstraints<double>& get_level_constraints(unsigned int level) const {
        return mg_constraints.get_level_constraints(level);
    }
    const dealii::MGConstrainedDoFs& get_mg_dofs() const {
        return mg_constraints;
    }

    // active level accessors
    const dealii::DoFHandler<dim>& get_dofs() const {
        return dof_handler;
    }
    const dealii::FiniteElement<dim>& get_fe() const {
        return dof_handler.get_fe();
    }
    unsigned int n_dofs() const {
        return dof_handler.n_dofs();
    }
    const dealii::AffineConstraints<double>& get_constraints() const{
        return constraints;
    }

private:
    dealii::DoFHandler<dim> dof_handler;
    dealii::MGConstrainedDoFs mg_constraints;
    dealii::AffineConstraints<double> constraints;
};

}
#endif //GPE_FE_SPACE_MG_H