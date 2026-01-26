#ifndef GPE_DOFS_HH
#define GPE_DOFS_HH

#include "option_types.h"

// step 2 -- dof libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools.h>

namespace gpe
{
using dealii::numbers::invalid_unsigned_int;

//! Renumber degrees of freedom for improved conditioning of system matrix.
//! @tparam dim Dimension of the domain
//! @param dof_handler
//! @param order A valid member of gpe::Ordering
//! @param use_constraints
//! @param reversed_numbering
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
        default:
            throw std::invalid_argument("unknown ordering");
    }
}

template <int dim>
class FeSpace
{
public:
    // Variable element (quads, simplex, ...) taken as argument
    FeSpace(const dealii::Triangulation<dim>& triangulation)
        : dof_handler(triangulation)
    {}
    FeSpace() {}
    FeSpace(const FeSpace&) = delete;
    FeSpace& operator=(const FeSpace&) = delete;

    void setup_dofs(const Ordering order, const dealii::FiniteElement<dim>& element)
    {
        // Distribute degrees of freedom according to (default or other) ordering,
        // such that a basis of V_h can be enumerated in a deterministic way
        // This function stores a copy of the finite element given as argument
        this->dof_handler.distribute_dofs(element);

        // Reorder degrees of freedom for improved conditioning of system matrix
        // (default: order vertices, faces, ... by refinement level)
        if (order != Ordering::DEFAULT) {
            renumber_dofs<dim>(dof_handler, order);
        }
    }

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
    dealii::AffineConstraints<double> constraints;
};

} // namespace gpe

#endif //GPE_DOFS_HH
