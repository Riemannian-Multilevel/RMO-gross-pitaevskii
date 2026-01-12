#ifndef GPE_DOFS_HH
#define GPE_DOFS_HH

#include <optional>

// step 2 -- dof libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools.h>

// step 16 -- multigrid libraries
#include <deal.II/multigrid/mg_tools.h>

namespace gpe
{
using dealii::numbers::invalid_unsigned_int;

// Every ordering should be compatible to (geometric) multigrid
enum class Ordering
{
    DEFAULT,
    RANDOM,
    CUTHILL_MCKEE
};

enum class BoundaryCondition
{
    NEUMANN,
    DIRICHLET,
    ROBIN
};

//! Renumber degrees of freedom for improved conditioning of system matrix.
//!
//! For some methods, the algorithm can be applied to a specified level in a multigrid hierarchy.
//! @tparam dim Dimension of the domain
//! @param dof_handler
//! @param order A valid member of gpe::Ordering
//! @param level Ordering for a specified level in multigrid
//! @param use_constraints
//! @param reversed_numbering
template <int dim>
void renumber_dofs(dealii::DoFHandler<dim>& dof_handler,
                   const Ordering order     = Ordering::CUTHILL_MCKEE,
                   unsigned int level       = invalid_unsigned_int,
                   bool use_constraints     = false,
                   bool reversed_numbering  = false)
{
    if (level != invalid_unsigned_int) {
        AssertIndexRange(level, dof_handler.get_triangulation().n_levels());
    }

    switch (order) {
        case Ordering::DEFAULT:
            break;
        case Ordering::RANDOM:
            if (level == invalid_unsigned_int) {
                dealii::DoFRenumbering::random(dof_handler);
            } else {
                dealii::DoFRenumbering::random(dof_handler, level);
            }
            break;
        case Ordering::CUTHILL_MCKEE:
            if (level == invalid_unsigned_int) {
                dealii::DoFRenumbering::Cuthill_McKee(dof_handler, reversed_numbering, use_constraints);
            } else {
                dealii::DoFRenumbering::Cuthill_McKee(dof_handler, level, reversed_numbering);
            }
            break;
        default:
            throw std::invalid_argument("unknown ordering");
    }
}

template <int dim>
class DiscreteProblem
{
public:
    DiscreteProblem(const dealii::Triangulation<dim>& triangulation, unsigned int degree)
        : dof_handler(triangulation), element(degree)
    {}
    virtual ~DiscreteProblem() = default;

    virtual void setup_dofs(Ordering) = 0;
    virtual void setup_constraints(BoundaryCondition) = 0;

    dealii::DynamicSparsityPattern
    make_sparsity_pattern(unsigned int level = invalid_unsigned_int,
                          bool keep_constrained_dofs = true)
    {
        // Create sparsity pattern based on dof numbering
        const unsigned int n = (level == invalid_unsigned_int)
            ? dof_handler.n_dofs()
            : dof_handler.n_dofs(level);
        dealii::DynamicSparsityPattern dsp(n, n);

        // There is no level-aware `make_sparsity_pattern` in DoFTools
        if (level == invalid_unsigned_int) {
            dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, keep_constrained_dofs);
        } else {
            dealii::MGTools::make_sparsity_pattern(dof_handler, dsp, level, constraints, keep_constrained_dofs);
        }
        return dsp;
    }

    dealii::DynamicSparsityPattern
    make_interface_sparsity_pattern(unsigned int level)
    {
        const unsigned int n = dof_handler.n_dofs(level);
        dealii::DynamicSparsityPattern dsp(n, n);

        dealii::MGTools::make_interface_sparsity_pattern(dof_handler, mg_constrained_dofs, dsp, level);
        return dsp;
    }

protected:
    dealii::DoFHandler<dim> dof_handler;
    const dealii::FE_Q<dim> element;

    dealii::AffineConstraints<double> constraints;
    dealii::MGConstrainedDoFs mg_constrained_dofs;
};


template <int dim>
class DiscreteProblemActive : public DiscreteProblem<dim>
{
public:
    DiscreteProblemActive(const dealii::Triangulation<dim>& triangulation, unsigned int degree)
        : DiscreteProblem<dim>(triangulation, degree)
    {}

    void setup_dofs(Ordering order) override
    {
        // Distribute degrees of freedom according to (default or other) ordering,
        // such that a basis of V_h can be enumerated in a deterministic way
        this->dof_handler.distribute_dofs(this->element);

        // Reorder degrees of freedom for improved conditioning of system matrix
        // (default: order vertices, faces, ... by refinement level)
        if (order != Ordering::DEFAULT) {
            renumber_dofs<dim>(this->dof_handler, order);
        }
    }

    void setup_constraints(BoundaryCondition bounds) override
    {
        dealii::Functions::ZeroFunction<dim> boundary_function(this->dof_handler.get_fe().n_components());

        // Define hanging nodes (optional for global refinement)
        this->constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);

        // Set boundary condition for linear system (after dof distribution)
        if (bounds == BoundaryCondition::DIRICHLET) {
            dealii::VectorTools::interpolate_boundary_values(this->dof_handler,
                0, boundary_function, this->constraints);
        }
        this->constraints.close();
    }
};


// TODO: when using geometric multigrid (eg. as preconditioner, in contrast to multilevel methods),
//       additional constraints are needed -- see step-16 assemble_multigrid()
template <int dim>
class DiscreteProblemMG : public DiscreteProblem<dim>
{
public:
    DiscreteProblemMG(const dealii::Triangulation<dim>& triangulation, unsigned int degree)
        : DiscreteProblem<dim>(triangulation, degree)
    {}

    void setup_dofs(Ordering order) override
    {
        const unsigned int n_levels = this->dof_handler.get_triangulation().n_levels();

        // Distribute degrees of freedom according to (default or other) ordering,
        // such that a basis of V_h can be enumerated in a deterministic way
        this->dof_handler.distribute_dofs(this->element);

        // Distribute level degrees of freedom on each level for geometric multigrid
        this->dof_handler.distribute_mg_dofs();

        // Reorder degrees of freedom for improved conditioning of system matrix
        // (default: order vertices, faces, ... by refinement level)
        if (order != Ordering::DEFAULT) {
            for (unsigned i = 0; i < n_levels; i++) {
                renumber_dofs<dim>(this->dof_handler, order, i);
            }
        }
    }

    void setup_constraints(BoundaryCondition bounds) override
    {
        dealii::Functions::ZeroFunction<dim> boundary_function(this->dof_handler.get_fe().n_components());

        // Define hanging nodes (optional for global refinement)
        this->constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);

        // MG constraints
        this->mg_constrained_dofs.clear();
        this->mg_constrained_dofs.initialize(this->dof_handler);

        // Set boundary condition for linear system (after dof distribution)
        if (bounds == BoundaryCondition::DIRICHLET) {
            dealii::VectorTools::interpolate_boundary_values(this->dof_handler,
                0, boundary_function, this->constraints);

            this->mg_constrained_dofs.make_zero_boundary_constraints(this->dof_handler, {0});
        }
        this->constraints.close();
    }
};

} // namespace gpe

#endif //GPE_DOFS_HH
