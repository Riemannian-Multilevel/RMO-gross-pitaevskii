#ifndef GPE_GPE_H
#define GPE_GPE_H

#include <deal.II/numerics/matrix_creator.h>

#include "lac.h"
#include "mesh.h"
#include "dofs.h"
#include "assemble.h"
#include <deal.II/numerics/vector_tools.h>

namespace gpe
{
// TODO: lac_types.h to easily change to different matrix implementation

// TODO: merge parts with main.cc
//! @tparam dim
template <int dim>
class GPE
{
public:
    //!
    //! @param n_levels_ Number of times the mesh is refined
    //! @param degree Degree of the Lagrange finite element
    //! @param left_ Left end-point of the rectangular domain
    //! @param right_ Opposite end-point of the rectangular domain
    //! @param order_ Ordering used for degrees of freedom
    GPE(const Point<dim>& left_, const Point<dim>& right_, const int n_levels_, const int degree,
        const Ordering order_ = Ordering::DEFAULT)
    :
        left(left_), right(right_), n_levels(n_levels_), order(order_),
        // Flag to allow multigrid algorithms
        triangulation(dealii::Triangulation<dim>::limit_level_difference_at_vertices),
        // DoFHandler<> has a deleted assignment operator, so initialize in the constructor
        element(degree), dof_handler(triangulation)
    {
        dimension = dim;
    }

    void make_rectangle()
    {
        // step 1 - make grid
        // rectangle consisting of precisely one cell
        std::cerr << "Dimension: " << dimension << std::endl;
        dealii::GridGenerator::hyper_rectangle(triangulation, left, right);

        // the number of cells increases by a factor of 2^(dim x times)
        // -> n_levels equals the number of refinements + 1
        triangulation.refine_global(n_levels-1);

        AssertDimension(n_levels, triangulation.n_levels());
        std::cerr << "Number of levels: " << triangulation.n_levels() << std::endl;
    }

    void dofs()
    {
        // step 2 - degrees of freedom
        distribute_dofs(dof_handler, element, order);
    }

    void dofs_mg()
    {
        // step 2 - degrees of freedom - ordering applied to every level
        distribute_mg_dofs(dof_handler, element, order, std::vector<bool>(n_levels, true));
    }

    SparsityPattern
    sparsity(unsigned int level = dealii::numbers::invalid_unsigned_int) const
    {
        return make_sparsity_pattern(dof_handler, level);
    }

    // TODO: move to dofs.h
    dealii::AffineConstraints<double>
    boundary(BoundaryCondition condition, const std::set<dealii::types::boundary_id>& dirichlet_ids = {0}) const
    {
        dealii::AffineConstraints<double> constraints;
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        dealii::Functions::ZeroFunction<dim> boundary_function(element.n_components());

        switch (condition) {
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
        constraints.close();
        return constraints;
    }

    // TODO: move to dofs.h
    dealii::MGConstrainedDoFs
    boundary_mg(BoundaryCondition bc, const std::set<dealii::types::boundary_id>& dirichlet_ids = {0}) const
    {
        dealii::MGConstrainedDoFs mg_constrained_dofs;
        mg_constrained_dofs.initialize(dof_handler);

        switch (bc) {
            case BoundaryCondition::NEUMANN:
                // Natural boundary conditions, hanging nodes only
                break;

            case BoundaryCondition::DIRICHLET:
                // Dirichlet boundary (zero-valued)
                mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_ids);
                break;

            default:
                throw std::invalid_argument("Unknown boundary condition");
        }
        return mg_constrained_dofs;
    }

    const dealii::DoFHandler<dim>& get_dof() const
    {
        return dof_handler;
    }
    const dealii::Triangulation<dim>& get_triangulation() const
    {
        return triangulation;
    }

private:
    // Rectangle bounds
    Point<dim> left;
    Point<dim> right;

    // Problem parameters
    int n_levels;
    int dimension;
    Ordering order;

    // Finite element containers
    dealii::Triangulation<dim>   triangulation; // copy stored by dof_handler
    const dealii::FE_Q<dim>      element;       // copy stored by dof_handler
    dealii::DoFHandler<dim>      dof_handler;
};

}
#endif //GPE_GPE_H