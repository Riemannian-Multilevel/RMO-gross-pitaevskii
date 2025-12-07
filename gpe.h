#ifndef GPE_GPE_H
#define GPE_GPE_H

#include <deal.II/numerics/matrix_creator.h>

#include "lac.h"
#include "mesh.h"
#include "dofs.h"
#include "assemble.h"

namespace gpe
{
// TODO: lac_types.h to easily change to different matrix implementation

struct GPE_Options
{
    int dimension;          // dimension of domain
    int n_levels;           // number of levels in finite element discretization
    int degree;             // degree of shape functions
    Ordering order;         // ordering for degrees of freedom
    BoundaryCondition bc;   // problem boundary conditions (dirichlet or neumann)
};

//! @tparam dim
template <int dim>
class GPE
{
public:
    //!
    //! @param left_ Left end-point of the rectangular domain
    //! @param right_ Opposite end-point of the rectangular domain
    //! @param options_ Parameters for GPE problem
    GPE(const Point<dim>& left_, const Point<dim>& right_, const GPE_Options& options_)
    :
        left(left_), right(right_), options(options_),
        // Flag to allow multigrid algorithms
        triangulation(dealii::Triangulation<dim>::limit_level_difference_at_vertices),
        // DoFHandler<> has a deleted assignment operator, so initialize in the constructor
        element(options.degree), dof_handler(triangulation)
    {
        options.dimension = dim;
    }

    void make_rectangle()
    {
        // step 1 - make grid
        // rectangle consisting of precisely one cell
        std::cerr << "Dimension: " << options.dimension << std::endl;
        dealii::GridGenerator::hyper_rectangle(triangulation, left, right);

        // the number of cells increases by a factor of 2^(dim x times)
        // -> n_levels equals the number of refinements + 1
        triangulation.refine_global(options.n_levels-1);

        AssertDimension(options.n_levels, triangulation.n_global_levels());
        std::cerr << "Number of levels: " << triangulation.n_global_levels() << std::endl;
    }

    void dofs()
    {
        // step 2 - degrees of freedom
        distribute_dofs(dof_handler, element, options.order);

        // step 6 - formulate constraints
        constraints = make_boundary(dof_handler, options.bc, {0});
        has_active_constraints = true;
    }

    void dofs_mg()
    {
        // step 2 - degrees of freedom - ordering applied to every level
        distribute_mg_dofs(dof_handler, element, options.order, std::vector<bool>(options.n_levels, true));

        // step 6 - formulate constraints
        mg_constrained_dofs = make_boundary_mg(dof_handler, options.bc, {0});
        has_mg_constraints = true;
    }

    const dealii::DoFHandler<dim>&
        get_dofs() const
    {
        return dof_handler;
    }
    const dealii::MGConstrainedDoFs&
        get_mg_dofs() const
    {
        return mg_constrained_dofs;
    }
    const dealii::Triangulation<dim>&
        get_triangulation() const
    {
        return triangulation;
    }
    const dealii::AffineConstraints<double>&
        get_constraints() const
    {
        if (!has_active_constraints) {
            throw dealii::ExcEmptyObject("GPE::get_constraints(): call dofs() first");
        }
        return constraints;
    }
    const dealii::AffineConstraints<double>&
        get_level_constraints(const unsigned level) const
    {
        if (!has_mg_constraints) {
            throw dealii::ExcEmptyObject("GPE::get_mg_constraints(): call dofs_mg() first");
        }
        return mg_constrained_dofs.get_level_constraints(level);
    }

    GPE_Options get_options() const
    {
        return options;
    }

private:
    // Problem parameters
    Point<dim> left;
    Point<dim> right;
    GPE_Options options;

    // Finite element containers
    dealii::Triangulation<dim>   triangulation; // copy stored by dof_handler
    const dealii::FE_Q<dim>      element;       // copy stored by dof_handler
    dealii::DoFHandler<dim>      dof_handler;

    // Constraints for active level or multigrid
    dealii::AffineConstraints<double> constraints;
    dealii::MGConstrainedDoFs mg_constrained_dofs;

    // Sanity check
    bool has_active_constraints = false;
    bool has_mg_constraints = false;
};

}
#endif //GPE_GPE_H