#ifndef GPE_GPE_H
#define GPE_GPE_H

#include <deal.II/numerics/matrix_creator.h>

#include "lac.h"
#include "mesh.h"
#include "dofs.h"

namespace gpe
{
// TODO: lac_types.h to easily change to different matrix implementation

struct GPE_Options
{
    int dimension;          // dimension of domain
    int n_levels;           // number of levels for global refinement
    int degree;             // degree of shape functions
    double radius;          // radius of the cube (square, line) domain
    double beta;            // factor for the non-linear term in GPE
    Ordering order;         // ordering for degrees of freedom
    BoundaryCondition bc;   // problem boundary conditions (dirichlet or neumann)
};

//! Wrapper object for GPE problem, which encodes order/dependencies of used methods
//! @tparam dim
template <int dim>
class GPE
{
public:
    // TODO: simplify to radius / hyper cube (later: allow passing in arbitrary meshes?)
    GPE(const GPE_Options& options_)
    :
        options(options_),
        // Flag to allow multigrid algorithms
        triangulation(dealii::Triangulation<dim>::limit_level_difference_at_vertices),
        // DoFHandler<> has a deleted assignment operator, so initialize in the constructor
        element(options.degree), dof_handler(triangulation)
    {
        // TODO: check values of options for validity
        options.dimension = dim;
        dirichlet_boundary_ids = {0};
    }

    void make_grid()
    {
        // step 1 - regularly refined mesh
        make_cube(triangulation, options.radius, options.n_levels);
        has_grid = true;

        std::cerr << "Number of levels: " << triangulation.n_global_levels() << std::endl;
        std::cerr << "Number of vertices: " << triangulation.n_vertices() << std::endl;
    }

    void plot_grid(const std::string& prefix) const
    {
        const std::string filename = prefix + "_" + std::to_string(dim) + "{}";
        if (dim == 2) {
            grid2file(filename + ".svg", triangulation, dealii::GridOut::OutputFormat::svg);
        }
        grid2file(filename + ".gnuplot", triangulation, dealii::GridOut::OutputFormat::gnuplot);
    }

    void dofs()
    {
        if (!has_grid) {
            throw dealii::ExcEmptyObject("GPE::dofs(): call make_grid() or make_grid_graded() first");
        }
        // step 2 - degrees of freedom
        distribute_dofs(dof_handler, element, options.order);

        // step 6 - formulate constraints
        constraints = make_boundary(dof_handler, options.bc, dirichlet_boundary_ids);
        constraints.close();
        has_active_constraints = true;
    }

    void dofs_mg()
    {
        if (!has_grid) {
            throw dealii::ExcEmptyObject("GPE::dofs_mg(): call make_grid() first");
        }
        // step 2 - degrees of freedom - ordering applied to every level
        distribute_mg_dofs(dof_handler, element, options.order);

        // step 6 - formulate constraints
        constraints = make_boundary(dof_handler, options.bc, dirichlet_boundary_ids);
        constraints.close();
        has_active_constraints = true;

        // step 16 - formulate multigrid constraints
        mg_constrained_dofs = make_boundary_mg(dof_handler, options.bc, dirichlet_boundary_ids);
        has_mg_constraints  = true;
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
    unsigned int n_levels() const
    {
        return triangulation.n_levels();
    }
    const dealii::AffineConstraints<double>&
        get_constraints() const
    {
        if (!has_active_constraints)
            throw dealii::ExcEmptyObject("GPE::get_constraints(): call dofs() or dofs_mg() first");

        return constraints;
    }
    const dealii::AffineConstraints<double>&
        get_level_constraints(const unsigned level) const
    {
        if (!has_mg_constraints)
            throw dealii::ExcEmptyObject("GPE::get_mg_constraints(): call dofs_mg() first");

        return mg_constrained_dofs.get_level_constraints(level);
    }

private:
    // Problem parameters
    GPE_Options options;

    // Finite element containers
    dealii::Triangulation<dim> triangulation; // copy stored by dof_handler
    const dealii::FE_Q<dim> element;          // copy stored by dof_handler
    dealii::DoFHandler<dim> dof_handler;

    // Constraints for active level or multigrid
    std::set<dealii::types::boundary_id> dirichlet_boundary_ids;
    dealii::AffineConstraints<double> constraints;
    dealii::MGConstrainedDoFs mg_constrained_dofs;

    // Checks & balances
    bool has_active_constraints = false;
    bool has_mg_constraints = false;
    bool has_grid = false;
};

}
#endif //GPE_GPE_H