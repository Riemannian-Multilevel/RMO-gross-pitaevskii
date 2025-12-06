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

template <typename T>
struct GPE_Mass
{
    // avoid issues with deleted copy/move constructors of SparsityPattern
    // SparseMatrix has defined move constructors, so we store them directly
    explicit GPE_Mass(const dealii::SparsityPattern& sparsity_)
        : sparsity(std::make_shared<dealii::SparsityPattern>())
    {
        // make an internal copy of the sparsity pattern
        sparsity->copy_from(sparsity_);

        // now initialize all matrices with this *owned* pattern
        M.reinit(*sparsity);
        A_0.reinit(*sparsity);
        Mpp.reinit(*sparsity);
    }

    SparseMatrix<T> M;
    SparseMatrix<T> A_0;
    SparseMatrix<T> Mpp;

private:
    std::shared_ptr<SparsityPattern> sparsity;
};

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
    GPE(const int n_levels_, const int degree, const Point<dim>& left_, const Point<dim>& right_,
        const Ordering order_ = Ordering::DEFAULT)
    :
        n_levels(n_levels_), order(order_), left(left_), right(right_),
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

    dealii::AffineConstraints<double>
    boundary(BoundaryCondition condition, std::set<dealii::types::boundary_id> dirichlet_ids = {0}) const
    {
        dealii::AffineConstraints<double> constraints;
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        switch (condition) {
            case BoundaryCondition::NEUMANN:
                // Natural boundary conditions
                break;

            case BoundaryCondition::DIRICHLET:
                // Dirichlet boundary (zero-valued)
                dealii::Functions::ZeroFunction<dim> boundary_function(element.n_components());

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

    dealii::MGConstrainedDoFs
    boundary_mg(BoundaryCondition condition, std::set<dealii::types::boundary_id> dirichlet_ids = {0}) const
    {
        dealii::MGConstrainedDoFs mg_constrained_dofs;
        mg_constrained_dofs.initialize(dof_handler);

        switch (condition) {
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

    void dofs()
    {
        // step 2 - degrees of freedom
        distribute_dofs(dof_handler, element, order);
    }

    void dofs_mg()
    {
        // step 2 - degrees of freedom - ordering applied to every level
        std::vector<bool> levels(n_levels, true);

        // DoFHandler::distribute_dofs, DoFHandler::distribute_mg_dofs
        distribute_mg_dofs(dof_handler, element, order, levels);
    }

    template <typename Function>
    GPE_Mass<double>
    assemble(Function&& V, const dealii::AffineConstraints<double>& constraints,
        unsigned int level = dealii::numbers::invalid_unsigned_int) const
    {
        // In-place construction of sparsity pattern
        // Handles multigrid transparently
        GPE_Mass<double> Mass(make_sparsity_pattern(dof_handler, level));

        // Compute values of mass matrix
        assemble_mass(Mass.M, dof_handler, constraints, level);

        // Compute values of stiffness + weighed mass matrix
        assemble_A0(Mass.A_0, dof_handler, V, constraints, level);

        return Mass; // XXX: or store in class object
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
    // Problem parameters
    int n_levels;
    int dimension;
    Ordering order;

    // Rectangle bounds
    Point<dim> left;
    Point<dim> right;

    // Finite element containers
    dealii::Triangulation<dim>   triangulation; // copy stored by dof_handler
    const dealii::FE_Q<dim>      element;       // copy stored by dof_handler
    dealii::DoFHandler<dim>      dof_handler;
};

}
#endif //GPE_GPE_H