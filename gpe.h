#ifndef GPE_GPE_H
#define GPE_GPE_H

#include "lac.h"
#include "mesh.h"
#include "dofs.h"
#include "assemble.h"

namespace gpe
{
// TODO: lac_types.h to easily change to different matrix implementation

template <typename T>
struct GPE_Mass
{
    explicit GPE_Mass(const dealii::SparsityPattern &sparsity_)
        // avoid issues with deleted copy/move constructors of SparsityPattern
        // SparseMatrix has defined move constructors, so we store them directly
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

//!
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
    std::vector<GPE_Mass<double> >
    assemble(Function&& V) const
    {
        // Use vector with one entry for interoperability with assemble_mg()
        std::vector<GPE_Mass<double> > Mass_v;
        // In-place construction of sparsity pattern
        Mass_v.emplace_back(make_sparsity_pattern(dof_handler));

        // Compute values of mass matrix
        assemble_mass(Mass_v[0].M, dof_handler);

        // Compute values of stiffness + weighed mass matrix
        assemble_A0(Mass_v[0].A_0, dof_handler, V);

        return Mass_v; // XXX: or store in class object
    }

    template <typename Function>
    std::vector<GPE_Mass<double> >
    assemble_mg(Function&& V) const
    {
        // Structure of arrays for multigrid
        //std::vector<SparseMatrix<double> > mass_matrix_weighed_v(n_levels);
        //std::vector<SparseMatrix<double> > stiffness_matrix_v(n_levels);
        std::vector<GPE_Mass<double> > Mass_v;

        // Iterate over cells in every level of the hierarchy
        for (int level = 0; level < n_levels; level++) {
            Mass_v.emplace_back(make_sparsity_pattern_mg(level, dof_handler));
        }
        for (int level = 0; level < n_levels; level++) {
            // Compute values of mass matrix for level
            assemble_mass(Mass_v[level].M, dof_handler, level);

            // Compute values of stiffness + weighed mass matrix
            assemble_A0(Mass_v[level].A_0, dof_handler, V, level);
        }
        return Mass_v; // XXX: or store in class (mg specialized?) object
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