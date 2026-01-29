//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "assemble.h"
#include "grid.h"
#include "sparsity.h"
#include "fe_space.h"
#include "descent.h"
#include "option_types.h"

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>  // for higher degree simplex elements with mass lumping

namespace gpe
{

// TODO: Base class to provide methods for:
//       Assembly of A0 (matrix-free or otherwise)
//       Preconditioning
//       Assembly of M_phiphi (matrix-free or otherwise)
// Move to lac.h?
struct LevelMatrix
{
    SparseMatrix<double> A0, M, Mpp;
    SparsityPattern sparsity_pattern;
    unsigned int level;

    void reinit(DynamicSparsityPattern&& sp)
    {
        sparsity_pattern.copy_from(sp);
        A0.reinit(sparsity_pattern);
        M.reinit(sparsity_pattern);
        Mpp.reinit(sparsity_pattern);
    }
};

template <int dim>
class GPE
{
public:
    GPE(const GPE_Options& options, unsigned int n_levels)
    // establish relations between objects
        : grid(options.radius, options.mesh_kind == MeshKind::SIMPLEX)
        , space(grid.triangulation)
        , system{}
    {
        // Note: assumes grid has no mixed cells (contains either quadrilaterals or simplices)
        if (grid.has_simplex) {
            mapping    = std::make_unique<dealii::MappingFE<dim>>(dealii::FE_SimplexP<dim>(1));
            if (options.degree > 1) {
                // add basis function corresponding to interpolation at the centroid (in 2d)
                // -> valid nodal quadrature formula for mass lumping
                // the polynomial degree is typically one higher than the specified degree
                // (for degree == 1, FE_SimplexP_Bubbles is equivalent to FE_SimplexP)
                element = std::make_unique<dealii::FE_SimplexP_Bubbles<dim>>(options.degree);
            } else {
                element = std::make_unique<dealii::FE_SimplexP<dim>>(options.degree);
            }
            quadrature = std::make_unique<dealii::QGaussSimplex<dim>>(options.degree + 1);
        }
        else {
            mapping    = std::make_unique<dealii::MappingQ1<dim>>();
            element    = std::make_unique<dealii::FE_Q<dim>>(options.degree);
            quadrature = std::make_unique<dealii::QGauss<dim>>(options.degree + 1);
        }
        grid.refine(n_levels);    // do the actual computations
        std::cerr << "Number of levels: " << grid.triangulation.n_global_levels() << std::endl;
        std::cerr << "Number of vertices: " << grid.triangulation.n_vertices() << std::endl;

        space.setup_dofs(options.order, *element);
        space.setup_constraints(options.bc);
    }

    template <typename Function>
    void assemble(Function&& V)
    {
        const auto& dof_handler = space.get_dofs();
        const auto& constraints = space.get_constraints();

        system.reinit(make_sparsity_pattern(dof_handler, constraints));

        // Fixed mass and stiffness matrix
        assemble_A0  (system.A0, V, dof_handler, *quadrature, *mapping, constraints);
        assemble_mass(system.M, dof_handler, *quadrature, *mapping, constraints);
    }

    [[maybe_unused]] Vector<double>
    run(const Vector<double>& x0, double beta, GdOptions options_rgd, std::ostream& os)
    {
        const auto& dof_handler = space.get_dofs();
        const auto& constraints = space.get_constraints();

        // Compute solution on most refined (active) level
        std::cerr << "Number of cells: " << grid.triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << space.n_dofs() << std::endl;

        // Weighed mass matrix for solution in every step
        auto update_mpp = [this, &dof_handler, &constraints](SparseMatrix<double>& matrix, const Vector<double>& x)
        {
            assemble_mass_phiphi(matrix, x, dof_handler, *quadrature, *mapping, constraints);
        };

        // Run gradient descent + enforce boundary conditions
        // TODO: abstraction leak `constraints`
        Vector<double> x = gp_energy_rgd<dim>(system.A0, system.M, system.Mpp, update_mpp,
            x0, beta, constraints, options_rgd, os);
        return x;
    }

    // Iteration with constant starting value
    [[maybe_unused]] Vector<double>
    run(const double x0d, double beta, GdOptions options_rgd, std::ostream& os)
    {
        // Define starting value
        Vector<double> x0(space.n_dofs());
        x0 = x0d;

        Vector<double> x = run(x0, beta, options_rgd, os);
        return x;
    }

    const FeSpace<dim>& fe_space() const { return space; }
    unsigned int n_dofs() const { return space.n_dofs(); }

    const dealii::DoFHandler<dim>& get_dofs() const { return space.get_dofs(); }
    const dealii::AffineConstraints<double>& get_constraints() const { return space.get_constraints(); }

private:
    HyperCube<dim> grid;    // cells
    FeSpace<dim>   space;   // degrees of freedom
    LevelMatrix    system;  // linear system

    // Variables for simplex or quadrilateral meshes
    std::unique_ptr<dealii::Mapping<dim>> mapping;
    std::unique_ptr<dealii::FiniteElement<dim>> element;
    std::unique_ptr<dealii::Quadrature<dim>> quadrature;
};

}

#endif //GPE_MAIN_H