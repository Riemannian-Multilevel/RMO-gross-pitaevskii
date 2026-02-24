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

template <int dim>
class GrossPitaevskiiProblem
{
public:
    template <typename Potential>
    GrossPitaevskiiProblem(const dealii::DoFHandler<dim>& dofs,
                           const dealii::Quadrature<dim>& quad,
                           const dealii::Mapping<dim>& map,
                           const dealii::AffineConstraints<double>& cstr,
                           Potential&& V): dof_handler(dofs)
    , quadrature(quad)
    , mapping(map)
    , constraints(cstr)
    {
        auto dsp = make_sparsity_pattern(dof_handler, constraints);
        sparsity_pattern.copy_from(dsp);

        // Assemble S (stiffness) + M_V (weighed mass)
        A0.reinit(sparsity_pattern);
        assemble_A0(A0, V, dof_handler, quadrature, mapping, constraints);

        // Assemble M (mass)
        M.reinit(sparsity_pattern);
        assemble_mass(M, dof_handler, quadrature, mapping, constraints);

        // Initialize non-linear term (varies between iterations)
        Mpp.reinit(sparsity_pattern);
    }

    void assemble_nonlinear_term(Vector<double>& x)
    {
        // Apply constraints to incumbent solution
        constraints.distribute(x);

        // A = A_0 + beta * M_xx
        assemble_mass_phiphi(Mpp, x, dof_handler, quadrature, mapping, constraints);
    }

    const SparseMatrix<double>& get_A0() const { return A0; }
    const SparseMatrix<double>& get_M() const { return M; }
    const SparseMatrix<double>& get_Mpp() const { return Mpp; }

private:
    // Finite element parameters
    // const FeSpace<dim>& space;
    const dealii::DoFHandler<dim>& dof_handler;
    const dealii::Quadrature<dim>& quadrature;
    const dealii::Mapping<dim>& mapping;
    const dealii::AffineConstraints<double>& constraints;

    // Data for discrete problem
    SparseMatrix<double> A0, M;
    SparseMatrix<double> Mpp;  // changes in every iteration step
    SparsityPattern sparsity_pattern;
};


// Class to discretize a domain and return a corresponding GrossPitaevskiiProblem
template <int dim>
class GrossPitaevskiiPackage
{
public:
    GrossPitaevskiiPackage(const GPE_Options& options, unsigned int n_levels)
    // establish relations between objects
        : grid(options.radius, options.mesh_kind == MeshKind::SIMPLEX)
        , space(grid.triangulation)
    {
        // Note: assumes grid has no mixed cells (contains either quadrilaterals or simplices)
        if (grid.has_simplex) {
            // MappingFE stores a reference to an FE_SimplexP object, which lifetime must
            // match the lifetime of the mapping object
            mapping_fe = std::make_unique<dealii::FE_SimplexP<dim>>(1);
            mapping    = std::make_unique<dealii::MappingFE<dim>>(*mapping_fe);

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
            mapping_fe = nullptr;
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

    template <typename Potential>
    GrossPitaevskiiProblem<dim> problem(Potential&& V) const
    {
        const auto& dof_handler = space.get_dofs();
        const auto& constraints = space.get_constraints();

        return GP(dof_handler, *quadrature, *mapping, constraints, V);
    }

    const FeSpace<dim>& get_space() const { return space; }
    const dealii::DoFHandler<dim>& get_dofs() const { return space.get_dofs(); }
    const dealii::AffineConstraints<double>& get_constraints() const { return space.get_constraints(); }

    const HyperCube<dim>& get_grid() const { return grid; }

private:
    HyperCube<dim>    grid;    // cell
    FeSpace<dim>      space;   // DoFHandler, AffineConstraints

    // Variables for simplex or quadrilateral meshes
    std::unique_ptr<dealii::FiniteElement<dim>> mapping_fe;
    std::unique_ptr<dealii::Mapping<dim>>       mapping;
    std::unique_ptr<dealii::FiniteElement<dim>> element;
    std::unique_ptr<dealii::Quadrature<dim>>    quadrature;
};


}

#endif //GPE_MAIN_H