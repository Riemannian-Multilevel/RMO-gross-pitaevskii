//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "assemble.h"
#include "grid.h"
#include "sparsity.h"
#include "fe_space.h"
#include "function.h"
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

template <int dim>
class CoarseMassOracle
{
public:
    // TODO: refactor from EnergyOracle
private:

};

template <int dim>
class CoarseActiveOracle
{
public:
    // TODO: refactor from EnergyOracle
private:

};

// TODO: common base class (virtual? constructor: `using Oracle::Oracle`)
template <int dim>
class EnergyOracle
{
public:
    using OperatorType  = LinearCombinationMatrix<SparseMatrix<double>, Vector<double>>;
    using InverseOpType = InverseMatrix<OperatorType, Vector<double>,dealii::SparseILU<double>>;

    // setup before step 1 (discretization)
    EnergyOracle(GrossPitaevskiiProblem<dim>& GP_, double beta_)
        : GP(GP_), beta(beta_)  // parameter beta stored for residual computations
    {
        // Set up sparse ILU for linear term
        // TODO: this decreases in effectiveness as beta increases
        precond.initialize(GP.get_A0());
        const Vector<double> prototype(GP.get_A0().m());

        // Set up the linear operator A = A0 + beta * Mpp
        // We do this ONCE, the operator class stores pointers.
        // When Mpp changes values later, this operator automatically uses the new values.
        // TODO: merge to GrossPitaevskiiProblem? (common between fine and coarse methods)
        Aop.add_component(1.0, GP.get_A0());
        Aop.add_component(beta, GP.get_Mpp());
        Aop.reinit(prototype);

        // Same for M
        Mop.add_component(1.0, GP.get_M());
        Mop.reinit(prototype);
    }

    // setup before step k
    void initialize(Vector<double>& x)
    {
        GP.assemble_nonlinear_term(x);   // also applies constraints to `x`

        iter_prev = iter;
        iter = energy::residual(x, GP.get_A0(), GP.get_Mpp(), GP.get_M(), beta);
    }

    double value(const Vector<double>& x) const
    {
        return energy::function_value(x, GP.get_A0(), GP.get_Mpp());
    }

    // TODO: generic return type (computation of gradient does not necessarily involve a linear system)
    unsigned gradient(const Vector<double>& x, Vector<double>& output, const GdOptions& options) const
    {
        // parameter beta is implicitly included with linear operator `Aop`
        InverseOpType Aop_inv(Aop, options.solver, precond, options.max_iter, options.tol_inner);

        energy::gradient(Aop_inv, Mop, x, output);
        return Aop_inv.control().last_step();
    }

    // Note: this writes the retraction to the base point x (x <- R_x(factor*z))
    // TODO: retractions at run-time (enum class)
    void retract(const Vector<double>& z, Vector<double>& x, double factor) const
    {
        energy::retract_by_norm(GP.get_M(), z, x, factor);
        //energy::retract_by_ortho(GP.get_M(), z, x, factor);
        //energy::retract_by_exp(GP.get_M(), z, x, factor);
    }

    // TODO: remove
    energy::State residual() const
    {
        return iter;
    }

    // TODO: move to function.h
    bool check_convergence(const GdOptions& options) const
    {
        const double lmb_diff   = std::abs(iter.lambda - iter_prev.lambda);
        const double lmb_factor = 1.0 + std::abs(iter.lambda);  // avoid numerical issues near lmb ~ 0

        if (lmb_diff < options.tol_lambda * lmb_factor && iter.residual < options.tol_residual) {
            return true;
        }
        return false;
    }

private:
    GrossPitaevskiiProblem<dim>& GP;
    double beta;
    dealii::SparseILU<double> precond;

    // Linear combinations for gradient descent / matrix inversion
    // This type contains weights and references to stored matrices.
    OperatorType Aop, Mop;

    // Information on last iteration
    energy::State iter, iter_prev;
};


// Put it all together
template <int dim>
class GPE
{
public:
    GPE(const GPE_Options& options, unsigned int n_levels)
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
    [[maybe_unused]] Vector<double>
    run(Potential&& V, const Vector<double>& x0, double beta, GdOptions options_rgd, std::ostream& os)
    {
        const auto& dof_handler = space.get_dofs();
        const auto& constraints = space.get_constraints();

        // Assemble matrices M, A0 = M_V + S
        GrossPitaevskiiProblem<dim> GP(dof_handler, *quadrature, *mapping, constraints, V);
        EnergyOracle<dim> oracle(GP, beta);

        // Compute solution on most refined (active) level
        std::cerr << "Number of cells: " << grid.triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << space.n_dofs() << std::endl;

        // Run gradient descent + enforce boundary conditions
        Vector<double> x = gradient_descent(oracle, x0, options_rgd, os);
        constraints.distribute(x);
        return x;
    }

    // Iteration with constant starting value
    template <typename Potential>
    [[maybe_unused]] Vector<double>
    run(Potential&& V, const double x0d, double beta, GdOptions options_rgd, std::ostream& os)
    {
        // Define starting value
        Vector<double> x0(space.n_dofs());
        x0 = x0d;

        Vector<double> x = run(V, x0, beta, options_rgd, os);
        return x;
    }

    const FeSpace<dim>& fe_space() const { return space; }
    unsigned int n_dofs() const { return space.n_dofs(); }

    const dealii::DoFHandler<dim>& get_dofs() const { return space.get_dofs(); }
    const dealii::AffineConstraints<double>& get_constraints() const { return space.get_constraints(); }

private:
    HyperCube<dim>    grid;    // cell
    FeSpace<dim>      space;   // degrees of freedom

    // Variables for simplex or quadrilateral meshes
    std::unique_ptr<dealii::FiniteElement<dim>> mapping_fe;
    std::unique_ptr<dealii::Mapping<dim>>       mapping;
    std::unique_ptr<dealii::FiniteElement<dim>> element;
    std::unique_ptr<dealii::Quadrature<dim>>    quadrature;
};

}

#endif //GPE_MAIN_H