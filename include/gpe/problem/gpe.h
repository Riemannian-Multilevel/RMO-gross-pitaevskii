//
// Created by Ferdinand Vanmaele on 12.01.26.
//
#ifndef GPE_GPE_H
#define GPE_GPE_H

#include <gpe/fe/assemble.h>
#include <gpe/fe/grid.h>
#include <gpe/fe/fe_space.h>
#include <gpe/ropt/descent.h>
#include <gpe/util/sparsity.h>
#include <gpe/option_types.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>  // for higher degree simplex elements with mass lumping

namespace gpe
{

/**
 * @brief Handles the assembly and storage of matrices for the Gross-Pitaevskii equation.
 * This class manages the linear and non-linear operators resulting from the discretization
 * of the GPE. It stores the time-independent parts (stiffness and potential) separately
 * from the non-linear term that depends on the current solution density.
 *
 * @tparam dim The spatial dimension of the problem.
 */
// TODO: use stored beta (e.g in options) and setter method to update it?
//       (consistency between calls of get_operator_A(), value(), directional_derivative())
template <int dim>
class GrossPitaevskiiProblem
{
public:
    /**
     * @brief Constructor that initializes sparsity patterns and assembles linear matrices.
     * Computes the initial system matrices:
     * - \f$ A_0 = S + M_V \f$ (Stiffness + Potential mass matrix)
     * - \f$ M \f$ (Standard mass matrix)
     *
     * @tparam Potential A functional or class representing the external potential \f$ V(x) \f$.
     * @param dofs The Degree of Freedom handler.
     * @param quad The quadrature formula for integration.
     * @param map The mapping from reference to real cells.
     * @param cstr Linear constraints (e.g., Dirichlet boundary conditions).
     * @param V The external potential object.
     */
    template <typename Potential>
    GrossPitaevskiiProblem(const dealii::DoFHandler<dim>& dofs,
                           const dealii::Quadrature<dim>& quad,
                           const dealii::Mapping<dim>& map,
                           const dealii::AffineConstraints<double>& cstr,
                           Potential&& V)
        : dof_handler(dofs)
        , quadrature(quad)
        , mapping(map)
        , constraints(cstr)
    {
        // Setup sparsity pattern
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

    /**
     * @brief Assembles the non-linear matrix term \f$ M_{\phi\phi} \f$ based on a solution.
     * In the GPE, the non-linearity usually takes the form \f$ \beta |\psi|^2 \f$.
     * This method updates the internal @ref Mpp matrix using the values in @p x.
     *
     * @param x The current solution vector.
     */
    void assemble_nonlinear_term(const Vector<double>& x) const
    {
        assemble_mass_phiphi(Mpp, x, dof_handler, quadrature, mapping, constraints);
    }

    /**
     * @brief Evaluates the energy functional for the Gross-Pitaevskii equation.
     * Computes the energy value:
     * \f[
     * E(x) = \frac{1}{2} x^T A_0 x + \frac{beta}{4} x^T M_{\phi\phi}(x) x
     * \f]
     */
    double value(const Vector<double>& x, const double beta) const
    {
        Vector<double> A0_x(x.size());
        A0.vmult(A0_x, x);
        A0_x *= 0.5;

        Vector<double> Mpp_x(x.size());
        Mpp.vmult(Mpp_x, x);

        A0_x.add(0.25*beta, Mpp_x);
        return x * A0_x;
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z,
                                  const double beta) const
    {
        Vector<double> A0_x(x.size());
        A0.vmult(A0_x, x);

        Vector<double> Mpp_x(x.size());
        Mpp.vmult(Mpp_x, x);

        A0_x.add(beta, Mpp_x);
        return A0_x * z;
    }

    // TODO: Euclidean gradient

    // Since LinearCombination stores pointers to matrices, these functions are lazy;
    // the (non-linear) terms can be assembled after calling this function.
    auto get_operator_A(const double beta) const
    {
        using Operator = LinearCombination<SparseMatrix<double>, Vector<double>>;
        Operator Aop;

        // Note: We pass pointers to our internal matrices.
        // The operator is valid as long as this Problem instance exists.
        Aop.add_component(1.0, A0);
        Aop.add_component(beta, Mpp);
        Aop.reinit(Vector<double>(A0.m()));

        return Aop;
    }

    auto get_operator_A(const Vector<double>& x, const double beta) const
    {
        assemble_nonlinear_term(x);
        return get_operator_A(beta);
    }

    auto get_operator_M() const
    {
        using Operator = LinearCombination<SparseMatrix<double>, Vector<double>>;
        Operator Mop;

        Mop.add_component(1.0, M);
        Mop.reinit(Vector<double>(A0.m()));

        return Mop;
    }

    /** @brief Returns the linear operator \f$ A_0 \f$. */
    const SparseMatrix<double>& get_A0() const { return A0; }

    /** @brief Returns the mass matrix \f$ M \f$. */
    const SparseMatrix<double>& get_M() const { return M; }

    /** @brief Returns the non-linear mass matrix \f$ M_{\phi\phi} \f$. */
    const SparseMatrix<double>& get_Mpp() const { return Mpp; }

    unsigned int n_dofs() const { return dof_handler.n_dofs(); }  // A0.m()

private:
    const dealii::DoFHandler<dim>& dof_handler;
    const dealii::Quadrature<dim>& quadrature;
    const dealii::Mapping<dim>& mapping;
    const dealii::AffineConstraints<double>& constraints;

    SparseMatrix<double> A0;  ///< Constant part of the operator (Laplacian + Potential).
    SparseMatrix<double> M;   ///< Standard mass matrix.

    /** @brief The non-linear matrix is mutable to facilitate "lazy assembly."
     * In the GPE, \f$ M_{pp} \f$ must be recomputed whenever the solution
     * density \f$ |\phi|^2 \f$ changes. Marking it mutable allows consumers
     * to trigger this assembly within logically @p const methods (like value
     * evaluation or gradient computation).
     */
    mutable SparseMatrix<double> Mpp; ///< Non-linear interaction matrix (changes every iteration).
    SparsityPattern sparsity_pattern;
};


/**
 * @brief Factory class to discretize a domain and produce a @ref GrossPitaevskiiProblem.
 * This class handles the setup phase:
 * 1. Generates the mesh (Simplex or Hypercube).
 * 2. Selects appropriate Finite Elements and Quadrature rules.
 * 3. Manages DoF distribution and boundary constraints.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim>
class GrossPitaevskiiPackage
{
public:
    /**
     * @brief Constructs the package and prepares the triangulation and FE space.
     * Branches logic based on @p options to handle either Simplicial or Quadrilateral
     * geometry. It performs the global mesh refinement and sets up DoFs.
     *
     * @param options Configuration options including degree, mesh type, and BCs.
     * @param n_levels Number of global refinement steps for the mesh.
     */
    GrossPitaevskiiPackage(const GPE_Options& options, unsigned int n_levels)
        : grid(options.radius, options.mesh_kind == MeshKind::SIMPLEX)
        , space(grid.triangulation)
    {
        if (grid.has_simplex)
        {
            // Set up simplicial mapping and finite elements
            mapping_fe = std::make_unique<dealii::FE_SimplexP<dim>>(1);
            mapping    = std::make_unique<dealii::MappingFE<dim>>(*mapping_fe);

            if (options.degree > 1) {
                // FE_SimplexP_Bubbles provides nodal quadrature nodes for mass lumping
                element = std::make_unique<dealii::FE_SimplexP_Bubbles<dim>>(options.degree);
            } else {
                element = std::make_unique<dealii::FE_SimplexP<dim>>(options.degree);
            }
            quadrature = std::make_unique<dealii::QGaussSimplex<dim>>(options.degree + 1);
        }
        else
        {
            // Set up standard hypercube mapping and finite elements
            mapping_fe = nullptr;
            mapping    = std::make_unique<dealii::MappingQ1<dim>>();
            element    = std::make_unique<dealii::FE_Q<dim>>(options.degree);
            quadrature = std::make_unique<dealii::QGauss<dim>>(options.degree + 1);
        }

        // Perform mesh refinement
        grid.refine(n_levels);

        std::cerr << "Number of levels: " << grid.triangulation.n_global_levels() << std::endl;
        std::cerr << "Number of vertices: " << grid.triangulation.n_vertices() << std::endl;

        // Distribute DoFs and build constraints
        space.setup_dofs(options.order, *element);
        space.setup_constraints(options.bc);
    }

    /**
     * @brief Generates a new problem instance for a specific potential.
     * @tparam Potential Type of the potential function.
     * @param V The potential function.
     * @return GrossPitaevskiiProblem<dim> The assembled problem object.
     */
    template <typename Potential>
    GrossPitaevskiiProblem<dim> problem(Potential&& V) const
    {
        const auto& dof_handler = space.get_dofs();
        const auto& constraints = space.get_constraints();

        return GrossPitaevskiiProblem<dim>(dof_handler, *quadrature, *mapping, constraints, V);
    }

    void distribute(Vector<double>& x) const
    {
        const auto& constraints = space.get_constraints();

        constraints.distribute(x);
    }

    /** @brief Access the underlying Finite Element space. */
    const FeSpace<dim>& get_space() const { return space; }

    /** @brief Access the DoF handler. */
    const dealii::DoFHandler<dim>& get_dofs() const { return space.get_dofs(); }

    /** @brief Access the number of degrees of freedom. */
    [[nodiscard]] unsigned int n_dofs() const { return space.get_dofs().n_dofs(); }

    /** @brief Access the constraints. */
    [[nodiscard]] const dealii::AffineConstraints<double>& get_constraints() const { return space.get_constraints(); }

    /** @brief Access the geometry/grid object. */
    const HyperCube<dim>& get_grid() const { return grid; }

private:
    HyperCube<dim>    grid;    ///< The geometry and triangulation.
    FeSpace<dim>      space;   ///< Wrapper for DoFHandler and AffineConstraints.

    // Using unique_ptr to handle polymorphic types (Simplex vs Q) and lifetime requirements
    std::unique_ptr<dealii::FiniteElement<dim>> mapping_fe; ///< Helper FE for Simplex mapping.
    std::unique_ptr<dealii::Mapping<dim>>       mapping;    ///< The geometric mapping.
    std::unique_ptr<dealii::FiniteElement<dim>> element;    ///< The finite element system.
    std::unique_ptr<dealii::Quadrature<dim>>    quadrature; ///< Integration quadrature.
};


/**
 * @brief Orchestrator for Gross-Pitaevskii simulations.
 * The @ref EnergySimulator manages the persistent @ref GrossPitaevskiiPackage
 * (discretization) and coordinates the execution of the energy minimization
 * using a given @ref Oracle.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim, typename Oracle>
class GrossPitaevskiiSimulator
{
public:
    static_assert(Oracle::dimension == dim);

    /**
     * @brief Constructor.
     * @tparam Potential Functor or class representing the external potential \f$ V(x) \f$.
     * @param V The potential object.
     * @param options General options for GPE discretization.
     * @param n_levels Number of global mesh refinements.
     */
    template <typename Potential>
    GrossPitaevskiiSimulator(Potential&& V, const GPE_Options& options, unsigned int n_levels)
        : package(options, n_levels)
        , problem(package.problem(std::forward<Potential>(V)))
        , options(options)
    {}

    // Allow to change the potential without re-discretizing the domain.
    template <typename Potential>
    void reinit(Potential&& V)
    {
        problem = package.problem(std::forward<Potential>(V));
    }

    void distribute(Vector<double>& x) const
    {
        package.distribute(x);
    }

    /**
     * @brief Runs the energy minimization for a given potential.
     * @param x0
     * @param beta The interaction strength constant.
     * @param options_inner
     * @param options_gd Options for the gradient descent algorithm.
     */
    // TODO factor this out to caller, see get_oracle()
    Vector<double>
    run(const Vector<double>& x0, double beta,
        const SolverOptions&  options_inner,
        const DescentOptions& options_gd, std::ostream& os) const
    {
        Assert(x0.size() == package.n_dofs(), dealii::ExcDimensionMismatch(x0.size(), package.n_dofs()));
        // Create the oracle (light-weight object, references problem matrices)
        Oracle oracle(problem, beta, options_inner);

        // Termination criterium
        auto conv_check = [&options_gd](const iteration::State& current, const iteration::State& previous)
        {
            const double lmb_diff   = std::abs(current.lambda - previous.lambda);
            const double lmb_factor = 1.0 + std::abs(current.lambda);

            return (lmb_diff < options_gd.tol_lambda * lmb_factor && current.residual < options_gd.tol_residual);
        };

        // Riemannian gradient descent
        // Note: the update strategy can be arbitrary complex (e.g. for multilevel algorithms)
        return gradient_descent(oracle, x0, options_gd, os, conv_check);
    }

    /** @brief Access the discretization package. */
    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }
    const GrossPitaevskiiProblem<dim>& get_problem() const { return problem; }

    unsigned int n_dofs() const { return package.n_dofs(); }

    const dealii::DoFHandler<dim>&
    get_dofs() const { return package.get_dofs(); }

    const dealii::AffineConstraints<double>&
    get_constraints() const { return package.get_constraints(); }

    Oracle get_oracle(double beta, SolverOptions options_gd) const
    {
        return Oracle(problem, beta, options_gd);
    }

    auto get_M() const { return problem.get_operator_M(); }
    auto get_A(double beta) const { return problem.get_operator_A(beta); }

private:
    /** @brief Persistent discretization infrastructure. */
    GrossPitaevskiiPackage<dim> package;
    /** @brief Assembly and storage of matrices. */
    GrossPitaevskiiProblem<dim> problem;
    /** @brief Problem configuration options. */
    GPE_Options options;
};

}

#endif //GPE_GPE_H