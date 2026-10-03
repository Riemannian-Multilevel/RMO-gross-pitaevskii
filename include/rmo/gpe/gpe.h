//
// Created by Ferdinand Vanmaele on 12.01.26.
//
#ifndef RMO_GPE_GPE_H
#define RMO_GPE_GPE_H

#include <rmo/lac.h>
#include <rmo/option_types.h>

#include <rmo/fe/assemble.h>
#include <rmo/fe/grid.h>
#include <rmo/fe/space.h>

#include <rmo/util/sparsity.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>  // for higher degree simplex elements with mass lumping

#include <memory>
#include <numbers>
#include <string>
#include <type_traits>
#include <variant>

namespace rmo::gpe
{

namespace potential
{
/**
 * @brief Functor computing the square of the Euclidean norm of a point.
 *
 * Computes \f$ f(p) = \sum_{d=0}^{dim-1} p_d^2 \f$.
 * Used primarily for initializing test cases or potentials.
 *
 * @tparam dim The spatial dimension of the point.
 */
template <int dim>
class Square
{
public:
    double operator()(const Point<dim>& p) const {
        typename Point<dim>::value_type out = 0.0;

        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    }
};


template <int dim>
class OpticalLattice
{
public:
    explicit OpticalLattice(const double nu = 100)
        : m_nu(nu)
    {}

    double operator()(const Point<dim>& p) const
    {
        typename Point<dim>::value_type out = 0.0;

        for (unsigned d = 0; d < dim; d++) {
            const double spx = std::sin(0.5*std::numbers::pi*p[d]);
            out += 0.5*p[d]*p[d] + m_nu*spx*spx;
        }
        return out;
    }

private:
    const double m_nu;
};


/**
 * @brief Potential given as an expression in the coordinates x[,y[,z]], e.g. "0.5*(x^2+y^2)".
 *
 * dealii::FunctionParser is neither copyable nor movable, so it is held through a shared_ptr to
 * keep this functor a value type: PVar is returned by value and forwarded into ModelBuilder.
 */
template <int dim>
class Expression
{
public:
    explicit Expression(const std::string& expr)
        : m_parser(std::make_shared<dealii::FunctionParser<dim>>(
              expr, "", dealii::FunctionParser<dim>::default_variable_names()))
    {}

    double operator()(const Point<dim>& p) const
    {
        return m_parser->value(p);
    }

private:
    std::shared_ptr<const dealii::FunctionParser<dim>> m_parser;
};


template <int dim>
using PVar = std::variant<Square<dim>, OpticalLattice<dim>, Expression<dim>>;

template <int dim>
PVar<dim>
get_potential(Potential potential_t, const std::string& expr = "") {
    switch (potential_t)
    {
        case Potential::SQUARE:
            return Square<dim>();
        case Potential::OPTICAL_LATTICE:
            return OpticalLattice<dim>();
        case Potential::EXPRESSION:
            AssertThrow(!expr.empty(), dealii::ExcMessage("empty potential expression"));
            return Expression<dim>(expr);
        default:
            throw std::invalid_argument("Unknown potential type");
    }
}

} // namespace potential


/**
 * @brief Matrices and operators shared by GrossPitaevskiiSystem and GrossPitaevskiiLumpedSystem.
 *
 * Stores the time-independent part \f$ A_0 \f$ (stiffness and potential), the mass matrix \f$ M \f$
 * and the non-linear term \f$ M_{\phi\phi} \f$, and builds LinearCombination operators from them.
 * Derived classes assemble the matrices and implement `assemble_nonlinear_term(x)`.
 *
 * @tparam dim The spatial dimension of the problem.
 * @tparam MassMatrixType Storage of \f$ M \f$ and \f$ M_{\phi\phi} \f$: `SparseMatrix<double>`
 * (consistent) or `DiagonalMatrix<Vector<double>>` (lumped).
 */
template <int dim, typename MassMatrixType>
class GrossPitaevskiiSystemBase
{
public:
    static constexpr int dimension = dim;

    using MassMatrix = MassMatrixType;
    using Operator   = std::conditional_t<std::is_same_v<MassMatrix, SparseMatrix<double>>,
                                          LinearCombination<Vector<double>, SparseMatrix<double>>,
                                          LinearCombination<Vector<double>, SparseMatrix<double>, MassMatrix>>;

    // Since LinearCombination stores pointers to matrices, these functions are lazy;
    // the (non-linear) terms can be assembled after calling this function.
    // TODO: rename to operator_A() or similar, since this creates a new object? (potential lifetime issues)
    /** @brief Operator \f$ w_{A_0} A_0 + w_{M_{\phi\phi}} M_{\phi\phi} \f$. */
    Operator get_operator_A(const double weight_Mpp, const double weight_A0 = 1.0) const
    {
        // Note: We pass pointers to our internal matrices.
        // The operator is valid as long as this Problem instance exists.
        Operator Aop;
        Aop.add_component(weight_A0, A0);
        Aop.add_component(weight_Mpp, Mpp);
        Aop.reinit(Vector<double>(A0.m()));

        return Aop;
    }

    // TODO: rename to operator_M() or similar, since this creates a new object? (potential lifetime issues)
    /** @brief Operator \f$ w_M M \f$. */
    Operator get_operator_M(const double weight_M = 1.0) const
    {
        Operator Mop;
        Mop.add_component(weight_M, M);
        Mop.reinit(Vector<double>(A0.m()));

        return Mop;
    }

    /** @brief Returns the linear operator \f$ A_0 \f$. */
    const SparseMatrix<double>& get_A0() const { return A0; }

    /** @brief Returns the mass matrix \f$ M \f$. */
    const MassMatrix& get_M() const { return M; }

    /** @brief Returns the non-linear mass matrix \f$ M_{\phi\phi} \f$. */
    const MassMatrix& get_Mpp() const { return Mpp; }

    unsigned int n_dofs() const { return dof_handler.n_dofs(); }  // A0.m()

    /** @brief Returns the linear constraints the operators were assembled with. */
    const dealii::AffineConstraints<double>& get_constraints() const { return constraints; }

protected:
    /** @brief Sets up the sparsity pattern and sizes \f$ A_0 \f$; derived classes assemble. */
    GrossPitaevskiiSystemBase(const dealii::DoFHandler<dim>& dofs,
                              const dealii::AffineConstraints<double>& cstr)
        : dof_handler(dofs)
        , constraints(cstr)
    {
        auto dsp = make_sparsity_pattern(dof_handler, constraints);
        sparsity_pattern.copy_from(dsp);
        A0.reinit(sparsity_pattern);
    }

    const dealii::DoFHandler<dim>& dof_handler;
    const dealii::AffineConstraints<double>& constraints;

    SparsityPattern sparsity_pattern;  ///< Declared before the matrices, which refer to it.
    SparseMatrix<double> A0;           ///< Constant part of the operator (Laplacian + Potential).
    MassMatrix M;                      ///< Mass matrix.

    /** @brief Non-linear term (changes every iteration).
     * In the GPE, \f$ M_{pp} \f$ must be recomputed whenever the solution
     * density \f$ |\phi|^2 \f$ changes.
     */
    MassMatrix Mpp;
};


/**
 * @brief Handles the assembly and storage of matrices for the Gross-Pitaevskii equation.
 * This class manages the linear and non-linear operators resulting from the discretization
 * of the GPE. It stores the time-independent parts (stiffness and potential) separately
 * from the non-linear term that depends on the current solution density.
 *
 * @tparam dim The spatial dimension of the problem.
 */
template <int dim>
class GrossPitaevskiiSystem : public GrossPitaevskiiSystemBase<dim, SparseMatrix<double>>
{
    using Base = GrossPitaevskiiSystemBase<dim, SparseMatrix<double>>;

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
    GrossPitaevskiiSystem(const dealii::DoFHandler<dim>& dofs,
                          const dealii::Quadrature<dim>& quad,
                          const dealii::Mapping<dim>& map,
                          const dealii::AffineConstraints<double>& cstr,
                          Potential&& V)
        : Base(dofs, cstr)
        , quadrature(quad)
        , mapping(map)
    {
        // Assemble S (stiffness) + M_V (weighed mass)
        fe::assemble_A0(this->A0, V, dofs, quadrature, mapping, cstr);

        // Assemble M (mass)
        this->M.reinit(this->sparsity_pattern);
        fe::assemble_mass(this->M, dofs, quadrature, mapping, cstr);

        // Initialize non-linear term (varies between iterations)
        this->Mpp.reinit(this->sparsity_pattern);
    }

    /**
     * @brief Assembles the non-linear matrix term \f$ M_{\phi\phi} \f$ based on a solution.
     * In the GPE, the non-linearity usually takes the form \f$ \beta |\psi|^2 \f$.
     * This method updates the internal @c Mpp matrix (see get_Mpp()) using the values in @p x.
     *
     * @param x The current solution vector.
     */
    // TODO: keep this method non-const so evaluative methods (e.g. value, gradient, ...) cannot accidentally
    //       call a matrix assembly. Future versions should implement a state pattern
    void assemble_nonlinear_term(const Vector<double>& x)
    {
        fe::assemble_mass_phiphi(this->Mpp, x, this->dof_handler, quadrature, mapping, this->constraints);
    }

private:
    const dealii::Quadrature<dim>& quadrature;
    const dealii::Mapping<dim>& mapping;
};


/**
 * @brief Gross-Pitaevskii system with lumped (diagonal) mass and nonlinear matrices.
 *
 * Same interface as GrossPitaevskiiSystem, with \f$ M \f$ and \f$ M_{\phi\phi} \f$ stored as
 * `DiagonalMatrix` and the potential term of \f$ A_0 = S + M_{V,L} \f$ lumped. The nodal quadrature
 * for the lumped terms is derived from the finite element (fe::make_nodal_quadrature()), so the
 * element must be lumpable and the constraints free of hanging nodes.
 *
 * @tparam dim The spatial dimension of the problem.
 */
template <int dim>
class GrossPitaevskiiLumpedSystem : public GrossPitaevskiiSystemBase<dim, DiagonalMatrix<Vector<double>>>
{
    using Base = GrossPitaevskiiSystemBase<dim, DiagonalMatrix<Vector<double>>>;

public:
    /**
     * @brief Assembles \f$ A_0 = S + M_{V,L} \f$ and the lumped mass matrix \f$ M_L \f$.
     *
     * @param dofs The Degree of Freedom handler.
     * @param quad The quadrature formula for the stiffness matrix.
     * @param map The mapping from reference to real cells.
     * @param cstr Linear constraints (e.g., Dirichlet boundary conditions).
     * @param V The external potential object.
     */
    template <typename Potential>
    GrossPitaevskiiLumpedSystem(const dealii::DoFHandler<dim>& dofs,
                                const dealii::Quadrature<dim>& quad,
                                const dealii::Mapping<dim>& map,
                                const dealii::AffineConstraints<double>& cstr,
                                Potential&& V)
        : Base(dofs, cstr)
    {
        // Nodal quadrature for the lumped terms; only needed during assembly
        const auto quad_nodal = fe::make_nodal_quadrature(dofs.get_fe());

        fe::assemble_A0_lumped(this->A0, V, dofs, quad, quad_nodal, map, cstr);

        this->M.get_vector().reinit(dofs.n_dofs());
        fe::assemble_mass_lumped(this->M, dofs, quad_nodal, map, cstr);

        this->Mpp.get_vector().reinit(dofs.n_dofs());
    }

    /**
     * @brief Updates \f$ (M_{\phi\phi})_{ii} = x_i^2 \, (M_L)_{ii} \f$ for the state @p x.
     *
     * Equals fe::assemble_mass_phiphi_lumped() on unconstrained rows, without a cell loop.
     */
    void assemble_nonlinear_term(const Vector<double>& x)
    {
        AssertDimension(x.size(), this->n_dofs());
        const auto& m = this->M.get_vector();
        auto& mpp     = this->Mpp.get_vector();

        for (unsigned int i = 0; i < x.size(); ++i) {
            mpp[i] = x[i] * x[i] * m[i];
        }
    }
};

/**
 * @brief Factory class to discretize a domain and produce a @ref GrossPitaevskiiSystem.
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
     * @brief Generates a problem instance for a specific potential.
     * @tparam System GrossPitaevskiiSystem (default) or GrossPitaevskiiLumpedSystem; both share
     * the constructor signature.
     * @tparam Potential Type of the potential function.
     * @param V The potential function.
     */
    template <typename System = GrossPitaevskiiSystem<dim>, typename Potential>
    System system(Potential&& V) const
    {
        static_assert(System::dimension == dim, "System dimension must match the package");
        return System(space.get_dofs(), *quadrature, *mapping, space.get_constraints(), std::forward<Potential>(V));
    }

    void distribute(Vector<double>& x) const
    {
        const auto& constraints = space.get_constraints();

        constraints.distribute(x);
    }

    /** @brief Access the underlying Finite Element space. */
    const fe::FeSpace<dim>& get_space() const { return space; }

    /** @brief Access the DoF handler. */
    const dealii::DoFHandler<dim>& get_dofs() const { return space.get_dofs(); }

    /** @brief Access the number of degrees of freedom. */
    [[nodiscard]] unsigned int n_dofs() const { return space.get_dofs().n_dofs(); }

    /** @brief Access the constraints. */
    [[nodiscard]] const dealii::AffineConstraints<double>& get_constraints() const { return space.get_constraints(); }

    /** @brief Access the geometry/grid object. */
    const fe::HyperCube<dim>& get_grid() const { return grid; }

    /** @brief Access the geometric mapping (reference cell to real cell). */
    const dealii::Mapping<dim>& get_mapping() const { return *mapping; }

private:
    fe::HyperCube<dim>    grid;    ///< The geometry and triangulation.
    fe::FeSpace<dim>      space;   ///< Wrapper for DoFHandler and AffineConstraints.

    // Using unique_ptr to handle polymorphic types (Simplex vs Q) and lifetime requirements
    std::unique_ptr<dealii::FiniteElement<dim>> mapping_fe; ///< Helper FE for Simplex mapping.
    std::unique_ptr<dealii::Mapping<dim>>       mapping;    ///< The geometric mapping.
    std::unique_ptr<dealii::FiniteElement<dim>> element;    ///< The finite element system.
    std::unique_ptr<dealii::Quadrature<dim>>    quadrature; ///< Integration quadrature.
};


// Class that represents the smooth objective function E(x) in ambient Euclidean space
template <int dim>
class GrossPitaevskiiFunctional
{
public:
    GrossPitaevskiiFunctional(GrossPitaevskiiSystem<dim>& system, double beta, SolverOptions options)
        : system(system)
        , beta(beta)
        , M(system.get_operator_M())
        , A(system.get_operator_A(beta))
        // Instantiate the solvers here so Oracle is a light-weight objects
        // (that can be instantiated inside loops if necessary)
        , M_inv(M, options)
        , A_inv(A, options)
    {
        A_inv.update_static(system.get_A0());
        M_inv.update_static(system.get_M());
        M_inv.update_dynamic(M.diagonal());
    }

    // Assembly of the non-linear matrix for value() / directional_derivative()
    void update(const Vector<double>& x)
    {
        // Updates A, M as references to system
        system.assemble_nonlinear_term(x);

        // TODO: lazy evaluation for preconditioner updates in A_inv
        A_inv.update_dynamic(A.diagonal());
    }

    /**
     * @brief Evaluates the energy functional for the Gross-Pitaevskii equation.
     * Computes the energy value:
     * \f[
     * E(x) = \frac{1}{2} x^T A_0 x + \frac{beta}{4} x^T M_{\phi\phi}(x) x
     * \f]
     */
    double value(const Vector<double>& x) const
    {
        auto A_eval = system.get_operator_A(beta*0.25, 0.5);

        Vector<double> Ax(x.size());
        A_eval.vmult(Ax, x);

        return x * Ax;
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const
    {
        Vector<double> Ax(x.size());
        A.vmult(Ax, x);

        return Ax * z;
    }

    void gradient(const Vector<double>& x, Vector<double>& output) const
    {
        A.vmult(output, x);
    }

    // Accessors
    unsigned n_dofs() const { return system.n_dofs(); }
    double get_beta() const { return beta; }

    const auto& get_M() const { return M; }
    const auto& get_A() const { return A; }
    const auto& get_A0() const { return system.get_A0(); }

    const InverseOpType& get_M_inv() const { return M_inv; }
    InverseOpType& get_M_inv() { return M_inv; }

    const InverseOpType& get_A_inv() const { return A_inv; }
    InverseOpType& get_A_inv() { return A_inv; }


private:
    GrossPitaevskiiSystem<dim>& system;
    double beta;
    OperatorType M, A;
    InverseOpType M_inv, A_inv;
};

} // namespace rmo::gpe
#endif //RMO_GPE_GPE_H