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
#include <rmo/fe/util.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>  // for higher degree simplex elements with mass lumping

#include <algorithm>
#include <memory>
#include <numbers>
#include <string>
#include <type_traits>
#include <variant>

/**
 * @file
 * @brief Discretization of the Gross-Pitaevskii energy
 * \f$ E(u) = \frac12 \int_\Omega |\nabla u|^2 + V u^2 \, dx + \frac\beta4 \int_\Omega u^4 \, dx \f$,
 * minimized on the unit-mass sphere \f$ x^\top M x = 1 \f$.
 *
 * - potentials \f$ V \f$ (namespace potential);
 * - the matrices \f$ A_0 = S + M_V \f$, \f$ M \f$ and the non-linear term \f$ M_{\phi\phi}(x) \f$, consistent
 *   (GrossPitaevskiiSystem) or lumped (GrossPitaevskiiLumpedSystem);
 * - mesh, element, mapping, quadrature and DoFs (GrossPitaevskiiPackage);
 * - the energy, its Euclidean gradient \f$ A(x) x \f$ with \f$ A(x) = A_0 + \beta M_{\phi\phi}(x) \f$, and the
 *   inverse operators of \f$ M \f$ and \f$ A(x) \f$ (GrossPitaevskiiFunctional).
 *
 * Systems refer to the DoFHandler, mapping, quadrature and constraints of their package, and functionals to
 * their system; both must outlive them.
 */
namespace rmo::gpe
{

/**
 * @brief External potentials \f$ V \f$, functors `double(const Point<dim>&)`: Zero, Constant, Square
 * (\f$ |x|^2 \f$), OpticalLattice (\f$ \sum_d \frac12 x_d^2 + \nu \sin^2(\pi x_d / 2) \f$) and Expression;
 * get_potential() selects one at runtime.
 */
namespace potential
{
template <int dim>
class Zero
{
public:
    double operator()(const Point<dim>& p) const
    {
        return 0.0;
    }
};

template <int dim>
class Constant
{
public:
    explicit Constant(const double a = 1.0)
        : m_a(a)
    {}

    double operator()(const Point<dim>& p) const
    {
        return m_a;
    }

private:
    double const m_a;
};

/** @brief Harmonic potential \f$ V(x) = |x|^2 \f$. */
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
 * The parser (neither copyable nor movable) is shared, so that the functor is a value type like the others.
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


//! Any of the potentials.
template <int dim>
using PVar = std::variant<Zero<dim>, Constant<dim>, Square<dim>, OpticalLattice<dim>, Expression<dim>>;

//! Potential of type @p potential_t; @p expr is the expression of Potential::EXPRESSION.
template <int dim>
PVar<dim>
get_potential(Potential potential_t, const std::string& expr = "") {
    switch (potential_t)
    {
        case Potential::ZERO:
            return Zero<dim>();
        // TODO: allow setting constructor variable
        case Potential::CONSTANT:
            return Constant<dim>();
        case Potential::SQUARE:
            return Square<dim>();
        // TODO: allow setting constructor variable
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
 * @brief Storage of \f$ A_0 \f$, \f$ M \f$ and \f$ M_{\phi\phi} \f$, and LinearCombination operators of them;
 * derived classes assemble the matrices and implement `assemble_nonlinear_term(x)`.
 *
 * @tparam MassMatrixType Storage of \f$ M \f$ and \f$ M_{\phi\phi} \f$: `SparseMatrix<double>` (consistent) or
 * `DiagonalMatrix<Vector<double>>` (lumped).
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
    /** @brief Operator \f$ w_{A_0} A_0 + w_{M_{\phi\phi}} M_{\phi\phi} \f$; refers to the current \f$ M_{\phi\phi} \f$. */
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

    /** @brief State-independent part \f$ A_0 = S + M_V \f$. */
    const SparseMatrix<double>& get_A0() const { return A0; }

    /** @brief Mass matrix \f$ M \f$. */
    const MassMatrix& get_M() const { return M; }

    /** @brief Non-linear term \f$ M_{\phi\phi} \f$ of the last assemble_nonlinear_term(). */
    const MassMatrix& get_Mpp() const { return Mpp; }

    unsigned int n_dofs() const { return dof_handler.n_dofs(); }  // A0.m()

    /** @brief Constraints the matrices were assembled with. */
    const dealii::AffineConstraints<double>& get_constraints() const { return constraints; }

protected:
    /** @brief Sets up the sparsity pattern and sizes \f$ A_0 \f$. */
    GrossPitaevskiiSystemBase(const dealii::DoFHandler<dim>& dofs,
                              const dealii::AffineConstraints<double>& cstr)
        : dof_handler(dofs)
        , constraints(cstr)
    {
        auto dsp = fe::make_sparsity_pattern(dof_handler, constraints);
        sparsity_pattern.copy_from(dsp);
        A0.reinit(sparsity_pattern);
    }

    const dealii::DoFHandler<dim>& dof_handler;
    const dealii::AffineConstraints<double>& constraints;

    SparsityPattern sparsity_pattern;  ///< Declared before the matrices, which refer to it.
    SparseMatrix<double> A0;           ///< \f$ A_0 = S + M_V \f$
    MassMatrix M;                      ///< \f$ M \f$
    MassMatrix Mpp;                    ///< \f$ M_{\phi\phi} \f$, reassembled for each state
};


/**
 * @brief Consistent matrices: \f$ A_0 = S + M_V \f$, \f$ M \f$ and
 * \f$ (M_{\phi\phi})_{ij} = \int_\Omega u_h^2 \phi_i \phi_j \, dx \f$ for the state \f$ u_h \f$.
 */
template <int dim>
class GrossPitaevskiiSystem : public GrossPitaevskiiSystemBase<dim, SparseMatrix<double>>
{
    using Base = GrossPitaevskiiSystemBase<dim, SparseMatrix<double>>;

public:
    /** @brief Assembles \f$ A_0 \f$ and \f$ M \f$ with quadrature @p quad, condensed with the constraints @p cstr. */
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

    /** @brief Assembles \f$ M_{\phi\phi} \f$ for the state @p x. */
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
 * `DiagonalMatrix`: \f$ M_L \f$ holds the row sums of \f$ M \f$, and the zeroth-order terms are evaluated
 * at the support points \f$ a_i \f$, \f$ (M_{V,L})_{ii} = V(a_i) (M_L)_{ii} \f$ in \f$ A_0 = S + M_{V,L} \f$
 * and \f$ (M_{\phi\phi})_{ii} = x_i^2 (M_L)_{ii} \f$. The row sums must be positive (checked; e.g. not
 * `FE_SimplexP(2)`), and the constraints free of hanging nodes.
 *
 * @tparam dim The spatial dimension of the problem.
 */
template <int dim>
class GrossPitaevskiiLumpedSystem : public GrossPitaevskiiSystemBase<dim, DiagonalMatrix<Vector<double>>>
{
    using Base = GrossPitaevskiiSystemBase<dim, DiagonalMatrix<Vector<double>>>;

public:
    /**
     * @brief Assembles \f$ M_L \f$ and \f$ A_0 = S + M_{V,L} \f$ with quadrature @p quad, condensed with the
     * constraints @p cstr.
     * @throws dealii::ExcMessage if the row sums of the mass matrix are not positive.
     */
    template <typename Potential>
    GrossPitaevskiiLumpedSystem(const dealii::DoFHandler<dim>& dofs,
                                const dealii::Quadrature<dim>& quad,
                                const dealii::Mapping<dim>& map,
                                const dealii::AffineConstraints<double>& cstr,
                                Potential&& V)
        : Base(dofs, cstr)
    {
        this->M.get_vector().reinit(dofs.n_dofs());
        fe::assemble_mass_lumped(this->M, dofs, quad, map, cstr);

        // Exactly zero row sums (e.g. FE_SimplexP(2)) are only zero up to rounding
        const auto& m = this->M.get_vector();
        AssertThrow(*std::ranges::min_element(m) > 1e-12 * m.linfty_norm(),
                    dealii::ExcMessage("element not lumpable (non-positive row sums of the mass matrix)"));

        fe::assemble_A0_lumped(this->A0, V, dofs, quad, map, cstr);

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
 * @brief Mesh, finite element, mapping, quadrature and DoFs of a problem, and factory of its systems.
 *
 * Quadrilateral meshes use `FE_Q(p)` with `MappingQ1` and `QGauss(p + 1)`; simplex meshes use `FE_SimplexP(1)`,
 * or `FE_SimplexP_Bubbles(p)` for \f$ p > 1 \f$, with a linear `MappingFE` and `QGaussSimplex(p + 1)`.
 */
template <int dim>
class GrossPitaevskiiPackage
{
public:
    /**
     * @param options Mesh kind, degree, radius, DoF ordering and boundary condition.
     * @param n_levels Number of mesh levels, i.e. `n_levels - 1` global refinements (see fe::HyperCube::refine()).
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
                // FE_SimplexP_Bubbles has positive row sums of the mass matrix (unlike FE_SimplexP(2)),
                // as required for mass lumping
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

    /** @brief Assembles a system (GrossPitaevskiiSystem or GrossPitaevskiiLumpedSystem) for the potential @p V. */
    // TODO: turn this into a free function
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

    const fe::FeSpace<dim>& get_space() const { return space; }
    const dealii::DoFHandler<dim>& get_dofs() const { return space.get_dofs(); }
    [[nodiscard]] unsigned int n_dofs() const { return space.get_dofs().n_dofs(); }
    [[nodiscard]] const dealii::AffineConstraints<double>& get_constraints() const { return space.get_constraints(); }
    const fe::HyperCube<dim>& get_grid() const { return grid; }
    const dealii::Mapping<dim>& get_mapping() const { return *mapping; }
    const dealii::Quadrature<dim>& get_quadrature() const { return *quadrature; }

private:
    fe::HyperCube<dim>    grid;    ///< The geometry and triangulation.
    fe::FeSpace<dim>      space;   ///< Wrapper for DoFHandler and AffineConstraints.

    // Using unique_ptr to handle polymorphic types (Simplex vs Q) and lifetime requirements
    std::unique_ptr<dealii::FiniteElement<dim>> mapping_fe; ///< Helper FE for Simplex mapping.
    std::unique_ptr<dealii::Mapping<dim>>       mapping;    ///< The geometric mapping.
    std::unique_ptr<dealii::FiniteElement<dim>> element;    ///< The finite element system.
    std::unique_ptr<dealii::Quadrature<dim>>    quadrature; ///< Integration quadrature.
};


/**
 * @brief Energy \f$ E(x) \f$ in ambient space, its Euclidean gradient \f$ A(x) x \f$, and the inverse operators of
 * \f$ M \f$ and \f$ A(x) \f$ (for a lumped system, \f$ M^{-1} \f$ is exact).
 *
 * update() reassembles \f$ M_{\phi\phi} \f$ of the system; all other members refer to the state of the last update.
 */
template <typename System>
class GrossPitaevskiiFunctional
{
public:
    static constexpr int dimension = System::dimension;
    static constexpr bool lumped   = is_diagonal_matrix_v<typename System::MassMatrix>;

    using SystemType = System;
    using Operator   = typename System::Operator;
    using InverseA   = PreconditionInverse<Operator, SparseMatrix<double>>;
    using InverseM   = std::conditional_t<lumped, DiagonalInverse, InverseA>;

    GrossPitaevskiiFunctional(System& system, double beta, SolverOptions options)
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
        // DiagonalInverse is exact and set up by its constructor
        if constexpr (!lumped) {
            M_inv.update_static(system.get_M());
            M_inv.update_dynamic(M.diagonal());
        }
    }

    /** @brief Reassembles \f$ M_{\phi\phi}(x) \f$ and the Jacobi preconditioner of \f$ A(x) \f$. */
    void update(const Vector<double>& x)
    {
        // Updates A, M as references to system
        system.assemble_nonlinear_term(x);

        // TODO: lazy evaluation for preconditioner updates in A_inv
        A_inv.update_dynamic(A.diagonal());
    }

    /** @brief \f$ E(x) = \frac12 x^\top A_0 x + \frac\beta4 x^\top M_{\phi\phi}(x) x \f$ */
    double value(const Vector<double>& x) const
    {
        auto A_eval = system.get_operator_A(beta*0.25, 0.5);

        Vector<double> Ax(x.size());
        A_eval.vmult(Ax, x);

        return x * Ax;
    }

    /** @brief \f$ (A(x) x)^\top z \f$ */
    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const
    {
        Vector<double> Ax(x.size());
        A.vmult(Ax, x);

        return Ax * z;
    }

    /** @brief Euclidean gradient \f$ A(x) x \f$. */
    void gradient(const Vector<double>& x, Vector<double>& output) const
    {
        A.vmult(output, x);
    }

    // Accessors
    unsigned n_dofs() const { return system.n_dofs(); }
    double get_beta() const { return beta; }

    const Operator& get_M() const { return M; }
    const Operator& get_A() const { return A; }
    const SparseMatrix<double>& get_A0() const { return system.get_A0(); }

    const InverseM& get_M_inv() const { return M_inv; }
    InverseM& get_M_inv() { return M_inv; }

    const InverseA& get_A_inv() const { return A_inv; }
    InverseA& get_A_inv() { return A_inv; }


private:
    System& system;
    double beta;
    Operator M, A;
    InverseM M_inv;
    InverseA A_inv;
};

} // namespace rmo::gpe


// Convenience overloads for assemble.h functions
namespace rmo::fe
{

template <int dim, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_mass(GlobalMatrix& system_matrix, const gpe::GrossPitaevskiiPackage<dim>& context,
                   unsigned int level = invalid_unsigned_int)
{
    assemble_mass(system_matrix, context.get_dofs(), context.get_quadrature(),
        context.get_mapping(), context.get_constraints(), level);
}

template <int dim, typename GlobalMatrix = dealii::DiagonalMatrix<dealii::Vector<double>>>
void assemble_mass_lumped(GlobalMatrix& system_matrix, const gpe::GrossPitaevskiiPackage<dim>& context,
                          unsigned int level = invalid_unsigned_int)
{
    assemble_mass_lumped(system_matrix, context.get_dofs(), context.get_quadrature(),
        context.get_mapping(), context.get_constraints(), level);
}

template <int dim, typename Function, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_mass_weighted(GlobalMatrix& system_matrix, Function&& V,
                            const gpe::GrossPitaevskiiPackage<dim>& context,
                            unsigned int level = invalid_unsigned_int)
{
    assemble_mass_weighted(system_matrix, V, context.get_dofs(), context.get_quadrature(),
        context.get_mapping(), context.get_constraints(), level);
}

template <int dim, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_stiffness(GlobalMatrix& system_matrix, const gpe::GrossPitaevskiiPackage<dim>& context,
                        unsigned int level = invalid_unsigned_int)
{
    assemble_stiffness(system_matrix, context.get_dofs(), context.get_quadrature(),
                context.get_mapping(), context.get_constraints(), level);
}

template <int dim, typename Function, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_A0(GlobalMatrix& system_matrix, Function&& V,
                 const gpe::GrossPitaevskiiPackage<dim>& context,
                 unsigned int level = invalid_unsigned_int)
{
    assemble_A0(system_matrix, V, context.get_dofs(), context.get_quadrature(),
        context.get_mapping(), context.get_constraints(), level);
}

template <int dim, typename Function, typename GlobalMatrix = dealii::SparseMatrix<double>>
void assemble_A0_lumped(GlobalMatrix& system_matrix, Function&& V,
                        const gpe::GrossPitaevskiiPackage<dim>& context,
                        unsigned int level = invalid_unsigned_int)
{
    assemble_A0_lumped(system_matrix, V, context.get_dofs(), context.get_quadrature(),
        context.get_mapping(), context.get_constraints(), level);
}

} // namespace rmo::fe

#endif //RMO_GPE_GPE_H