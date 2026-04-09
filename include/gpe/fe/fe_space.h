#ifndef GPE_DOFS_HH
#define GPE_DOFS_HH

#include <gpe/option_types.h>

// step 2 -- dof libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools.h>


namespace gpe
{
using dealii::numbers::invalid_unsigned_int;

/**
 * @brief Renumbers degrees of freedom to optimize the sparsity pattern of the system matrix.
 *
 * This function applies a reordering algorithm to the degrees of freedom managed by
 * the DoFHandler. Reordering is relevant for iterative solvers as it can significantly
 * reduce the bandwidth or profile of the sparse matrix, improving cache locality and
 * preconditioning performance.
 *
 * Supported algorithms:
 * - `Ordering::CUTHILL_MCKEE`: (Reverse) Cuthill-McKee algorithm. generally the best choice
 * for reducing bandwidth.
 * - `Ordering::RANDOM`: Random shuffling (mostly for testing/debugging).
 * - `Ordering::DEFAULT`: No action.
 *
 * @tparam dim The spatial dimension of the domain.
 * @param[in,out] dof_handler The DoFHandler object to be renumbered.
 * @param[in] order The desired reordering algorithm (default: Cuthill-McKee).
 * @param[in] use_constraints If true, the connectivity graph used for reordering will
 * respect constraints (e.g., hanging nodes), leading to a better ordering at the
 * cost of computation time.
 * @param[in] reversed_numbering If true, applies the Reverse Cuthill-McKee (RCM) ordering.
 * RCM is typically preferred over standard CM for minimizing bandwidth. See:
 * W. Liu, A. Sherman. Comparative Analysis of the Cuthill-Mckee and the Reverse Cuthill-Mckee
 * Ordering Algorithms for Sparse Matrices, SIAM, 1976.
 *
 * @throws std::invalid_argument if an unknown ordering type is passed.
 */
template <int dim>
void renumber_dofs(dealii::DoFHandler<dim>& dof_handler,
                   const Ordering order     = Ordering::CUTHILL_MCKEE,
                   bool use_constraints     = false,
                   bool reversed_numbering  = false)
{
    switch (order) {
        case Ordering::DEFAULT:
            break;
        case Ordering::RANDOM:
            dealii::DoFRenumbering::random(dof_handler);
            break;
        case Ordering::CUTHILL_MCKEE:
            dealii::DoFRenumbering::Cuthill_McKee(dof_handler, reversed_numbering, use_constraints);
            break;
        default:
            throw std::invalid_argument("unknown ordering");
    }
}

/**
 * @brief Renumbers degrees of freedom on a specific multigrid level.
 *
 * Similar to renumber_dofs(), but acts on the level-specific DoFs required for
 * geometric multigrid (GMG) smoothers. This ensures that the local smoothing matrices
 * on each level have optimized bandwidths.
 *
 * @tparam dim The spatial dimension of the domain.
 * @param[in,out] dof_handler The DoFHandler object managing the MG hierarchy.
 * @param[in] level The specific multigrid level (0 to n_levels-1) to renumber.
 * @param[in] order The desired reordering algorithm (default: Cuthill-McKee).
 * @param[in] reversed_numbering If true, applies Reverse Cuthill-McKee (RCM).
 *
 * @throws std::invalid_argument if an unknown ordering type is passed.
 */
template <int dim>
void renumber_dofs_mg(dealii::DoFHandler<dim>& dof_handler, unsigned int level,
                      const Ordering order = Ordering::CUTHILL_MCKEE,
                      bool reversed_numbering = false)
{
    switch (order) {
        case Ordering::DEFAULT:
            break;
        case Ordering::RANDOM:
            dealii::DoFRenumbering::random(dof_handler, level);
            break;
        case Ordering::CUTHILL_MCKEE:
            dealii::DoFRenumbering::Cuthill_McKee(dof_handler, level, reversed_numbering);
            break;
        default:
            throw std::invalid_argument("unknown ordering");
    }
}

/**
 * @brief Manages the Finite Element space, DoF distribution, and constraints.
 *
 * This class acts as a wrapper around `dealii::DoFHandler` and `dealii::AffineConstraints`.
 * It handles the initialization of the finite element space, including distributing
 * degrees of freedom, applying renumbering for performance, and computing constraints
 * arising from hanging nodes and boundary conditions.
 *
 * Usage:
 * 1. Construct with a valid triangulation.
 * 2. Call `setup_dofs()` to associate a FiniteElement and distribute indices.
 * 3. Call `setup_constraints()` to apply boundary conditions and hanging nodes.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim>
class FeSpace
{
public:
    /**
     * @brief Constructor initializing the DoFHandler with a triangulation.
     * @param triangulation The underlying mesh.
     */
    FeSpace(const dealii::Triangulation<dim>& triangulation)
        : dof_handler(triangulation)
    {}

    FeSpace() {}
    FeSpace(const FeSpace&) = delete;
    FeSpace& operator=(const FeSpace&) = delete;

    /**
     * @brief Distributes DoFs and optionally renumbers them.
     *
     * This function associates the given finite element with the DoFHandler,
     * allocates memory for DoF indices, and optionally renumbers them to reduce
     * matrix bandwidth.
     *
     * @param order The reordering strategy (e.g., CUTHILL_MCKEE).
     * @param element The finite element description (e.g., FE_Q, FE_SimplexP).
     */
    void setup_dofs(const Ordering order, const dealii::FiniteElement<dim>& element)
    {
        // Distribute degrees of freedom according to (default or other) ordering,
        // such that a basis of V_h can be enumerated in a deterministic way
        // This function stores a copy of the finite element given as argument
        dof_handler.distribute_dofs(element);

        // Reorder degrees of freedom for improved conditioning of system matrix
        // (default: order vertices, faces, ... by refinement level)
        if (order != Ordering::DEFAULT) {
            renumber_dofs<dim>(dof_handler, order);
        }
    }

    /**
     * @brief Computes constraints for hanging nodes and boundary conditions.
     *
     * Populates the `AffineConstraints` object. It first computes hanging node
     * constraints (essential for adaptive refinement) and then applies Dirichlet
     * boundary values if specified.
     *
     * @param bounds The type of boundary condition to apply (e.g., DIRICHLET).
     */
    void setup_constraints(const BoundaryCondition bounds)
    {
        dealii::Functions::ZeroFunction<dim> boundary_function(dof_handler.get_fe().n_components());

        // Define hanging nodes (optional for global refinement)
        constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        // Set boundary condition for linear system (after dof distribution)
        if (bounds == BoundaryCondition::DIRICHLET) {
            dealii::VectorTools::interpolate_boundary_values(dof_handler,
                0, boundary_function, constraints);
        }
        constraints.close();
    }

    /** @return Const reference to the underlying DoFHandler. */
    const dealii::DoFHandler<dim>& get_dofs() const {
        return dof_handler;
    }

    /** @return Const reference to the FiniteElement used. */
    const dealii::FiniteElement<dim>& get_fe() const {
        return dof_handler.get_fe();
    }

    /** @return The global number of degrees of freedom. */
    unsigned int n_dofs() const {
        return dof_handler.n_dofs();
    }

    /** @return Const reference to the computed constraints (hanging nodes + BCs). */
    const dealii::AffineConstraints<double>& get_constraints() const{
        return constraints;
    }

private:
    dealii::DoFHandler<dim> dof_handler;
    dealii::AffineConstraints<double> constraints;
};


/**
 * @brief Manages the Finite Element space for Geometric Multigrid (GMG) methods.
 *
 * Extends the functionality of `FeSpace` to support Multigrid hierarchies.
 * In addition to global DoFs, this class manages level-specific DoFs and constraints
 * via `dealii::MGConstrainedDoFs`. This is required for level transfer operators
 * and smoothers in MG solvers.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim>
class FeSpaceMG
{
public:
    /**
     * @brief Constructor initializing the DoFHandler with a triangulation.
     * @param triangulation The underlying mesh.
     */
    FeSpaceMG(const dealii::Triangulation<dim>& triangulation)
        : dof_handler(triangulation)
    {}

    FeSpaceMG() {}
    FeSpaceMG(const FeSpaceMG&) = delete;
    FeSpaceMG& operator=(const FeSpaceMG&) = delete;

    /**
     * @brief Distributes global and level-wise DoFs, and applies renumbering.
     *
     * This calls `distribute_dofs` for the active mesh and `distribute_mg_dofs`
     * for the multigrid levels. If an ordering strategy is provided, it is applied
     * to every level in the hierarchy to ensure consistent efficiency across levels.
     *
     * @param order The reordering strategy.
     * @param element The finite element description.
     */
    void setup_dofs(const Ordering order, const dealii::FiniteElement<dim>& element)
    {
        const unsigned int n_levels = dof_handler.get_triangulation().n_levels();

        // Distribute degrees of freedom according to (default or other) ordering,
        // such that a basis of V_h can be enumerated in a deterministic way
        dof_handler.distribute_dofs(element);

        // Distribute level degrees of freedom on each level for geometric multigrid
        dof_handler.distribute_mg_dofs();

        // Reorder degrees of freedom for improved conditioning of system matrix
        // (default: order vertices, faces, ... by refinement level)
        if (order != Ordering::DEFAULT) {
            for (unsigned i = 0; i < n_levels; i++) {
                renumber_dofs_mg<dim>(dof_handler, i, order);
            }
        }
    }

    /**
     * @brief Computes global and level-wise constraints.
     *
     * 1. Computes global hanging node and boundary constraints.
     * 2. Initializes `MGConstrainedDoFs` to handle interface constraints between levels.
     * 3. Applies homogeneous Dirichlet BCs to the multigrid levels if specified.
     *
     * @param bounds The type of boundary condition to apply.
     */
    void setup_constraints(const BoundaryCondition bounds)
    {
        dealii::Functions::ZeroFunction<dim> boundary_function(dof_handler.get_fe().n_components());

        // Define hanging nodes (optional for global refinement)
        constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        // MG constraints
        mg_constraints.clear();
        mg_constraints.initialize(dof_handler);

        // Set boundary condition for linear system (after dof distribution)
        if (bounds == BoundaryCondition::DIRICHLET) {
            dealii::VectorTools::interpolate_boundary_values(dof_handler,
                0, boundary_function, constraints);

            // Apply Zero BCs to MG levels (essential for defect correction in MG)
            mg_constraints.make_zero_boundary_constraints(dof_handler, {0});
        }
        constraints.close();
    }

    /**
     * @brief Accessor for constraints on a specific multigrid level.
     * @param level The multigrid level index.
     * @return Constraints object for that level (usually contains boundary constraints).
     */
    const dealii::AffineConstraints<double>& get_level_constraints(unsigned int level) const {
        return mg_constraints.get_level_constraints(level);
    }

    /** @return Const reference to the Multigrid constraint handler. */
    const dealii::MGConstrainedDoFs& get_mg_dofs() const {
        return mg_constraints;
    }

    /** @return Const reference to the underlying DoFHandler. */
    const dealii::DoFHandler<dim>& get_dofs() const {
        return dof_handler;
    }

    /** @return Const reference to the FiniteElement used. */
    const dealii::FiniteElement<dim>& get_fe() const {
        return dof_handler.get_fe();
    }

    /** @return The global number of degrees of freedom. */
    unsigned int n_dofs() const {
        return dof_handler.n_dofs();
    }

    /** @return Const reference to the computed constraints (hanging nodes + BCs). */
    const dealii::AffineConstraints<double>& get_constraints() const{
        return constraints;
    }

private:
    dealii::DoFHandler<dim> dof_handler;
    dealii::MGConstrainedDoFs mg_constraints;
    dealii::AffineConstraints<double> constraints;
};

} // namespace gpe
#endif //GPE_DOFS_HH
