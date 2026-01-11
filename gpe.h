#ifndef GPE_GPE_H
#define GPE_GPE_H
#define EXECUTION_POLICY gpe::execution::seq

#include <deal.II/numerics/matrix_creator.h>

#include "lac.h"
#include "mesh.h"
#include "dofs.h"
#include "assemble.h"

namespace gpe
{
// TODO: lac_types.h to easily change to different matrix implementation

struct GPE_Options
{
    int dimension;          // dimension of domain
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
    // TODO: allow passing in meshes
    //       constructor which accepts finite element (+ObserverPointer, step7)
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
    virtual ~GPE() = default;

    void make_grid(unsigned int n_levels)
    {
        // step 1 - regularly refined mesh
        make_cube(triangulation, options.radius, n_levels);
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
        has_active_constraints = true;

        // step 16 - formulate multigrid constraints
        mg_constrained_dofs = make_boundary_mg(dof_handler, options.bc, dirichlet_boundary_ids);
        has_mg_constraints  = true;
    }

    // TODO: set execution policy in macro, instead of at runtime
    template <typename Function>
    void assemble(Function&& V, SparseMatrix<double>& A0, SparseMatrix<double>& M,
        SparsityPattern& sparsity_pattern) const
    {
        const SparsityPattern sp = make_sparsity_pattern(dof_handler, constraints);
        sparsity_pattern.copy_from(sp);

        // A0: used for preconditioning (incomplete LU) and sum A0 + M_xx
        A0.reinit(sparsity_pattern);
        assemble_A0(EXECUTION_POLICY, A0, V, dof_handler, constraints);

        // M: used for energy metric
        M.reinit(sparsity_pattern);
        assemble_mass(EXECUTION_POLICY, M, dof_handler, constraints);
    }

    template <typename Function>
    void assemble_mg(Function&& V, SparseMatrix<double>& A0, SparseMatrix<double>& M,
        SparsityPattern& sparsity_pattern, unsigned int level) const
    {
        // "Note that there is [no] need to consider hanging nodes on the typical level matrices,
        // since only one level is considered."
        // TODO: we may still want to pass on Dirichlet constraints
        const SparsityPattern sp = make_sparsity_pattern_mg(dof_handler, level, {});
        sparsity_pattern.copy_from(sp);

        // A0: used for preconditioning (incomplete LU) and sum A0 + M_xx
        A0.reinit(sparsity_pattern);
        assemble_A0(EXECUTION_POLICY, A0, V, dof_handler, get_level_constraints(level), level);

        // M: used for energy metric
        M.reinit(sparsity_pattern);
        assemble_mass(EXECUTION_POLICY, M, dof_handler, get_level_constraints(level), level);
    }

    void assemble_phiphi(SparseMatrix<double>& Mpp, const Vector<double>& x) const
    {
        assemble_mass_phiphi<dim>(EXECUTION_POLICY, Mpp, x, dof_handler, constraints);
    }

    void assemble_phiphi_mg(SparseMatrix<double>& Mpp, const Vector<double>& x, unsigned int level) const
    {
        assemble_mass_phiphi<dim>(EXECUTION_POLICY, Mpp, x, dof_handler, get_level_constraints(level), level);
    }

    const dealii::AffineConstraints<double>&
    get_constraints() const
    {
        if (!has_active_constraints) {
            throw dealii::ExcEmptyObject("GPE::get_constraints(): call dofs() or dofs_mg() first");
        }
        return constraints;
    }

    const dealii::AffineConstraints<double>&
    get_level_constraints(const unsigned level) const
    {
        if (!has_mg_constraints) {
            throw dealii::ExcEmptyObject("GPE::get_mg_constraints(): call dofs_mg() first");
        }
        return mg_constrained_dofs.get_level_constraints(level);
    }

    unsigned int n_levels() const {
        return triangulation.n_levels();
    }
    unsigned int n_levels(unsigned int level) const {
        return triangulation.n_levels(level);
    }
    unsigned int n_cells() const {
        return triangulation.n_cells();
    }
    unsigned int n_cells(unsigned int level) const {
        return triangulation.n_cells(level);
    }
    unsigned int n_active_cells() const {
        return triangulation.n_active_cells();
    }
    unsigned int n_dofs() const {
        return dof_handler.n_dofs();
    }
    unsigned int n_dofs(unsigned int level) const {
        return dof_handler.n_dofs(level);
    }
    const dealii::DoFHandler<dim>& get_dofs() const {
        return dof_handler;
    }
    const dealii::MGConstrainedDoFs& get_mg_dofs() const {
        return mg_constrained_dofs;
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