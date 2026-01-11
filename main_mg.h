#ifndef GPE_MAIN_MG_H
#define GPE_MAIN_MG_H
#include "operators.h"
#include "gpe.h"
#include "descent.h"
#include "util.h"

#include <deal.II/multigrid/mg_transfer.h>

namespace gpe
{
using dealii::AffineConstraints;
using dealii::Triangulation;
using dealii::DoFHandler;
using dealii::MGLevelObject;
using dealii::MGTransferPrebuilt;


// TODO: inherit from GPE base class
template <int dim, typename ExecutionPolicy>
class GPE_Solve_MG
{
public:
    using solver_kind = mg_solver_tag;

    explicit GPE_Solve_MG(const GPE_Options& options, const unsigned int n_levels_,
        const unsigned int min_level_,
        const unsigned int max_level_)
    :
        problem(options), n_levels(n_levels_), min_level(min_level_), max_level(max_level_),
        A0_v(min_level, max_level-1),
        M_v (min_level, max_level-1),
        sp_v(min_level, max_level-1)
    {
        AssertIndexRange(max_level-1, n_levels);
    }

    void setup()
    {
        problem.make_grid(n_levels);
        problem.dofs_mg();

        for (unsigned int level = min_level; level < max_level; level++) {
            const auto sp = make_sparsity_pattern_mg(problem.get_dofs(), level, {});
            sp_v[level].copy_from(sp);
        }
    }

    template <typename Function>
    void assemble_matrix(Function&& V)
    {
        const DoFHandler<dim>& dof_handler = problem.get_dofs();

        // "Note that there is [no] need to consider hanging nodes on the typical level matrices,
        // since only one level is considered."
        // TODO: we may still want to pass on Dirichlet constraints
        for (unsigned int level = min_level; level < max_level; ++level) {
            const AffineConstraints<double>& level_constraints = problem.get_level_constraints(level);

            A0_v[level].reinit(sp_v[level]);
            assemble_A0(policy, A0_v[level], dof_handler, V, level_constraints, level);

            M_v[level].reinit(sp_v[level]);
            assemble_mass(policy, M_v[level], dof_handler, level_constraints, level);
        }
        is_assembled = true;
    }

    [[maybe_unused]] MGLevelObject<Vector<double>>
    run(const double x0d, double beta, GdOptions options_rgd, std::ostream& os) const
    {
        if (!is_assembled)
            throw dealii::ExcEmptyObject("GPE_Solve_MG::run(): call assemble_matrix() first");

        const Triangulation<dim>& triangulation = problem.get_triangulation();
        const DoFHandler<dim>& dof_handler = problem.get_dofs();

        // Build transfer operators
        MGTransferPrebuilt<Vector<double> > mg_transfer(problem.get_mg_dofs());
        mg_transfer.build(dof_handler);

        // Iterate over levels
        MGLevelObject<Vector<double> > x_v(min_level, max_level-1);

        for (unsigned int level = min_level; level < max_level; level++) {
            std::cerr << "Level: " << level << std::endl;
            std::cerr << "Number of cells: " << triangulation.n_cells(level) << std::endl;
            std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs(level) << std::endl;

            // Update weighed matrix for current solution + boundary conditions
            const AffineConstraints<double>& level_constraints = problem.get_level_constraints(level);

            // Define starting value
            // TODO: take MGLevelObject of starting vectors, then overload on constant value
            Vector<double> x0(dof_handler.n_dofs(level));
            x0 = x0d;

            // TODO: check for missing steps in tutorial/step-16
            // if (level == min_level) {
            //     // Constant value on coarsest level
            //     x0 = x0d;
            // }
            // else if (level+1 < max_level) {
            //     // Linear interpolation of solution on previous level
            //     mg_transfer.prolongate(level, x0, x_v[level-1]);
            //
            //     // Apply fine level constraints
            //     level_constraints.distribute(x0);
            //
            //     // Renormalize in M-norm
            //     Vector<double> Mx0(x0.size());
            //     lm_v[level].M.vmult(Mx0, x0);
            //     x0 /= std::sqrt(x0 * Mx0);
            // }

            auto update_mpp_level = [&dof_handler, level, &level_constraints](
                SparseMatrix<double>& Mpp, const Vector<double>& x)
            {
                assemble_mass_phiphi<dim>(policy, Mpp, dof_handler, x, level_constraints, level);
            };

            // Weighed mass matrix for solution in every step
            SparseMatrix<double> Mpp(sp_v[level]);

            // Gradient descent + enforce boundary conditions
            x_v[level] = gp_energy_rgd<dim>(A0_v[level], M_v[level], Mpp,
                update_mpp_level, x0, beta, level_constraints, options_rgd, os);
            std::cerr << std::endl;
        }
        return x_v;
    }

    const DoFHandler<dim>& get_dofs() const
    {
        return problem.get_dofs();
    }

private:
    // Problem parameters
    GPE<dim> problem;
    static constexpr ExecutionPolicy policy{};
    unsigned int n_levels, min_level, max_level;

    MGLevelObject<SparseMatrix<double>> A0_v;  // inclusive range
    MGLevelObject<SparseMatrix<double>> M_v;
    MGLevelObject<SparsityPattern> sp_v;
    bool is_assembled = false;
};

}
#endif //GPE_MAIN_MG_H