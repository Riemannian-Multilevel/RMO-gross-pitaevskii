#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "operators.h"
#include "gpe.h"
#include "descent.h"
#include "util.h"

#include <deal.II/numerics/fe_field_function.h>

namespace gpe
{
using dealii::AffineConstraints;
using dealii::Triangulation;
using dealii::DoFHandler;

template <int dim, typename ExecutionPolicy>
class GPE_Solve
{
public:
    using solver_kind = plain_solver_tag;

    explicit GPE_Solve(const GPE_Options& options, const unsigned int n_levels_)
        : problem(options), n_levels(n_levels_)
    {}

    void setup()
    {
        // TODO: grid based on KellyErrorEstimator (run after different solves, so setup outside the constructor)
        problem.make_grid(n_levels);
        problem.dofs();

        const auto sp = make_sparsity_pattern(problem.get_dofs(),
            problem.get_constraints());
        sparsity_pattern.copy_from(sp);
    }

    // Populate matrix A_0 = M_V + S based on boundary conditions
    template <typename Function>
    void assemble_matrix(Function&& V)
    {
        const AffineConstraints<double>& constraints = problem.get_constraints();

        // A0: used for preconditioning (incomplete LU) and sum A0 + M_xx
        A0.reinit(sparsity_pattern);
        assemble_A0(policy, A0, problem.get_dofs(), V, constraints);

        // M: used for energy metric
        M.reinit(sparsity_pattern);
        assemble_mass(policy, M, problem.get_dofs(), constraints);
        is_assembled = true;
    }

    [[maybe_unused]] Vector<double>
    run(const Vector<double>& x0, double beta, GdOptions options_rgd, int n_check_res, std::ostream& os) const
    {
        if (!is_assembled)
            throw dealii::ExcEmptyObject("GPE_Solve::run(): call assemble_matrix() first");

        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        const Triangulation<dim>& triangulation = problem.get_triangulation();
        const AffineConstraints<double>& constraints = problem.get_constraints();

        // Compute solution on most refined (active) level
        std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        auto update_mpp = [&dof_handler, &constraints](
            SparseMatrix<double>& Mpp, const Vector<double>& x)
        {
            assemble_mass_phiphi<dim>(policy, Mpp, dof_handler, x, constraints);
        };

        // Weighed mass matrix for solution in every step
        SparseMatrix<double> Mpp(sparsity_pattern);

        // Run gradient descent + enforce boundary conditions
        Vector<double> x = gp_energy_rgd<dim>(A0, M, Mpp,
            update_mpp, x0, beta, constraints, options_rgd, n_check_res, os);
        return x;
    }

    // Iteration with constant starting value
    [[maybe_unused]] Vector<double>
    run(const double x0d, double beta, GdOptions options_rgd, int n_check_res, std::ostream& os) const
    {
        // Define starting value
        Vector<double> x0(problem.get_dofs().n_dofs());
        x0 = x0d;

        Vector<double> x = run(x0, beta, options_rgd, n_check_res, os);
        return x;
    }

    const DoFHandler<dim>& get_dofs() const
    {
        return problem.get_dofs();
    }
    [[nodiscard]] unsigned int n_dofs() const
    {
        return problem.n_dofs();
    }
    [[nodiscard]] const AffineConstraints<double>& get_constraints() const
    {
        return problem.get_constraints();
    }
private:
    // Problem parameters
    GPE<dim> problem;
    static constexpr ExecutionPolicy policy{};
    unsigned int n_levels;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> A0;
    SparseMatrix<double> M;
    bool is_assembled = false;
};

}

#endif //GPE_MAIN_H