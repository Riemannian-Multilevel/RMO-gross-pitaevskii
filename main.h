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
    explicit GPE_Solve(const GPE_Options& options)
        : problem(options)
    {}

    void setup()
    {
        // TODO: grid based on KellyErrorEstimator (run after different solves, so setup outside the constructor)
        //       make_grid_graded() leads to increased conditioning, due to hanging node constraints
        problem.make_grid();
        problem.dofs();
    }

    // Populate matrix A_0 = M_V + S based on boundary conditions
    template <typename Function>
    void get_matrix(Function&& V, LevelMatrix& lm) const
    {
        // Construct sparsity pattern
        const AffineConstraints<double>& constraints = problem.get_constraints();
        lm.reinit(make_sparsity_pattern(problem.get_dofs(), constraints));

        // Assemble matrix + boundary conditions
        assemble_mass(policy, lm.M, problem.get_dofs(), constraints);
        assemble_A0(policy, lm.A0, problem.get_dofs(), V, constraints);
    }

    template <typename Function>
    [[maybe_unused]] Vector<double>
    run(Function&& V, const Vector<double>& x0, double beta, GdOptions options_rgd, int n_check_res = 5) const
    {
        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        const Triangulation<dim>& triangulation = problem.get_triangulation();

        // Compute solution on most refined (active) level
        std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Populate matrices
        // TODO: store in class object? (access for normalization etc.)
        LevelMatrix lm;
        this->get_matrix(V, lm);

        // Update weighed matrix for current solution + boundary conditions
        const AffineConstraints<double>& constraints = problem.get_constraints();

        auto update_mpp = [&dof_handler, &constraints](
            SparseMatrix<double>& Mpp, const Vector<double>& x)
        {
            assemble_mass_phiphi<dim>(policy, Mpp, dof_handler, x, constraints);
        };

        // Run gradient descent + enforce boundary conditions
        Vector<double> x = gp_energy_rgd<dim>(lm.A0, lm.M, lm.Mpp,
            update_mpp, x0, beta, constraints, options_rgd, n_check_res);
        return x;
    }

    // Iteration with constant starting value
    template <typename Function>
    [[maybe_unused]] Vector<double>
    run(Function&& V, const double x0d, double beta, GdOptions options_rgd, int n_check_res = 5) const
    {
        // Define starting value
        Vector<double> x0(problem.get_dofs().n_dofs());
        x0 = x0d;

        Vector<double> x = run(V, x0, beta, options_rgd, n_check_res);
        return x;
    }

    const DoFHandler<dim>& get_dofs() const
    {
        return problem.get_dofs();
    }
    unsigned int n_dofs() const
    {
        return problem.n_dofs();
    }
    const AffineConstraints<double>& get_constraints() const
    {
        return problem.get_constraints();
    }
private:
    // Problem parameters
    GPE<dim> problem;
    static constexpr ExecutionPolicy policy{};
};

}

#endif //GPE_MAIN_H