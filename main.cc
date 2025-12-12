//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include "function.h"
#include "operators.h"
#include "gpe.h"
#include "descent.h"
#include "util.h"
#include "options.h"

#include <deal.II/numerics/fe_field_function.h>
#include <iostream>
#include <fmt/format.h>

using namespace gpe;
using namespace dealii;


template <int dim, typename ExecutionPolicy>
class GPE_Solve
{
public:
    explicit GPE_Solve(const GPE_Options& options)
        : problem(options.radius, options.degree)
    {
        // TODO: grid based on KellyErrorEstimator (run after different solves, so move this outside the constructor?)
        //       make_grid_graded() leads to increased conditioning, due to hanging node constraints
        problem.make_grid(options.n_levels);
        problem.dofs(options.order, options.bc);
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

    // Begin iteration with constant starting value
    template <typename Function>
    [[maybe_unused]] Vector<double>
    run(Function&& V, const double x0d, double beta, GdOptions options_rgd, int n_check_res = 5) const
    {
        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        const Triangulation<dim>& triangulation = problem.get_triangulation();

        // Compute solution on most refined (active) level
        std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Define starting value
        Vector<double> x0(dof_handler.n_dofs());
        x0 = x0d;

        // Populate matrices
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

    void output(const Vector<double>& x)
    {
        using DataOutBase::OutputFormat::vtu;
        output_results(x, problem.get_dofs(), vtu, fmt::format("solution_{}d.vtu", dim));
    }
    const DoFHandler<dim>& get_dofs() const
    {
        return problem.get_dofs();
    }

private:
    // Problem parameters
    GPE<dim> problem;
    static constexpr ExecutionPolicy policy{};
};

template <int dim, typename ExecutionPolicy>
void package(const GPE_Options& options, const GdOptions& options_rgd)
{
    Square<dim> V;
    GPE_Solve<dim, ExecutionPolicy> GS(options);

    auto x = GS.run(V, 1.0, options.beta, options_rgd);
    GS.output(x);
}

template <int dim>
void run_package(bool parallel, const GPE_Options& options, const GdOptions& options_rgd)
{
    if (parallel) {
        package<dim, execution::par_t>(options, options_rgd);
    } else {
        package<dim, execution::seq_t>(options, options_rgd);
    }
}

int main(int argc, char* argv[])
{
    GPE_Options options{};
    GdOptions   options_rgd{};
    MG_Options  options_mg{};

    // TODO: add configuration file (cf. boost tutorial)
    try {
        add_options(argc, argv, options, options_rgd, options_mg);

        if (options_mg.multigrid) {
            throw std::logic_error("Multigrid is not supported in this program!");
        }

        with_dimension(options.dimension, [&](auto D)
        {
            constexpr int dim = decltype(D)::value;

            run_package<dim>(options_mg.parallel, options, options_rgd);
        });
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }
}
