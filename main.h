#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "assemble.h"
#include "gpe.h"
#include "descent.h"
#include "util.h"

#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/numerics/fe_field_function.h>

namespace gpe
{
using dealii::AffineConstraints;
using dealii::Triangulation;
using dealii::DoFHandler;
using dealii::MGLevelObject;
using dealii::MGTransferPrebuilt;

template <int dim>
class GPE_Solve : public DiscreteProblemActive<dim>
{
public:
    using solver_kind = plain_solver_tag;

    template <typename Function>
    explicit GPE_Solve(const GPE_Options& options, Function&& V, const unsigned int n_levels_)
        : GPE<dim>(options), n_levels(n_levels_)
    {
        // `this` is required when inheriting from a templated base class
        this->make_grid(n_levels);
        this->dofs();
        this->assemble(V, A0, M, sparsity_pattern);

        n_active_cells = this->get_triangulation().n_active_cells();
        n_dofs = this->get_dofs().n_dofs();
    }

    void plot_grid(const std::string& prefix) const
    {
        const std::string filename = prefix + "_" + std::to_string(dim) + "{}";
        if (dim == 2) {
            grid2file(filename + ".svg", triangulation, dealii::GridOut::OutputFormat::svg);
        }
        grid2file(filename + ".gnuplot", triangulation, dealii::GridOut::OutputFormat::gnuplot);
    }

    [[maybe_unused]] Vector<double>
    run(const Vector<double>& x0, double beta, GdOptions options_rgd, std::ostream& os) const
    {
        // Compute solution on most refined (active) level
        std::cerr << "Number of cells: " << n_active_cells << std::endl;
        std::cerr << "Number of degrees of freedom: " << n_dofs << std::endl;

        // Weighed mass matrix for solution in every step
        SparseMatrix<double> Mpp(sparsity_pattern);

        auto update_mpp = [this](SparseMatrix<double>& matrix, const Vector<double>& x)
        {
            this->assemble_phiphi(matrix, x);
        };

        // Run gradient descent + enforce boundary conditions
        // TODO: abstraction leak get_constraints()
        Vector<double> x = gp_energy_rgd<dim>(A0, M, Mpp, update_mpp,
            x0, beta, this->get_constraints(), options_rgd, os);
        return x;
    }

    // Iteration with constant starting value
    [[maybe_unused]] Vector<double>
    run(const double x0d, double beta, GdOptions options_rgd, std::ostream& os) const
    {
        // Define starting value
        Vector<double> x0(n_dofs);
        x0 = x0d;

        Vector<double> x = run(x0, beta, options_rgd, os);
        return x;
    }

private:
    unsigned int n_levels, n_dofs, n_active_cells;

    // Linear system parameters
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> A0;
    SparseMatrix<double> M;
};

template <int dim>
class GPE_Solve_MG : public GPE<dim>
{
public:
    using solver_kind = mg_solver_tag;

    template <typename Function>
    explicit GPE_Solve_MG(const GPE_Options& options, Function&& V, const unsigned int n_levels_,
        const unsigned int min_level_,
        const unsigned int max_level_)
    :
        GPE<dim>(options), n_levels(n_levels_), min_level(min_level_), max_level(max_level_),
        A0_v(min_level, max_level-1),
         M_v(min_level, max_level-1),
        sp_v(min_level, max_level-1)
    {
        AssertIndexRange(max_level-1, n_levels);
        this->make_grid(n_levels);
        this->dofs_mg();

        for (unsigned int level = min_level; level < max_level; level++) {
            // TODO: ensure consistency between level and passed on matrices/sparsity patterns
            this->assemble_mg(V, A0_v[level], M_v[level], sp_v[level], level);
        }

    }

    [[maybe_unused]] MGLevelObject<Vector<double>>
    run(const double x0d, double beta, GdOptions options_rgd, std::ostream& os) const
    {
        // Build transfer operators
        MGTransferPrebuilt<Vector<double> > mg_transfer(this->get_mg_dofs());
        mg_transfer.build(this->get_dofs());

        // Iterate over levels
        MGLevelObject<Vector<double> > x_v(min_level, max_level-1);

        for (unsigned int level = min_level; level < max_level; level++) {
            unsigned int n_level_dofs = this->get_dofs().n_dofs(level);
            unsigned int n_level_cells = this->get_triangulation().n_cells(level);

            std::cerr << "Level: " << level << std::endl;
            std::cerr << "Number of cells: " << n_level_cells << std::endl;
            std::cerr << "Number of degrees of freedom: " << n_level_dofs << std::endl;

            // Define starting value
            // TODO: take MGLevelObject of starting vectors, then overload on constant value
            Vector<double> x0(n_level_dofs);
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

            // Weighed mass matrix for solution in every step
            SparseMatrix<double> Mpp(sp_v[level]);

            auto update_mpp_level = [this, level](SparseMatrix<double>& Mpp, const Vector<double>& x)
            {
                this->assemble_phiphi_mg(Mpp, x, level);
            };

            // Gradient descent + enforce boundary conditions
            // TODO: abstraction leak get_level_constraints()
            x_v[level] = gp_energy_rgd<dim>(A0_v[level], M_v[level], Mpp,
                update_mpp_level, x0, beta, this->get_level_constraints(level), options_rgd, os);
            std::cerr << std::endl;
        }
        return x_v;
    }

private:
    unsigned int n_levels, min_level, max_level;

    // Linear system parameters
    MGLevelObject<SparseMatrix<double>> A0_v;  // inclusive range
    MGLevelObject<SparseMatrix<double>> M_v;
    MGLevelObject<SparsityPattern> sp_v;
};

}

#endif //GPE_MAIN_H