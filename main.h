//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "assemble.h"
#include "grid.h"
#include "sparsity.h"
#include "fe_space.h"
#include "descent.h"
#include "option_types.h"

namespace gpe
{
struct LevelMatrix
{
    SparseMatrix<double> A0, M, Mpp;
    SparsityPattern sparsity_pattern;
    unsigned int level;

    void reinit(DynamicSparsityPattern&& sp)
    {
        sparsity_pattern.copy_from(sp);
        A0.reinit(sparsity_pattern);
        M.reinit(sparsity_pattern);
        Mpp.reinit(sparsity_pattern);
    }
};

template <int dim>
class GPE
{
public:
    GPE(const GPE_Options& options, unsigned int n_levels)
        : grid{}, system{}, space(grid.triangulation, options.degree)   // establish relations between objects
    {
        grid.setup_grid(options.radius, n_levels);    // do the actual computations
        std::cerr << "Number of levels: " << grid.triangulation.n_global_levels() << std::endl;
        std::cerr << "Number of vertices: " << grid.triangulation.n_vertices() << std::endl;

        space.setup_dofs(options.order);
        space.setup_constraints(options.bc);
    }

    template <typename Function>
    void assemble(Function&& V)
    {
        const auto& dof_handler = space.get_dofs();
        const auto& constraints = space.get_constraints();

        system.reinit(make_sparsity_pattern(dof_handler, constraints));

        // Fixed mass and stiffness matrix
        assemble_A0  (system.A0, V, dof_handler, constraints);
        assemble_mass(system.M, dof_handler, constraints);
    }

    [[maybe_unused]] Vector<double>
    run(const Vector<double>& x0, double beta, GdOptions options_rgd, std::ostream& os)
    {
        const auto& dof_handler = space.get_dofs();
        const auto& constraints = space.get_constraints();

        // Compute solution on most refined (active) level
        std::cerr << "Number of cells: " << grid.triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << space.n_dofs() << std::endl;

        // Weighed mass matrix for solution in every step
        auto update_mpp = [&dof_handler, &constraints](SparseMatrix<double>& matrix, const Vector<double>& x)
        {
            assemble_mass_phiphi(matrix, x, dof_handler, constraints);
        };

        // Run gradient descent + enforce boundary conditions
        // TODO: abstraction leak `constraints`
        Vector<double> x = gp_energy_rgd<dim>(system.A0, system.M, system.Mpp, update_mpp,
            x0, beta, constraints, options_rgd, os);
        return x;
    }

    // Iteration with constant starting value
    [[maybe_unused]] Vector<double>
    run(const double x0d, double beta, GdOptions options_rgd, std::ostream& os)
    {
        // Define starting value
        Vector<double> x0(space.n_dofs());
        x0 = x0d;

        Vector<double> x = run(x0, beta, options_rgd, os);
        return x;
    }

    const FeSpace<dim,dealii::FE_Q<dim>>& fe_space() const { return space; }
    unsigned int n_dofs() const { return space.n_dofs(); }
    const dealii::DoFHandler<dim>& get_dofs() const { return space.get_dofs(); }
    const dealii::AffineConstraints<double>& get_constraints() const { return space.get_constraints(); }

private:
    HyperCube<dim> grid;
    LevelMatrix    system;
    FeSpace<dim,dealii::FE_Q<dim>>   space;
};

}

#endif //GPE_MAIN_H