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

template <int dim>
class GPE
{
public:
    GPE(const GPE_Options& options, unsigned int n_levels)
        : grid(options.radius)
        , space(grid.get_triangulation(), options.degree)   // establish relations between objects
    {
        grid.setup_grid(n_levels);    // do the actual computations
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
        std::cerr << "Number of cells: " << grid.get_triangulation().n_active_cells() << std::endl;
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

    const FeSpace<dim>& fe_space() const { return space; }

private:
    HyperCube<dim> grid;
    FeSpace<dim>   space;
    LevelMatrix    system;
};

}

#endif //GPE_MAIN_H