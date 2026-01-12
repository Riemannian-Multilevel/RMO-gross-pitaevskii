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
class GPE : public HyperCube<dim>, public FeSpace<dim>
{
public:
    GPE(const GPE_Options& options, unsigned int n_levels)
        : HyperCube<dim>(options.radius)
        , FeSpace<dim>(HyperCube<dim>::get_triangulation(), options.degree)   // establish relations between objects
        , options_(options)
    {
        this->setup_grid(n_levels);    // do the actual computations
        this->setup_dofs(options.order);
        this->setup_constraints(options.bc);
    }

    template <typename Function>
    void assemble(Function&& V)
    {
        system.reinit(make_sparsity_pattern(this->dof_handler, this->constraints));

        // Fixed mass and stiffness matrix
        assemble_A0  (system.A0, V, this->dof_handler, this->constraints);
        assemble_mass(system.M, this->dof_handler, this->constraints);
    }

    [[maybe_unused]] Vector<double>
    run(const Vector<double>& x0, double beta, GdOptions options_rgd, std::ostream& os)
    {
        // Compute solution on most refined (active) level
        std::cerr << "Number of cells: " << this->triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << this->dof_handler.n_dofs() << std::endl;

        // Weighed mass matrix for solution in every step
        auto update_mpp = [this](SparseMatrix<double>& matrix, const Vector<double>& x)
        {
            assemble_mass_phiphi(matrix, x, this->dof_handler, this->constraints);
        };

        // Run gradient descent + enforce boundary conditions
        // TODO: abstraction leak `constraints`
        Vector<double> x = gp_energy_rgd<dim>(system.A0, system.M, system.Mpp, update_mpp,
            x0, beta, this->constraints, options_rgd, os);
        return x;
    }

    // Iteration with constant starting value
    [[maybe_unused]] Vector<double>
    run(const double x0d, double beta, GdOptions options_rgd, std::ostream& os)
    {
        // Define starting value
        Vector<double> x0(this->dof_handler.n_dofs());
        x0 = x0d;

        Vector<double> x = run(x0, beta, options_rgd, os);
        return x;
    }

private:
    GPE_Options options_;
    LevelMatrix system;
};

}

#endif //GPE_MAIN_H