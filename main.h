#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "assemble.h"
#include "fe_space.h"
#include "sparsity.h"
#include "descent.h"
#include "util.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/grid/grid_out.h>

namespace gpe
{
using dealii::AffineConstraints;
using dealii::Triangulation;
using dealii::DoFHandler;
using dealii::MGLevelObject;
using dealii::MGTransferPrebuilt;


template <int dim>
class GPE_Grid
{
public:
    GPE_Grid(double radius_)
        : triangulation(dealii::Triangulation<dim>::limit_level_difference_at_vertices), radius(radius_)
    {}
    virtual ~GPE_Grid() = default;

    void setup_grid(unsigned int n_levels)
    {
        // step 1 - regularly refined mesh
        dealii::GridGenerator::hyper_cube(triangulation, -radius, radius);

        // the number of cells increases by a factor of 2^(dim x times)
        triangulation.refine_global(n_levels-1);
        AssertDimension(n_levels, triangulation.n_global_levels());

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

    const Triangulation<dim>& get_triangulation() const {
        return triangulation;
    }

protected:
    Triangulation<dim> triangulation;
    double radius;
};

template <int dim>
class GPE : public GPE_Grid<dim>, public FeSpace<dim>
{
public:
    using solver_kind = plain_solver_tag;

    GPE(double radius, const unsigned int degree, Ordering order, BoundaryCondition bounds,
        const unsigned int n_levels)
        : GPE_Grid<dim>(radius)
        , FeSpace<dim>(GPE_Grid<dim>::get_triangulation(), degree)  // establish relations between objects
    {
        this->setup_grid(n_levels);    // do the actual computations
        this->setup_dofs(order);
        this->setup_constraints(bounds);
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
    LevelMatrix system;
};

}

#endif //GPE_MAIN_H