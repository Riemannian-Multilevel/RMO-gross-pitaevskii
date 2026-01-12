#ifndef GPE_MAIN_H
#define GPE_MAIN_H
#define EXECUTION_POLICY gpe::execution::seq

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
class GPE : public GPE_Grid<dim>, public FeSpaceActive<dim>
{
public:
    using solver_kind = plain_solver_tag;

    GPE(double radius, const unsigned int degree, Ordering order, BoundaryCondition bounds,
        const unsigned int n_levels)
        : GPE_Grid<dim>(radius)
        , FeSpaceActive<dim>(GPE_Grid<dim>::get_triangulation(), degree)  // establish relations between objects
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
        assemble_A0  (EXECUTION_POLICY, system.A0, V,
            this->dof_handler, this->constraints);
        assemble_mass(EXECUTION_POLICY, system.M,
            this->dof_handler, this->constraints);
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
            assemble_mass_phiphi(EXECUTION_POLICY, matrix, x, this->dof_handler, this->constraints);
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

template <int dim>
class GPE_MG : public GPE_Grid<dim>, public FeSpaceMG<dim>
{
public:
    using solver_kind = mg_solver_tag;

    explicit GPE_MG(double radius, const unsigned int degree, Ordering order, BoundaryCondition bounds,
                    const unsigned int n_levels,
                    const unsigned int min_l,
                    const unsigned int max_l)
        : GPE_Grid<dim>(radius)
        , FeSpaceMG<dim>(GPE_Grid<dim>::get_triangulation(), degree)  // establish relations between objects
        , min_level(min_l)
        , max_level(max_l)
    {
        this->setup_grid(n_levels);
        this->setup_dofs(order);
        this->setup_constraints(bounds);

        system_v.resize(min_level, max_level-1);
    }

    template <typename Function>
    void assemble(Function&& V)
    {
        for (unsigned int i = min_level; i < max_level; i++) {
            system_v[i].level = i;
            system_v[i].reinit(make_sparsity_pattern_mg(this->dof_handler, this->get_mg_dofs(), i));

            // Fixed mass and stiffness matrix
            // TODO: ensure consistency between level and passed on matrices/sparsity patterns
            assemble_A0(EXECUTION_POLICY, system_v[i].A0, V,
                this->dof_handler, this->get_level_constraints(i), i);
            assemble_mass(EXECUTION_POLICY, system_v[i].M,
                this->dof_handler, this->get_level_constraints(i), i);
        }
    }

    [[maybe_unused]] MGLevelObject<Vector<double>>
    run(const double x0d, double beta, GdOptions options_rgd, std::ostream& os)
    {
        // Build transfer operators
        MGTransferPrebuilt<Vector<double>> mg_transfer(this->get_mg_dofs());
        mg_transfer.build(this->get_dofs());

        // Iterate over levels
        MGLevelObject<Vector<double> > x_v(min_level, max_level-1);

        for (unsigned int level = min_level; level < max_level; level++) {
            unsigned int n_level_dofs  = this->get_dofs().n_dofs(level);
            unsigned int n_level_cells = this->get_triangulation().n_cells(level);

            std::cerr << "Level: " << level << std::endl;
            std::cerr << "Number of cells: " << n_level_cells << std::endl;
            std::cerr << "Number of degrees of freedom: " << n_level_dofs << std::endl;

            // Define starting value
            // TODO: take MGLevelObject of starting vectors, then overload on constant value
            Vector<double> x0(n_level_dofs);
            x0 = x0d;

            // Weighed mass matrix for solution in every step
            auto update_mpp_level = [this, level](SparseMatrix<double>& matrix, const Vector<double>& x)
            {
                assemble_mass_phiphi(EXECUTION_POLICY, matrix, x,
                    this->dof_handler, this->get_level_constraints(level), level);
            };

            // Gradient descent + enforce boundary conditions
            // TODO: abstraction leak get_level_constraints()
            x_v[level] = gp_energy_rgd<dim>(system_v[level].A0, system_v[level].M, system_v[level].Mpp,
                update_mpp_level, x0, beta, this->get_level_constraints(level), options_rgd, os);
            std::cerr << std::endl;
        }
        return x_v;
    }

private:
    MGLevelObject<LevelMatrix> system_v;
    unsigned int min_level;
    unsigned int max_level;
};

}

#endif //GPE_MAIN_H