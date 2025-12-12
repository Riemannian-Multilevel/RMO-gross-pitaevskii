#include "mesh.h"
#include "dofs.h"
#include "gpe.h"
#include "options.h"
#include "util.h"

#include <fmt/format.h>
#include <iostream>

using namespace gpe;
using namespace dealii;

template <int dim>
class GPE_Sparsity
{
public:
    explicit GPE_Sparsity(const GPE_Options& options)
    :
        problem(options.radius, options.degree)
    {
        problem.make_grid(options.n_levels);
        problem.dofs(options.order, options.bc);
    }

    void run(const std::string& prefix) const
    {
        const Triangulation<dim>& triangulation = problem.get_triangulation();
        std::cerr << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of levels: " << triangulation.n_levels() << std::endl;

        problem.plot_grid(prefix);

        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        const AffineConstraints<double>& constraints = problem.get_constraints();

        write_dof_locations(dof_handler, fmt::format("{}_{}d_dof.gnuplot", prefix, dim));

        auto Sd = make_sparsity_pattern(dof_handler, constraints);
        {
            std::ofstream out(fmt::format("{}_{}d_sparsity.svg", prefix, dim));
            Sd.print_svg(out);
        }
    }

private:
    GPE<dim> problem;
};

template <int dim>
class GPE_Sparsity_MG
{
public:
    explicit GPE_Sparsity_MG(const GPE_Options& options,
        unsigned int min_level_ = 0,
        unsigned int max_level_ = numbers::invalid_unsigned_int)
    :
        problem(options.radius, options.degree), min_level(min_level_), max_level(max_level_)
    {
        problem.make_grid(options.n_levels);
        problem.dofs_mg(options.order, options.bc);

        if (max_level == numbers::invalid_unsigned_int) {
            max_level = problem.get_triangulation().n_levels();
        }
    }

    // TODO: multigrid/colored print_grid
    void run(const std::string& prefix)
    {
        const Triangulation<dim>& triangulation = problem.get_triangulation();
        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        std::cerr << "Number of levels: " << max_level-min_level << std::endl;

        for (unsigned level = min_level; level < max_level; level++) {
            std::cerr << fmt::format("Number of cells (level {}): ", level)
                      << triangulation.n_cells(level) << std::endl;

            write_level_vertex_points(dof_handler, level,
                fmt::format("{}_{}d_dof_l{}.gnuplot", prefix, dim, level));

            auto Sd = make_sparsity_pattern_mg(dof_handler, level, {});
            {
                std::ofstream out(fmt::format("{}_{}d_sparsity_l{}.svg", prefix, dim, level));
                Sd.print_svg(out);
            }
        }
    }

private:
    GPE<dim> problem;
    unsigned min_level, max_level;
};

template <int dim>
void run_package(bool multigrid, unsigned min_level, unsigned max_level, const GPE_Options& options)
{
    if (multigrid) {
        GPE_Sparsity_MG<dim> GSM(options, min_level, max_level);
        GSM.run("domain");
    }
    else {
        GPE_Sparsity<dim> GS(options);
        GS.run("domain");
    }
}

int main(int argc, char** argv)
{
    GPE_Options options{};
    MG_Options options_mg{};

    try {
        add_options(argc, argv, options, {}, options_mg);

        with_dimension(options.dimension, [&](auto D)
        {
            constexpr int dim = decltype(D)::value;

            run_package<dim>(options_mg.multigrid, options_mg.min_level, options_mg.max_level, options);
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