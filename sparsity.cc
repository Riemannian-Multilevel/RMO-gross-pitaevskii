#include "mesh.h"
#include "discrete.h"
#include "gpe.h"
#include "options.h"
#include "util.h"

#include <fmt/format.h>
#include <iostream>

using namespace gpe;
using namespace dealii;

// TODO: use gpe::DiscreteProblem (dofs.h)
template <int dim>
class GPE_Sparsity
{
public:
    explicit GPE_Sparsity(const GPE_Options& options)
    :
        problem(options)
    {}

    void setup()
    {
        problem.make_grid();
        problem.dofs();

        auto sp = make_sparsity_pattern(problem.get_dofs(), problem.get_constraints());
        sparsity_pattern.copy_from(sp);
    }

    void run(const std::string& prefix) const
    {
        const Triangulation<dim>& triangulation = problem.get_triangulation();
        std::cerr << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of levels: " << triangulation.n_levels() << std::endl;

        problem.plot_grid(prefix);

        write_dof_locations(problem.get_dofs(), fmt::format("{}_{}d_dof.gnuplot", prefix, dim));
        {
            std::ofstream out(fmt::format("{}_{}d_sparsity.svg", prefix, dim));
            sparsity_pattern.print_svg(out);
        }
    }

private:
    GPE<dim> problem;
    SparsityPattern sparsity_pattern;
};

template <int dim>
class GPE_Sparsity_MG
{
public:
    explicit GPE_Sparsity_MG(const GPE_Options& options,
        unsigned int min_level_,
        unsigned int max_level_)
    :
        problem(options), min_level(min_level_), max_level(max_level_)
    {
        AssertIndexRange(min_level, problem.get_triangulation().n_global_levels());
        AssertIndexRange(max_level, problem.get_triangulation().n_global_levels());
    }

    void setup()
    {
        problem.make_grid();
        problem.dofs_mg();

        for (unsigned level = min_level; level < max_level; level++) {
            auto Sd = make_sparsity_pattern_mg(problem.get_dofs(), level, {});
            sparsity_pattern_v[level].copy_from(Sd);
        }
    }

    void run(const std::string& prefix) const
    {
        const Triangulation<dim>& triangulation = problem.get_triangulation();
        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        std::cerr << "Number of levels: " << max_level-min_level << std::endl;

        for (unsigned level = min_level; level < max_level; level++) {
            std::cerr << fmt::format("Number of cells (level {}): ", level)
                      << triangulation.n_cells(level) << std::endl;

            write_level_vertex_points(dof_handler, level,
                fmt::format("{}_{}d_dof_l{}.gnuplot", prefix, dim, level));

            std::ofstream out(fmt::format("{}_{}d_sparsity_l{}.svg", prefix, dim, level));
            sparsity_pattern_v[level].print_svg(out);
        }
    }

private:
    GPE<dim> problem;
    unsigned min_level, max_level;
    MGLevelObject<SparsityPattern> sparsity_pattern_v;
};

int main(int argc, char** argv)
{
    GPE_Options options{};
    MG_Options  options_mg{};

    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(mg_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, options);
        apply_mg_options(vm, options_mg);

        with_dimension(options.dimension, [&](auto D)
        {
            constexpr int dim = decltype(D)::value;

            if (options_mg.multigrid) {
                GPE_Sparsity_MG<dim> GS(options, options_mg.min_level, options_mg.max_level);
                GS.run("domain");
            }
            else {
                GPE_Sparsity<dim> GS(options);
                GS.run("domain");
            }
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