#include "fe_space.h"
#include "grid.h"
#include "sparsity.h"
#include "option.h"
#include "util.h"

#include <fmt/format.h>
#include <iostream>

using namespace gpe;
using namespace dealii;

// TODO: use gpe::DiscreteProblem (dofs.h)
template <int dim>
class Sparsity
{
public:
    Sparsity(GPE_Options options, const unsigned int n_levels)
        : grid{}, space(grid.triangulation, options.degree)   // establish relations between objects
    {
        grid.setup_grid(options.radius, n_levels);    // do the actual computations
        space.setup_dofs(options.order);
        space.setup_constraints(options.bc);

        auto dsp = make_sparsity_pattern(space.get_dofs(), space.get_constraints());
        sparsity_pattern.copy_from(dsp);
    }

    void run(const std::string& prefix, unsigned int level) const
    {
        std::cerr << "Number of active cells: " << grid.triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of levels: " << grid.triangulation.n_levels() << std::endl;

        plot_grid(grid.triangulation, prefix);

        write_dof_locations(space.get_dofs(), fmt::format("{}_{}d_dof.gnuplot", prefix, dim));
        {
            std::ofstream out(fmt::format("{}_{}d_lvl{}_sparsity.svg", prefix, dim, level));
            sparsity_pattern.print_svg(out);
        }
    }

private:
    HyperCube<dim> grid;
    SparsityPattern sparsity_pattern;
    FeSpace<dim> space;
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
            unsigned int min_level = options_mg.multilevel ? options_mg.min_level : options_mg.max_level-1;
            unsigned int max_level = options_mg.max_level;

            for (unsigned int i = min_level; i < max_level; ++i) {
                Sparsity<dim> GS(options, i+1);
                GS.run("domain", i);
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