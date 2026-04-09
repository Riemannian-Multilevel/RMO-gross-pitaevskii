#include <gpe/fe/fe_space.h>
#include <gpe/fe/grid.h>
#include <gpe/util/sparsity.h>
#include <gpe/util/util.h>
#include <gpe/option.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <fmt/format.h>
#include <iostream>

using namespace gpe;
using namespace dealii;

template <int dim>
class Sparsity
{
public:
    Sparsity(GPE_Options options, const unsigned int n_levels)
        : grid(options.radius, options.mesh_kind == MeshKind::SIMPLEX)
        , space(grid.triangulation)   // establish relations between objects
    {
        // Note: assumes grid has no mixed cells (contains either quadrilaterals or simplices)
        if (grid.has_simplex) {
            mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
            element = std::make_unique<FE_SimplexP<dim>>(options.degree);
        }
        else {
            mapping = std::make_unique<MappingQ1<dim>>();
            element = std::make_unique<FE_Q<dim>>(options.degree);
        }
        grid.refine(n_levels);   // do the actual computations
        std::cerr << "Number of active cells: " << grid.triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of levels: " << grid.triangulation.n_levels() << std::endl;

        space.setup_dofs(options.order, *element);
        space.setup_constraints(options.bc);

        auto dsp = make_sparsity_pattern(space.get_dofs(), space.get_constraints());
        sparsity_pattern.copy_from(dsp);
    }

    void run(const std::string& prefix, unsigned int level) const
    {
        std::string mesh = grid.has_simplex ? "simplex" : "quads";
        std::string grid_file = fmt::format("{}_{}d_{}_lvl{}", prefix, dim, mesh, level);

        if (dim == 2) {
            write_grid<dim>(grid_file + ".svg", grid.triangulation, dealii::GridOut::OutputFormat::svg);
            std::cerr << "Saving " + grid_file + ".svg" << std::endl;
        }
        // only for quadrilateral mesh
        // write_grid<dim>(grid_file + ".gnuplot", grid.triangulation, dealii::GridOut::OutputFormat::gnuplot);
        // std::cerr << "Saving " + grid_file + ".svg" << std::endl;

        std::string dof_file = fmt::format("{}_{}d_{}_dof.gnuplot", prefix, dim, mesh);
        write_dof_locations(space.get_dofs(), dof_file, *mapping);
        std::cerr << "Saving " + dof_file << std::endl;

        std::string sparsity_file = fmt::format("{}_{}d_{}_lvl{}_sparsity.svg", prefix, dim, mesh, level);
        std::ofstream out(sparsity_file);
        sparsity_pattern.print_svg(out);
        std::cerr << "Saving " + sparsity_file << std::endl;
    }

private:
    HyperCube<dim> grid;
    SparsityPattern sparsity_pattern;
    FeSpace<dim> space;

    // Variable for simplex or quadrilateral meshes
    std::unique_ptr<Mapping<dim>> mapping;
    std::unique_ptr<FiniteElement<dim>> element;
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

        with_dimension(options.dimension, [&]<typename T0>(T0 D)
        {
            constexpr int dim = T0::value;
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