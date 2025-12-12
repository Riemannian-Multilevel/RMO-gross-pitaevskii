#include "mesh.h"
#include "dofs.h"
#include "gpe.h"

#include <fmt/format.h>
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>
namespace po = boost::program_options;

using namespace gpe;
using namespace dealii;

//! Retrieve and visualize sparsity pattern for a given DoF ordering
//! @tparam dim Dimension of the domain
//! @param handler DoFHandler
//! @return
template <int dim>
SparsityPattern
dofs_sparsity(const DoFHandler<dim>& handler, const AffineConstraints<double>& constraints,
    std::string prefix = "domain")
{
    assert(handler.has_active_dofs());
    write_dof_locations(handler, fmt::format("{}_{}d_dof.gnuplot", prefix, handler.dimension));

    auto Sd = make_sparsity_pattern(handler, constraints);
    {
        std::ofstream out(fmt::format("{}_{}d_sparsity.svg", prefix, handler.dimension));
        Sd.print_svg(out);
    }
    return Sd;
}

//! Retrieve and visualize sparsity pattern for a given DoF ordering
//! @tparam dim Dimension of the domain
//! @param dof_handler DoFHandler
//! @return
template <int dim>
// TODO: MGLevelObject for min/max level
std::vector<SparsityPattern>
dofs_mg_sparsity(const DoFHandler<dim>& dof_handler, std::string prefix = "domain")
{
    assert(dof_handler.has_level_dofs());
    const int n_levels = dof_handler.get_triangulation().n_levels();

    // TODO: dof for every level (coloring?)
    // write_dof_locations(handler, fmt::format("{}_{}d_dof.gnuplot", prefix, handler.dimension));
    std::vector<SparsityPattern> Sd_v(n_levels);

    for (int level = 0; level < n_levels; level++) {
        write_level_vertex_points(dof_handler, level, fmt::format("{}_{}d_dof_l{}.gnuplot",
            prefix, dof_handler.dimension, level));
    }
    for (int level = 0; level < n_levels; ++level) {
        auto Sd = make_sparsity_pattern_mg(dof_handler, level);
        Sd_v[level].copy_from(Sd);

        std::ofstream out(fmt::format("{}_{}d_sparsity_l{}.svg",
            prefix, dof_handler.dimension, level));
        Sd.print_svg(out);
    }
    return Sd_v;
}

// TODO: dirichlet boundary and hanging-node constraints for sparsity pattern
//       inner/outer radius for adaptive refinement
template <int dim>
class GPE_Sparsity
{
public:
    GPE_Sparsity(GPE_Options options_, const double radius_)
    :
        options(options_), radius(radius_),
        triangulation(Triangulation<dim>::limit_level_difference_at_vertices),
        element(options.degree), dof_handler(triangulation)
    {}

    void plot() const
    {
        if (dim == 2) {
            grid2file(fmt::format("rectangle_{}d.svg",dim), triangulation, GridOut::OutputFormat::svg);
        }
        grid2file(fmt::format("rectangle_{}d.gnuplot",dim), triangulation, GridOut::OutputFormat::gnuplot);
    }

    [[maybe_unused]] SparsityPattern
    dofs(const AffineConstraints<double>& constraints = {})
    {
        // step 2 - degrees of freedom
        distribute_dofs(dof_handler, element, options.order);

        std::cerr << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of levels: " << triangulation.n_levels() << std::endl;

        return dofs_sparsity(dof_handler, constraints);
    }

    // TODO: MGLevelObject for min/max level
    [[maybe_unused]] std::vector<SparsityPattern>
    dofs_mg()
    {
        // step 2 - degrees of freedom - ordering applied to every level
        std::vector<bool> levels(options.n_levels, true);

        // DoFHandler::distribute_dofs, DoFHandler::distribute_mg_dofs
        distribute_mg_dofs(dof_handler, element, options.order, levels);

        std::cerr << "Number of levels: " << options.n_levels << std::endl;
        for (int i = 0; i < options.n_levels; i++) {
            std::cerr << "Number of cells (level " << i << "): " << triangulation.n_cells(i) << std::endl;
        }
        return dofs_mg_sparsity(dof_handler);
    }

    void run(bool multigrid, bool adaptive, int n_levels_adaptive = 0)
    {
        if (adaptive) {
            make_cube_graded(triangulation, radius, options.n_levels, n_levels_adaptive);
        } else {
            make_cube(triangulation, radius, options.n_levels);
        }
        if (multigrid) {
            this->dofs_mg();
        } else {
            this->dofs();
        }
        this->plot();
    }

private:
    // Problem parameters
    GPE_Options options;
    double radius;

    // Finite element containers
    Triangulation<dim>   triangulation; // copy stored by dof_handler
    const FE_Q<dim>      element;       // copy stored by dof_handler
    DoFHandler<dim>      dof_handler;
};

int main(int argc, char** argv)
{
    GPE_Options options{};
    int min_level, max_level;
    bool multigrid, adaptive = false;
    double radius;

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("degree", po::value<int>()->default_value(1),
             "polynomial degree for finite element")
            ("levels", po::value<int>()->default_value(3),
             "number of times to regularly refine the mesh")
            ("levels-adaptive", po::value<int>()->default_value(2),
                "number of times to refine the mesh towards the origin")
            ("radius-inner", po::value<double>()->default_value(-1.0),
                "inner radius for mesh refinement")
            ("radius-outer", po::value<double>()->default_value(-1.0),
                "outer radius for mesh refinement")
            ("min-level", po::value<int>()->default_value(0),
                "minimum multigrid level")
            ("max-level", po::value<int>()->default_value(0),
                "maximum multigrid level")
            ("multigrid", po::bool_switch(&multigrid),
             "enable multigrid")
            ("dimension", po::value<int>()->default_value(2),
             "problem dimension")
            ("ordering", po::value<std::string>()->default_value("default"),
             "ordering for degrees of freedom (default|random|cuthill_mckee|king|min_deg)")
            ("adaptive", po::bool_switch(&adaptive),
                "non-uniform mesh with higher refinement around the origin")
            ("radius", po::value<double>()->default_value(10.0),
                "radius for the hypercube domain");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        auto order_str = vm["ordering"].as<std::string>();
        std::transform(order_str.begin(), order_str.end(), order_str.begin(), ::toupper);

        options.order = select_order(order_str);
        options.degree = vm["degree"].as<int>();
        options.n_levels = vm["levels"].as<int>();
        options.dimension = vm["dimension"].as<int>();
        options.bc = BoundaryCondition::NEUMANN;

        radius    = vm["radius"].as<double>();
        int n_levels_adaptive = vm["levels-adaptive"].as<int>();
        min_level = vm["min-level"].as<int>();
        max_level = vm["max-level"].as<int>();
        max_level = max_level == 0 ? options.n_levels : max_level;

        switch (options.dimension) {
            case 1:
                {
                    GPE_Sparsity<1> Problem(options, radius);
                    Problem.run(multigrid, adaptive, n_levels_adaptive);
                }
                break;
            case 2:
                {
                    GPE_Sparsity<2> Problem(options, radius);
                    Problem.run(multigrid, adaptive, n_levels_adaptive);
                }
                break;
            case 3:
                {
                    GPE_Sparsity<3> Problem(options, radius);
                    Problem.run(multigrid, adaptive, n_levels_adaptive);
                }
                break;
            default:
                throw std::invalid_argument("dimension must be 1, 2 or 3");
        }
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