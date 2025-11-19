#include "mesh.h"
#include "dofs.h"

#include <fmt/format.h>
#include <iostream>
#include <sstream>

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
dofs_sparsity(const DoFHandler<dim>& handler, std::string prefix = "domain")
{
    assert(handler.has_active_dofs());
    write_dof_locations(handler, fmt::format("{}_{}d_dof.gnuplot", prefix, handler.dimension));

    auto Sd = make_sparsity_pattern(handler);
    {
        std::ofstream out(fmt::format("{}_{}d_sparsity.svg", prefix, handler.dimension));
        Sd.print_svg(out);
    }
    return Sd;
}

//! Retrieve and visualize sparsity pattern for a given DoF ordering
//! @tparam dim Dimension of the domain
//! @param handler DoFHandler
//! @return
template <int dim>
std::vector<SparsityPattern>
dofs_mg_sparsity(const DoFHandler<dim>& handler, std::string prefix = "domain")
{
    assert(handler.has_level_dofs());
    const int n_levels = handler.get_triangulation().n_levels();

    // TODO: dof for every level (coloring?)
    // write_dof_locations(handler, fmt::format("{}_{}d_dof.gnuplot", prefix, handler.dimension));
    std::vector<SparsityPattern> Sd_v;

    for (int i = 1; i <= n_levels; i++) {
        write_level_vertex_points(handler, i, fmt::format("{}_{}d_dof_l{}.gnuplot", prefix, handler.dimension, i));
    }
    for (int i = 1; i <= n_levels; ++i) {
        auto Sd = make_sparsity_pattern_mg(i, handler);
        Sd_v.push_back(Sd);

        std::ofstream out(fmt::format("{}_{}d_sparsity_l{}.svg", prefix, handler.dimension, i));
        Sd.print_svg(out);
    }
    return Sd_v;
}

// TODO: class structure
template <int dim>
void make_rectangle(const int degree, const int n_levels, const Point<dim>& left, const Point<dim>& right)
{
    Triangulation<dim> triangulation(Triangulation<dim>::limit_level_difference_at_vertices); // multigrid compatibility
    const FE_Q<dim>    element(degree); // element for defining degrees of freedom (degree 1 -> vertices are DoFs)
    DoFHandler<dim>    dof_handler(triangulation);  // associate triangulation to DoF object

    // generate rectangular domain
    std::cerr << "Dimension: " << dim << std::endl;
    dealii::GridGenerator::hyper_rectangle(triangulation, left, right);
    // the number of cells increases by a factor of 2^(dim x times)
    triangulation.refine_global(n_levels);

    // visualize grid
    grid2file(fmt::format("rectangle_{}d.gnuplot",dim), triangulation, exportFormat::GNUPLOT);
    if (dim == 2) {
        grid2file(fmt::format("rectangle_{}d.svg",dim), triangulation, exportFormat::SVG);
    }
    std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
    std::cerr << "Number of levels: " << triangulation.n_levels() << std::endl;
}

int main(int argc, char** argv)
{
    int degree, n_levels, dimension;
    bool multigrid;
    Ordering order;
    std::string left_str, right_str;

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("degree", po::value<int>()->default_value(1),
             "polynomial degree for finite element")
            ("levels", po::value<int>()->default_value(2),
             "number of times to globally refine the mesh")
            ("multigrid", po::value<bool>()->default_value(false),
             "enable multigrid")
            ("dimension", po::value<int>()->default_value(2),
             "problem dimension")
            ("ordering", po::value<std::string>()->default_value("default"),
             "ordering for degrees of freedom (default|random|cuthill_mckee|king|min_deg)")
            ("left", po::value<std::string>()->default_value(""),
                "left point of the mesh")
            ("right", po::value<std::string>()->default_value(""),
                "right point of the mesh")

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        auto order_str = vm["ordering"].as<std::string>();
        std::transform(order_str.begin(), order_str.end(), order_str.begin(), ::toupper);

        order     = select_order(order_str);
        degree    = vm["degree"].as<int>();
        n_levels  = vm["levels"].as<int>();
        dimension = vm["dimension"].as<int>();
        multigrid = vm["multigrid"].as<bool>();
        left_str  = vm["left"].as<std::string>();
        right_str = vm["right"].as<std::string>();
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }

    // TODO: populate DoFs according to order, print sparsity patterns
}