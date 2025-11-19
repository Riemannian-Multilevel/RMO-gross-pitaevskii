#include "mesh.h"
#include "dofs.h"
#include "util.h"

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
// Use vector with one entry for interoperability with dofs_mg_sparsity
// (SparsityPattern has no copy or move constructors)
std::vector<SparsityPattern>
dofs_sparsity(const DoFHandler<dim>& handler, std::string prefix = "domain")
{
    assert(handler.has_active_dofs());
    write_dof_locations(handler, fmt::format("{}_{}d_dof.gnuplot", prefix, handler.dimension));
    std::vector<SparsityPattern> Sd_v(1);

    auto Sd = make_sparsity_pattern(handler);
    {
        std::ofstream out(fmt::format("{}_{}d_sparsity.svg", prefix, handler.dimension));
        Sd.print_svg(out);
    }
    Sd_v[0].copy_from(Sd);
    return Sd_v;
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
    std::vector<SparsityPattern> Sd_v(n_levels);

    for (int i = 0; i < n_levels; i++) {
        write_level_vertex_points(handler, i, fmt::format("{}_{}d_dof_l{}.gnuplot", prefix, handler.dimension, i));
    }
    for (int i = 0; i < n_levels; ++i) {
        auto Sd = make_sparsity_pattern_mg(i, handler);
        Sd_v[i].copy_from(Sd);

        std::ofstream out(fmt::format("{}_{}d_sparsity_l{}.svg", prefix, handler.dimension, i));
        Sd.print_svg(out);
    }
    return Sd_v;
}

template <int dim>
class GPE_Sparsity
{
public:
    GPE_Sparsity(const int n_levels_, const int degree,
        const Point<dim>& left_, const Point<dim>& right_,
        const Ordering order_ = Ordering::DEFAULT)
    :
        n_levels(n_levels_), order(order_), left(left_), right(right_),
        triangulation(Triangulation<dim>::limit_level_difference_at_vertices), element(degree), dof_handler(triangulation)
    {}

    // generate rectangular domain
    void make_rectangle()
    {
        std::cerr << "Dimension: " << dim << std::endl;
        dealii::GridGenerator::hyper_rectangle(triangulation, left, right);
        // the number of cells increases by a factor of 2^(dim x times)
        triangulation.refine_global(n_levels-1);

        // visualize grid
        grid2file(fmt::format("rectangle_{}d.gnuplot",dim), triangulation, exportFormat::GNUPLOT);
        if (dim == 2) {
            grid2file(fmt::format("rectangle_{}d.svg",dim), triangulation, exportFormat::SVG);
        }
        AssertDimension(n_levels, triangulation.n_levels());
    }

    [[maybe_unused]] std::vector<SparsityPattern> dofs()
    {
        // step 2 - degrees of freedom
        distribute_dofs(dof_handler, element, order);

        std::cerr << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of levels: " << triangulation.n_levels() << std::endl;

        return dofs_sparsity(dof_handler);
    }

    [[maybe_unused]] std::vector<SparsityPattern> dofs_mg()
    {
        // step 2 - degrees of freedom - ordering applied to every level
        std::vector<int> levels(n_levels);
        std::iota(levels.begin(), levels.end(), 1);

        // DoFHandler::distribute_dofs, DoFHandler::distribute_mg_dofs
        distribute_mg_dofs(dof_handler, element, order, levels);

        std::cerr << "Number of levels: " << n_levels << std::endl;
        for (int i = 0; i < n_levels; i++) {
            std::cerr << "Number of cells (level " << i << "): " << triangulation.n_cells(i) << std::endl;
        }
        return dofs_mg_sparsity(dof_handler);
    }

    void run(bool multigrid)
    {
        this->make_rectangle();
        if (multigrid)
            this->dofs_mg();
        else
            this->dofs();
    }

private:
    // Problem parameters
    int n_levels;
    int dimension;
    Ordering order;

    // Rectangle bounds
    Point<dim> left;
    Point<dim> right;

    // Finite element containers
    Triangulation<dim>   triangulation; // copy stored by dof_handler
    const FE_Q<dim>      element;       // copy stored by dof_handler
    DoFHandler<dim>      dof_handler;
};

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
            ("levels", po::value<int>()->default_value(3),
             "number of times to globally refine the mesh")
            ("multigrid", po::bool_switch(&multigrid),
             "enable multigrid")
            ("dimension", po::value<int>()->default_value(2),
             "problem dimension")
            ("ordering", po::value<std::string>()->default_value("default"),
             "ordering for degrees of freedom (default|random|cuthill_mckee|king|min_deg)")
            ("left", po::value<std::string>()->default_value(""),
                "left point of the mesh")
            ("right", po::value<std::string>()->default_value(""),
                "right point of the mesh");

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

    switch (dimension) {
    case 1:
        {
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10"; right_str = "10";
            }
            GPE_Sparsity<1> Problem(n_levels, degree, str_to_point<1>(left_str), str_to_point<1>(right_str), order);
            Problem.run(multigrid);
        }
        break;
    case 2:
        {
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10"; right_str = "10,10";
            }
            GPE_Sparsity<2> Problem(n_levels, degree, str_to_point<2>(left_str), str_to_point<2>(right_str), order);
            Problem.run(multigrid);
        }
        break;
    case 3:
        {
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10,-10"; right_str = "10,10,10";
            }
            GPE_Sparsity<3> Problem(n_levels, degree, str_to_point<3>(left_str), str_to_point<3>(right_str), order);
            Problem.run(multigrid);
        }
        break;
    default:
        throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}