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
//! @param dof_handler DoFHandler
//! @return
template <int dim>
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
        auto Sd = make_sparsity_pattern(dof_handler, level);
        Sd_v[level].copy_from(Sd);

        std::ofstream out(fmt::format("{}_{}d_sparsity_l{}.svg",
            prefix, dof_handler.dimension, level));
        Sd.print_svg(out);
    }
    return Sd_v;
}

template <int dim>
class GPE_Sparsity
{
public:
    GPE_Sparsity(const unsigned int n_levels_, const unsigned int degree, double radius_,
        const Ordering order_ = Ordering::DEFAULT)
    :
        n_levels(n_levels_), order(order_), radius(radius_),
        triangulation(Triangulation<dim>::limit_level_difference_at_vertices),
        element(degree), dof_handler(triangulation)
    {}

    void plot()
    {
        // visualize grid
        if (dim == 2) {
            grid2file(fmt::format("rectangle_{}d.svg",dim), triangulation, GridOut::OutputFormat::svg);
        }
        grid2file(fmt::format("rectangle_{}d.gnuplot",dim), triangulation, GridOut::OutputFormat::gnuplot);
    }

    // generate rectangular domain
    void make_cube()
    {
        dealii::GridGenerator::hyper_cube(triangulation, -radius, radius);

        // the number of cells increases by a factor of 2^(dim x times)
        triangulation.refine_global(n_levels-1);
        AssertDimension(n_levels, triangulation.n_global_levels());
    }

    // generate rectangular domain, refined around the origin
    // TODO: make R_outer / R_inner adjustable
    void make_cube_graded()
    {
        if (n_levels <= 2) throw ExcInternalError("n_levels must be at least 2");

        // Parameters controlling grading strength
        const double R_outer = radius*std::sqrt(static_cast<double>(dim));  // start refining inside this radius
        const double R_inner = 0.5; // target radius for finest cells

        // Geometric shrink factor for the refinement radius per cycle
        const double shrink = std::pow(R_inner / R_outer, 1.0 / n_levels);

        // Start from a very coarse mesh on [-L, L]^dim
        GridGenerator::hyper_cube(triangulation, -radius, radius);

        triangulation.refine_global(2);
        double current_radius = R_outer;

        for (unsigned int cycle = 0; cycle < n_levels; ++cycle)
        {
            // Mark cells for refinement if their center is closer
            // than current_radius to the origin
            for (const auto &cell : triangulation.active_cell_iterators())
            {
                if (const double r = cell->center().norm(); r < current_radius) {
                    cell->set_refine_flag();
                }
            }
            triangulation.execute_coarsening_and_refinement();
            current_radius *= shrink;
        }
    }

    // TODO: MGLevelObject vs. std::vector
    [[maybe_unused]] std::vector<SparsityPattern> dofs()
    {
        // step 2 - degrees of freedom
        distribute_dofs(dof_handler, element, order);

        std::cerr << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of levels: " << triangulation.n_levels() << std::endl;

        return dofs_sparsity(dof_handler);
    }

    // TODO: MGLevelObject vs. std::vector
    [[maybe_unused]] std::vector<SparsityPattern> dofs_mg()
    {
        // step 2 - degrees of freedom - ordering applied to every level
        std::vector<bool> levels(n_levels, true);

        // DoFHandler::distribute_dofs, DoFHandler::distribute_mg_dofs
        distribute_mg_dofs(dof_handler, element, order, levels);

        std::cerr << "Number of levels: " << n_levels << std::endl;
        for (int i = 0; i < n_levels; i++) {
            std::cerr << "Number of cells (level " << i << "): " << triangulation.n_cells(i) << std::endl;
        }
        return dofs_mg_sparsity(dof_handler);
    }

    void run(bool multigrid, bool adaptive)
    {
        adaptive  ? this->make_cube_graded() : this->make_cube();
        multigrid ? this->dofs_mg() : this->dofs();

        this->plot();
    }

private:
    // Problem parameters
    unsigned int n_levels;
    Ordering order;
    double radius;

    // Finite element containers
    Triangulation<dim>   triangulation; // copy stored by dof_handler
    const FE_Q<dim>      element;       // copy stored by dof_handler
    DoFHandler<dim>      dof_handler;
};

int main(int argc, char** argv)
{
    int degree, n_levels, dimension;
    int min_level, max_level;
    bool multigrid, adaptive = false;
    Ordering order;
    std::string radius_str;

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("degree", po::value<int>()->default_value(1),
             "polynomial degree for finite element")
            ("levels", po::value<int>()->default_value(3),
             "number of times to globally refine the mesh")
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
            ("radius", po::value<double>()->default_value(10),
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

        order      = select_order(order_str);
        degree     = vm["degree"].as<int>();
        n_levels   = vm["levels"].as<int>();
        dimension  = vm["dimension"].as<int>();
        radius_str = vm["radius"].as<std::string>();
        min_level  = vm["min-level"].as<int>();
        max_level  = vm["max-level"].as<int>();
        max_level  = max_level == 0 ? n_levels : max_level;
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
            GPE_Sparsity<1> Problem(n_levels, degree, std::stod(radius_str), order);
            Problem.run(multigrid, adaptive);
        }
        break;
    case 2:
        {
            GPE_Sparsity<2> Problem(n_levels, degree, std::stod(radius_str), order);
            Problem.run(multigrid, adaptive);
        }
        break;
    case 3:
        {
            GPE_Sparsity<3> Problem(n_levels, degree, std::stod(radius_str), order);
            Problem.run(multigrid, adaptive);
        }
        break;
    default:
        throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}