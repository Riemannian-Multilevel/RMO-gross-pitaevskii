//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include "mesh.h"
#include "dofs.h"
#include "assembly.h"
#include "lac.h"

#include <iostream>
#include <sstream>
#include <fmt/format.h>

// Formulate possible extensions as command-line options.
#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>
namespace po = boost::program_options;

using namespace gpe;
using namespace dealii;

// Utility function for taking boundary points as strings "x,y,z" from the command-line
template <int dim>
Point<dim> str_to_point(const std::string& s, const char sep=',') {
    Point<dim> p;
    std::stringstream ss(s);
    std::string item;

    int i = 0;
    while (std::getline(ss, item, sep) && i < dim) {
        p[i++] = std::stod(item);
    }
    assert(i == dim);
    return p;
}

//!
//! @tparam dim Dimension of the domain
//! @param handler DoFHandler
//! @param element Finite element used
//! @param order DoF renumbering
//! @return
template <int dim>
SparsityPattern
rectangle_dofs(DoFHandler<dim>& handler, const FE_Q<dim>& element, Ordering order)
{
    distribute_dofs(handler, element, order);
    write_dof_locations(handler, fmt::format("rectangle_{}d_dof.gnuplot", handler.dimension));

    auto Sd = make_sparsity_pattern(handler);
    {
        std::ofstream out(fmt::format("rectangle_{}d_sparsity.svg", handler.dimension));
        Sd.print_svg(out);
    }
    return Sd;
}

//!
//! @tparam dim Dimension of the domain
//! @param handler DoFHandler
//! @param element Finite element used
//! @param order DoF renumbering
//! @return
template <int dim>
std::vector<SparsityPattern>
rectangle_dofs_mg(DoFHandler<dim>& handler, const FE_Q<dim>& element, Ordering order)
{
    int n_levels = handler.n_levels();
    std::vector<SparsityPattern> Sd_v;
    std::vector<int> levels(n_levels);

    std::iota(levels.begin(), levels.end(), 1);
    distribute_dofs_mg(handler, element, order, levels);

    // TODO: dof for every level (coloring?)
    // write_dof_locations(handler, fmt::format("rectangle_{}d_dof.gnuplot", handler.dimension));

    for (int i = 1; i <= n_levels; i++) {
        write_level_vertex_points(handler, i, fmt::format("rectangle_{}d_dof_l{}.gnuplot", handler.dimension, i));
    }
    for (int i = 1; i <= n_levels; ++i) {
        auto Sd = make_sparsity_pattern_mg(handler, i);
        Sd_v.push_back(Sd);

        std::ofstream out(fmt::format("rectangle_{}d_sparsity_l{}.svg", handler.dimension, i));
        Sd.print_svg(out);
    }
    return Sd_v;
}

Ordering select_order(const std::string& order_str)
{
    if (order_str == "DEFAULT") {
        return Ordering::DEFAULT;
    }
    else if (order_str == "RANDOM") {
        return Ordering::RANDOM;
    }
    else if (order_str == "CUTHILL_MCKEE") {
        return Ordering::CUTHILL_MCKEE;
    }
    else if (order_str == "KING") {
        return Ordering::KING;
    }
    else if (order_str == "MIN_DEG") {
        return Ordering::MIN_DEG;
    }
    else {
        throw std::runtime_error(order_str + ": invalid ordering");
    }
}

template <typename T>
struct GPE_Mass
{
    SparsityPattern sparsity_pattern;
    SparseMatrix<T> Mv;
    SparseMatrix<T> S;
    SparseMatrix<T> Mpp;
};

//!
//! @tparam dim
template <int dim>
class GPE
{
public:
    //!
    //! @param n_levels_ Number of times the mesh is refined
    //! @param degree Degree of the Lagrange finite element
    //! @param left_ Left end-point of the rectangular domain
    //! @param right_ Opposite end-point of the rectangular domain
    //! @param multigrid_ Enable multigrid (default = falser)
    //! @param order_ Ordering used for degrees of freedom
    GPE(const int n_levels_, const int degree, const Point<dim>& left_, const Point<dim>& right_,
        const Ordering order_ = Ordering::DEFAULT)
    :
        n_levels(n_levels_), order(order_), left(left_), right(right_),
        triangulation(Triangulation<dim>::limit_level_difference_at_vertices),
    // DoFHandler<> has a deleted assignment operator, so initialize in the constructor
        element(degree), dof_handler(triangulation)
    {
        dimension = dim;
    }

    void setup()
    {
        // step 1 - make grid
        std::cerr << "Dimension: " << dimension << std::endl;
        // rectangle consisting of precisely one cell
        dealii::GridGenerator::hyper_rectangle(triangulation, left, right);
        // the number of cells increases by a factor of 2^(dim x times)
        triangulation.refine_global(n_levels);

        // visualize grid
        grid2file(fmt::format("rectangle_{}d.gnuplot",dimension), triangulation, exportFormat::GNUPLOT);
        if (dimension == 2) {
            grid2file(fmt::format("rectangle_{}d.svg",dimension), triangulation, exportFormat::SVG);
        }
        std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of levels: " << triangulation.n_levels() << std::endl;
    }

    template <typename Function>
    GPE_Mass<double>
    mass(Function&& V) const
    {
        // Step 2 - degrees of freedom
        //SparsityPattern      sparsity_pattern;
        //SparseMatrix<double> mass_matrix_weighed; // M_v, V(x)*phi_i*phi_j*dx
        //SparseMatrix<double> stiffness_matrix;    // S, grad phi_i * grad phi_j * dx
        GPE_Mass<double> Mass;
        Mass.sparsity_pattern = rectangle_dofs(dof_handler, element, order);

        assert(dof_handler.has_active_dofs());
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Step 3 - linear system
        // Compute values of mass matrix
        Mass.Mv.reinit(Mass.sparsity_pattern);
        assemble_mass_weighed(Mass.Mv, dof_handler, V);

        // Compute values of stiffness matrix
        Mass.S.reinit(Mass.sparsity_pattern);
        assemble_stiffness(Mass.S, dof_handler);

        Mass.Mpp.reinit(Mass.sparsity_pattern);
        // values assembled with update() function
        return Mass; // XXX: or store in class object
    }

    template <typename Function>
    std::vector<GPE_Mass<double> >
    mass_mg(Function&& V) const
    {
        // Structure of arrays for multigrid
        //std::vector<SparseMatrix<double> > mass_matrix_weighed_v(n_levels);
        //std::vector<SparseMatrix<double> > stiffness_matrix_v(n_levels);
        std::vector<GPE_Mass<double> > Mass_v(n_levels);

        // Populate degrees of freedom on every level of the hierarchy
        auto sparsity_pattern_v= rectangle_dofs_mg(dof_handler, element, order);

        assert(sparsity_pattern_v.size() == n_levels);
        assert(dof_handler.has_level_dofs());
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Iterate over cells in every level of the hierarchy
        for (int i = 0; i < sparsity_pattern_v.size(); i++) {
            int mg_level = i+1;
            Mass_v.at(i).sparsity_pattern = std::move(sparsity_pattern_v.at(i));

            // compute values of mass matrix for level i
            Mass_v.at(i).Mv.reinit(sparsity_pattern_v[i]);
            assemble_mass_weighed(Mass_v.at(i).Mv, dof_handler, V, mg_level);

            // compute values of stiffness matrix for level i
            Mass_v.at(i).S.reinit(sparsity_pattern_v[i]);
            assemble_stiffness(Mass_v.at(i).S, dof_handler, mg_level);
        }
        return Mass_v; // XXX: or store in class (mg specialized?) object
    }

    void update_with_solution(SparseMatrix<double>& M, const Vector<double>& u) const
    {
        // Note: does not reinitialize matrix, sparsity assumed the same between iterations/update calls
        assemble_mass_phiphi(M, dof_handler, u);
    }

    const DoFHandler<dim>& get_dof() const {
        return dof_handler;
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

template <int dim>
void inverse_iteration(const GPE<dim>& Problem, const Vector<double>& x0)
{
    auto V = [](const Point<dim>& p) {
        Point<dim> out;
        for (unsigned d = 0; d < dim; d++) {
            out[d] = std::pow(p[d],2);
        }
        return out;
    };
    // Generate sparsity pattern, weighed mass matrix, and stiffness matrix
    GPE_Mass<double> Mass = Problem.mass(V);
}

template <int dim>
void inverse_iteration_mg(const GPE<dim>& Problem, const Vector<double>& x0)
{

}

int main(int argc, char* argv[])
{
    int degree, n_levels, dimension;
    bool multigrid;
    Ordering order;
    std::string left_str, right_str;

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

    // XX: use std::variant / std::visit
    switch (dimension) {
    case 1:
        {
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10";
                right_str = "10";
            }
            GPE<1> Problem(n_levels, degree, str_to_point<1>(left_str), str_to_point<1>(right_str), order);
            Problem.setup();
        }
        break;
    case 2:
        {
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10";
                right_str = "10,10";
            }
            GPE<2> Problem(n_levels, degree, str_to_point<2>(left_str), str_to_point<2>(right_str), order);
            Problem.setup();
        }
        break;
    case 3:
        {
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10,-10";
                right_str = "10,10,10";
            }
            GPE<3> Problem(n_levels, degree, str_to_point<3>(left_str), str_to_point<3>(right_str), order);
            Problem.setup();
        }
        break;
    default:
        throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}
