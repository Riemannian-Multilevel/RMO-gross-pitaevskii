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

//! Retrieve and visualize sparsity pattern for a given DoF ordering
//! @tparam dim Dimension of the domain
//! @param handler DoFHandler
//! @param element Finite element used
//! @param order DoF renumbering
//! @return
template <int dim>
SparsityPattern
rectangle_dofs(const DoFHandler<dim>& handler)
{
    assert(handler.has_active_dofs());
    write_dof_locations(handler, fmt::format("rectangle_{}d_dof.gnuplot", handler.dimension));

    auto Sd = make_sparsity_pattern(handler);
    {
        std::ofstream out(fmt::format("rectangle_{}d_sparsity.svg", handler.dimension));
        Sd.print_svg(out);
    }
    return Sd;
}

//! Retrieve and visualize sparsity pattern for a given DoF ordering
//! @tparam dim Dimension of the domain
//! @param handler DoFHandler
//! @param element Finite element used
//! @param order DoF renumbering
//! @return
template <int dim>
std::vector<SparsityPattern>
rectangle_mg_dofs(const DoFHandler<dim>& handler)
{
    assert(handler.has_level_dofs());
    const int n_levels = handler.get_triangulation().n_levels();

    // TODO: dof for every level (coloring?)
    // write_dof_locations(handler, fmt::format("rectangle_{}d_dof.gnuplot", handler.dimension));
    std::vector<SparsityPattern> Sd_v;

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

// XXX: refactor to options.h
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

SolverMethod select_solver(const std::string& solver_str)
{
    if (solver_str == "GMRES") {
        return SolverMethod::GMRES;
    }
    else if (solver_str == "MINRES") {
        return SolverMethod::MINRES;
    }
    else if (solver_str == "CG") {
        return SolverMethod::CG;
    }
    else {
        throw std::runtime_error(solver_str + ": invalid solver");
    }
}

template <typename T>
struct GPE_Mass
{
    SparseMatrix<T> Mv;
    SparseMatrix<T> S;
    SparseMatrix<T> Mpp;
    SparsityPattern sparsity_pattern;
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
    //! @param multigrid_ Enable multigrid (default = false)
    //! @param order_ Ordering used for degrees of freedom
    GPE(const int n_levels_, const int degree, const Point<dim>& left_, const Point<dim>& right_,
        const Ordering order_ = Ordering::DEFAULT)
    :
        n_levels(n_levels_), order(order_), left(left_), right(right_),
        // Flag to allow multigrid algorithms
        triangulation(Triangulation<dim>::limit_level_difference_at_vertices),
        // DoFHandler<> has a deleted assignment operator, so initialize in the constructor
        element(degree), dof_handler(triangulation)
    {
        dimension = dim;
    }

    void step1()
    {
        // step 1 - make grid
        // rectangle consisting of precisely one cell
        std::cerr << "Dimension: " << dimension << std::endl;
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

    // XXX: virtual function, overridden in MG_GPE
    void step2()
    {
        // step 2 - degrees of freedom
        distribute_dofs(dof_handler, element, order);
    }

    void step2_mg()
    {
        // step 2 - degrees of freedom - applied to every level
        std::vector<int> levels(n_levels);
        std::iota(levels.begin(), levels.end(), 1);

        // DoFHandler::distribute_dofs, DoFHandler::distribute_mg_dofs
        distribute_mg_dofs(dof_handler, element, order, levels);
    }

    template <typename Function>
    GPE_Mass<double>
    step3(Function&& V) const
    {
        // Step 2 - degrees of freedom
        GPE_Mass<double> Mass;
        { // XXX: in-place construction of sparsity pattern?
            const SparsityPattern sparsity_pattern = rectangle_dofs(dof_handler);
            Mass.sparsity_pattern.copy_from(sparsity_pattern);
            std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
        }
        // Step 3 - linear system
        // Compute values of mass matrix
        Mass.Mv.reinit(Mass.sparsity_pattern);
        assemble_mass_weighed(Mass.Mv, dof_handler, element, V);

        // Compute values of stiffness matrix
        Mass.S.reinit(Mass.sparsity_pattern);
        assemble_stiffness(Mass.S, dof_handler, element);

        Mass.Mpp.reinit(Mass.sparsity_pattern);
        // values assembled with update() function
        return Mass; // XXX: or store in class object
    }

    template <typename Function>
    std::vector<GPE_Mass<double> >
    step3_mg(Function&& V) const
    {
        // Structure of arrays for multigrid
        //std::vector<SparseMatrix<double> > mass_matrix_weighed_v(n_levels);
        //std::vector<SparseMatrix<double> > stiffness_matrix_v(n_levels);
        std::vector<GPE_Mass<double> > Mass_v(n_levels);

        // Populate degrees of freedom on every level of the hierarchy
        {
            auto sparsity_pattern_v= rectangle_mg_dofs(dof_handler);
            assert(sparsity_pattern_v.size() == n_levels);
            std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

            // Iterate over cells in every level of the hierarchy
            for (int i = 0; i < n_levels; i++) {
                // XXX: in-place construction of sparsity pattern?
                Mass_v.at(i).sparsity_pattern.copy_from(sparsity_pattern_v.at(i));
            }
        }
        for (int i = 0; i < n_levels; i++) {
            // compute values of mass matrix for level i
            // TODO: multigrid dof_handler iteration
            Mass_v.at(i).Mv.reinit(Mass_v.at(i).sparsity_pattern);
            assemble_mass_weighed(Mass_v.at(i).Mv, dof_handler, element, V);

            // compute values of stiffness matrix for level i
            // TODO: multigrid dof_handler iteration
            Mass_v.at(i).S.reinit(Mass_v.at(i).sparsity_pattern);
            assemble_stiffness(Mass_v.at(i).S, dof_handler, element);
        }
        return Mass_v; // XXX: or store in class (mg specialized?) object
    }

    const DoFHandler<dim>& get_dof() const {
        return dof_handler;
    }
    const FE_Q<dim>& get_element() const {
        return element;
    }
    const Triangulation<dim>& get_triangulation() const {
        return triangulation;
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
std::tuple<Vector<double>,double,double>
// TODO: step size argument (double)
rgd_fixed_step(GPE_Mass<double>& Mass, const DoFHandler<dim>& dof_handler, const FE_Q<dim>& element,
    const Vector<double>& x0, double beta, SolverMethod solver, int max_iter, double reltol)
{
    // Generate sparsity pattern, weighed mass matrix, and stiffness matrix
    SparseMatrix<double> A_0;
    A_0.reinit(Mass.S);
    A_0.copy_from(Mass.S);
    A_0.add(1.0, Mass.Mv);

    // Begin inverse iteration
    Vector<double> x(x0);
    for (int i = 0; i < max_iter; i++) {
        SparseMatrix<double> A;
        A.reinit(A_0);
        A.copy_from(A_0);
        assemble_mass_phiphi(Mass.Mpp, dof_handler, element, x);
        A.add(beta, Mass.Mpp);  // A = A_0 + beta * M_phiphi

        // Invert A
        Vector<double> y = solve_sparse(A, x, solver, max_iter, reltol);
        double denom1 = x * y;
        assert(denom1 > 0);
        x  = y;
        x /= denom1;

        // Normalize in energy norm
        Vector<double> tmp(x.size());
        Mass.Mpp.vmult(tmp, x);
        double denom2 = std::sqrt(x * tmp);
        x /= denom2;
    }

    // Compute residual
    SparseMatrix<double> A;
    A.reinit(A_0);
    A.copy_from(A_0);
    assemble_mass_phiphi<dim>(Mass.Mpp, dof_handler, element, x);

    Vector<double> res(x.size());
    A.add(beta, Mass.Mpp);
    A.vmult(res, x); // A*x

    Vector<double> tmp(x);
    tmp *= (x*res); // (x'*A*x)*x
    res.add(-1.0, tmp); // A*x - (x'*A*x)*x

    // Compute energy
    SparseMatrix<double> B;
    B.reinit(A_0);
    B.copy_from(A_0);
    B *= 0.5;
    B.add(0.25, Mass.Mpp);

    Vector<double> tmp1(x.size());
    B.vmult(tmp1, x);
    double energy = x * tmp1;

    return std::make_tuple(x, res.l2_norm(), energy);
}

// template <int dim>
// void inverse_iteration_mg(const GPE<dim>&, const Vector<double>&)
// {
//     throw std::logic_error("inverse_iteration_mg not implemented");
// }

template <int dim>
void experiment1(int n_levels, int degree, const std::string& left_str, const std::string& right_str,
    double beta, Ordering order, bool multigrid, SolverMethod solver, int max_iter, double reltol)
{
    // Set up grid
    GPE<dim> Problem(n_levels, degree, str_to_point<dim>(left_str), str_to_point<dim>(right_str), order);
    Problem.step1();

    // Set up degrees of freedom (multigrid-dependent)
    if (multigrid) {
        Problem.step2_mg();
    } else {
        Problem.step2();
    }

    // Define potential function
    auto V = [](const Point<dim>& p) {
        typename Point<dim>::value_type out = 0.0;
        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    };

    // Set up mass and stiffness matrices (multigrid-dependent)
    if (multigrid) {
        // TODO: populate matrices on every level
        throw std::logic_error("step3_mg not implemented");
    }
    GPE_Mass Mass = Problem.step3(V);

    // Set initial value
    Vector<double> x0(Problem.get_dof().n_dofs());
    x0 = 1.0;

    // Run iteration, update M_pp in every step
    const DoFHandler<dim>& dof_handler = Problem.get_dof();
    const FE_Q<dim>& element = Problem.get_element();
    std::tuple<Vector<double>, double, double> results;

    if (multigrid) {
        // TODO: run iteration for all levels
        throw std::logic_error("inverse_iteration_mg not implemented");
    } else {
        results = rgd_fixed_step<dim>(Mass, dof_handler, element, x0, beta, solver, max_iter, reltol);
    }
    std::cerr << "Residual: " << std::get<1>(results) << std::endl;
    std::cerr << "Energy: " << std::get<2>(results) << std::endl;
}

int main(int argc, char* argv[])
{
    int degree, n_levels, dimension, max_iter;
    bool multigrid;
    Ordering order;
    std::string left_str, right_str;
    SolverMethod solver;
    double beta, reltol, step_size;

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
            ("beta", po::value<double>()->default_value(100),
                "non-linearity factor")
            ("solver", po::value<std::string>()->default_value("gmres"),
                "sparse solver (gmres|minres|cg)")
            ("max_iter", po::value<int>()->default_value(100),
                "maximum number of iterations for sparse solver")
            ("reltol", po::value<double>()->default_value(1e-6),
                "relative tolerance for sparse solver");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        auto order_str = vm["ordering"].as<std::string>();
        auto solver_str = vm["solver"].as<std::string>();
        std::transform(order_str.begin(), order_str.end(), order_str.begin(), ::toupper);
        std::transform(solver_str.begin(), solver_str.end(), solver_str.begin(), ::toupper);

        order     = select_order(order_str);
        degree    = vm["degree"].as<int>();
        n_levels  = vm["levels"].as<int>();
        dimension = vm["dimension"].as<int>();
        multigrid = vm["multigrid"].as<bool>();
        left_str  = vm["left"].as<std::string>();
        right_str = vm["right"].as<std::string>();
        beta      = vm["beta"].as<double>();
        solver    = select_solver(solver_str);
        max_iter  = vm["max_iter"].as<int>();
        reltol    = vm["reltol"].as<double>();
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
            experiment1<1>(n_levels, degree, left_str, right_str, beta, order, multigrid, solver, max_iter, reltol);
        }
        break;
    case 2:
        {
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10";
                right_str = "10,10";
            }
            experiment1<2>(n_levels, degree, left_str, right_str, beta, order, multigrid, solver, max_iter, reltol);
        }
        break;
    case 3:
        {
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10,-10";
                right_str = "10,10,10";
            }
            experiment1<3>(n_levels, degree, left_str, right_str, beta, order, multigrid, solver, max_iter, reltol);
        }
        break;
    default:
        throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}
