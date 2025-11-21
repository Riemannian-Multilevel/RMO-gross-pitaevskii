//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include "gpe.h"
#include "energy.h"
#include "util.h"

#include <deal.II/numerics/data_out.h>
#include <iostream>
#include <fmt/format.h>

// Formulate possible extensions as command-line options.
#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>
namespace po = boost::program_options;

using namespace gpe;
using namespace dealii;

//!
//! @tparam dim Problem dimension
//! @param solution
//! @param dof_handler
//! @param format
//! @param filename
template <int dim>
void output_results(const Vector<double>& solution, const DoFHandler<dim>& dof_handler,
    const exportFormat format, const std::string& filename)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();

    std::ofstream output(filename);
    switch (format) {
    case exportFormat::SVG:
        data_out.write_svg(output);
        break;
    case exportFormat::VTK:
        data_out.write_vtk(output);
        break;
    case exportFormat::GNUPLOT:
        data_out.write_gnuplot(output);
        break;
    default:
        throw std::invalid_argument("unknown exportFormat");
    }
}

template <int dim>
void experiment1(int n_levels, int degree, const std::string& left_str, const std::string& right_str,
    double beta, double h, Ordering order, bool multigrid,
    SolverMethod solver, const GdOptions& options)
{
    // Define potential function
    auto V = [](const Point<dim>& p) {
        typename Point<dim>::value_type out = 0.0;
        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    };

    // Set up grid
    GPE<dim> Problem(n_levels, degree, str_to_point<dim>(left_str), str_to_point<dim>(right_str), order);
    Problem.make_rectangle();

    // Set up mass and stiffness matrices
    std::vector<GPE_Mass<double> > Mass_v;
    if (multigrid) {
        Problem.dofs_mg();
        Mass_v = Problem.assemble_mg(V);
    } else {
        Problem.dofs();
        Mass_v = Problem.assemble(V);
    }

    // Run iteration, update M_phiphi in every step
    const DoFHandler<dim>& dof_handler = Problem.get_dof();
    const Triangulation<dim>& triangulation = Problem.get_triangulation();

    if (multigrid) {
        for (int level = 0; level < n_levels; level++) {
            std::cerr << "Level: " << level << std::endl;
            std::cerr << "Number of cells: " << triangulation.n_cells(level) << std::endl;
            std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs(level) << std::endl;

            // Set initial value on level
            Vector<double> x0(Problem.get_dof().n_dofs(level));
            x0 = 1.0;

            // Function object for updating M_phiphi
            auto update_mpp_level = [&dof_handler, level](SparseMatrix<double>& Mpp, const Vector<double>& x)
            {
                assemble_mass_phiphi<dim>(Mpp, dof_handler, x, level);
            };

            Vector<double> x = energy_rgd<dim>(Mass_v[level].A_0, Mass_v[level].M, Mass_v[level].Mpp,
                update_mpp_level, x0, beta, h, solver, options, 5);

            output_results(x, dof_handler, exportFormat::VTK,
                fmt::format("solution_{}d_lvl{}.vtk", dim, level));
            std::cerr << std::endl;
        }
    }
    else {
        std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Set initial value
        Vector<double> x0(Problem.get_dof().n_dofs());
        x0 = 1.0;

        // Function object for updating M_phiphi
        auto update_mpp = [&dof_handler](SparseMatrix<double>& Mpp, const Vector<double>& x)
        {
            assemble_mass_phiphi<dim>(Mpp, dof_handler, x);
        };

        Vector<double> x = energy_rgd<dim>(Mass_v[0].A_0, Mass_v[0].M, Mass_v[0].Mpp,
                update_mpp, x0, beta, h, solver, options, 5);

        output_results(x, dof_handler, exportFormat::VTK, fmt::format("solution_{}d.vtk",dim));
    }
}

int main(int argc, char* argv[])
{
    int degree, n_levels, dimension;
    bool multigrid;
    Ordering order;
    std::string left_str, right_str;
    SolverMethod solver;
    double beta, step_size;
    GdOptions options{};

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
            ("order", po::value<std::string>()->default_value("default"),
             "ordering for degrees of freedom (default|random|cuthill_mckee|king|min_deg)")
            ("left", po::value<std::string>()->default_value(""),
                "left point of the mesh")
            ("right", po::value<std::string>()->default_value(""),
                "right point of the mesh")
            ("beta", po::value<double>()->default_value(100.0),
                "non-linearity factor")
            ("solver", po::value<std::string>()->default_value("gmres"),
                "sparse solver (gmres|minres|cg)")
            ("max_iter", po::value<int>()->default_value(25),
                "maximum number of iterations")
            ("max_inner", po::value<int>()->default_value(100),
                "maximum number of iterations for sparse solver")
            ("tol_residual", po::value<double>()->default_value(1e-4),
                "tolerance for M-residual")
            ("tol_inner", po::value<double>()->default_value(1e-6),
                "tolerance for sparse solver, relative to right-hand side")
            ("tol_lambda", po::value<double>()->default_value(1e-8),
                "tolerance for rayleigh quotient")
            ("step_size", po::value<double>()->default_value(1.0),
                "step size for RGD");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        auto order_str = vm["order"].as<std::string>();
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
        step_size = vm["step_size"].as<double>();
        options.max_iter     = vm["max_iter"].as<int>();
        options.max_inner    = vm["max_inner"].as<int>();
        options.tol_inner    = vm["tol_inner"].as<double>();
        options.tol_residual = vm["tol_residual"].as<double>();
        options.tol_lambda   = vm["tol_lambda"].as<double>();
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
            // Default endpoints of rectangle
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10"; right_str = "10";
            }
            experiment1<1>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, multigrid, solver, options);
        }
        break;
    case 2:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10"; right_str = "10,10";
            }
            experiment1<2>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, multigrid, solver, options);
        }
        break;
    case 3:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10,-10"; right_str = "10,10,10";
            }
            experiment1<3>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, multigrid, solver, options);
        }
        break;
    default:
        throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}
