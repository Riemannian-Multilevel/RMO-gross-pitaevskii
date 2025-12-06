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
    const DataOutBase::OutputFormat format, const std::string& filename)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "psi");
    data_out.build_patches(dof_handler.get_fe().degree);

    std::ofstream output(filename);
    data_out.write(output, format);
}

template <int dim>
void output_hdf5(const Vector<double>& solution, const DoFHandler<dim>& dof_handler, const std::string& filename_h5)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "psi");
    data_out.build_patches(dof_handler.get_fe().degree);

    DataOutBase::DataOutFilterFlags flags(true, true);
    DataOutBase::DataOutFilter data_filter(flags);
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
}

template <int dim>
struct GPE_Options
{
    Point<dim> left;    // bottom-left rectangle point
    Point<dim> right;   // upper-right rectangle point
    double beta;        // non-linearity factor
    int n_levels;       // number of levels in finite element discretization
    int degree;         // degree of shape functions
    Ordering order;     // ordering for degrees of freedom
};

template <int dim, typename Function>
class Experiment1
{
public:
    Experiment1(GPE_Options<dim> options_) : options(options_)
    {
        problem = GPE<dim>(options.n_levels, options.degree,
            options.left, options.right, options.order);
        problem.make_rectangle();
    }

    // Populate matrix A_0 = M_V + S, fixed for iteration on every level,
    // based on imposed boundary conditions
    void mass_matrix(Function&& V, BoundaryCondition condition)
    {
        problem.dofs();
        Mass_v.clear();

        constraints = problem.boundary(condition, {0});
        Mass_v.emplace_back(problem.assemble(V, constraints));
    }

    void mass_matrix_mg(Function&& V, BoundaryCondition condition)
    {
        problem.dofs_mg();
        Mass_v.clear();

        mg_constrained_dofs = problem.boundary_mg(condition, {0});

        for (int level = 0; level < options.n_levels; level++) {
            const AffineConstraints<double> &level_constraints =
                mg_constrained_dofs.get_level_constraints(level);

            Mass_v.emplace_back(problem.assemble(V, level_constraints, level));
        }
    }

    Vector<double>
    solve(const Vector<double>& x0, GdOptions& options_rgd, SolverMethod solver) const
    {
        Vector<double> x(x0);
        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        const Triangulation<dim>& triangulation = dof_handler.get_triangulation();

        std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Function object for updating M_phiphi + boundary conditions
        auto update_mpp = [this, dof_handler](SparseMatrix<double>& Mpp, const Vector<double>& x)
        {
            constraints.distribute(x);

            assemble_mass_phiphi<dim>(Mpp, dof_handler, x, constraints);
        };

        x = gp_energy_rgd<dim>(Mass_v[0].A_0, Mass_v[0].M, Mass_v[0].Mpp,update_mpp, x0,
                               options.beta, solver, options_rgd, 5);

        output_results(x, dof_handler, DataOutBase::OutputFormat::vtu, fmt::format("solution_{}d.vtu",dim));
        return x;
    }

    // TODO: MGLevelObject (minimal and maximum level)
    std::vector<Vector<double> >
    solve_mg(const std::vector<Vector<double> >& x0_v, GdOptions& options_rgd, SolverMethod solver) const
    {
        std::vector<Vector<double> > x_v;
        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        const Triangulation<dim>& triangulation = dof_handler.get_triangulation();

        for (int level = 0; level < options.n_levels; level++) {
            std::cerr << "Level: " << level << std::endl;
            std::cerr << "Number of cells: " << triangulation.n_cells(level) << std::endl;
            std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs(level) << std::endl;

            // Retrieve initial value on level
            Vector<double> x0(x0_v[level]);

            // Function object for updating M_phiphi + boundary conditions
            auto update_mpp_level = [this, level, dof_handler](SparseMatrix<double>& Mpp, const Vector<double>& x)
            {
                // Impose boundary conditions on current solution (dirichlet + hanging nodes)
                const AffineConstraints<double> &level_constraints = mg_constrained_dofs.get_level_constraints(level);
                level_constraints.distribute(x);

                // Update weighed matrix for current solution
                assemble_mass_phiphi<dim>(Mpp, dof_handler, x, level_constraints, level);
            };

            auto x = gp_energy_rgd<dim>(Mass_v[level].A_0, Mass_v[level].M, Mass_v[level].Mpp,
                update_mpp_level, x0, options.beta, solver, options_rgd, 5);
            x_v.emplace_back(x);

            output_results(x, dof_handler, DataOutBase::OutputFormat::vtu,
                fmt::format("solution_{}d_lvl{}.vtu", dim, level));
            std::cerr << std::endl;
        }
        return x_v;
    }

    void run(bool multigrid)
    {
        // Define potential function
        auto V = [](const Point<dim>& p) {
            typename Point<dim>::value_type out = 0.0;
            for (unsigned d = 0; d < dim; d++) {
                out += p[d]*p[d];
            }
            return out;
        };
    }

private:
    // Problem parameters
    GPE_Options<dim> options;
    GPE<dim> problem;

    // Mass matrices for every level
    std::vector<GPE_Mass<double> > Mass_v;

    // Constraints for active level or multigrid
    AffineConstraints<double> constraints;
    MGConstrainedDoFs mg_constrained_dofs;
};

// TODO: Write experiment inside a class
template <int dim>
void experiment1(GPE_Options<dim> options_gpe, BoundaryCondition boundary, bool multigrid,
    SolverMethod solver, const GdOptions& options_rgd)
{


    // TODO: options

    // Run iteration, update M_phiphi in every step
    if (multigrid) {



    } else {


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
    BoundaryCondition boundary;

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
            ("constraints", po::value<std::string>()->default_value("neumann"),
                "boundary constraints (neumann|dirichlet)")
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
        auto boundary_str = vm["boundary"].as<std::string>();
        std::transform(order_str.begin(), order_str.end(), order_str.begin(), ::toupper);
        std::transform(solver_str.begin(), solver_str.end(), solver_str.begin(), ::toupper);
        std::transform(boundary_str.begin(), boundary_str.end(), boundary_str.begin(), ::toupper);

        order     = select_order(order_str);
        boundary  = select_boundary_condition(boundary_str);
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
                           order, boundary, multigrid, solver, options);
        }
        break;
    case 2:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10"; right_str = "10,10";
            }
            experiment1<2>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, boundary, multigrid, solver, options);
        }
        break;
    case 3:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10,-10"; right_str = "10,10,10";
            }
            experiment1<3>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, boundary, multigrid, solver, options);
        }
        break;
    default:
        throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}
