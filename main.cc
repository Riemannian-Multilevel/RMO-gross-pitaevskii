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
    // TODO: add_mg_data_vector()
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

// SparsityPattern objects need to have the same lifetime as SparseMatrix ones
struct LevelMatrix
{
    void reinit(const SparsityPattern& sparsity)
    {
        sp.copy_from(sparsity);
        M.reinit(sp);
        A0.reinit(sp);
        Mpp.reinit(sp);
    }

    SparsityPattern sp;
    SparseMatrix<double> M;
    SparseMatrix<double> A0;
    SparseMatrix<double> Mpp;
};

template <int dim>
class GPE_Solve
{
public:
    GPE_Solve(Point<dim> left_, Point<dim> right_, double beta_, const GPE_Options& options_)
        : problem(left_, right_, options_), beta(beta_)
    {
        problem.make_rectangle();
        problem.dofs();
    }

    // Populate matrix A_0 = M_V + S based on boundary conditions
    template <typename Function>
    void matrix(Function&& V, LevelMatrix& lm) const
    {
        // Construct sparsity pattern
        lm.reinit(make_sparsity_pattern(problem.get_dof()));

        // Assemble matrix + boundary conditions
        const AffineConstraints<double>& constraints = problem.get_constraints();
        assemble_mass(lm.M, problem.get_dof(), constraints);
        assemble_A0(lm.A0, problem.get_dof(), V, constraints);
    }

    // Begin iteration with constant starting value
    template <typename Function>
    [[maybe_unused]] Vector<double>
    run(Function&& V, const double x0d, GdOptions options_rgd, SolverMethod solver, int n_check_res = 5) const
    {
        const DoFHandler<dim>& dof_handler = problem.get_dof();
        const Triangulation<dim>& triangulation = problem.get_triangulation();

        // Compute solution on most refined (active) level
        std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Define starting value
        Vector<double> x0(dof_handler.n_dofs());
        x0 = x0d;

        // Populate matrices
        LevelMatrix lm;
        this->matrix(V, lm);

        // Update weighed matrix for current solution + boundary conditions
        const AffineConstraints<double>& constraints = problem.get_constraints();

        auto update_mpp = [&dof_handler, &constraints](
            SparseMatrix<double>& Mpp, const Vector<double>& x)
        {
            assemble_mass_phiphi<dim>(Mpp, dof_handler, x, constraints);
        };

        // Run gradient descent + enforce boundary conditions
        Vector<double> x = gp_energy_rgd<dim>(lm.A0, lm.M, lm.Mpp,
            update_mpp, x0, constraints, beta, solver, options_rgd, n_check_res);
        return x;
    }

    void output(const Vector<double>& x)
    {
        using DataOutBase::OutputFormat::vtu;
        output_results(x, problem.get_dof(), vtu, fmt::format("solution_{}d.vtu", dim));
    }

private:
    // Problem parameters
    GPE<dim> problem;
    double beta;
};

// TODO: inherit from GPE base class
template <int dim>
class GPE_Solve_MG
{
public:
    GPE_Solve_MG(Point<dim> left_, Point<dim> right_, double beta_, const GPE_Options& options_)
        : problem(left_, right_, options_), beta(beta_)
    {
        problem.make_rectangle();
        problem.dofs_mg();
    }

    template <typename Function>
    void matrix(Function&& V, LevelMatrix& lm, unsigned int level) const
    {
        AssertIndexRange(level, problem.get_triangulation().n_global_levels());

        // Initialize sparsity pattern and compute entries (level)
        const DoFHandler<dim>& dof_handler = problem.get_dof();
        lm.reinit(make_sparsity_pattern(dof_handler, level));

        const AffineConstraints<double>& level_constraints = problem.get_level_constraints(level);
        assemble_mass(lm.M, problem.get_dof(), level_constraints, level);
        assemble_A0(lm.A0, problem.get_dof(), V, level_constraints, level);
    }

    template <typename Function>
    [[maybe_unused]] std::vector<Vector<double> >
    run(Function&& V, const double x0d, GdOptions options_rgd, SolverMethod solver, int n_check_res = 5) const
    {
        const Triangulation<dim>& triangulation = problem.get_triangulation();
        const unsigned int n_levels = triangulation.n_global_levels();

        // FE matrices for every multigrid level
        std::vector<LevelMatrix> lm_v(n_levels);

        for (unsigned int level = 0; level < n_levels; ++level) {
            this->matrix(V, lm_v[level], level);
        }

        // Iterate over levels
        std::vector<Vector<double> > x_v(n_levels);
        const DoFHandler<dim>& dof_handler = problem.get_dof();

        for (unsigned int level = 0; level < n_levels; level++) {
            std::cerr << "Level: " << level << std::endl;
            std::cerr << "Number of cells: " << triangulation.n_cells(level) << std::endl;
            std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs(level) << std::endl;

            // Define starting value
            Vector<double> x0(dof_handler.n_dofs(level));
            x0 = x0d;

            // Update weighed matrix for current solution + boundary conditions
            const AffineConstraints<double>& level_constraints = problem.get_level_constraints(level);

            auto update_mpp_level = [&dof_handler, level, &level_constraints](
                SparseMatrix<double>& Mpp, const Vector<double>& x)
            {
                assemble_mass_phiphi<dim>(Mpp, dof_handler, x, level_constraints, level);
            };

            // Gradient descent + enforce boundary conditions
            x_v[level] = gp_energy_rgd<dim>(lm_v[level].A0, lm_v[level].M, lm_v[level].Mpp,
                update_mpp_level, x0, level_constraints, beta, solver, options_rgd, n_check_res);
            std::cerr << std::endl;
        }
        return x_v;
    }

    void output(unsigned int level)
    {

    }

private:
    // Problem parameters
    GPE<dim> problem;
    double beta;
};

// Example potential function (later: potential.h, different functions)
template <int dim>
class Square
{
public:
    double operator()(const Point<dim>& p) const {
        typename Point<dim>::value_type out = 0.0;
        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    }
};

// Put it all together
template <int dim>
void package(Point<dim> left, Point<dim> right, double beta, GPE_Options opt_gpe, GdOptions opt_rgd,
    SolverMethod solver, bool multigrid)
{
    Square<dim> V;

    if (multigrid) {
        GPE_Solve_MG<dim> GSM(left, right, beta, opt_gpe);
        auto res_mg = GSM.run(V, 1.0, opt_rgd, solver);

        // TODO: output results
        // dof_handler on mg vectors requires special treatment (add_mg_data_vector())
    }
    else {
        GPE_Solve<dim> GS(left, right, beta, opt_gpe);
        auto res = GS.run(V, 1.0, opt_rgd, solver);

        // TODO: output results
    }
}

int main(int argc, char* argv[])
{
    bool         multigrid;
    std::string  left_str, right_str;
    SolverMethod solver;
    GdOptions    opt_rgd{};
    GPE_Options  opt_gpe{};
    double beta;

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
            ("boundary", po::value<std::string>()->default_value("neumann"),
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
        std::string order_str = vm["order"].as<std::string>();
        std::transform(order_str.begin(), order_str.end(), order_str.begin(), ::toupper);

        std::string solver_str = vm["solver"].as<std::string>();
        std::transform(solver_str.begin(), solver_str.end(), solver_str.begin(), ::toupper);

        std::string boundary_str = vm["boundary"].as<std::string>();
        std::transform(boundary_str.begin(), boundary_str.end(), boundary_str.begin(), ::toupper);

        // General problem parameters
        opt_gpe.order     = select_order(order_str);
        opt_gpe.bc        = select_boundary_condition(boundary_str);
        opt_gpe.degree    = vm["degree"].as<int>();
        opt_gpe.n_levels  = vm["levels"].as<int>();
        opt_gpe.dimension = vm["dimension"].as<int>();

        // Domain parameters
        multigrid = vm["multigrid"].as<bool>();
        left_str  = vm["left"].as<std::string>();
        right_str = vm["right"].as<std::string>();

        // Gradient descent parameters
        opt_rgd.step_size    = vm["step_size"].as<double>();
        opt_rgd.max_iter     = vm["max_iter"].as<int>();
        opt_rgd.max_inner    = vm["max_inner"].as<int>();
        opt_rgd.tol_inner    = vm["tol_inner"].as<double>();
        opt_rgd.tol_residual = vm["tol_residual"].as<double>();
        opt_rgd.tol_lambda   = vm["tol_lambda"].as<double>();
        solver = select_solver(solver_str);
        beta   = vm["beta"].as<double>();
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }

    switch (opt_gpe.dimension) {
    case 1:
        {
            // Default endpoints of rectangle
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10"; right_str = "10";
            }
            package<1>(str_to_point<1>(left_str), str_to_point<1>(right_str), beta, opt_gpe, opt_rgd, solver, multigrid);
        }
        break;
    case 2:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10"; right_str = "10,10";
            }
            package<2>(str_to_point<2>(left_str), str_to_point<2>(right_str), beta, opt_gpe, opt_rgd, solver, multigrid);
        }
        break;
    case 3:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10,-10"; right_str = "10,10,10";
            }
            package<3>(str_to_point<3>(left_str), str_to_point<3>(right_str), beta, opt_gpe, opt_rgd, solver, multigrid);
        }
        break;
    default:
        throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}
