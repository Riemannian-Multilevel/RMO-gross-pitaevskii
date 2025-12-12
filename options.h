#ifndef GPE_OPTIONS_H
#define GPE_OPTIONS_H

#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>
namespace po = boost::program_options;  // XXX: move to gpe namespace?

namespace gpe
{

struct GPE_Options
{
    int dimension;          // dimension of domain
    int n_levels;           // number of levels for global refinement
    int degree;             // degree of shape functions
    double radius;          // radius of the cube (square, line) domain
    double beta;            // factor for the non-linear term in GPE
    Ordering order;         // ordering for degrees of freedom
    BoundaryCondition bc;   // problem boundary conditions (dirichlet or neumann)
};

// TODO: parallel/multigrid to other struct
struct MG_Options
{
    bool parallel{false};   // parallelize matrix assembly
    bool multigrid{false};  // build a multigrid hierarchy
    int min_level{0};       // minimum level for multigrid algorithms
    int max_level{0};       // maximum level for multigrid algorithms
};

inline SolverMethod
select_solver(const std::string& solver_str)
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

inline Ordering
select_order(const std::string& order_str)
{
    if (order_str == "DEFAULT") {
        return Ordering::DEFAULT;
    }
    if (order_str == "RANDOM") {
        return Ordering::RANDOM;
    }
    if (order_str == "CUTHILL_MCKEE") {
        return Ordering::CUTHILL_MCKEE;
    }
    if (order_str == "KING") {
        return Ordering::KING;
    }
    if (order_str == "MIN_DEG") {
        return Ordering::MIN_DEG;
    }
    throw std::runtime_error(order_str + ": invalid ordering");
}

inline BoundaryCondition
select_boundary_condition(const std::string& boundary_str)
{
    if (boundary_str == "NEUMANN") {
        return BoundaryCondition::NEUMANN;
    }
    if (boundary_str == "DIRICHLET") {
        return BoundaryCondition::DIRICHLET;
    }
    if (boundary_str == "ROBIN") {
        return BoundaryCondition::ROBIN;
    }
    throw std::runtime_error(boundary_str + ": invalid boundary condition");
}

inline void
add_options(int argc, char* argv[], GPE_Options& options, GdOptions& options_rgd, MG_Options& options_mg)
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("degree", po::value<int>()->default_value(1),
         "polynomial degree for finite element")
        ("levels", po::value<int>()->default_value(3),
         "number of times to globally refine the mesh")
        ("multigrid", po::bool_switch(&options_mg.multigrid),
         "enable multigrid")
        ("dimension", po::value<int>()->default_value(2),
         "problem dimension")
        ("order", po::value<std::string>()->default_value("default"),
         "ordering for degrees of freedom (default|random|cuthill_mckee|king|min_deg)")
        ("boundary", po::value<std::string>()->default_value("neumann"),
            "boundary constraints (neumann|dirichlet)")
        ("radius", po::value<double>()->default_value(10.0),
            "default radius of the cube domain")
        ("beta", po::value<double>()->default_value(100.0),
            "non-linearity factor")
        ("solver", po::value<std::string>()->default_value("gmres"),
            "sparse solver (gmres|minres|cg)")
        ("max-iter", po::value<int>()->default_value(25),
            "maximum number of iterations")
        ("max-inner", po::value<int>()->default_value(100),
            "maximum number of iterations for sparse solver")
        ("tol-residual", po::value<double>()->default_value(1e-4),
            "tolerance for M-residual")
        ("tol-inner", po::value<double>()->default_value(1e-6),
            "tolerance for sparse solver, relative to right-hand side")
        ("tol-lambda", po::value<double>()->default_value(1e-8),
            "tolerance for rayleigh quotient")
        ("step-size", po::value<double>()->default_value(1.0),
            "step size for RGD")
        ("min-level", po::value<int>()->default_value(0),
            "minimal level for multigrid")
        ("max-level", po::value<int>()->default_value(0),
            "maximal level for multigrid")
        ("parallel", po::bool_switch(&options_mg.parallel),
            "parallel assembly for system matrices");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return;
    }
    std::string order_str = vm["order"].as<std::string>();
    std::transform(order_str.begin(), order_str.end(), order_str.begin(), ::toupper);

    std::string solver_str = vm["solver"].as<std::string>();
    std::transform(solver_str.begin(), solver_str.end(), solver_str.begin(), ::toupper);

    std::string boundary_str = vm["boundary"].as<std::string>();
    std::transform(boundary_str.begin(), boundary_str.end(), boundary_str.begin(), ::toupper);

    // General problem parameters
    options.order     = select_order(order_str);
    options.bc        = select_boundary_condition(boundary_str);
    options.degree    = vm["degree"].as<int>();
    options.n_levels  = vm["levels"].as<int>();
    options.dimension = vm["dimension"].as<int>();
    options.beta      = vm["beta"].as<double>();
    options.radius    = vm["radius"].as<double>();

    // Gradient descent parameters
    options_rgd.step_size    = vm["step-size"].as<double>();
    options_rgd.max_iter     = vm["max-iter"].as<int>();
    options_rgd.max_inner    = vm["max-inner"].as<int>();
    options_rgd.tol_inner    = vm["tol-inner"].as<double>();
    options_rgd.tol_residual = vm["tol-residual"].as<double>();
    options_rgd.tol_lambda   = vm["tol-lambda"].as<double>();
    options_rgd.solver       = select_solver(solver_str);

    // Multigrid options
    options_mg.min_level = vm["min-level"].as<int>();
    options_mg.max_level = vm["max-level"].as<int>();
    options_mg.max_level = options_mg.max_level == 0 ? options.n_levels : options_mg.max_level;
}

} // namespace gpe

#endif //GPE_OPTIONS_H