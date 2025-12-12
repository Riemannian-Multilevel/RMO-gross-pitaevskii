#ifndef GPE_OPTIONS_H
#define GPE_OPTIONS_H

#include "gpe.h"
#include "descent.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>
namespace po = boost::program_options;  // XXX: move to gpe namespace?

namespace gpe
{

// TODO: parallel/multigrid to other struct
struct MG_Options
{
    bool parallel;      // parallelize matrix assembly
    bool multigrid;     // build a multigrid hierarchy
    int min_level;      // minimum level for multigrid algorithms
    int max_level;      // maximum level for multigrid algorithms
};

inline SolverMethod
select_solver(const std::string& solver_str)
{
    if (solver_str == "GMRES") {
        return SolverMethod::GMRES;
    }
    if (solver_str == "MINRES") {
        return SolverMethod::MINRES;
    }
    if (solver_str == "CG") {
        return SolverMethod::CG;
    }
    throw std::runtime_error(solver_str + ": invalid solver");
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

inline std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    return s;
}


// ---------- MG_Options ----------
inline po::options_description mg_cli_options() {
    po::options_description d("Multigrid options");
    d.add_options()
        ("multigrid", po::value<bool>()->default_value(false)->implicit_value(true),
         "enable multigrid (0|1)")
        ("min-level", po::value<unsigned>()->default_value(0),
         "minimal level for multigrid")
        ("max-level", po::value<unsigned>()->default_value(dealii::numbers::invalid_unsigned_int),
         "maximal level for multigrid")
        ("parallel", po::value<bool>()->default_value(false)->implicit_value(true),
         "parallel assembly for system matrices (0|1)");
    return d;
}

inline void apply_mg_options(const po::variables_map& vm, MG_Options& mg)
{
    mg.multigrid = vm["multigrid"].as<bool>();
    mg.parallel  = vm["parallel"].as<bool>();
    mg.min_level = vm["min-level"].as<unsigned>();
    mg.max_level = vm["max-level"].as<unsigned>();
}


// ---------- GPE_Options ----------
inline po::options_description gpe_cli_options() {
    po::options_description d("General problem options");
    d.add_options()
        ("degree", po::value<int>()->default_value(1),
         "polynomial degree for finite element")
        ("levels", po::value<int>()->default_value(3),
         "number of times to globally refine the mesh")
        ("dimension", po::value<int>()->default_value(2),
         "problem dimension")
        ("order", po::value<std::string>()->default_value("default"),
         "ordering for degrees of freedom (default|random|cuthill_mckee|king|min_deg)")
        ("boundary", po::value<std::string>()->default_value("neumann"),
         "boundary constraints (neumann|dirichlet)")
        ("radius", po::value<double>()->default_value(10.0),
         "default radius of the cube domain")
        ("beta", po::value<double>()->default_value(100.0),
         "non-linearity factor");
    return d;
}

inline void apply_gpe_options(const po::variables_map& vm, GPE_Options& options) {
    const auto order_str    = upper(vm["order"].as<std::string>());
    const auto boundary_str = upper(vm["boundary"].as<std::string>());

    options.order     = select_order(order_str);
    options.bc        = select_boundary_condition(boundary_str);
    options.degree    = vm["degree"].as<int>();
    options.n_levels  = vm["levels"].as<int>();
    options.dimension = vm["dimension"].as<int>();
    options.beta      = vm["beta"].as<double>();
    options.radius    = vm["radius"].as<double>();
}


// ---------- GdOptions ----------
inline po::options_description gd_cli_options() {
    po::options_description d("RGD options");
    d.add_options()
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
         "step size for RGD");
    return d;
}

inline void apply_gd_options(const po::variables_map& vm, GdOptions& options_rgd) {
    const auto solver_str = upper(vm["solver"].as<std::string>());

    options_rgd.step_size    = vm["step-size"].as<double>();
    options_rgd.max_iter     = vm["max-iter"].as<int>();
    options_rgd.max_inner    = vm["max-inner"].as<int>();
    options_rgd.tol_inner    = vm["tol-inner"].as<double>();
    options_rgd.tol_residual = vm["tol-residual"].as<double>();
    options_rgd.tol_lambda   = vm["tol-lambda"].as<double>();
    options_rgd.solver       = select_solver(solver_str);
}

} // namespace gpe

#endif //GPE_OPTIONS_H