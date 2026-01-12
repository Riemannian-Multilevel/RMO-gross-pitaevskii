#ifndef GPE_OPTIONS_H
#define GPE_OPTIONS_H

#include "gpe.h"
#include "descent.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>
namespace po = boost::program_options;  // XXX: move to gpe namespace?

namespace gpe
{

struct MG_Options
{
    bool multigrid;             // build a multigrid hierarchy
    unsigned int n_levels;      // number of levels for global refinement
    unsigned int min_level;     // minimum level for multigrid algorithms
    unsigned int max_level;     // maximum level for multigrid algorithms
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
        ("levels", po::value<int>()->default_value(3),
        "number of times to globally refine the mesh")
        ("multigrid", po::value<bool>()->default_value(false)->implicit_value(true),
         "enable multigrid (0|1)")
        ("min-level", po::value<int>()->default_value(0),
         "minimal level for multigrid")
        ("max-level", po::value<int>()->default_value(0),
         "maximal level for multigrid");
    return d;
}

// Integer validation
static unsigned int to_unsigned_nonneg(int v, const char* opt_name) {
    if (v < 0)
        throw po::validation_error(po::validation_error::invalid_option_value, opt_name,
                                   std::to_string(v));
    return static_cast<unsigned int>(v);
}

inline void apply_mg_options(const po::variables_map& vm, MG_Options& mg)
{
    mg.multigrid = vm["multigrid"].as<bool>();
    mg.n_levels  = vm["levels"].as<int>();

    // min_level >= 0
    const int min_i = vm["min-level"].as<int>();
    mg.min_level = to_unsigned_nonneg(min_i, "min-level");

    // max_level >= 0, default n_levels
    const int max_i = vm["max-level"].as<int>();
    const unsigned max_u = to_unsigned_nonneg(max_i, "max-level");
    mg.max_level = (max_u == 0) ? mg.n_levels : max_u;

    // min_level <= max_level
    if (mg.max_level < mg.min_level) {
        throw po::validation_error(po::validation_error::invalid_option_value,
                                   "max-level",
                                   "must be >= min-level");
    }
}


// ---------- GPE_Options ----------
inline po::options_description gpe_cli_options() {
    po::options_description d("General problem options");
    d.add_options()
        ("degree", po::value<int>()->default_value(1),
         "polynomial degree for finite element")
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