#ifndef GPE_OPTION_CLI_H
#define GPE_OPTION_CLI_H

#include "option_types.h"
#include "util.h"

#include <boost/program_options.hpp>

namespace gpe
{
namespace po = boost::program_options;

BOOST_DESCRIBE_STRUCT(DescentOptions, (),
    (tol_lambda, tol_residual, step_size, max_iter));
BOOST_DESCRIBE_STRUCT(SolverOptions, (),
    (tol_inner, max_inner, solver, precond));
BOOST_DESCRIBE_STRUCT(MG_Options, (),
    (multilevel, n_levels, min_level, max_level));
BOOST_DESCRIBE_STRUCT(GPE_Options, (),
    (dimension, degree, radius, beta, order, bc, mesh_kind));

BOOST_DESCRIBE_ENUM(Ordering, DEFAULT, RANDOM, CUTHILL_MCKEE);
BOOST_DESCRIBE_ENUM(BoundaryCondition, NEUMANN, DIRICHLET);
BOOST_DESCRIBE_ENUM(SolverMethod, GMRES, MINRES, CG);
BOOST_DESCRIBE_ENUM(Precondition, NONE, DIAGONAL, SPARSE_ILU, AMG);
BOOST_DESCRIBE_ENUM(MeshKind, QUADRILATERAL, SIMPLEX);

// ---------- MG_Options ----------
inline po::options_description mg_cli_options() {
    po::options_description d("multilevel options");
    d.add_options()
        ("levels", po::value<int>()->default_value(3),
            "number of times to globally refine the mesh")
        ("multilevel", po::value<bool>()->default_value(false)->implicit_value(true),
            "enable multilevel (0|1)")
        ("min-level", po::value<int>()->default_value(0),
            "minimal level for multilevel")
        ("max-level", po::value<int>()->default_value(0),
            "maximal level for multilevel");
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
    mg.multilevel = vm["multilevel"].as<bool>();
    mg.n_levels   = vm["levels"].as<int>();

    // min_level >= 0
    const int min_i = vm["min-level"].as<int>();
    mg.min_level    = to_unsigned_nonneg(min_i, "min-level");

    // max_level >= 0, default n_levels
    const int max_i      = vm["max-level"].as<int>();
    const unsigned max_u = to_unsigned_nonneg(max_i, "max-level");
    mg.max_level         = (max_u == 0) ? mg.n_levels : max_u;

    // min_level <= max_level
    if (mg.max_level < mg.min_level) {
        throw po::validation_error(po::validation_error::invalid_option_value,
            "max-level", "must be >= min-level");
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
            "non-linearity factor")
        ("mesh", po::value<std::string>()->default_value("quadrilateral"),
            "type of mesh elements used (quadrilateral|simplex)");
    return d;
}

inline void apply_gpe_options(const po::variables_map& vm, GPE_Options& options) {
    const auto order_str    = upper(vm["order"].as<std::string>());
    const auto boundary_str = upper(vm["boundary"].as<std::string>());
    const auto mesh_str     = upper(vm["mesh"].as<std::string>());

    options.order     = string_to_enum<Ordering>(order_str);
    options.bc        = string_to_enum<BoundaryCondition>(boundary_str);
    options.mesh_kind = string_to_enum<MeshKind>(mesh_str);
    options.degree    = vm["degree"].as<int>();
    options.dimension = vm["dimension"].as<int>();
    options.beta      = vm["beta"].as<double>();
    options.radius    = vm["radius"].as<double>();
}


// TODO: encode default values in option_type.h and remove default_value()?
// ---------- DescentOptions ----------
inline po::options_description descent_cli_options() {
    po::options_description d("RGD options");
    d.add_options()
        ("max-iter", po::value<int>()->default_value(25),
            "maximum number of iterations")
        ("max-search", po::value<int>()->default_value(20),
            "maximum number of iterations for line search")
        ("tol-residual", po::value<double>()->default_value(1e-4),
            "tolerance for M-residual")
        ("tol-lambda", po::value<double>()->default_value(1e-8),
            "tolerance for rayleigh quotient")
        ("step-size", po::value<double>()->default_value(1.0),
            "step size for RGD")
        ("line-search", po::value<bool>()->default_value(false)->implicit_value(true),
            "use armijo line search")
        ("ls-alpha", po::value<double>()->default_value(1.0),
            "alpha for armijo line search")
        ("ls-beta", po::value<double>()->default_value(0.5),
            "beta for armijo line search")
        ("ls-sigma", po::value<double>()->default_value(0.4),
            "sigma for armijo line search")
        ("ls-min", po::value<double>()->default_value(1e-4),
            "minimal step size for armijo line search");
    return d;
}

inline void apply_descent_options(const po::variables_map& vm, DescentOptions& options_rgd) {
    options_rgd.step_size    = vm["step-size"].as<double>();
    options_rgd.max_iter     = vm["max-iter"].as<int>();
    options_rgd.max_search   = vm["max-search"].as<int>();
    options_rgd.tol_residual = vm["tol-residual"].as<double>();
    options_rgd.tol_lambda   = vm["tol-lambda"].as<double>();
    options_rgd.line_search  = vm["line-search"].as<bool>();
    options_rgd.ls_alpha     = vm["ls-alpha"].as<double>();
    options_rgd.ls_beta      = vm["ls-beta"].as<double>();
    options_rgd.ls_sigma     = vm["ls-sigma"].as<double>();
    options_rgd.ls_min       = vm["ls-min"].as<double>();
}


// ---------- SolverOptions ----------
inline void apply_inner_options(const po::variables_map& vm, SolverOptions& options_slv) {
    const auto solver_str = upper(vm["solver"].as<std::string>());
    const auto precond_str= upper(vm["precond"].as<std::string>());

    options_slv.max_inner    = vm["max-inner"].as<int>();
    options_slv.tol_inner    = vm["tol-inner"].as<double>();
    options_slv.solver       = string_to_enum<SolverMethod>(solver_str);
    options_slv.precond      = string_to_enum<Precondition>(precond_str);
}

inline po::options_description inner_cli_options() {
    po::options_description d("Inner solver options");
    d.add_options()
        ("solver", po::value<std::string>()->default_value("gmres"),
            "sparse solver (gmres|minres|cg)")
        ("precond", po::value<std::string>()->default_value("none"),
            "preconditioner (none|diagonal|sparse_ilu|amg)")
        ("max-inner", po::value<int>()->default_value(100),
            "maximum number of iterations for sparse solver")
        ("tol-inner", po::value<double>()->default_value(1e-6),
            "tolerance for sparse solver, relative to right-hand side");
    return d;
}

}

#endif //GPE_OPTION_CLI_H