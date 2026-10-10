#ifndef RMO_OPTION_H
#define RMO_OPTION_H

#include <rmo/option_types.h>
#include <rmo/util/random.h>
#include <rmo/util/util.h>

#include <boost/program_options.hpp>
#include <charconv>
#include <ranges>
#include <string_view>

/**
 * @file
 * @brief Command-line options: option descriptions, parsing into the structs of option_types.h, and validation.
 */
namespace rmo
{
namespace po = boost::program_options;

BOOST_DESCRIBE_STRUCT(DescentOptions, (),
    (tol_residual, step_size, max_iter, line_search, ls));
BOOST_DESCRIBE_STRUCT(DescentOptions::LineSearchOptions, (),
    (max_iter, alpha, beta, sigma, min));
BOOST_DESCRIBE_STRUCT(SolverOptions, (),
    (tol_inner, max_inner, solver, precond));
BOOST_DESCRIBE_STRUCT(MG_Options, (),
    (n_levels, v_levels));
BOOST_DESCRIBE_STRUCT(OutputOptions, (),
    (output_bin, bin_filename, output_vtk, vtk_filename, output_every));
BOOST_DESCRIBE_STRUCT(FAS_Options, (),
    (kappa, eps, coarse_every));


BOOST_DESCRIBE_ENUM(Ordering, DEFAULT, RANDOM, CUTHILL_MCKEE, KING, MIN_DEG);
BOOST_DESCRIBE_ENUM(BoundaryCondition, NEUMANN, DIRICHLET);
BOOST_DESCRIBE_ENUM(SolverMethod, GMRES, MINRES, CG);
BOOST_DESCRIBE_ENUM(Precondition, NONE, DIAGONAL, SPARSE_ILU, AMG);
BOOST_DESCRIBE_ENUM(MeshKind, QUADRILATERAL, SIMPLEX);
BOOST_DESCRIBE_ENUM(MetricKind, NONE, FROBENIUS, MASS, ENERGY_ADAPTIVE)


// ---------- MG_Options ----------
inline po::options_description mg_cli_options() {
    po::options_description d("multilevel options");
    d.add_options()
        ("levels", po::value<int>(),
            "number of times to globally refine the mesh")
        ("multilevel", po::value<std::string>(),
            "levels for the multilevel hierarchy");
        // ("min-level", po::value<int>()->default_value(0),
        //     "minimal level for multilevel")
        // ("max-level", po::value<int>()->default_value(0),
        //     "maximal level for multilevel");
    return d;
}

// Integer validation
static unsigned int to_unsigned_nonneg(int v, const char* opt_name) {
    if (v < 0) {
        throw po::validation_error(po::validation_error::invalid_option_value, opt_name,
                                   std::to_string(v));
    }
    return static_cast<unsigned int>(v);
}

inline void apply_mg_options(const po::variables_map& vm, MG_Options& mg)
{
    if (!vm.contains("levels") && !vm.contains("multilevel")) {
        throw po::required_option("levels or multilevel");
    }

    std::vector<unsigned> v_levels;
    if (vm.contains("multilevel")) {
        auto level_str = vm["multilevel"].as<std::string>();

        for (const auto word: std::views::split(level_str,',')) {
            std::string_view sv(word.begin(), word.end());
            if (sv.empty()) continue;   // handles default "" case

            int val = 0;
            std::from_chars(sv.data(), sv.data() + sv.size(), val);
            v_levels.emplace_back(to_unsigned_nonneg(val, "multilevel"));
        }
    }

    unsigned n_levels = 0;
    if (vm.contains("levels")) {
        n_levels = to_unsigned_nonneg(vm["levels"].as<int>(), "levels");
    }

    // min_level, max_level >= 0
    if (v_levels.empty()) {
        AssertThrow(n_levels > 0, dealii::ExcMessage("--levels must be greater 0"));

        mg.v_levels = {n_levels};
        mg.n_levels = n_levels;
    }
    else {
        //const unsigned min_u = *std::ranges::min_element(v_levels);
        const unsigned max_u = *std::ranges::max_element(v_levels);
        AssertThrow(max_u > 0, dealii::ExcMessage("--multilevel must contain a level greater 0"));

        mg.v_levels = v_levels;

        if (n_levels == 0) {
            mg.n_levels = max_u;
        }
        else {
            AssertThrow(n_levels == max_u, dealii::ExcDimensionMismatch(n_levels, max_u));
            mg.n_levels = n_levels;
        }
    }
}


// ---------- DescentOptions ----------
inline po::options_description descent_cli_options() {
    po::options_description d("RGD options");
    d.add_options()
        ("max-iter", po::value<int>()->default_value(25),
            "maximum number of iterations")
        ("tol-residual", po::value<double>()->default_value(1e-4),
            "tolerance for M-residual")
        ("step-size", po::value<double>()->default_value(1.0),
            "step size for RGD")
        ("line-search", po::value<bool>()->default_value(false)->implicit_value(true),
            "use armijo line search")
        ("ls-max-iter", po::value<int>()->default_value(3),
            "maximum number of iterations for line search")
        ("ls-alpha", po::value<double>()->default_value(1.0),
            "alpha for armijo line search")
        ("ls-beta", po::value<double>()->default_value(0.6),
            "beta for armijo line search")
        ("ls-sigma", po::value<double>()->default_value(0.2),
            "sigma for armijo line search")
        ("ls-min", po::value<double>()->default_value(1e-1),
            "minimal step size for armijo line search");
    return d;
}

inline void apply_descent_options(const po::variables_map& vm, DescentOptions& options_rgd) {
    options_rgd.step_size    = vm["step-size"].as<double>();
    options_rgd.max_iter     = vm["max-iter"].as<int>();
    options_rgd.tol_residual = vm["tol-residual"].as<double>();
    options_rgd.line_search  = vm["line-search"].as<bool>();
    options_rgd.ls.max_iter  = vm["ls-max-iter"].as<int>();
    options_rgd.ls.alpha     = vm["ls-alpha"].as<double>();
    options_rgd.ls.beta      = vm["ls-beta"].as<double>();
    options_rgd.ls.sigma     = vm["ls-sigma"].as<double>();
    options_rgd.ls.min       = vm["ls-min"].as<double>();
}


// ---------- SolverOptions ----------
inline void apply_inner_options(const po::variables_map& vm, SolverOptions& options_slv) {
    const auto solver_str = upper(vm["solver"].as<std::string>());
    const auto precond_str= upper(vm["precond"].as<std::string>());

    options_slv.max_inner     = vm["max-inner"].as<int>();
    options_slv.tol_inner     = vm["tol-inner"].as<double>();
    options_slv.tol_inner_res = vm["tol-inner-res"].as<double>();
    options_slv.solver        = string_to_enum<SolverMethod>(solver_str);
    options_slv.precond       = string_to_enum<Precondition>(precond_str);
}

inline po::options_description inner_cli_options() {
    po::options_description d("Inner solver options");
    d.add_options()
        ("solver", po::value<std::string>()->default_value("cg"),
            "sparse solver (gmres|minres|cg)")
        ("precond", po::value<std::string>()->default_value("none"),
            "preconditioner (none|diagonal|sparse_ilu|amg)")
        ("max-inner", po::value<int>()->default_value(500),
            "maximum number of iterations for sparse solver")
        ("tol-inner", po::value<double>()->default_value(1e-6),
            "tolerance for sparse solver, relative to right-hand side")
        ("tol-inner-res", po::value<double>()->default_value(1e-2),
            "tolerance for sparse solver, relative to residual");
    return d;
}


// ---------- FAS_Options ----------
inline void apply_fas_options(const po::variables_map& vm, FAS_Options& options_fas)
{
    options_fas.kappa        = vm["kappa"].as<double>();
    options_fas.eps          = vm["eps"].as<double>();
    options_fas.coarse_every = vm["coarse-every"].as<unsigned>();
    // options_fas.coarse_energy_adaptive = vm["coarse-energy-adaptive"].as<bool>();
}

inline po::options_description fas_cli_options()
{
    po::options_description d("FAS options");
    d.add_options()
        ("kappa", po::value<double>()->default_value(0.8),\
            "weight for ratio of restricted and coarse gradient")
        ("eps", po::value<double>()->default_value(2e-8),
            "minimum norm of restricted gradient")
        ("coarse-every", po::value<unsigned>()->default_value(2),
            "minimum number of fine steps before coarse step is taken");
        // ("coarse-energy-adaptive", po::value<bool>()->default_value(false)->implicit_value(true),
        //     "solve coarse model with energy-adaptive gradient")
    return d;
}

// ---------- OutputOptions ----------
inline po::options_description output_cli_options()
{
    po::options_description d("Output options");
    d.add_options()
        ("output-bin", po::value<std::string>()->implicit_value(""),
            "write all iterates in binary format, optionally with base name (--output-bin=<name>)")
        ("output-vtk", po::value<std::string>()->implicit_value(""),
            "write the solution in VTK format, optionally with base name (--output-vtk=<name>)")
        ("output-every", po::value<unsigned>()->default_value(0),
            "with --output-vtk, also write every k-th and the final iterate as a series (0: final iterate only)");
    return d;
}

inline void apply_output_options(const po::variables_map& vm, OutputOptions& options_out)
{
    options_out.output_bin = vm.contains("output-bin");
    if (options_out.output_bin) {
        options_out.bin_filename = vm["output-bin"].as<std::string>();
    }

    options_out.output_vtk = vm.contains("output-vtk");
    if (options_out.output_vtk) {
        options_out.vtk_filename = vm["output-vtk"].as<std::string>();
    }
    options_out.output_every = vm["output-every"].as<unsigned>();
}


// ---------- Gross-Pitaevskii problem ----------
namespace gpe
{

BOOST_DESCRIBE_STRUCT(GPE_Options, (),
    (dimension, degree, radius, beta, order, bc, mesh_kind, mass_lumping, initial, initial_arg, seed));
BOOST_DESCRIBE_ENUM(Potential, ZERO, CONSTANT, SQUARE, OPTICAL_LATTICE, EXPRESSION);
BOOST_DESCRIBE_ENUM(InitialValue, CONSTANT, COSINE, RANDOM);
BOOST_DESCRIBE_STRUCT(CoarseModelOptions, (),
    (metric_t, transport_t));
BOOST_DESCRIBE_ENUM(Transport, FROBENIUS, MASS, DIFFERENTIAL, ADJOINT_RESTRICTION, ADJOINT_DIFFERENTIAL,
                    ADJOINT_RESTRICTION_FROBENIUS, ADJOINT_DIFFERENTIAL_FROBENIUS, DIFFERENTIAL_FROBENIUS);


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
            "type of mesh elements used (quadrilateral|simplex)")
        ("potential", po::value<std::string>()->default_value("square"),
            "used potential (zero|constant|square|optical_lattice|expression)")
        ("potential-expr", po::value<std::string>()->default_value(""),
            "potential as an expression in the coordinates x[,y[,z]] (e.g. \"0.5*(x^2+y^2)\"); "
            "implies --potential expression")
        ("mass-lumping", po::value<bool>()->default_value(false)->implicit_value(true),
            "use lumped (diagonal) mass matrices; requires a lumpable element "
            "(quadrilateral, or simplex with degree <= 2)")
        ("initial", po::value<std::string>()->default_value("constant"),
            "initial value (constant|cosine|random)")
        ("initial-arg", po::value<std::string>()->default_value(""),
            "value (constant, default 1) or bump radius (cosine, default --radius)")
        ("seed", po::value<unsigned>()->default_value(default_seed),
            "seed of the random number generator");
    return d;
}

inline void apply_gpe_options(const po::variables_map& vm, GPE_Options& options) {
    const auto order_str    = upper(vm["order"].as<std::string>());
    const auto boundary_str = upper(vm["boundary"].as<std::string>());
    const auto mesh_str     = upper(vm["mesh"].as<std::string>());
    const auto potential_str= upper(vm["potential"].as<std::string>());
    const auto initial_str  = upper(vm["initial"].as<std::string>());
    const auto initial_arg_str = vm["initial-arg"].as<std::string>();

    options.order     = string_to_enum<Ordering>(order_str);
    options.bc        = string_to_enum<BoundaryCondition>(boundary_str);
    options.mesh_kind = string_to_enum<MeshKind>(mesh_str);
    options.potential = string_to_enum<Potential>(potential_str);
    options.initial   = string_to_enum<InitialValue>(initial_str);

    options.potential_expr = vm["potential-expr"].as<std::string>();
    options.mass_lumping   = vm["mass-lumping"].as<bool>();

    options.seed = vm["seed"].as<unsigned>();

    options.initial_arg.reset();
    if (!initial_arg_str.empty()) {
        if (options.initial == InitialValue::RANDOM) {
            throw std::invalid_argument("--initial-arg: not used by --initial random (see --seed)");
        }
        try {
            std::size_t pos = 0;
            options.initial_arg = std::stod(initial_arg_str, &pos);
            if (pos != initial_arg_str.size()) {
                throw std::invalid_argument(initial_arg_str);
            }
        }
        catch (const std::logic_error&) {  // invalid_argument or out_of_range
            throw std::invalid_argument("--initial-arg: not a number: " + initial_arg_str);
        }
    }

    // A supplied expression selects the parsed potential; asking for it without one is an error
    if (!options.potential_expr.empty()) {
        options.potential = Potential::EXPRESSION;
    }
    else if (options.potential == Potential::EXPRESSION) {
        throw std::invalid_argument("--potential expression requires --potential-expr");
    }
    options.degree    = vm["degree"].as<int>();
    options.dimension = vm["dimension"].as<int>();
    options.beta      = vm["beta"].as<double>();
    options.radius    = vm["radius"].as<double>();
}


// TODO: encode default values in option_type.h and remove default_value()?


// ---------- CoarseModelOptions ----------
inline po::options_description coarse_model_cli_options()
{
    po::options_description d("Coarse model options");
    d.add_options()
        ("metric", po::value<std::string>()->default_value("mass"),
            "metric for coarse model (none|frobenius|mass)")
        ("metric-cond", po::value<std::string>()->default_value("none"),
            "metric for coarse condition evaluation (none|frobenius|mass)")
        // ("metric-smooth", po::value<std::string>()->default_value("energy_adaptive"),
        //     "metric for smoother (energy_adaptive|mass|frobenius)")
        ("transport", po::value<std::string>()->default_value("mass"),
            "vector transport operator (frobenius|mass|differential|adjoint_restriction|adjoint_differential|"
            "adjoint_restriction_frobenius|adjoint_differential_frobenius|differential_frobenius)");
    return d;
}

inline void apply_coarse_model_options(const po::variables_map& vm, CoarseModelOptions& options_cm)
{
    const auto metric_str = upper(vm["metric"].as<std::string>());
    const auto ccond_str  = upper(vm["metric-cond"].as<std::string>());
    const auto transp_str = upper(vm["transport"].as<std::string>());
    // const auto smooth_str = upper(vm["metric-smooth"].as<std::string>());

    options_cm.metric_t    = string_to_enum<MetricKind>(metric_str);
    options_cm.ccond_t     = string_to_enum<MetricKind>(ccond_str);
    // options_cm.smooth_t    = string_to_enum<MetricKind>(smooth_str);
    options_cm.transport_t = string_to_enum<Transport>(transp_str);
}

} // namespace gpe

} // namespace rmo

#endif //RMO_OPTION_H
