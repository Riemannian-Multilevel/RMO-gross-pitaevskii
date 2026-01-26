#ifndef GPE_OPTIONS_H
#define GPE_OPTIONS_H

#include "option_types.h"

#include <stdexcept>
#include <string>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/describe.hpp>
#include <boost/mp11.hpp>

namespace po = boost::program_options;  // XXX: move to gpe namespace?

namespace gpe
{
BOOST_DESCRIBE_STRUCT(GdOptions, (),
    (tol_inner, tol_lambda, tol_residual, step_size, max_iter, max_inner));
BOOST_DESCRIBE_STRUCT(MG_Options, (),
    (multilevel, n_levels, min_level, max_level));
BOOST_DESCRIBE_STRUCT(GPE_Options, (),
    (dimension, degree, radius, beta));

BOOST_DESCRIBE_ENUM(Ordering, DEFAULT, RANDOM, CUTHILL_MCKEE);
BOOST_DESCRIBE_ENUM(BoundaryCondition, NEUMANN, DIRICHLET);
BOOST_DESCRIBE_ENUM(SolverMethod, GMRES, MINRES, CG);
BOOST_DESCRIBE_ENUM(MeshKind, QUADRILATERAL, SIMPLEX);

template<class E>
std::string enum_to_string(E v) {
    using namespace boost::describe;
    bool found = false;
    std::string result;

    // Iterate over enumerators to find the matching value
    using DescribedEnum = describe_enumerators<E>;
    boost::mp11::mp_for_each<DescribedEnum>([&](auto D) {
        if (!found && D.value == v) {
            result = D.name;
            found = true;
        }
    });
    return found ? result : "UNKNOWN";
}

template<class E>
E string_to_enum(const std::string& name) {
    using namespace boost::describe;
    bool found = false;
    E result = {};

    // Iterate over all enumerators of E
    boost::mp11::mp_for_each<describe_enumerators<E>>([&](auto D) {
        // D.name is a const char*, so it compares easily with std::string
        if (!found && D.name == name) {
            result = D.value;
            found = true;
        }
    });

    if (found) return result;

    throw std::runtime_error(name + ": invalid enum value");
}

template<class T>
void dump_options(const T& obj, std::ostream& out)
{
    using namespace boost::describe;
    using namespace boost::mp11;

    mp_for_each<describe_members<T, mod_public>>([&](auto D) {
        auto value = obj.*D.pointer;
        out << D.name << " = ";

        // 1. Check if the member is a float or double
        if constexpr (std::is_floating_point_v<decltype(value)>) {
            // Save current stream state to restore it later
            std::ios old_state(nullptr);
            old_state.copyfmt(out);

            // Set to scientific notation with maximum precision
            out << std::scientific << std::setprecision(6);
            out << value;

            // Restore old state (so integers/enums don't get messed up later)
            out.copyfmt(old_state);
        }
        else {
            // Print everything else normally
            out << value;
        }

        out << std::endl;
    });
}

inline std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned const char c){ return std::toupper(c); });
    return s;
}


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
    options_rgd.solver       = string_to_enum<SolverMethod>(solver_str);
}

} // namespace gpe

#endif //GPE_OPTIONS_H