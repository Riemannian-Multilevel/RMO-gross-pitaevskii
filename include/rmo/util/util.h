#ifndef RMO_UTIL_UTIL_H
#define RMO_UTIL_UTIL_H

#include <deal.II/base/point.h>
#include <deal.II/numerics/data_out.h>

#include <boost/describe.hpp>
#include <boost/mp11.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

/**
 * @file
 * @brief Utilities of the driver programs: runtime selection of the dimension and the system type, solution
 * output, enum/string conversion of the options and option dumps.
 */
namespace rmo
{

namespace gpe
{
// Forward declarations for with_system(); defined in rmo/gpe/gpe.h
template <int dim> class GrossPitaevskiiSystem;
template <int dim> class GrossPitaevskiiLumpedSystem;
} // namespace gpe

//! Point from a string of @p sep separated coordinates, e.g. "x,y,z" from the command line.
template <int dim>
dealii::Point<dim> str_to_point(const std::string& s, const char sep=',') {
    dealii::Point<dim> p;
    std::stringstream ss(s);
    std::string item;

    int i = 0;
    while (std::getline(ss, item, sep) && i < dim) {
        p[i++] = std::stod(item);
    }
    assert(i == dim);
    return p;
}

//! Calls `f(std::integral_constant<int, dim>{})` for the runtime @p dim (1, 2 or 3).
template <class F>
decltype(auto) with_dimension(unsigned dim, F&& f)
{
    switch (dim)
    {
        case 1: return std::forward<F>(f)(std::integral_constant<int, 1>{});
        case 2: return std::forward<F>(f)(std::integral_constant<int, 2>{});
        case 3: return std::forward<F>(f)(std::integral_constant<int, 3>{});
        default:
            throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}

//! Calls `f.template operator()<System>()` with the system type selected at runtime: GrossPitaevskiiLumpedSystem
//! if @p mass_lumping is set, GrossPitaevskiiSystem otherwise; e.g. `f = [&]<typename System>() { ... }`.
template <int dim, class F>
decltype(auto) with_system(bool mass_lumping, F&& f)
{
    if (mass_lumping) {
        return f.template operator()<gpe::GrossPitaevskiiLumpedSystem<dim>>();
    }
    return f.template operator()<gpe::GrossPitaevskiiSystem<dim>>();
}

//! Writes @p solution (named "psi") to @p filename in @p format.
template <int dim>
void output_results(const dealii::Vector<double>& solution, const dealii::DoFHandler<dim>& dof_handler,
    const dealii::DataOutBase::OutputFormat format, const std::string& filename)
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "psi");
    data_out.build_patches(dof_handler.get_fe().degree);

    std::ofstream output(filename);
    data_out.write(output, format);
}

//! Writes the final iterate to `basename.vtk` and, if @p every > 0, also every k-th and the final iterate to
//! `basename_iter<i>.vtk`, which ParaView reads as one time series.
template <int dim>
void output_vtk(const std::vector<dealii::Vector<double>>& history, const dealii::DoFHandler<dim>& dof_handler,
                const std::string& basename, unsigned every)
{
    AssertThrow(!history.empty(), dealii::ExcMessage("no iterates to write"));

    output_results(history.back(), dof_handler, dealii::DataOutBase::vtk, basename + ".vtk");
    if (every == 0) {
        return;
    }

    const std::size_t last = history.size() - 1;
    for (std::size_t i = 0; i <= last; ++i) {
        if (i == last || i % every == 0) {
            output_results(history[i], dof_handler, dealii::DataOutBase::vtk,
                           basename + "_iter" + std::to_string(i) + ".vtk");
        }
    }
}

//! Upper-case copy of @p s.
inline std::string upper(std::string s) {
    std::ranges::transform(s, s.begin(),
                           [](unsigned const char c){ return std::toupper(c); });
    return s;
}

//! Name of the enumerator @p v of a described enum (BOOST_DESCRIBE_ENUM), or "UNKNOWN".
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

//! Enumerator of a described enum (BOOST_DESCRIBE_ENUM) with name @p name.
//! @throws std::runtime_error if there is none.
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

//! Writes the public members of a described struct (BOOST_DESCRIBE_STRUCT) as `name = value` lines.
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

} // namespace rmo

#endif //RMO_UTIL_UTIL_H