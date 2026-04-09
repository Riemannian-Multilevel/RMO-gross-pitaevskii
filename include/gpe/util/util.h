#ifndef GPE_UTIL_H
#define GPE_UTIL_H

#include <deal.II/base/point.h>
#include <deal.II/numerics/data_out.h>

#include <boost/describe.hpp>
#include <boost/mp11.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace gpe
{

//! Take boundary points as strings "x,y,z" from the command-line
//! @tparam dim
//! @param s
//! @param sep
//! @return
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

//! Select dimension (compile-time) at runtime
//! @tparam F
//! @param dim
//! @param f
//! @return
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

//!
//! @tparam dim Problem dimension
//! @param solution
//! @param dof_handler
//! @param format
//! @param filename
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

//!
//! @param s
//! @return
inline std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned const char c){ return std::toupper(c); });
    return s;
}

//!
//! @tparam E
//! @param v
//! @return
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

//!
//! @tparam E
//! @param name
//! @return
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

//!
//! @tparam T
//! @param obj
//! @param out
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

} //namespace gpe

#endif //GPE_UTIL_H