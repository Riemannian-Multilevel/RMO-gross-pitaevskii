#ifndef GPE_OPTIONS_H
#define GPE_OPTIONS_H

#include <stdexcept>
#include <string>

#include <boost/describe.hpp>
#include <boost/mp11.hpp>

namespace gpe
{
inline std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned const char c){ return std::toupper(c); });
    return s;
}

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

} // namespace gpe

#endif //GPE_OPTIONS_H