#ifndef GPE_UTIL_H
#define GPE_UTIL_H

#include <deal.II/base/point.h>
#include <sstream>

namespace gpe
{

// Utility function for taking boundary points as strings "x,y,z" from the command-line
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

} //namespace gpe

#endif //GPE_UTIL_H