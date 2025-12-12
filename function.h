#ifndef GPE_FUNCTIONS_H
#define GPE_FUNCTIONS_H

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

namespace gpe
{
using dealii::Point;

// TODO: include other potentials
//       use deal.II function objects
template <int dim>
class Square
{
public:
    double operator()(const Point<dim>& p) const {
        typename Point<dim>::value_type out = 0.0;
        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    }
};

} // namespace gpe

#endif //GPE_FUNCTIONS_H