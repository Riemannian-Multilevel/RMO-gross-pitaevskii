#ifndef GPE_FUNCTION_H
#define GPE_FUNCTION_H

#include "lac.h"

namespace gpe
{
/**
 * @brief Functor computing the square of the Euclidean norm of a point.
 *
 * Computes \f$ f(p) = \sum_{d=0}^{dim-1} p_d^2 \f$.
 * Used primarily for initializing test cases or potentials.
 *
 * @tparam dim The spatial dimension of the point.
 */
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

template <int dim>
class GrossPitaevskiiEnergy
{
public:
    // TODO: constructor which takes a LinearCombination (object with matrix/operator pointers)
    //       such that matrices can be assembled elsewhere, and value/gradient remain accurate
    double value(const Vector<double>&)
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    Vector<double> gradient(const Vector<double>&)
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }
};

template <int dim>
class NashCoarseModel
{
    double value(const Vector<double>&)
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    Vector<double> gradient(const Vector<double>&)
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }
};

} // namespace gpe

#endif //GPE_FUNCTION_H