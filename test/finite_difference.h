#ifndef RMO_TEST_FINITE_DIFFERENCE_H
#define RMO_TEST_FINITE_DIFFERENCE_H

#include <deal.II/lac/vector.h>

/**
 * @file
 * @brief Difference quotients of a function @p f along a direction @p d, to check directional derivatives.
 */
namespace rmo::test
{

//! \f$ (f(x + h d) - f(x)) / h \f$ with \f$ f(x) \f$ given as @p fx, accurate to \f$ O(h) \f$.
template <typename Function>
double forward_difference(Function&& f, const dealii::Vector<double>& x, const dealii::Vector<double>& d,
                          double h, double fx)
{
    dealii::Vector<double> x_h(x);
    x_h.add(h, d);
    return (f(x_h) - fx) / h;
}

//! \f$ (f(x + h d) - f(x - h d)) / (2h) \f$, accurate to \f$ O(h^2) \f$.
template <typename Function>
double central_difference(Function&& f, const dealii::Vector<double>& x, const dealii::Vector<double>& d, double h)
{
    dealii::Vector<double> x_h(x);
    x_h.add(h, d);
    const double f_plus = f(x_h);

    x_h = x;
    x_h.add(-h, d);
    return (f_plus - f(x_h)) / (2 * h);
}

} // namespace rmo::test

#endif // RMO_TEST_FINITE_DIFFERENCE_H
