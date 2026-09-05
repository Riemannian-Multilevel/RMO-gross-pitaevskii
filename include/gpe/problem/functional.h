#ifndef GPE_FUNCTIONAL_H
#define GPE_FUNCTIONAL_H

#include <gpe/lac.h>

namespace gpe
{

class FunctionalBase
{
public:
    virtual ~FunctionalBase() = default;

    virtual void update(const Vector<double>&)
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    virtual double value(const Vector<double>& x) const = 0;
    virtual double directional_derivative(const Vector<double>&, const Vector<double>&) const = 0;

    virtual void gradient(const Vector<double>&, Vector<double>&) const = 0;
    virtual unsigned n_dofs() const = 0;
};

} // namespace gpe

#endif //GPE_FUNCTIONAL_H
