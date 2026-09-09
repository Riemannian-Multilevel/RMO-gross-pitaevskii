//
// Created by alad on 9/9/26.
//

#ifndef RMO_INTERPOLATE_H
#define RMO_INTERPOLATE_H

#include <rmo/lac.h>

namespace rmo
{

class LinearTransferBase
{
public:
    virtual ~LinearTransferBase() = default;

    // TODO: match vmult() interface (output&, const input&)
    virtual void to_coarse_mesh(const Vector<double>&, Vector<double>&) const = 0;

    virtual void to_fine_mesh(const Vector<double>&, Vector<double>&) const = 0;

    // Transpose of to_coarse_mesh() (for matrix implementations)
    virtual void Tcoarse(const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    // Transpose of to_fine_mesh()
    virtual void Tfine(const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    virtual unsigned n_coarse() const = 0;
    virtual unsigned n_fine() const = 0;
};


}
#endif //RMO_INTERPOLATE_H
