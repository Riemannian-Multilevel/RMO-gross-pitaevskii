#ifndef RMO_ROPT_RESIDUAL_H
#define RMO_ROPT_RESIDUAL_H

#include <rmo/lac.h>

/**
 * @file
 * @brief ResidualBase: the residual of the problem on one level, the solvers' stopping criterion.
 */
namespace rmo
{

//! Residual of the problem on one level, e.g. the eigenvalue residual of a constrained minimization.
//! Used by the solvers as stopping criterion; independent of the metric of an oracle.
class ResidualBase
{
public:
    virtual ~ResidualBase() = default;

    [[nodiscard]] virtual double residual(const Vector<double>& x) const = 0;

    //! Dimension of the problem; solvers check it against the oracle (debug mode)
    [[nodiscard]] virtual unsigned n_dofs() const = 0;
};

} // namespace rmo

#endif //RMO_ROPT_RESIDUAL_H
