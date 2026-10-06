#ifndef RMO_ROPT_RESIDUAL_H
#define RMO_ROPT_RESIDUAL_H

#include <rmo/lac.h>
#include <algorithm>

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


//! Exception of AssertResidualAt: the residual passed, and the residual at x
DeclException2(ExcResidualNotAt, double, double,
               << "residual " << arg1 << " was not evaluated at x, where R(x) = " << arg2);

//! True if residual is the value of R at x, up to rounding; see AssertResidualAt
[[nodiscard]] inline bool
residual_at(const ResidualBase& R, const Vector<double>& x, const double residual)
{
    return std::abs(R.residual(x) - residual) <= 1e-12 * std::max(1.0, std::abs(residual));
}

} // namespace rmo

//! Debug check that `value` is the residual of the ResidualBase `R` at `x`, e.g. a residual passed to
//! OracleBase::gradient().
#define AssertResidualAt(R, x, value) \
    Assert(::rmo::residual_at((R), (x), (value)), ::rmo::ExcResidualNotAt((value), (R).residual(x)))

#endif //RMO_ROPT_RESIDUAL_H
