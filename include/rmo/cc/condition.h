//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_CONDITION_H
#define RMO_CC_CONDITION_H

#include <rmo/ropt/condition.h>

namespace rmo::cc
{

/**
 * @brief Coarse-correction trigger implementing eq. (16) as written:
 * @f$ \|R\operatorname{grad}f_h\|_\psi \geq \max(\eta\|\operatorname{grad}f_h\|_\phi, \mu) @f$,
 * with @f$ \eta = @f$ @c options.kappa and @f$ \mu = @f$ @c options.eps -- i.e. @f$ \mu @f$
 * bounds the *coarse* (restricted) norm, not the fine one. Equivalent to the conjunction of
 * the scaled-comparison and the gate, both evaluated on @c norm_coarse.
 *
 * The @c grid_scale factor multiplying @c norm_coarse before either comparison matches the
 * reference implementation's @c operators.get_grid_scale: it is a reference-side compensation
 * for the norm contraction/growth a given restriction operator introduces (for Option 1/4's
 * @f$ R=4F_h^H @f$ over one factor-2 step, the reference uses @c grid_scale=2), not something
 * in the paper itself -- eq. (16) has no such factor.
 *
 * Three variants of the mu gate were compared (REVIEW-continuous-cuts.md \S3.3): the paper's
 * (implemented here), the reference's (Euclidean norm of the fine gradient,
 * @c multilevel.py:104-106), and the port's previous default (@ref DefaultCoarseCondition,
 * fine Fisher-Rao norm > eps). All three give the same final energy on the synthetic problem;
 * only the number of coarse corrections differs (paper's form: 4, vs 3 for the other two).
 */
class ScaledCoarseCondition : public CoarseConditionBase
{
public:
    explicit ScaledCoarseCondition(double grid_scale) : m_grid_scale(grid_scale) {}

    [[nodiscard]] bool trigger(double norm_level, double norm_coarse, const FAS_Options& options) const override
    {
        const double norm_coarse_scaled = m_grid_scale * norm_coarse;
        return norm_coarse_scaled >= options.kappa * norm_level && norm_coarse_scaled >= options.eps;
    }

private:
    double m_grid_scale;
};

} // namespace rmo::cc

#endif //RMO_CC_CONDITION_H
