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
 *
 * @c min_fine_norm adds a floor on the *fine* norm itself, below which no correction is
 * considered regardless of how the coarse side compares -- eq. (16) has no such term either,
 * but GPE's own @ref DefaultCoarseCondition has always had one (its @c norm_level > options.eps
 * clause; @c --eps on the GPE CLI, @c option.h). It matters on problems where @c norm_coarse
 * stays many orders of magnitude above @c norm_level for the entire run rather than the two
 * converging together, so the @f$\max(\eta\|g\|,\mu)@f$ gate alone never shuts off: measured
 * on the paper's 960x1280 cow image (REVIEW-continuous-cuts.md \S8.2), the scaled restricted
 * norm stayed 5000-9000x the fine norm through 300 iterations while the fine norm itself
 * decayed steadily (139 -> 6.5) -- 145 of 150 eligible iterations triggered a correction, most
 * of them past the point of doing any further good. Default 0: disabled, matching this
 * class's behaviour before this parameter existed.
 */
class ScaledCoarseCondition : public CoarseConditionBase
{
public:
    explicit ScaledCoarseCondition(double grid_scale, double min_fine_norm = 0.0)
        : m_grid_scale(grid_scale), m_min_fine_norm(min_fine_norm)
    {}

    [[nodiscard]] bool trigger(double norm_level, double norm_coarse, const FAS_Options& options) const override
    {
        if (norm_level <= m_min_fine_norm) return false;
        const double norm_coarse_scaled = m_grid_scale * norm_coarse;
        return norm_coarse_scaled >= options.kappa * norm_level && norm_coarse_scaled >= options.eps;
    }

private:
    double m_grid_scale;
    double m_min_fine_norm;
};

} // namespace rmo::cc

#endif //RMO_CC_CONDITION_H
