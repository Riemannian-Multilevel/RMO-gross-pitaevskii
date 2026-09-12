//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_ROPT_CONDITION_H
#define RMO_ROPT_CONDITION_H

#include <rmo/option_types.h>

namespace rmo
{

/**
 * @brief Coarse-correction trigger (eq.~16), pluggable per level.
 * @ref FullApproximationScheme calls this with the two norms it has already computed
 * via its (also pluggable) @c cond_norm -- implementations decide only how to combine
 * them, not how they are measured.
 */
class CoarseConditionBase
{
public:
    virtual ~CoarseConditionBase() = default;

    [[nodiscard]] virtual bool trigger(double norm_level, double norm_coarse, const FAS_Options& options) const = 0;
};

/**
 * @brief @f$ \|R\operatorname{grad}f_h\| \geq \kappa\|\operatorname{grad}f_h\| @f$, gated by
 * @f$ \|\operatorname{grad}f_h\| > \varepsilon @f$. The condition @ref FullApproximationScheme
 * used before this class existed; the default when no level-specific condition is given.
 */
class DefaultCoarseCondition : public CoarseConditionBase
{
public:
    [[nodiscard]] bool trigger(double norm_level, double norm_coarse, const FAS_Options& options) const override
    {
        return norm_coarse >= options.kappa * norm_level && norm_level > options.eps;
    }
};

} // namespace rmo

#endif //RMO_ROPT_CONDITION_H
