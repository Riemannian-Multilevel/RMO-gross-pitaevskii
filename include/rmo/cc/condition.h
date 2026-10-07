//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_CONDITION_H
#define RMO_CC_CONDITION_H

#include <rmo/ropt/condition.h>

namespace rmo::cc
{

/**
 * @brief Coarse-correction trigger with a per-transition @c grid_scale factor
 * (@f$ \|R\operatorname{grad}f_h\| \cdot \text{grid\_scale} \geq \kappa\|\operatorname{grad}f_h\| @f$,
 * gated by @f$ \|\operatorname{grad}f_h\| > \varepsilon @f$), matching the reference
 * implementation's @c operators.get_grid_scale: it compensates for the norm contraction or
 * growth a given restriction operator introduces, so that @f$ \kappa @f$ carries the same
 * meaning across restriction choices. For Option 1/4's @f$ R=4F_h^H @f$ over one factor-2
 * step, the reference uses @c grid_scale=2.
 */
class ScaledCoarseCondition : public CoarseConditionBase
{
public:
    explicit ScaledCoarseCondition(double grid_scale) : m_grid_scale(grid_scale) {}

    [[nodiscard]] bool trigger(double norm_level, double norm_coarse, const FAS_Options& options) const override
    {
        return m_grid_scale * norm_coarse >= options.kappa * norm_level && norm_level > options.eps;
    }

private:
    double m_grid_scale;
};

} // namespace rmo::cc

#endif //RMO_CC_CONDITION_H
