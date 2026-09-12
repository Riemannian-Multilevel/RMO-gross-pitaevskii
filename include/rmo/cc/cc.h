//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_CC_H
#define RMO_CC_CC_H

#include <rmo/cc/grid.h>
#include <rmo/lac.h>

#include <cmath>
#include <utility>

namespace rmo::cc
{

/**
 * @brief Ambient-space (Euclidean) smoothed continuous-cuts objective (eq. (42), \S 6.3.1):
 * @f[
 *   E_\varepsilon^{CC}(\phi) = \rho^\top\phi + \alpha\sum_i\sqrt{(D_1\phi)_i^2+(D_2\phi)_i^2+\varepsilon^2}.
 * @f]
 *
 * @ref update caches @f$ D\phi @f$ so that @ref value, @ref gradient, and
 * @ref directional_derivative do not each re-apply the sparse operator @f$ D @f$;
 * as with @ref rmo::gpe::GrossPitaevskiiFunctional, they assume @p phi matches the
 * point passed to the most recent @ref update call.
 */
class ContinuousCutsFunctional
{
public:
    ContinuousCutsFunctional(const ForwardDifference& D, Vector<double> rho, double alpha, double epsilon)
        : m_D(D), m_rho(std::move(rho)), m_alpha(alpha), m_epsilon(epsilon)
    {
        AssertDimension(m_rho.size(), m_D.n_cols());
    }

    void update(const Vector<double>& phi)
    {
        m_phi = phi;
        m_Dphi.reinit(m_D.n_rows());
        m_D.matrix().vmult(m_Dphi, phi);
    }

    double value(const Vector<double>& phi) const
    {
        double tv = 0.0;
        for (unsigned int i = 0; i < n_dofs(); i++) {
            tv += 1.0 / weight_at(i);
        }
        return m_rho * phi + m_alpha * tv;
    }

    double directional_derivative(const Vector<double>& /* phi */, const Vector<double>& z) const
    {
        const unsigned int n = n_dofs();

        Vector<double> Dz(m_D.n_rows());
        m_D.matrix().vmult(Dz, z);

        double s = m_rho * z;
        for (unsigned int i = 0; i < n; i++) {
            s += m_alpha * weight_at(i) * (m_Dphi[i] * Dz[i] + m_Dphi[n + i] * Dz[n + i]);
        }
        return s;
    }

    /**
     * @brief Ambient (Euclidean) gradient: @f$ \rho + \alpha D^\top(I_2\otimes\operatorname{diag}(\omega))D\phi @f$,
     * with @f$ \omega_i = 1/\sqrt{(D_1\phi)_i^2+(D_2\phi)_i^2+\varepsilon^2} @f$.
     */
    void gradient(const Vector<double>& /* phi */, Vector<double>& dst) const
    {
        const unsigned int n = n_dofs();

        Vector<double> weighted(m_Dphi);
        for (unsigned int i = 0; i < n; i++) {
            const double omega_i = weight_at(i);
            weighted[i]     *= omega_i;
            weighted[n + i] *= omega_i;
        }

        dst.reinit(n);
        m_D.matrix().Tvmult(dst, weighted);
        dst *= m_alpha;
        dst += m_rho;
    }

    [[nodiscard]] unsigned int n_dofs() const { return m_D.n_cols(); }

    /**
     * @brief The point passed to the most recent @ref update. State-dependent quantities
     * that need a "current point" but take none of their own (@ref FisherRaoOracle's
     * norm/inner/apply_metric, per the @ref OracleBase interface) read it from here rather
     * than caching their own copy, so that two oracle instances wrapping the same
     * functional -- as @ref FullApproximationScheme::cycle constructs for its @c O_level
     * and @c T_level -- see the same current point after either one calls @c update.
     */
    [[nodiscard]] const Vector<double>& get_phi() const { return m_phi; }

    [[nodiscard]] const ForwardDifference& get_D() const { return m_D; }
    [[nodiscard]] const Vector<double>& get_rho() const { return m_rho; }
    [[nodiscard]] double get_alpha() const { return m_alpha; }
    [[nodiscard]] double get_epsilon() const { return m_epsilon; }

private:
    /** @brief @f$ \omega_i = 1/\sqrt{(D_1\phi)_i^2+(D_2\phi)_i^2+\varepsilon^2} @f$ from the cached @f$ D\phi @f$. */
    [[nodiscard]] double weight_at(unsigned int i) const
    {
        const unsigned int n = n_dofs();
        return 1.0 / std::sqrt(m_Dphi[i] * m_Dphi[i] + m_Dphi[n + i] * m_Dphi[n + i] + m_epsilon * m_epsilon);
    }

    const ForwardDifference& m_D;
    Vector<double> m_rho;
    double m_alpha, m_epsilon;

    Vector<double> m_phi;
    Vector<double> m_Dphi;
};

} // namespace rmo::cc

#endif //RMO_CC_CC_H
