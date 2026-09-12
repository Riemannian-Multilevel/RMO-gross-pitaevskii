//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_ORACLE_COARSE_H
#define RMO_CC_ORACLE_COARSE_H

#include <rmo/cc/manifold.h>
#include <rmo/cc/metric.h>
#include <rmo/cc/oracle.h>

#include <rmo/ropt/oracle_coarse_base.h>

namespace rmo::cc
{

namespace detail
{

/**
 * @brief Coarse model value: @f$ q_k(z) = E_H(z) - w_k^\top(\operatorname{logit}(z) - \operatorname{logit}(\psi_k)) @f$.
 * Metric-independent and affine in logit coordinates: the Fisher-Rao weighting in
 * @f$ \langle w_k, \mathcal L_{\psi_k}(z)\rangle_{\psi_k} @f$ cancels exactly against the one
 * built into the lifting map @f$ \mathcal L_{\psi_k} @f$ itself.
 */
inline double coarse_value(const Vector<double>& z, const Vector<double>& psi,
                           const Vector<double>& w, double energy)
{
    double correction = 0.0;
    for (unsigned int i = 0; i < z.size(); i++) {
        correction += w[i] * (bernoulli::logit(z[i]) - bernoulli::logit(psi[i]));
    }
    return energy - correction;
}

/** @brief @f$ \mathrm D q_k(z)[v] = \mathrm D E_H(z)[v] - w_k^\top(v/(z(1-z))) @f$. */
inline double coarse_dir_deriv(double ambient_dir_deriv, const Vector<double>& z,
                               const Vector<double>& w, const Vector<double>& v)
{
    double s = ambient_dir_deriv;
    for (unsigned int i = 0; i < z.size(); i++) {
        s -= w[i] * v[i] / metric::fisher_rao::weight(z[i]);
    }
    return s;
}

/**
 * @brief Riemannian coarse gradient: @f$ \operatorname{grad} q_k(z) = \operatorname{diag}(z(1-z))\,\operatorname{grad}_{\mathrm{amb}}E_H(z) - w_k @f$.
 *
 * No metric weighting is applied to @f$ w_k @f$ itself. The general Riemannian coarse model
 * pulls @f$ w_k @f$ back through the metric-adjoint of the differentiated lifting map,
 * @f$ \mathrm D\mathcal L_{\psi_k}(z)^* @f$ (w.r.t. @f$ \langle\cdot,\cdot\rangle_{\psi_k} @f$ and
 * @f$ \langle\cdot,\cdot\rangle_z @f$) -- and from its defining property
 * @f$ \langle \mathrm D\mathcal L_{\psi_k}(z)[u], w\rangle_{\psi_k} = \langle u, \mathrm D\mathcal L_{\psi_k}(z)^*[w]\rangle_z @f$
 * this adjoint is exactly the identity map here (verified numerically against the reference
 * implementation), unlike @ref bernoulli::retract_inv_diff_adjoint, which is the *Euclidean*
 * self-adjoint of the same map and does not apply in this construction.
 */
inline void coarse_grad(const Vector<double>& ambient_grad, const Vector<double>& z,
                        const Vector<double>& w, Vector<double>& dst)
{
    dst.reinit(z.size());
    for (unsigned int i = 0; i < z.size(); i++) {
        dst[i] = metric::fisher_rao::weight(z[i]) * ambient_grad[i] - w[i];
    }
}

} // namespace detail


/**
 * @brief Coarse model oracle for continuous cuts, evaluating @f$ q_k @f$ and its Riemannian
 * gradient under the Fisher-Rao metric. Only one metric is used throughout this problem, so
 * this single class plays both the @c TiltCoarseOracleType and @c CoarseModelType roles
 * @ref FullApproximationScheme::cycle expects.
 */
class ContinuousCutsCoarseOracle : public OracleBase
{
public:
    const char* id() const override { return "CCC"; }
    static constexpr auto metric_t = MetricKind::FISHER_RAO;

    ContinuousCutsCoarseOracle(CoarseOracleBase& model, SolverOptions = {})
        : m_model(model)
    {
        AssertThrow(model.coarse().get_metric() == metric_t,
            dealii::ExcInternalError("Fisher-Rao metric expected"));
    }

    // Update for _evaluation_ of the coarse model at the trial point z; distinct from
    // CoarseOracleBase::update_model(x), which updates the fixed correction psi_k, w_k.
    void update(const Vector<double>& z) override
    {
        m_model.coarse().update(z);
    }

    [[nodiscard]] double value(const Vector<double>& z) const override
    {
        const auto& state = m_model.get_state();
        return detail::coarse_value(z, state.y, state.w, m_model.coarse().value(z));
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& z, const Vector<double>& v) const override
    {
        const auto& state = m_model.get_state();
        return detail::coarse_dir_deriv(m_model.coarse().directional_derivative(z, v), z, state.w, v);
    }

    GradInfo gradient(const Vector<double>& z, Vector<double>& output) const override
    {
        OracleBase& T_coarse = m_model.coarse();

        Vector<double> riemannian(z.size());
        auto info = T_coarse.gradient(z, riemannian);

        // Undo diag(z(1-z)) to recover the ambient gradient of E_H at z.
        Vector<double> ambient(z.size());
        T_coarse.apply_metric(riemannian, ambient);

        detail::coarse_grad(ambient, z, m_model.get_state().w, output);
        return info;
    }

    // No linear solve is involved, so the tolerance argument is unused.
    GradInfo gradient(const Vector<double>& z, Vector<double>& output, double) const override
    {
        return gradient(z, output);
    }

    [[nodiscard]] double residual(const Vector<double>& z) const override
    {
        Vector<double> g(z.size());
        gradient(z, g);

        return metric::fisher_rao::norm(g, z);
    }

    [[nodiscard]] double norm(const Vector<double>& v) const override { return m_model.norm(v); }

    [[nodiscard]] double inner(const Vector<double>& u, const Vector<double>& v) const override
    {
        return m_model.metric(u, v);
    }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        m_model.apply_metric(src, dst);
    }

    [[nodiscard]] MetricKind get_metric() const override { return metric_t; }
    [[nodiscard]] unsigned n_dofs() const override { return m_model.coarse().n_dofs(); }

private:
    CoarseOracleBase& m_model;
};

} // namespace rmo::cc

#endif //RMO_CC_ORACLE_COARSE_H
