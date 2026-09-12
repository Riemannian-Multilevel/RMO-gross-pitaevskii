//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_ORACLE_H
#define RMO_CC_ORACLE_H

#include <rmo/cc/cc.h>
#include <rmo/cc/metric.h>

#include <rmo/ropt/oracle_base.h>

#include <deal.II/base/timer.h>

namespace rmo::cc
{

namespace detail
{

/** @brief Riemannian gradient (Table 9): @f$ \operatorname{diag}(\phi(1-\phi)) \cdot (\text{ambient gradient}) @f$. */
inline void grad_fisher_rao(const Vector<double>& ambient_grad, const Vector<double>& phi, Vector<double>& dst)
{
    AssertDimension(ambient_grad.size(), phi.size());
    dst.reinit(phi.size());

    for (unsigned int i = 0; i < phi.size(); i++) {
        dst[i] = metric::fisher_rao::weight(phi[i]) * ambient_grad[i];
    }
}

} // namespace detail


/** @brief Stationarity measure @f$ \|\operatorname{grad} E_\varepsilon^{CC}(\phi)\|_\phi @f$, used as the residual. */
class ContinuousCutsResidual
{
public:
    explicit ContinuousCutsResidual(const ContinuousCutsFunctional& func)
        : m_func(func)
    {}

    // Assumes m_func.update(phi) has already been called.
    [[nodiscard]] double residual(const Vector<double>& phi) const
    {
        Vector<double> ambient(phi.size());
        m_func.gradient(phi, ambient);

        Vector<double> riemannian(phi.size());
        detail::grad_fisher_rao(ambient, phi, riemannian);

        return metric::fisher_rao::norm(riemannian, phi);
    }

private:
    const ContinuousCutsFunctional& m_func;
};


/** @brief @ref OracleBase for the continuous-cuts objective under the Fisher-Rao metric. */
class FisherRaoOracle : public OracleBase
{
public:
    const char* id() const override { return "FR"; }
    static constexpr auto metric_t = MetricKind::FISHER_RAO;

    // The SolverOptions parameter is unused (no linear solve is involved), but required so
    // FisherRaoOracle satisfies TiltOracle<FisherRaoOracle, ContinuousCutsFunctional> for
    // FullApproximationScheme, matching rmo::gpe::FrobeniusOracle's constructor.
    FisherRaoOracle(ContinuousCutsFunctional& func, SolverOptions = {})
        : m_func(func), m_res(func)
    {}

    void update(const Vector<double>& phi) override
    {
        m_func.update(phi);
    }

    [[nodiscard]] double value(const Vector<double>& phi) const override
    {
        return m_func.value(phi);
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& phi, const Vector<double>& z) const override
    {
        return m_func.directional_derivative(phi, z);
    }

    GradInfo gradient(const Vector<double>& phi, Vector<double>& output) const override
    {
        dealii::Timer timer;
        timer.start();

        Vector<double> ambient(phi.size());
        m_func.gradient(phi, ambient);
        detail::grad_fisher_rao(ambient, phi, output);

        timer.stop();
        GradInfo info{};
        info.elapsed_time = timer.cpu_time();
        return info;
    }

    // No linear solve is involved, so the tolerance argument is unused.
    GradInfo gradient(const Vector<double>& phi, Vector<double>& output, double) const override
    {
        return gradient(phi, output);
    }

    [[nodiscard]] double residual(const Vector<double>& phi) const override
    {
        return m_res.residual(phi);
    }

    [[nodiscard]] double norm(const Vector<double>& v) const override
    {
        return metric::fisher_rao::norm(v, m_func.get_phi());
    }

    [[nodiscard]] double inner(const Vector<double>& u, const Vector<double>& v) const override
    {
        return metric::fisher_rao::inner(u, v, m_func.get_phi());
    }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        const Vector<double>& phi = m_func.get_phi();
        AssertDimension(src.size(), phi.size());
        dst.reinit(src.size());

        for (unsigned int i = 0; i < src.size(); i++) {
            dst[i] = src[i] / metric::fisher_rao::weight(phi[i]);
        }
    }

    [[nodiscard]] MetricKind get_metric() const override { return metric_t; }
    [[nodiscard]] unsigned n_dofs() const override { return m_func.n_dofs(); }

private:
    ContinuousCutsFunctional& m_func;
    ContinuousCutsResidual m_res;
};

} // namespace rmo::cc

#endif //RMO_CC_ORACLE_H
