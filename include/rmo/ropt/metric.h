#ifndef RMO_ROPT_METRIC_H
#define RMO_ROPT_METRIC_H

#include <rmo/lac.h>
#include <rmo/option_types.h>

#include <cmath>

/**
 * @file
 * @brief Riemannian metrics \f$ g(u, v) = u^\top G v \f$ on the ambient space: MetricBase, OperatorMetric (an
 * operator \f$ G \f$, e.g. \f$ M \f$ or \f$ A(x) \f$) and EuclideanMetric (the F-metric).
 */
namespace rmo
{

//! Riemannian metric on the ambient space, g(u, v) = u^T G v with G symmetric positive definite.
//! A state-dependent G (e.g. the energy-adaptive A_x) refers to the point of the last oracle update.
class MetricBase
{
public:
    virtual ~MetricBase() = default;

    [[nodiscard]] virtual double inner(const Vector<double>& u, const Vector<double>& v) const = 0;

    [[nodiscard]] double norm(const Vector<double>& v) const { return std::sqrt(inner(v, v)); }

    //! dst = G src
    virtual void apply(const Vector<double>& src, Vector<double>& dst) const = 0;

    [[nodiscard]] virtual MetricKind kind() const = 0;
};


//! Metric of an operator G, which is referenced (not owned) and may be updated after construction.
template <typename Operator>
class OperatorMetric : public MetricBase
{
public:
    OperatorMetric(const Operator& G, MetricKind kind)
        : m_G(G)
        , m_kind(kind)
    {
        AssertDimension(G.m(), G.n());
    }

    //! Rejects temporaries: m_G would dangle after the constructing full-expression, e.g. for the
    //! LinearCombination returned by value from get_operator_M(). Pass an operator that outlives the
    //! metric, such as GrossPitaevskiiFunctional::get_M().
    OperatorMetric(const Operator&& G, MetricKind kind) = delete;

    [[nodiscard]] double inner(const Vector<double>& u, const Vector<double>& v) const override
    {
        AssertDimension(u.size(), v.size());
        Vector<double> Gv(v.size());
        m_G.vmult(Gv, v);
        return u * Gv;
    }

    void apply(const Vector<double>& src, Vector<double>& dst) const override
    {
        m_G.vmult(dst, src);
    }

    [[nodiscard]] MetricKind kind() const override { return m_kind; }

private:
    const Operator& m_G;
    MetricKind m_kind;
};


//! Euclidean metric g(u, v) = u^T v (the F-metric).
class EuclideanMetric : public MetricBase
{
public:
    [[nodiscard]] double inner(const Vector<double>& u, const Vector<double>& v) const override
    {
        AssertDimension(u.size(), v.size());
        return u * v;
    }

    void apply(const Vector<double>& src, Vector<double>& dst) const override
    {
        dst = src;
    }

    [[nodiscard]] MetricKind kind() const override { return MetricKind::FROBENIUS; }
};

} // namespace rmo

#endif //RMO_ROPT_METRIC_H
