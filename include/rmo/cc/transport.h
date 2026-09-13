//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_TRANSPORT_H
#define RMO_CC_TRANSPORT_H

#include <rmo/cc/interpolate.h>
#include <rmo/cc/manifold.h>
#include <rmo/cc/metric.h>

#include <rmo/ropt/transport.h>

namespace rmo::cc
{

/**
 * @brief Point restriction/prolongation @f$ r,p @f$ (\S 6.3.2). @f$ p @f$ always uses
 * bilinear interpolation in logit coordinates; @f$ r @f$ uses either injection (which
 * commutes with the logit change of coordinates, so no round trip through it is needed)
 * or full-weighting @f$ F_h^H = \tfrac14 (B_H^h)^\top @f$, i.e. @ref BernoulliGridTransfer::Tfine
 * scaled by @f$ \tfrac14 @f$.
 */
class BernoulliPointTransfer : public ManifoldTransferBase
{
public:
    enum class Restriction { INJECTION, FULL_WEIGHTING };

    // transfer may be a single BernoulliGridTransfer hop or a ComposedGridTransfer (several
    // hops treated as one coarse-model transition); both implement LinearTransferBase, and
    // restriction/prolongation below only ever call that interface.
    BernoulliPointTransfer(const LinearTransferBase& transfer, Restriction kind)
        : ManifoldTransferBase(transfer), m_transfer(transfer), m_kind(kind)
    {}

    void restriction(const Vector<double>& phi, Vector<double>& psi) const override
    {
        if (m_kind == Restriction::INJECTION) {
            m_transfer.to_coarse_mesh(phi, psi);
            return;
        }

        Vector<double> logit_phi;
        bernoulli::logit(phi, logit_phi);

        Vector<double> logit_psi(m_transfer.n_coarse());
        m_transfer.Tfine(logit_phi, logit_psi);
        logit_psi *= 0.25;

        bernoulli::sigmoid(logit_psi, psi);
    }

    void prolongation(const Vector<double>& psi, Vector<double>& phi) const override
    {
        Vector<double> logit_psi;
        bernoulli::logit(psi, logit_psi);

        Vector<double> logit_phi(m_transfer.n_fine());
        m_transfer.to_fine_mesh(logit_psi, logit_phi);

        bernoulli::sigmoid(logit_phi, phi);
    }

private:
    const LinearTransferBase& m_transfer;
    Restriction m_kind;
};


/**
 * @brief Vector transport shared by Table 10's Options 1 and 4 (they differ only in the
 * point restriction used to build @f$ \psi=r(\phi) @f$, see @ref BernoulliPointTransfer,
 * not in @f$ P,R @f$):
 * @f[
 *   P^\phi_\psi(v) = G(\phi)^{-1}\big(B_H^h\,G(\psi)v\big), \qquad
 *   R^\psi_\phi(v) = (B_H^h)^\top v = 4\,F_h^H v.
 * @f]
 */
class GeometricBilinearTransport : public VectorTransportBase
{
public:
    static constexpr auto id = "gB/4F";

    explicit GeometricBilinearTransport(const BernoulliGridTransfer& transfer)
        : m_transfer(transfer)
    {}

    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        Vector<double> Gv(v_coarse.size());
        for (unsigned int i = 0; i < v_coarse.size(); i++) {
            Gv[i] = v_coarse[i] / metric::fisher_rao::weight(y_coarse[i]);
        }

        Vector<double> Bv(x_fine.size());
        m_transfer.to_fine_mesh(Gv, Bv);

        dst.reinit(x_fine.size());
        for (unsigned int i = 0; i < x_fine.size(); i++) {
            dst[i] = metric::fisher_rao::weight(x_fine[i]) * Bv[i];
        }
    }

    void vector_restriction(const Vector<double>& /* y_coarse */, const Vector<double>& /* x_fine */,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        m_transfer.Tfine(v_fine, dst);
    }

private:
    const BernoulliGridTransfer& m_transfer;
};


/**
 * @brief @ref GeometricBilinearTransport composed over several factor-2 hops (Option 1/4),
 * matching the reference's @c operators.make_composed_ops applied to @c R_inj_bil/P_inj_bil.
 *
 * Unlike @ref ComposedGridTransfer (which @ref GeometricBilinearTransport itself could be
 * reused with, since @c to_fine_mesh/Tfine are metric-free), the metric weighting here is
 * @em not simply "once at the two endpoints": the reference's @c P_n applies @c apply_G /
 * @c apply_G_inv at @em every intermediate grid in the chain (each hop's own fine/coarse
 * pair), not just the true finest/coarsest points -- @c R_inj_bil ignores its @c phi/psi
 * arguments entirely, so no such per-hop weighting is needed on the restriction side, but
 * @c P_inj_bil's @f$ G(\phi)^{-1}(\cdot) @f$ is evaluated at @em each hop's local @f$\phi@f$.
 * This class therefore recomputes the intermediate points by injection from @c x_fine (the
 * same @c _intermediate_grids(phi) the reference recomputes fresh on every call, rather than
 * being handed them) and applies @ref GeometricBilinearTransport's per-hop formula at each
 * one -- passing a @ref ComposedGridTransfer built from the same hops to a plain
 * @ref GeometricBilinearTransport would silently skip this intermediate weighting.
 */
class ComposedVectorTransport : public VectorTransportBase
{
public:
    // hops[0]: finest <-> first intermediate, ..., hops.back(): last intermediate <-> coarsest.
    explicit ComposedVectorTransport(std::vector<const BernoulliGridTransfer*> hops)
        : m_hops(std::move(hops))
    {
        Assert(!m_hops.empty(), dealii::ExcMessage("at least one hop required"));
    }

    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& /* y_coarse */,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        const auto grids = intermediate_grids(x_fine);   // grids[0]=x_fine ... grids[N]=coarsest

        Vector<double> v_curr = v_coarse;
        for (std::size_t k = m_hops.size(); k-- > 0;) {
            const Vector<double>& phi_here = grids[k];
            const Vector<double>& psi_here = grids[k + 1];

            Vector<double> Gv(v_curr.size());
            for (unsigned int i = 0; i < v_curr.size(); i++) {
                Gv[i] = v_curr[i] / metric::fisher_rao::weight(psi_here[i]);
            }
            Vector<double> Bv(phi_here.size());
            m_hops[k]->to_fine_mesh(Gv, Bv);

            Vector<double> v_next(phi_here.size());
            for (unsigned int i = 0; i < phi_here.size(); i++) {
                v_next[i] = metric::fisher_rao::weight(phi_here[i]) * Bv[i];
            }
            v_curr = std::move(v_next);
        }
        dst = std::move(v_curr);
    }

    // R_inj_bil ignores phi/psi entirely, so composing it is just chained Tfine -- no
    // intermediate points needed, unlike vector_prolongation above.
    void vector_restriction(const Vector<double>& /* y_coarse */, const Vector<double>& /* x_fine */,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        Vector<double> curr = v_fine;
        for (const auto* hop : m_hops) {
            Vector<double> next;
            hop->Tfine(curr, next);
            curr = std::move(next);
        }
        dst = std::move(curr);
    }

private:
    [[nodiscard]] std::vector<Vector<double>> intermediate_grids(const Vector<double>& x_fine) const
    {
        std::vector<Vector<double>> grids{x_fine};
        for (const auto* hop : m_hops) {
            Vector<double> next;
            hop->to_coarse_mesh(grids.back(), next);
            grids.push_back(std::move(next));
        }
        return grids;
    }

    std::vector<const BernoulliGridTransfer*> m_hops;
};

} // namespace rmo::cc

#endif //RMO_CC_TRANSPORT_H
