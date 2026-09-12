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

    BernoulliPointTransfer(const BernoulliGridTransfer& transfer, Restriction kind)
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
    const BernoulliGridTransfer& m_transfer;
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

} // namespace rmo::cc

#endif //RMO_CC_TRANSPORT_H
