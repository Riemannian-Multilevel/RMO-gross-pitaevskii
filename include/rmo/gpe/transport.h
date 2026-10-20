#ifndef RMO_GPE_TRANSPORT_H
#define RMO_GPE_TRANSPORT_H

#include <rmo/gpe/manifold.h>
#include <rmo/gpe/metric.h>

#include <rmo/ropt/transport.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

/**
 * @file
 * @brief Transfers between the fine and coarse unit-mass spheres \f$ \mathcal{S}_h \f$, \f$ \mathcal{S}_H \f$: the point
 * maps of ManifoldTransfer and the vector transports implementing VectorTransportBase.
 *
 * Notation: \f$ x \in \mathcal{S}_h \f$ is the fine and \f$ y \in \mathcal{S}_H \f$ the coarse point (arguments
 * `x_fine`, `y_coarse`); \f$ I_H^h \f$ and \f$ I_h^H \f$ are the prolongation and restriction of the
 * LinearTransferBase, \f$ \Pi^M \f$ and \f$ \Pi^F \f$ the \f$ M \f$- and F-orthogonal projections onto the tangent
 * space at the subscripted point (namespace metric), and \f$ \tilde y = I_H^h y \f$, \f$ n = \|\tilde y\|_{M_h} \f$.
 *
 * A vector transport maps \f$ \mathcal{P}: T_y \mathcal{S}_H \to T_x \mathcal{S}_h \f$ (vector_prolongation()) and
 * \f$ \mathcal{R}: T_x \mathcal{S}_h \to T_y \mathcal{S}_H \f$ (vector_restriction()):
 * - MassProjectionTransport: \f$ \mathcal{P} v = \Pi^M_x I_H^h v \f$, \f$ \mathcal{R} v = \Pi^M_y I_h^H v \f$;
 * - FrobeniusProjectionTransport (= FrobeniusAdjointRestrictionTransport): \f$ \mathcal{P} v = \Pi^F_x I_H^h v \f$,
 *   \f$ \mathcal{R} v = \Pi^F_y (I_H^h)^\top v \f$;
 * - DifferentialTransport: \f$ \mathcal{P} v = \Pi^M_x \mathrm{D}p(y)[v] \f$, \f$ \mathcal{R} v = \Pi^M_y \mathrm{D}r(x)[v] \f$
 *   (FrobeniusDifferentialTransport: with \f$ \Pi^F \f$);
 * - AdjointRestrictionTransport: \f$ \mathcal{P} v = \Pi^M_x I_H^h v \f$,
 *   \f$ \mathcal{R} v = \Pi^M_y M_H^{-1} (I_H^h)^\top M_h v \f$;
 * - AdjointDifferentialTransport: \f$ \mathcal{P} v = \Pi^M_x \mathrm{D}p(y)[v] \f$,
 *   \f$ \mathcal{R} v = \frac1n \Pi^M_y M_H^{-1} (I_H^h)^\top M_h \big( v - \frac{\tilde y^\top M_h v}{n^2} \tilde y \big) \f$;
 * - FrobeniusAdjointDifferentialTransport: \f$ \mathcal{P} v = \Pi^F_x \mathrm{D}p(y)[v] \f$,
 *   \f$ \mathcal{R} v = \frac1n \Pi^F_y (I_H^h)^\top \big( v - \frac{\tilde y^\top v}{n^2} M_h \tilde y \big) \f$.
 *
 * The restriction of AdjointDifferentialTransport is ManifoldTransferBase::diff_prolongation_adjoint() of the point
 * transfer (ManifoldTransfer with the inverse coarse mass matrix, LogManifoldTransfer, LogThresholdManifoldTransfer).
 *
 * For the differential transports, \f$ \mathcal{R} \f$ is not the adjoint of \f$ \mathcal{P} \f$: both are
 * differentials of independent point maps.
 */
namespace rmo::gpe
{

namespace transfer_detail
{
//! a <- exp(a - max_i a_i): exponential of logarithms, shifted by their maximum against overflow
inline void exp_shifted(Vector<double>& a)
{
    const double a_max = *std::max_element(a.begin(), a.end());
    for (auto& ai : a) {
        ai = std::exp(ai - a_max);
    }
}
} // namespace transfer_detail

/**
 * @brief Point maps \f$ r \f$, \f$ p \f$ between the spheres and their differentials (metric-independent). With the
 * inverse coarse mass matrix @p M_inv_coarse (type InverseMatrixType), also the \f$ M \f$-adjoint of the differential of
 * \f$ p \f$, see diff_prolongation_adjoint().
 */
template <typename MatrixType, typename InverseMatrixType = void>
class ManifoldTransfer : public ManifoldTransferBase
{
public:
    ManifoldTransfer(const LinearTransferBase& transfer,
                     const MatrixType& M_coarse,
                     const MatrixType& M_fine)
        : ManifoldTransferBase(transfer), M_coarse(M_coarse), M_fine(M_fine)
    {}

    template <typename Inverse = InverseMatrixType>
        requires (!std::is_void_v<Inverse>)
    ManifoldTransfer(const LinearTransferBase& transfer,
                     const MatrixType& M_coarse,
                     const MatrixType& M_fine,
                     const Inverse& M_inv_coarse)
        : ManifoldTransferBase(transfer), M_coarse(M_coarse), M_fine(M_fine), M_inv_coarse(&M_inv_coarse)
    {}

    /** @brief \f$ y = r(x) = I_h^H x / \|I_h^H x\|_{M_H} \f$ */
    void restriction(const Vector<double>& x_fine, Vector<double>& y_coarse) const override
    {
        transfer.to_coarse_mesh(x_fine, y_coarse);

        ellipsoid::retract_by_norm(M_coarse, y_coarse);
    }

    /** @brief \f$ x = p(y) = I_H^h y / \|I_H^h y\|_{M_h} \f$ */
    void prolongation(const Vector<double>& y_coarse, Vector<double>& x_fine) const override
    {
        transfer.to_fine_mesh(y_coarse, x_fine);

        ellipsoid::retract_by_norm(M_fine, x_fine);
    }

    /**
     * @brief Differential of \f$ r \f$ at @p x_fine in the direction \f$ v \in T_x \mathcal{S}_h \f$, with
     * \f$ \hat x = I_h^H x \f$, \f$ v_H = I_h^H v \f$ and \f$ n = \|\hat x\|_{M_H} \f$:
     * \f[ \mathrm{D}r(x)[v] = \frac{1}{n} \Big( v_H - \frac{\hat x^\top M_H v_H}{n^2} \, \hat x \Big) \f]
     */
    void diff_restriction(const Vector<double>& x_fine, const Vector<double>& v, Vector<double>& dst) const override
    {
        // Linear restriction of the base point: I_h^H(y)
        Vector<double> x_lin(transfer.n_coarse());
        transfer.to_coarse_mesh(x_fine, x_lin);

        // Norm of the linear point: ||x_lin||_{M_H}^2
        Vector<double> M_x_lin(transfer.n_coarse());
        M_coarse.vmult(M_x_lin, x_lin);
        const double n_H_sq = x_lin * M_x_lin;
        const double n_H = std::sqrt(n_H_sq);

        // Linear restriction of the tangent vector: I_h^H(v)
        Vector<double> v_H(transfer.n_coarse());
        transfer.to_coarse_mesh(v, v_H);

        // Inner product: (I_h^H y)^T M_H (I_h^H v)
        Vector<double> M_v_H(transfer.n_coarse());
        M_coarse.vmult(M_v_H, v_H);
        const double inner_prod = x_lin * M_v_H;

        // Final assembly
        dst = v_H;
        dst.add(-inner_prod / n_H_sq, x_lin);
        dst /= n_H;
    }

    /**
     * @brief Differential of \f$ p \f$ at @p y_coarse in the direction \f$ v \in T_y \mathcal{S}_H \f$, with
     * \f$ \hat y = I_H^h y \f$, \f$ v_h = I_H^h v \f$ and \f$ n = \|\hat y\|_{M_h} \f$:
     * \f[ \mathrm{D}p(y)[v] = \frac{1}{n} \Big( v_h - \frac{\hat y^\top M_h v_h}{n^2} \, \hat y \Big) \f]
     */
    void diff_prolongation(const Vector<double>& y_coarse, const Vector<double>& v, Vector<double>& dst) const override
    {
        // 1. Linear prolongation of the base point: I_H^h(x)
        Vector<double> y_lin(transfer.n_fine());
        transfer.to_fine_mesh(y_coarse, y_lin);

        // 2. Norm of the linear point: ||y_lin||_{M_h}^2
        Vector<double> M_y_lin(transfer.n_fine());
        M_fine.vmult(M_y_lin, y_lin);
        const double n_h_sq = y_lin * M_y_lin;
        const double n_h = std::sqrt(n_h_sq);

        // 3. Linear prolongation of the tangent vector: I_H^h(v)
        Vector<double> v_h(transfer.n_fine());
        transfer.to_fine_mesh(v, v_h);

        // 4. Inner product: (I_H^h x)^T M_h (I_H^h v)
        Vector<double> M_v_h(transfer.n_fine());
        M_fine.vmult(M_v_h, v_h);
        const double inner_prod = y_lin * M_v_h;

        // 5. Final assembly
        dst = v_h;
        dst.add(-inner_prod / n_h_sq, y_lin);
        dst /= n_h;
    }

    /**
     * @brief \f$ \mathrm{D}p(y)^* u = \frac1n M_H^{-1} (I_H^h)^\top M_h \Pi_{p(y)} u
     * = \frac1n M_H^{-1} (I_H^h)^\top \Big( M_h u - \frac{\hat y^\top M_h u}{n^2} M_h \hat y \Big) \f$, with
     * \f$ \hat y = I_H^h y \f$ and \f$ n = \|\hat y\|_{M_h} \f$, in the operation order of AdjointDifferentialTransport.
     */
    void diff_prolongation_adjoint(const Vector<double>& y_coarse, const Vector<double>& v_fine,
                                   Vector<double>& dst) const override
    {
        if constexpr (std::is_void_v<InverseMatrixType>) {
            throw dealii::ExcMessage("ManifoldTransfer: the adjoint differential needs the inverse coarse mass matrix");
        }
        else {
            // Let \psi = y_coarse
            // 1. Compute \tilde{\psi} = I_H^h \psi (Prolonged base point)
            Vector<double> psi_tilde(transfer.n_fine());
            transfer.to_fine_mesh(y_coarse, psi_tilde);

            // 2. Compute M_h * \tilde{\psi}
            Vector<double> M_psi_tilde(transfer.n_fine());
            M_fine.vmult(M_psi_tilde, psi_tilde);

            // 3. Compute norm squared: ||I_H^h \psi||_{M_h}^2  and inner product (I_H^h \psi)^T M_h v_fine
            const double norm_sq = psi_tilde * M_psi_tilde;
            const double norm    = std::sqrt(norm_sq);

            Vector<double> M_v(transfer.n_fine());
            M_fine.vmult(M_v, v_fine);
            const double inner_prod = psi_tilde * M_v;

            // 4. Evaluate M_h * [ (I - \Pi) v_fine ]
            Vector<double> M_v_proj = M_v;
            M_v_proj.add(-inner_prod / norm_sq, M_psi_tilde);

            // 5. Apply transpose of prolongation: (I_H^h)^T [ M_h (I - \Pi) v_fine ]
            Vector<double> IT_M_v_proj(transfer.n_coarse());
            transfer.Tfine(M_v_proj, IT_M_v_proj);

            // 6. Apply inverse coarse mass matrix: M_H^{-1} * (I_H^h)^T ...
            //    XXX: Reset to a relative tolerance, so that restriction solve does not
            //    inherit the absolute tolerance left by the coarse oracle's gradient()
            //    on the shared M_inv_coarse
            M_inv_coarse->set_tol(0.0);
            dst.reinit(transfer.n_coarse());
            M_inv_coarse->vmult(dst, IT_M_v_proj);

            // 7. Scale by (1 / ||I_H^h \psi||_{M_h})
            dst /= norm;
        }
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    // Inverse coarse mass matrix for the adjoint (none for InverseMatrixType = void)
    using InverseStorage = std::conditional_t<std::is_void_v<InverseMatrixType>, char, InverseMatrixType>;
    const InverseStorage* M_inv_coarse = nullptr;
};

/** @brief Transport by linear transfer and \f$ M \f$-orthogonal projection, see the file documentation. */
template <typename MatrixType>
class MassProjectionTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "M";

    MassProjectionTransport(const LinearTransferBase& I,
                            const MatrixType& M_coarse,
                            const MatrixType& M_fine)
        : M_coarse(M_coarse), M_fine(M_fine), transfer(I)
    {}

    void vector_prolongation(const Vector<double>& x_fine,
                             [[maybe_unused]] const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse,
                             Vector<double>& dst) const override
    {
        // Tangent vector interpolated to fine ambient space
        Vector<double> Iv(transfer.n_fine());
        transfer.to_fine_mesh(v_coarse, Iv);

        // Orthogonal projection I(v \in T_x S_H) -> T_p(x) S_h in M-metric
        metric::mass::project_onto_tangent_space(x_fine, M_fine, Iv, dst);
    }

    void vector_restriction(const Vector<double>& y_coarse,
                            [[maybe_unused]] const Vector<double>& x_fine,
                            const Vector<double>& v_fine,
                            Vector<double>& dst) const override
    {
        // Tangent vector restricted to coarse ambient space
        Vector<double> Iv(transfer.n_coarse());
        transfer.to_coarse_mesh(v_fine, Iv);

        // Orthogonal projection I(v \in T_y S_h) -> T_r(y) S_H
        metric::mass::project_onto_tangent_space(y_coarse, M_coarse, Iv, dst);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const LinearTransferBase& transfer;
};


/** @brief Transport by linear transfer and F-orthogonal projection, see the file documentation. */
template <typename MatrixType>
class FrobeniusProjectionTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "F";

    explicit FrobeniusProjectionTransport(const LinearTransferBase& I,
                                          const MatrixType& M_coarse,
                                          const MatrixType& M_fine)
        : M_coarse(M_coarse), M_fine(M_fine), transfer(I)
    {}

    void vector_prolongation(const Vector<double>& x_fine,
                             [[maybe_unused]] const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse,
                             Vector<double>& dst) const override
    {
        // Tangent vector interpolated to fine ambient space
        Vector<double> Iv(transfer.n_fine());
        transfer.to_fine_mesh(v_coarse, Iv);

        // F-orthogonal projection on the fine grid
        metric::frobenius::project_onto_tangent_space(x_fine, M_fine, Iv, dst);
    }

    void vector_restriction(const Vector<double>& y_coarse,
                            [[maybe_unused]] const Vector<double>& x_fine,
                            const Vector<double>& v_fine,
                            Vector<double>& dst) const override
    {
        // Tangent vector restricted to coarse ambient space
        Vector<double> Iv(transfer.n_coarse());
        //transfer.to_coarse_mesh(v_fine, Iv);
        transfer.Tfine(v_fine, Iv);

        // F-Orthogonal projection I(v \in T_y S_h) -> T_r(y) S_H
        metric::frobenius::project_onto_tangent_space(y_coarse, M_coarse, Iv, dst);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const LinearTransferBase& transfer;
};

/**
 * @brief F-metric counterpart of AdjointRestrictionTransport. In the F-metric, the adjoint of the prolongation is
 * its transpose \f$ (I_H^h)^\top \f$, which FrobeniusProjectionTransport already uses for the restriction.
 */
template <typename MatrixType>
using FrobeniusAdjointRestrictionTransport = FrobeniusProjectionTransport<MatrixType>;


/** @brief Transport by the differentials of the point maps and \f$ M \f$-orthogonal projection. */
template <typename MatrixType>
class DifferentialTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "D";

    explicit DifferentialTransport(const ManifoldTransferBase& pt,
                                   const MatrixType& M_coarse,
                                   const MatrixType& M_fine)
        : M_coarse(M_coarse), M_fine(M_fine), point_transfer(pt)
    {}

    void vector_prolongation(const Vector<double>& y_coarse, const Vector<double>& v_coarse,
                             Vector<double>& dst) const
    {
        point_transfer.diff_prolongation(y_coarse, v_coarse, dst);
    }

    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Differential D_p(x): T_x S_H -> T_p(x) S_h
        Vector<double> D_px(point_transfer.n_fine());
        vector_prolongation(y_coarse, v_coarse, D_px);

        // Vector transport T_p(x) S_h -> T_y S_h
        // TODO: which metric to choose for orthogonal projection?
        // -> Use inheritance from DifferentialTransport base class (FrobeniusDifferentialTransport, MassDifferentialTransport)
        metric::mass::project_onto_tangent_space(x_fine, M_fine, D_px, dst);
    }

    void vector_restriction(const Vector<double>& x_fine, const Vector<double>& v_fine,
                            Vector<double>& dst) const
    {
        point_transfer.diff_restriction(x_fine, v_fine, dst);
    }

    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& x_fine,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        // Differential D_r(y): T_y S_h -> T_r(y) S_H
        Vector<double> D_ry(point_transfer.n_coarse());
        vector_restriction(x_fine, v_fine, D_ry);

        // Vector transport T_r(y) S_H -> T_x S_H
        // TODO: which metric to choose for orthogonal projection?
        metric::mass::project_onto_tangent_space(y_coarse, M_coarse, D_ry, dst);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const ManifoldTransferBase& point_transfer;
};


/** @brief Transport by the differentials of the point maps and F-orthogonal projection. */
template <typename MatrixType>
class FrobeniusDifferentialTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "FD";

    explicit FrobeniusDifferentialTransport(const ManifoldTransferBase& pt,
                                            const MatrixType& M_coarse,
                                            const MatrixType& M_fine)
        : M_coarse(M_coarse), M_fine(M_fine), point_transfer(pt)
    {}

    void vector_prolongation(const Vector<double>& y_coarse, const Vector<double>& v_coarse,
                             Vector<double>& dst) const
    {
        point_transfer.diff_prolongation(y_coarse, v_coarse, dst);
    }

    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Differential D_p(x): T_x S_H -> T_p(x) S_h
        Vector<double> D_px(point_transfer.n_fine());
        vector_prolongation(y_coarse, v_coarse, D_px);

        // Vector transport T_p(x) S_h -> T_y S_h using F-orthogonal projection
        metric::frobenius::project_onto_tangent_space(x_fine, M_fine, D_px, dst);
    }

    void vector_restriction(const Vector<double>& x_fine, const Vector<double>& v_fine,
                            Vector<double>& dst) const
    {
        point_transfer.diff_restriction(x_fine, v_fine, dst);
    }

    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& x_fine,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        // Differential D_r(y): T_y S_h -> T_r(y) S_H
        Vector<double> D_ry(point_transfer.n_coarse());
        vector_restriction(x_fine, v_fine, D_ry);

        // Vector transport T_r(y) S_H -> T_x S_H using F-orthogonal projection
        metric::frobenius::project_onto_tangent_space(y_coarse, M_coarse, D_ry, dst);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const ManifoldTransferBase& point_transfer;
};


/** @brief Transport whose restriction is the \f$ M \f$-adjoint of the linear prolongation. */
template <typename MatrixType, typename InverseMatrixType>
class AdjointRestrictionTransport : public VectorTransportBase
{
public:
    static constexpr auto id = "V2Restr";

    // TODO: set tolerance inside vector_restriction() instead of global tolerance
    AdjointRestrictionTransport(const LinearTransferBase& I,
                                const MatrixType& M_coarse,
                                const MatrixType& M_fine,
                                const InverseMatrixType& M_inv_coarse)
        : transfer(I),
          M_coarse(M_coarse),
          M_fine(M_fine),
          M_inv_coarse(M_inv_coarse)
    {}

    void vector_prolongation(const Vector<double>& x_fine,
                             [[maybe_unused]] const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse,
                             Vector<double>& dst) const override
    {
        // 1. Linear prolongation to ambient space
        Vector<double> Iv(transfer.n_fine());
        transfer.to_fine_mesh(v_coarse, Iv);

        // 2. Orthogonal projection onto the target tangent space
        // TODO: which metric to choose for orthogonal projection?
        // -> Use inheritance from DifferentialTransport base class (FrobeniusDifferentialTransport, MassDifferentialTransport)
        metric::mass::project_onto_tangent_space(x_fine, M_fine, Iv, dst);
    }

    void vector_restriction(const Vector<double>& y_coarse,
                            [[maybe_unused]] const Vector<double>& x_fine,
                            const Vector<double>& v_fine,
                            Vector<double>& dst) const override
    {
        // 1. Compute M_h * v_fine
        Vector<double> M_v(transfer.n_fine());
        M_fine.vmult(M_v, v_fine);

        // 2. Apply transpose of prolongation: (I_H^h)^T (M_h * v_fine)
        Vector<double> IT_M_v(transfer.n_coarse());
        transfer.Tfine(M_v, IT_M_v);

        // 3. Apply inverse coarse mass matrix: M_H^{-1} * IT_M_v
        //    XXX: Reset to a relative tolerance, so that restriction solve does not
        //    inherit the absolute tolerance left by the coarse oracle's gradient()
        //    on the shared M_inv_coarse
        M_inv_coarse.set_tol(0.0);
        Vector<double> w(transfer.n_coarse());
        M_inv_coarse.vmult(w, IT_M_v);

        // 4. Project onto target tangent space T_\psi S_H
        // TODO: which metric to choose for orthogonal projection?
        // -> Use inheritance from DifferentialTransport base class (FrobeniusDifferentialTransport, MassDifferentialTransport)
        metric::mass::project_onto_tangent_space(y_coarse, M_coarse, w, dst);
    }

protected:
    const LinearTransferBase& transfer;
    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    const InverseMatrixType& M_inv_coarse;
};


/** @brief Transport whose restriction is the \f$ M \f$-adjoint of the differential of \f$ p \f$. */
template <typename MatrixType, typename InverseMatrixType>
class
AdjointDifferentialTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "V5restr";

    // The adjoint is ManifoldTransferBase::diff_prolongation_adjoint() of the point transfer, which holds
    // M_H^{-1}; the linear transfer and the inverse are not used, the arguments are kept for the callers
    explicit AdjointDifferentialTransport(const LinearTransferBase& /*transfer*/,
                                          const ManifoldTransferBase& pt,
                                          const MatrixType& M_coarse,
                                          const MatrixType& M_fine,
                                          const InverseMatrixType& /*M_inv_coarse*/)
        : point_transfer(pt),
          M_coarse(M_coarse), M_fine(M_fine)
    {}

    void vector_prolongation(const Vector<double>& y_coarse, const Vector<double>& v_coarse,
                             Vector<double>& dst) const
    {
        point_transfer.diff_prolongation(y_coarse, v_coarse, dst);
    }

    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Differential D_p(x): T_x S_H -> T_p(x) S_h
        Vector<double> D_px(point_transfer.n_fine());
        vector_prolongation(y_coarse, v_coarse, D_px);

        // Vector transport T_p(x) S_h -> T_y S_h using mass metric projection
        metric::mass::project_onto_tangent_space(x_fine, M_fine, D_px, dst);
    }

    /**
     * @brief \f$ \mathcal{R} v = \Pi_y \mathrm{D}p(y)^* v \f$, the \f$ M \f$-adjoint of vector_prolongation() on
     * \f$ T_x \f$, from ManifoldTransferBase::diff_prolongation_adjoint() of the point transfer (ManifoldTransfer,
     * LogManifoldTransfer).
     */
    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& /*x_fine*/,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        Vector<double> w(point_transfer.n_coarse());
        point_transfer.diff_prolongation_adjoint(y_coarse, v_fine, w);

        // Project onto the target tangent space T_\psi S_H
        metric::mass::project_onto_tangent_space(y_coarse, M_coarse, w, dst);
    }

private:
    const ManifoldTransferBase& point_transfer;

    const MatrixType& M_coarse;
    const MatrixType& M_fine;
};


/** @brief F-metric counterpart of AdjointDifferentialTransport. */
template <typename MatrixType>
class FrobeniusAdjointDifferentialTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "FV5restr";

    explicit FrobeniusAdjointDifferentialTransport(const LinearTransferBase& transfer,
                                                   const ManifoldTransferBase& pt,
                                                   const MatrixType& M_coarse,
                                                   const MatrixType& M_fine)
        : transfer(transfer), point_transfer(pt),
          M_coarse(M_coarse), M_fine(M_fine)
    {}

    void vector_prolongation(const Vector<double>& y_coarse, const Vector<double>& v_coarse,
                             Vector<double>& dst) const
    {
        point_transfer.diff_prolongation(y_coarse, v_coarse, dst);
    }

    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Differential D_p(x): T_x S_H -> T_p(x) S_h
        Vector<double> D_px(point_transfer.n_fine());
        vector_prolongation(y_coarse, v_coarse, D_px);

        // Vector transport T_p(x) S_h -> T_y S_h using F-orthogonal projection
        metric::frobenius::project_onto_tangent_space(x_fine, M_fine, D_px, dst);
    }

    void vector_restriction(const Vector<double>& y_coarse, [[maybe_unused]] const Vector<double>& x_fine,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        // Let \psi = y_coarse
        // 1. Compute \tilde{\psi} = I_H^h \psi (Prolonged base point)
        Vector<double> psi_tilde(transfer.n_fine());
        transfer.to_fine_mesh(y_coarse, psi_tilde);

        // 2. Compute M_h * \tilde{\psi}, and norm_sq = ||I_H^h \psi||_{M_h}^2
        Vector<double> M_psi_tilde(transfer.n_fine());
        M_fine.vmult(M_psi_tilde, psi_tilde);
        const double norm_sq = psi_tilde * M_psi_tilde;
        const double norm    = std::sqrt(norm_sq);

        // 3. Q(v_fine) = v_fine - (\tilde{\psi} . v_fine / norm_sq) * M_h \tilde{\psi}
        //    Note: the inner product \tilde{\psi} . v_fine is unweighted (F-metric),
        //    unlike the mass-metric version which uses \tilde{\psi}^T M_h v_fine.
        const double inner_prod = psi_tilde * v_fine;
        Vector<double> Qv = v_fine;
        Qv.add(-inner_prod / norm_sq, M_psi_tilde);

        // 4. Apply transpose of prolongation: (I_H^h)^T Q(v_fine)
        Vector<double> IT_Qv(transfer.n_coarse());
        transfer.Tfine(Qv, IT_Qv);

        // 5. Scale by (1 / ||I_H^h \psi||_{M_h})
        IT_Qv /= norm;

        // 6. Project onto the target tangent space T_\psi S_H
        metric::frobenius::project_onto_tangent_space(y_coarse, M_coarse, IT_Qv, dst);
    }

private:
    const LinearTransferBase& transfer;
    const ManifoldTransferBase& point_transfer;

    const MatrixType& M_coarse;
    const MatrixType& M_fine;
};


/**
 * @brief Point maps of the positive sphere by linear interpolation of logarithms (tex/drafts/positive_multilevel.tex,
 * (E1)-(E2)): \f$ r(x) = N_H(e^{I_h^H \log x}) \f$, \f$ p(y) = N_h(e^{I_H^h \log y}) \f$ with the normalization
 * \f$ N(a) = a / \|a\|_M \f$. Positive for any linear transfer (also with negative weights); exact for exponential
 * tails. For the injection \f$ I_h^H \f$ of LinearTransferMG, \f$ r \f$ is the restriction of ManifoldTransfer.
 * Evaluated in logarithms shifted by their maximum; components below the smallest normal number are raised to it.
 */
template <typename MatrixType, typename InverseMatrixType>
class LogManifoldTransfer : public ManifoldTransferBase
{
public:
    LogManifoldTransfer(const LinearTransferBase& transfer, const MatrixType& M_coarse, const MatrixType& M_fine,
                        const InverseMatrixType& M_inv_coarse)
        : ManifoldTransferBase(transfer), M_coarse(M_coarse), M_fine(M_fine), M_inv_coarse(M_inv_coarse)
    {}

    /** @brief \f$ y = r(x) = N_H(e^{I_h^H \log x}) \f$ */
    void restriction(const Vector<double>& x_fine, Vector<double>& y_coarse) const override
    {
        Vector<double> s(x_fine.size());
        log_of(x_fine, s);
        transfer.to_coarse_mesh(s, y_coarse);
        exp_normalized(M_coarse, y_coarse);
    }

    /** @brief \f$ x = p(y) = N_h(e^{I_H^h \log y}) \f$ */
    void prolongation(const Vector<double>& y_coarse, Vector<double>& x_fine) const override
    {
        Vector<double> s(y_coarse.size());
        log_of(y_coarse, s);
        transfer.to_fine_mesh(s, x_fine);
        exp_normalized(M_fine, x_fine);
    }

    /** @brief \f$ \mathrm{D}r(x)[v] = \Pi_{\hat y} (\hat y \circ I_h^H (v / x)) \f$, \f$ \hat y = r(x) \f$ */
    void diff_restriction(const Vector<double>& x_fine, const Vector<double>& v, Vector<double>& dst) const override
    {
        Vector<double> y_hat(transfer.n_coarse());
        restriction(x_fine, y_hat);
        Vector<double> rel(v.size());
        for (unsigned i = 0; i < v.size(); ++i) {
            rel[i] = v[i] / x_fine[i];
        }
        Vector<double> t(transfer.n_coarse());
        transfer.to_coarse_mesh(rel, t);
        t.scale(y_hat);
        metric::mass::project_onto_tangent_space(y_hat, M_coarse, t, dst);
    }

    /** @brief \f$ \mathrm{D}p(y)[v] = \Pi_{\hat x} (\hat x \circ I_H^h (v / y)) \f$, \f$ \hat x = p(y) \f$ */
    void diff_prolongation(const Vector<double>& y_coarse, const Vector<double>& v, Vector<double>& dst) const override
    {
        Vector<double> x_hat(transfer.n_fine());
        prolongation(y_coarse, x_hat);
        Vector<double> rel(v.size());
        for (unsigned j = 0; j < v.size(); ++j) {
            rel[j] = v[j] / y_coarse[j];
        }
        Vector<double> t(transfer.n_fine());
        transfer.to_fine_mesh(rel, t);
        t.scale(x_hat);
        metric::mass::project_onto_tangent_space(x_hat, M_fine, t, dst);
    }

    /** @brief \f$ \mathrm{D}p(y)^* u = Y^{-1} M_H^{-1} (I_H^h)^\top \hat X M_h \Pi_{\hat x} u \f$ (lumped \f$ M_H \f$) */
    void diff_prolongation_adjoint(const Vector<double>& y_coarse, const Vector<double>& u,
                                   Vector<double>& dst) const override
    {
        Vector<double> x_hat(transfer.n_fine());
        prolongation(y_coarse, x_hat);
        Vector<double> proj(u.size()), M_proj(u.size());
        metric::mass::project_onto_tangent_space(x_hat, M_fine, u, proj);
        M_fine.vmult(M_proj, proj);
        M_proj.scale(x_hat);
        Vector<double> dual(transfer.n_coarse());
        transfer.Tfine(M_proj, dual);
        M_inv_coarse.set_tol(0.0);  // relative tolerance, as in ManifoldTransfer (shared inverse)
        dst.reinit(transfer.n_coarse());
        M_inv_coarse.vmult(dst, dual);
        for (unsigned j = 0; j < dst.size(); ++j) {
            dst[j] /= y_coarse[j];
        }
    }

private:
    static void log_of(const Vector<double>& x, Vector<double>& s)
    {
        AssertThrow(std::all_of(x.begin(), x.end(), [](double a) { return a > 0; }),
                    dealii::ExcMessage("LogManifoldTransfer: point has a non-positive component"));
        s.reinit(x.size(), true);
        for (unsigned i = 0; i < x.size(); ++i) {
            s[i] = std::log(x[i]);
        }
    }

    //! a <- N(exp(a)), with the logarithms a shifted by their maximum against overflow
    static void exp_normalized(const MatrixType& M, Vector<double>& a)
    {
        transfer_detail::exp_shifted(a);
        ellipsoid::retract_by_norm(M, a);
        ellipsoid_positive::threshold(a, std::numeric_limits<double>::min());
    }

    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    const InverseMatrixType& M_inv_coarse;
};


/**
 * @brief Logarithmic point maps on points floored at a fixed @p eps (md/positive_multilevel_benchmark.md, step 3b):
 * \f$ r(x) = N_H(e^{I_h^H \log \max(x, \varepsilon)}) \f$, \f$ p(y) = N_h(e^{I_H^h \log \max(y, \varepsilon)}) \f$
 * with the normalization \f$ N(a) = a / \|a\|_M \f$.
 *
 * Only the input is clamped: the interpolated logarithms are at least \f$ \log \varepsilon \f$ (for an interpolation
 * with nonnegative weights summing to 1; the \f$ Q_2 \f$ embedding extrapolates slightly), so the outputs are at least
 * \f$ c \varepsilon \f$, \f$ c \f$ the normalization factor (close to 1). Clamping the output as well would put every
 * component coming from clamped inputs (\f$ \approx c \varepsilon \f$) on the kink of the clamp, so that the maps would
 * not be differentiable in the tails.
 *
 * The differentials are those of the clamped maps, with the input mask \f$ m = [\mathrm{src} > \varepsilon] \f$
 * (components at or below \f$ \varepsilon \f$ are frozen):
 * \f$ \mathrm{D}p(y)[v] = \Pi_{\hat x}(\hat x \circ I_H^h(m \circ v / y)) \f$, \f$ \hat x = p(y) \f$, and
 * \f$ \mathrm{D}r \f$ likewise. The adjoint is the \f$ M \f$-transpose of this differential,
 * \f$ \mathrm{D}p(y)^* u = M_H^{-1}(m \circ Y^{-1} (I_H^h)^\top (\hat x \circ M_h \Pi_{\hat x} u)) \f$. Its terms have
 * the factors \f$ \hat x_i \ge c \varepsilon \f$ and \f$ u_i \f$, and the divisor \f$ y_j > \varepsilon \f$: with the
 * default \f$ \varepsilon = 10^{-140} \f$ nothing overflows, and a product underflows only for \f$ u_i \f$ far below
 * \f$ \hat x_i \f$. Unlike LogManifoldTransfer, components of the input at or below \f$ \varepsilon \f$ (also zero or
 * negative ones) are allowed; they are treated as \f$ \varepsilon \f$.
 */
template <typename MatrixType, typename InverseMatrixType>
class LogThresholdManifoldTransfer : public ManifoldTransferBase
{
public:
    static constexpr double default_eps = 1e-140;

    LogThresholdManifoldTransfer(const LinearTransferBase& transfer, const MatrixType& M_coarse,
                                 const MatrixType& M_fine, const InverseMatrixType& M_inv_coarse,
                                 double eps = default_eps)
        : ManifoldTransferBase(transfer), M_coarse(M_coarse), M_fine(M_fine), M_inv_coarse(M_inv_coarse), eps(eps)
    {
        AssertThrow(eps > 0, dealii::ExcMessage("LogThresholdManifoldTransfer: eps must be positive"));
    }

    /** @brief \f$ y = r(x) = N_H(e^{I_h^H \log \max(x, \varepsilon)}) \f$ */
    void restriction(const Vector<double>& x_fine, Vector<double>& y_coarse) const override
    {
        map(x_fine, y_coarse, false);
    }

    /** @brief \f$ x = p(y) = N_h(e^{I_H^h \log \max(y, \varepsilon)}) \f$ */
    void prolongation(const Vector<double>& y_coarse, Vector<double>& x_fine) const override
    {
        map(y_coarse, x_fine, true);
    }

    /** @brief \f$ \mathrm{D}r(x)[v] = \Pi_{\hat y}(\hat y \circ I_h^H(m \circ v / x)) \f$, \f$ \hat y = r(x) \f$ */
    void diff_restriction(const Vector<double>& x_fine, const Vector<double>& v, Vector<double>& dst) const override
    {
        diff(x_fine, v, dst, false);
    }

    /** @brief \f$ \mathrm{D}p(y)[v] = \Pi_{\hat x}(\hat x \circ I_H^h(m \circ v / y)) \f$, \f$ \hat x = p(y) \f$ */
    void diff_prolongation(const Vector<double>& y_coarse, const Vector<double>& v, Vector<double>& dst) const override
    {
        diff(y_coarse, v, dst, true);
    }

    /** @brief \f$ \mathrm{D}p(y)^* u = M_H^{-1}(m \circ Y^{-1} (I_H^h)^\top (\hat x \circ M_h \Pi_{\hat x} u)) \f$ */
    void diff_prolongation_adjoint(const Vector<double>& y_coarse, const Vector<double>& u,
                                   Vector<double>& dst) const override
    {
        Vector<double> x_hat(transfer.n_fine());
        map(y_coarse, x_hat, true);
        Vector<double> proj(u.size()), M_proj(u.size());
        metric::mass::project_onto_tangent_space(x_hat, M_fine, u, proj);
        M_fine.vmult(M_proj, proj);
        M_proj.scale(x_hat);
        Vector<double> dual(transfer.n_coarse());
        transfer.Tfine(M_proj, dual);
        for (unsigned j = 0; j < dual.size(); ++j) {
            dual[j] = y_coarse[j] > eps ? dual[j] / y_coarse[j] : 0.0;
        }
        M_inv_coarse.set_tol(0.0);  // relative tolerance, as in ManifoldTransfer (shared inverse)
        dst.reinit(transfer.n_coarse());
        M_inv_coarse.vmult(dst, dual);
    }

    [[nodiscard]] double get_eps() const { return eps; }

private:
    //! dst = N(exp(I log max(src, eps))) on the fine (@p to_fine) or the coarse mesh
    void map(const Vector<double>& src, Vector<double>& dst, bool to_fine) const
    {
        Vector<double> s(src.size());
        for (unsigned i = 0; i < src.size(); ++i) {
            s[i] = std::log(std::max(src[i], eps));
        }
        dst.reinit(to_fine ? transfer.n_fine() : transfer.n_coarse());
        if (to_fine) {
            transfer.to_fine_mesh(s, dst);
        } else {
            transfer.to_coarse_mesh(s, dst);
        }
        transfer_detail::exp_shifted(dst);
        ellipsoid::retract_by_norm(to_fine ? M_fine : M_coarse, dst);
    }

    //! Pi_f(f (I (m v / src))), f = map(src), m = [src > eps]
    void diff(const Vector<double>& src, const Vector<double>& v, Vector<double>& dst, bool to_fine) const
    {
        const unsigned n = to_fine ? transfer.n_fine() : transfer.n_coarse();
        Vector<double> f(n);
        map(src, f, to_fine);
        Vector<double> rel(src.size());
        for (unsigned i = 0; i < src.size(); ++i) {
            rel[i] = src[i] > eps ? v[i] / src[i] : 0.0;
        }
        Vector<double> t(n);
        if (to_fine) {
            transfer.to_fine_mesh(rel, t);
        } else {
            transfer.to_coarse_mesh(rel, t);
        }
        t.scale(f);
        metric::mass::project_onto_tangent_space(f, to_fine ? M_fine : M_coarse, t, dst);
    }

    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    const InverseMatrixType& M_inv_coarse;
    double eps;
};


/**
 * @brief Whether the prolongation \f$ I_H^h \f$ of @p transfer has no entry below \f$ -\mathrm{tol} \f$, from its
 * columns \f$ I_H^h e_j \f$ (\f$ n_H \f$ prolongations): then ManifoldTransfer maps the positive sphere into itself
 * (with the injection as restriction). Fails for the \f$ Q_2 \f$ embedding (entries \f$ -1/8 \f$).
 */
inline bool is_nonnegative(const LinearTransferBase& transfer, double tol = 1e-14)
{
    Vector<double> e(transfer.n_coarse()), col(transfer.n_fine());
    for (unsigned j = 0; j < transfer.n_coarse(); ++j) {
        e = 0.0;
        e[j] = 1.0;
        transfer.to_fine_mesh(e, col);
        if (*std::min_element(col.begin(), col.end()) < -tol) {
            return false;
        }
    }
    return true;
}

} // namespace rmo::gpe

#endif //RMO_GPE_TRANSPORT_H
