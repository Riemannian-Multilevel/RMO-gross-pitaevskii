#ifndef RMO_GPE_TRANSPORT_H
#define RMO_GPE_TRANSPORT_H

#include <rmo/gpe/manifold.h>
#include <rmo/gpe/metric.h>

#include <rmo/ropt/transport.h>

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
 * For the differential transports, \f$ \mathcal{R} \f$ is not the adjoint of \f$ \mathcal{P} \f$: both are
 * differentials of independent point maps.
 */
namespace rmo::gpe
{

/** @brief Point maps \f$ r \f$, \f$ p \f$ between the spheres and their differentials (metric-independent). */
template <typename MatrixType>
class ManifoldTransfer : public ManifoldTransferBase
{
public:
    ManifoldTransfer(const LinearTransferBase& transfer,
                     const MatrixType& M_coarse,
                     const MatrixType& M_fine)
        : ManifoldTransferBase(transfer), M_coarse(M_coarse), M_fine(M_fine)
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

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;
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
class AdjointDifferentialTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "V5restr";

    // ADDED: LinearTransferBase and InverseMatrixType are required to compute
    // the adjoint restriction (transpose of I_H^h and M_H^{-1}.)
    explicit AdjointDifferentialTransport(const LinearTransferBase& transfer,
                                          const ManifoldTransferBase& pt,
                                          const MatrixType& M_coarse,
                                          const MatrixType& M_fine,
                                          const InverseMatrixType& M_inv_coarse)
        : transfer(transfer), point_transfer(pt),
          M_coarse(M_coarse), M_fine(M_fine), M_inv_coarse(M_inv_coarse)
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

    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& x_fine,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
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
        // Equivalent to M_h v_fine - [((I_H^h \psi)^T M_h v_fine) / ||I_H^h \psi||_{M_h}^2] * (M_h I_H^h \psi)
        Vector<double> M_v_proj = M_v;
        M_v_proj.add(-inner_prod / norm_sq, M_psi_tilde);

        // 5. Apply transpose of prolongation: (I_H^h)^T [ M_h (I - \Pi) v_fine ]
        Vector<double> IT_M_v_proj(transfer.n_coarse());
        transfer.Tfine(M_v_proj, IT_M_v_proj);

        // 6. Apply inverse coarse mass matrix: M_H^{-1} * (I_H^h)^T ...
        //    XXX: Reset to a relative tolerance, so that restriction solve does not
        //    inherit the absolute tolerance left by the coarse oracle's gradient()
        //    on the shared M_inv_coarse
        M_inv_coarse.set_tol(0.0);
        Vector<double> w(transfer.n_coarse());
        M_inv_coarse.vmult(w, IT_M_v_proj);

        // 7. Scale by (1 / ||I_H^h \psi||_{M_h})
        w /= norm;

        // 8. Project onto the target tangent space T_\psi S_H
        metric::mass::project_onto_tangent_space(y_coarse, M_coarse, w, dst);
    }

private:
    const LinearTransferBase& transfer;
    const ManifoldTransferBase& point_transfer;

    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    const InverseMatrixType& M_inv_coarse;
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

} // namespace rmo::gpe

#endif //RMO_GPE_TRANSPORT_H
