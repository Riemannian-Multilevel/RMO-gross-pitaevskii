#ifndef GPE_GRID_OPERATORS_H
#define GPE_GRID_OPERATORS_H

#include <gpe/fe/interpolate.h>
#include <gpe/ropt/manifold.h>
#include <gpe/option_types.h>

namespace gpe
{

class ManifoldTransferBase
{
public:
    ManifoldTransferBase(const LinearTransferBase& transfer_)
        : transfer(transfer_) {}
    virtual ~ManifoldTransferBase() = default;

    // Mandatory operations
    virtual void restriction(const Vector<double>&, Vector<double>&) const = 0;

    virtual void prolongation(const Vector<double>&, Vector<double>&) const = 0;

    // Optional operations
    virtual void diff_restriction(const Vector<double>&, const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented("Differential not implemented for manifold transfer");
    }

    virtual void diff_prolongation(const Vector<double>&, const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented("Differential not implemented for manifold transfer");
    }

    // Dimension
    unsigned n_fine() const { return transfer.n_fine(); }
    unsigned n_coarse() const { return transfer.n_coarse(); }

protected:
    // For now, every manifold transfer assumes an underlying (linear) interpolation for the embedding space
    const LinearTransferBase& transfer;
};


// Metric-independent transfers for the fine/coarse manifolds S_h/S_H
template <typename MatrixType>
class ManifoldTransfer : public ManifoldTransferBase
{
public:
    ManifoldTransfer(const LinearTransferBase& transfer,
                     const MatrixType& M_coarse,
                     const MatrixType& M_fine)
        : ManifoldTransferBase(transfer), M_coarse(M_coarse), M_fine(M_fine)
    {}

    /**
     * @brief Restricts a point from the fine manifold to the coarse manifold.
     *
     * The point restriction map $r(y)$ transfers a state vector $y \in \mathcal{S}_{M_h}$
     * on the fine grid to a state vector on the coarse grid $x \in \mathcal{S}_{M_H}$.
     * It does this by first applying the standard linear restriction operator $I_h^H$,
     * and then projecting (retracting) the result back onto the mass-weighted unit sphere
     * of the coarse space.
     * * Mathematically:
     * \f[ r(y) = \frac{I_h^H y}{\|I_h^H y\|_{M_H}} \f]
     *
     * @param x_fine The base point $y \in \mathcal{S}_{M_h}$ on the fine grid.
     * @param y_coarse [out] The restricted point $r(y) \in \mathcal{S}_{M_H}$ on the coarse grid.
     */
    void restriction(const Vector<double>& x_fine, Vector<double>& y_coarse) const override
    {
        transfer.to_coarse_mesh(x_fine, y_coarse);
        ellipsoid::retract_by_norm(M_coarse, y_coarse);
    }

    /**
     * @brief Prolongs a point from the coarse manifold to the fine manifold.
     *
     * The point prolongation map $p(x)$ transfers a state vector $x \in \mathcal{S}_{M_H}$
     * on the coarse grid to a state vector on the fine grid $y \in \mathcal{S}_{M_h}$.
     * It does this by first applying the standard linear prolongation operator $I_H^h$,
     * and then projecting (retracting) the result back onto the mass-weighted unit sphere
     * of the fine space.
     * * Mathematically:
     * \f[ p(x) = \frac{I_H^h x}{\|I_H^h x\|_{M_h}} \f]
     *
     * @param y_coarse The base point $x \in \mathcal{S}_{M_H}$ on the coarse grid.
     * @param x_fine [out] The prolonged point $p(x) \in \mathcal{S}_{M_h}$ on the fine grid.
     */
    void prolongation(const Vector<double>& y_coarse, Vector<double>& x_fine) const override
    {
        transfer.to_fine_mesh(y_coarse, x_fine);

        ellipsoid::retract_by_norm(M_fine, x_fine);
    }

    /**
     * @brief Computes the differential of the restriction map.
     *
     * The differential of the restriction map $r(y)$ evaluates the pushforward
     * of a tangent vector $v \in T_y \mathcal{S}_{M_h}$ onto the coarse tangent space.
     * It is computed in five steps:
     * 1. **Linear restriction of the base point:** $\hat{x} = I_h^H y$
     * 2. **Norm of the linear point:** $n_H = \|\hat{x}\|_{M_H}$
     * 3. **Linear restriction of the tangent vector:** $v_H = I_h^H v$
     * 4. **Mass-weighted inner product:** $\langle \hat{x}, v_H \rangle_{M_H} = \hat{x}^\top M_H v_H$
     * 5. **Assembly:** $\Drm r(y)[v] = \frac{1}{n_H} \left( v_H - \frac{\langle \hat{x}, v_H \rangle_{M_H}}{n_H^2} \hat{x} \right)$
     *
     * @param x_fine The base point $y \in \mathcal{S}_{M_h}$ on the fine grid.
     * @param v The tangent vector $v \in T_y \mathcal{S}_{M_h}$.
     * @param dst The mapped tangent vector $\Drm r(y)[v] \in T_{r(y)} \mathcal{S}_{M_H}$.
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
     * @brief Computes the differential of the prolongation map.
     *
     * The differential of the prolongation map $p(x)$ evaluates the pushforward
     * of a tangent vector $v \in T_x \mathcal{S}_{M_H}$ onto the fine tangent space.
     * It is computed in five steps:
     * 1. **Linear prolongation of the base point:** $\hat{y} = I_H^h x$
     * 2. **Norm of the linear point:** $n_h = \|\hat{y}\|_{M_h}$
     * 3. **Linear prolongation of the tangent vector:** $v_h = I_H^h v$
     * 4. **Mass-weighted inner product:** $\langle \hat{y}, v_h \rangle_{M_h} = \hat{y}^\top M_h v_h$
     * 5. **Assembly:** $\Drm p(x)[v] = \frac{1}{n_h} \left( v_h - \frac{\langle \hat{y}, v_h \rangle_{M_h}}{n_h^2} \hat{y} \right)$
     *
     * @param y_coarse The base point $x \in \mathcal{S}_{M_H}$ on the coarse grid.
     * @param v The tangent vector $v \in T_x \mathcal{S}_{M_H}$.
     * @param dst The mapped tangent vector $\Drm p(x)[v] \in T_{p(x)} \mathcal{S}_{M_h}$.
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


class VectorTransportBase
{
public:
    static constexpr const char* id = "";
    virtual ~VectorTransportBase() = default;

    /**
     * @brief Prolongs a tangent vector from the coarse grid to the fine grid.
     * @param x_fine The actual fine iterate (defines the target tangent space).
     * @param y_coarse The coarse iterate (defines the source tangent space).
     * @param v_coarse The vector in T_x S_H to be prolonged.
     * @param dst_fine [out] The prolonged vector, guaranteed to be in T_y S_h.
     */
    virtual void vector_prolongation(const Vector<double>& x_fine,
                                     const Vector<double>& y_coarse,
                                     const Vector<double>& v_coarse,
                                     Vector<double>& dst_fine) const = 0;

    /**
     * @brief Restricts a tangent vector from the fine grid to the coarse grid.
     * @param y_coarse The actual coarse iterate (defines the target tangent space).
     * @param x_fine The fine iterate (defines the source tangent space).
     * @param v_fine The vector in T_y S_h to be restricted.
     * @param dst_coarse [out] The restricted vector, guaranteed to be in T_x S_H.
     */
    virtual void vector_restriction(const Vector<double>& y_coarse,
                                    const Vector<double>& x_fine,
                                    const Vector<double>& v_fine,
                                    Vector<double>& dst_coarse) const = 0;
};


/**
 * @brief Strategy for vector transport via ambient space transfer and orthogonal projection.
 * This class implements the `VectorTransportBase` interface using the standard projection
 * strategy. Tangent vectors are first transferred as standard Euclidean vectors in the
 * ambient space $\mathbb{R}^n$, and then forcefully projected onto the target tangent
 * space using the mass-metric orthogonal projector.
 *
 * @note Because the ambient interpolation transfers the vector directly to the target space
 * grid, this strategy only requires the *target* base point to define the final tangent space.
 * The *source* base points in the overridden methods are ignored.
 *
 * @tparam dim The spatial dimension of the problem.
 * @tparam MatrixType The matrix type defining the ambient space metric (typically the Mass matrix).
 */
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

    /**
     * @brief Prolongs a tangent vector using ambient interpolation and M-metric projection.
     * Computes:
     * $$\mathcal{T}_{H \to h}(v) = P_{T_y \mathcal{S}_h} (I_H^h v)$$
     * where $I_H^h$ is the standard linear prolongation operator.
     */
    void vector_prolongation(const Vector<double>& x_fine,
                             [[maybe_unused]] const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse,
                             Vector<double>& dst) const override
    {
        // Tangent vector interpolated to fine ambient space
        Vector<double> Iv(transfer.n_fine());
        transfer.to_fine_mesh(v_coarse, Iv);

        // Orthogonal projection I(v \in T_x S_H) -> T_p(x) S_h in M-metric
        ellipsoid::mass::project_onto_tangent_space(x_fine, M_fine, Iv, dst);
    }

    /**
     * @brief Restricts a tangent vector using ambient restriction and M-metric projection.
     * Computes:
     * $$\mathcal{T}_{h \to H}(v) = P_{T_x \mathcal{S}_H} (I_h^H v)$$
     * where $I_h^H$ is the standard linear restriction operator.
     */
    void vector_restriction(const Vector<double>& y_coarse,
                            [[maybe_unused]] const Vector<double>& x_fine,
                            const Vector<double>& v_fine,
                            Vector<double>& dst) const override
    {
        // Tangent vector restricted to coarse ambient space
        Vector<double> Iv(transfer.n_coarse());
        transfer.to_coarse_mesh(v_fine, Iv);

        // Orthogonal projection I(v \in T_y S_h) -> T_r(y) S_H
        ellipsoid::mass::project_onto_tangent_space(y_coarse, M_coarse, Iv, dst);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const LinearTransferBase& transfer;
};


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
        ellipsoid::frobenius::project_onto_tangent_space(x_fine, M_fine, Iv, dst);
    }

    void vector_restriction(const Vector<double>& y_coarse,
                            [[maybe_unused]] const Vector<double>& x_fine,
                            const Vector<double>& v_fine,
                            Vector<double>& dst) const override
    {
        // Tangent vector restricted to coarse ambient space
        Vector<double> Iv(transfer.n_coarse());
        transfer.to_coarse_mesh(v_fine, Iv);

        // F-Orthogonal projection I(v \in T_y S_h) -> T_r(y) S_H
        ellipsoid::frobenius::project_onto_tangent_space(y_coarse, M_coarse, Iv, dst);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const LinearTransferBase& transfer;
};


/**
 * @brief Strategy for vector transport via differentials.
 * This class implements the `VectorTransportBase` interface by computing the differential
 * of the manifold point transfer maps (restriction $r$ and prolongation $p$).
 *
 * The push-forward of a vector maps it to the tangent space of the
 * mapped point (e.g., $D p(x)[v] \in T_{p(x)} \mathcal{S}_h$).
 * Since the target point of the solver might differ (e.g., $y_h \neq p(x_H)$),
 * this class employs a two-step interface:
 * 1. Evaluate the differential.
 * 2. Apply a corrective orthogonal projection to move the vector to the target tangent space.
 *
 * @tparam dim The spatial dimension of the problem.
 * @tparam MatrixType The matrix type defining the metric.
 */
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

    /**
     * @brief Prolongs a tangent vector by computing the differential of the prolongation map.
     * This two-step method computes the differential and then performs a corrective
     * vector transport to the fine target space:
     * $$\hat{v} = D p(x_H)[v_H] \quad \in T_{p(x_H)} \mathcal{S}_h$$
     * $$\mathcal{T}_{H \to h}(v_H) = P_{T_y \mathcal{S}_h} (\hat{v})$$
     */
    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Differential D_p(x): T_x S_H -> T_p(x) S_h
        Vector<double> D_px(point_transfer.n_fine());
        vector_prolongation(y_coarse, v_coarse, D_px);

        // Vector transport T_p(x) S_h -> T_y S_h
        // TODO: which metric to choose for orthogonal projection?
        // -> Use inheritance from DifferentialTransport base class (FrobeniusDifferentialTransport, MassDifferentialTransport)
        ellipsoid::mass::project_onto_tangent_space(x_fine, M_fine, D_px, dst);
    }

    void vector_restriction(const Vector<double>& x_fine, const Vector<double>& v_fine,
                            Vector<double>& dst) const
    {
        point_transfer.diff_restriction(x_fine, v_fine, dst);
    }

    /**
     * @brief Restricts a tangent vector by computing the differential of the restriction map.
     * This two-step method computes the differential and then performs a corrective
     * vector transport to the coarse target space:
     * $$\hat{v} = D r(y_h)[v_h] \quad \in T_{r(y_h)} \mathcal{S}_H$$
     * $$\mathcal{T}_{h \to H}(v_h) = P_{T_x \mathcal{S}_H} (\hat{v})$$
     */
    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& x_fine,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        // Differential D_r(y): T_y S_h -> T_r(y) S_H
        Vector<double> D_ry(point_transfer.n_coarse());
        vector_restriction(x_fine, v_fine, D_ry);

        // Vector transport T_r(y) S_H -> T_x S_H
        // TODO: which metric to choose for orthogonal projection?
        ellipsoid::mass::project_onto_tangent_space(y_coarse, M_coarse, D_ry, dst);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const ManifoldTransferBase& point_transfer;
};


/**
 * @brief Frobenius-metric counterpart of DifferentialTransport.
 *
 * @note As with DifferentialTransport, R^y_x is *not* the adjoint of P^x_y
 * (both are differentials of independent point maps), so the geometric
 * Galerkin condition (5) is not satisfied in either metric.
 */
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

    /**
     * @brief Prolongs a tangent vector by computing the differential of the prolongation map,
     * then transporting it to the fine target tangent space via F-orthogonal projection:
     * $$\hat{v} = D p(x_H)[v_H] \in T_{p(x_H)} \mathcal{S}_h, \qquad \mathcal{T}_{H \to h}(v_H) = \Pi_x^F(\hat{v}).$$
     */
    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Differential D_p(x): T_x S_H -> T_p(x) S_h
        Vector<double> D_px(point_transfer.n_fine());
        vector_prolongation(y_coarse, v_coarse, D_px);

        // Vector transport T_p(x) S_h -> T_y S_h using F-orthogonal projection
        ellipsoid::frobenius::project_onto_tangent_space(x_fine, M_fine, D_px, dst);
    }

    void vector_restriction(const Vector<double>& x_fine, const Vector<double>& v_fine,
                            Vector<double>& dst) const
    {
        point_transfer.diff_restriction(x_fine, v_fine, dst);
    }

    /**
     * @brief Restricts a tangent vector by computing the differential of the restriction map,
     * then transporting it to the coarse tangent space via F-orthogonal projection:
     * $$\hat{v} = D r(y_h)[v_h] \in T_{r(y_h)} \mathcal{S}_H, \qquad \mathcal{T}_{h \to H}(v_h) = \Pi_y^F(\hat{v}).$$
     */
    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& x_fine,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        // Differential D_r(y): T_y S_h -> T_r(y) S_H
        Vector<double> D_ry(point_transfer.n_coarse());
        vector_restriction(x_fine, v_fine, D_ry);

        // Vector transport T_r(y) S_H -> T_x S_H using F-orthogonal projection
        ellipsoid::frobenius::project_onto_tangent_space(y_coarse, M_coarse, D_ry, dst);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const ManifoldTransferBase& point_transfer;
};


/**
 * @brief Adjoint-based Vector Transport implementing Version II.
 *
 * This class computes the mathematical adjoint of the corresponding
 * prolongation maps in the mass-weighted metric.
 */
template <typename MatrixType, typename InverseMatrixType>
class AdjointRestrictionTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "V2Restr";

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

    /**
     * @brief Version II Prolongation: P(v) = Pi_\phi ( I_H^h v )
     */
    virtual void vector_prolongation(const Vector<double>& x_fine,
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
        ellipsoid::mass::project_onto_tangent_space(x_fine, M_fine, Iv, dst);
    }

    /**
     * @brief Restricts a tangent vector using Version II or Version V.
     * * Version II: R(v) = (I - \psi \psi^T M_H) M_H^{-1} (I_H^h)^T M_h v
     * Version V:  R(v) = (1 / ||I_H^h \psi||_{M_h}) * Version II
     */
    virtual void vector_restriction(const Vector<double>& y_coarse,
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
        Vector<double> w(transfer.n_coarse());
        M_inv_coarse.vmult(w, IT_M_v);

        // 4. Project onto target tangent space T_\psi S_H
        // TODO: which metric to choose for orthogonal projection?
        // -> Use inheritance from DifferentialTransport base class (FrobeniusDifferentialTransport, MassDifferentialTransport)
        ellipsoid::mass::project_onto_tangent_space(y_coarse, M_coarse, w, dst);
    }

protected:
    const LinearTransferBase& transfer;
    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    const InverseMatrixType& M_inv_coarse;
};


/**
 * @brief Frobenius-metric counterpart of AdjointRestrictionTransport
 *
 * Both P^x_y and R^y_x are defined exactly as in AdjointRestrictionTransport,
 * but with the orthogonal projection Pi taken in the Frobenius (unweighted
 * Euclidean) metric instead of the mass metric.
 */
template <typename MatrixType>
class FrobeniusAdjointRestrictionTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "FV2Restr";

    FrobeniusAdjointRestrictionTransport(const LinearTransferBase& I,
                                         const MatrixType& M_coarse,
                                         const MatrixType& M_fine)
        : transfer(I), M_coarse(M_coarse), M_fine(M_fine)
    {}

    /**
     * @brief Prolongation: P(v) = Pi_x^F( I_H^h v )
     */
    void vector_prolongation(const Vector<double>& x_fine,
                             [[maybe_unused]] const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse,
                             Vector<double>& dst) const override
    {
        // 1. Linear prolongation to ambient space
        Vector<double> Iv(transfer.n_fine());
        transfer.to_fine_mesh(v_coarse, Iv);

        // 2. F-orthogonal projection onto the target tangent space
        ellipsoid::frobenius::project_onto_tangent_space(x_fine, M_fine, Iv, dst);
    }

    /**
     * @brief Restriction: R(v) = Pi_y^F( (I_H^h)^T v )
     */
    void vector_restriction(const Vector<double>& y_coarse,
                            [[maybe_unused]] const Vector<double>& x_fine,
                            const Vector<double>& v_fine,
                            Vector<double>& dst) const override
    {
        // 1. Apply transpose of prolongation (no M_h/M_H weighting under the F-metric)
        Vector<double> IT_v(transfer.n_coarse());
        transfer.Tfine(v_fine, IT_v);

        // 2. Project onto target tangent space T_y S_H
        ellipsoid::frobenius::project_onto_tangent_space(y_coarse, M_coarse, IT_v, dst);
    }

private:
    const LinearTransferBase& transfer;
    const MatrixType& M_coarse;
    const MatrixType& M_fine;
};


/**
 * @brief Adjoint-based Vector Transport implementing Version V Restriction.
 */
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

    /**
     * @brief Prolongs a tangent vector by computing the differential of the prolongation map.
     */
    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Differential D_p(x): T_x S_H -> T_p(x) S_h
        Vector<double> D_px(point_transfer.n_fine());
        vector_prolongation(y_coarse, v_coarse, D_px);

        // Vector transport T_p(x) S_h -> T_y S_h using mass metric projection
        ellipsoid::mass::project_onto_tangent_space(x_fine, M_fine, D_px, dst);
    }

    /**
     * @brief Version V Restriction:
     * R(v) = (1 / ||I_H^h \psi||_{M_h}) * (I - \psi\psi^T M_H) M_H^{-1} (I_H^h)^T M_h (I - \Pi_{I_H^h \psi, M_h}) v
     */
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
        Vector<double> w(transfer.n_coarse());
        M_inv_coarse.vmult(w, IT_M_v_proj);

        // 7. Scale by (1 / ||I_H^h \psi||_{M_h})
        w /= norm;

        // 8. Project onto the target tangent space T_\psi S_H
        ellipsoid::mass::project_onto_tangent_space(y_coarse, M_coarse, w, dst);
    }

private:
    const LinearTransferBase& transfer;
    const ManifoldTransferBase& point_transfer;

    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    const InverseMatrixType& M_inv_coarse;
};


/**
 * @brief Frobenius-metric counterpart of AdjointDifferentialTransport.
 */
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

    /**
     * @brief Prolongs a tangent vector by computing the differential of the prolongation map,
     * then transporting it to the fine tangent space via F-orthogonal projection:
     * $$\hat{v} = D p(x_H)[v_H] \in T_{p(x_H)} \mathcal{S}_h, \qquad \mathcal{T}_{H \to h}(v_H) = \Pi_x^F(\hat{v}).$$
     */
    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Differential D_p(x): T_x S_H -> T_p(x) S_h
        Vector<double> D_px(point_transfer.n_fine());
        vector_prolongation(y_coarse, v_coarse, D_px);

        // Vector transport T_p(x) S_h -> T_y S_h using F-orthogonal projection
        ellipsoid::frobenius::project_onto_tangent_space(x_fine, M_fine, D_px, dst);
    }

    /**
     * @brief Restriction: $$R(v) = \frac{1}{\|I_H^h \psi\|_{M_h}} \Pi_\psi^F\big((I_H^h)^\top Q(v)\big),
     * \quad Q(v) = v - \frac{\tilde\psi \cdot v}{\|\tilde\psi\|_{M_h}^2}\, M_h \tilde\psi, \quad
     * \tilde\psi = I_H^h \psi.$$
     */
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
        ellipsoid::frobenius::project_onto_tangent_space(y_coarse, M_coarse, IT_Qv, dst);
    }

private:
    const LinearTransferBase& transfer;
    const ManifoldTransferBase& point_transfer;

    const MatrixType& M_coarse;
    const MatrixType& M_fine;
};

} // namespace gpe

#endif //GPE_GRID_OPERATORS_H