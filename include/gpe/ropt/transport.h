#ifndef GPE_GRID_OPERATORS_H
#define GPE_GRID_OPERATORS_H

#include <gpe/fe/fe_interpolate.h>
#include <gpe/ropt/manifold.h>
#include <gpe/option_types.h>

namespace gpe
{

template <typename MatrixType>
struct MatrixContext
{
    const MatrixType& M_c;
    const MatrixType& M_f;
    const MatrixType& A_c;
    const MatrixType& A_f;

    MatrixContext(const MatrixType& M_c, const MatrixType& M_f,
                  const MatrixType& A_c, const MatrixType& A_f)
        : M_c(M_c), M_f(M_f), A_c(A_c), A_f(A_f)
    {}
};


template <typename InverseMatrixType>
struct InverseMatrixContext
{
    const InverseMatrixType& M_inv_c;
    const InverseMatrixType& M_inv_f;
    const InverseMatrixType& A_inv_c;
    const InverseMatrixType& A_inv_f;

    InverseMatrixContext(const InverseMatrixType& M_inv_c, const InverseMatrixType& M_inv_f,
                         const InverseMatrixType& A_inv_c, const InverseMatrixType& A_inv_f)
        : M_inv_c(M_inv_c), M_inv_f(M_inv_f), A_inv_c(A_inv_c), A_inv_f(A_inv_f)
    {}
};


// Metric-independent transfers for the fine/coarse manifolds S_h/S_H
template <typename MatrixType>
class ManifoldTransfer
{
public:
    using Context = MatrixContext<MatrixType>;

    ManifoldTransfer(const Context& mtx, const LinearTransferBase& I)
        : M_coarse(mtx.M_c), M_fine(mtx.M_f), transfer(I)
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
    void restriction(const Vector<double>& x_fine, Vector<double>& y_coarse) const
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
    void prolongation(const Vector<double>& y_coarse, Vector<double>& x_fine) const
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
    void diff_restriction(const Vector<double>& x_fine, const Vector<double>& v, Vector<double>& dst) const
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
    void diff_prolongation(const Vector<double>& y_coarse, const Vector<double>& v, Vector<double>& dst) const
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

    unsigned n_fine() const { return transfer.n_fine(); }
    unsigned n_coarse() const { return transfer.n_coarse(); }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    const LinearTransferBase& transfer;
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
    static constexpr bool requires_inverse = false;
    using Context = MatrixContext<MatrixType>;

    MassProjectionTransport(const Context& mtx,
                            const LinearTransferBase& I,
                            const ManifoldTransfer<MatrixType>& pt)
        : M_coarse(mtx.M_c), M_fine(mtx.M_f), transfer(I), point_transfer(pt)
    {}

    /**
     * @brief Prolongs a tangent vector using ambient interpolation and M-metric projection.
     * Computes:
     * $$\mathcal{T}_{H \to h}(v) = P_{T_y \mathcal{S}_h} (I_H^h v)$$
     * where $I_H^h$ is the standard linear prolongation operator.
     */
    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
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
    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& x_fine,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
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
    const ManifoldTransfer<MatrixType>& point_transfer;
};


template <typename MatrixType>
class FrobeniusProjectionTransport : public VectorTransportBase
{
public:
    using Context = MatrixContext<MatrixType>;
    static constexpr const char* id = "F";
    static constexpr bool requires_inverse = false;

    explicit FrobeniusProjectionTransport(const Context& mtx,
                                          const LinearTransferBase& I,
                                          const ManifoldTransfer<MatrixType>& pt)
        : M_coarse(mtx.M_c), M_fine(mtx.M_f), transfer(I), point_transfer(pt)
    {}

    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Tangent vector interpolated to fine ambient space
        Vector<double> Iv(transfer.n_fine());
        transfer.to_fine_mesh(v_coarse, Iv);

        // F-orthogonal projection on the fine grid
        ellipsoid::frobenius::project_onto_tangent_space(x_fine, M_fine, Iv, dst);
    }

    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& x_fine,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
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
    const ManifoldTransfer<MatrixType>& point_transfer;
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
    using Context = MatrixContext<MatrixType>;
    static constexpr const char* id = "D";
    static constexpr bool requires_inverse = false;

    explicit DifferentialTransport(const Context& ctx,
                                   const LinearTransferBase& I,
                                   const ManifoldTransfer<MatrixType>& pt)
        : transfer(I), point_transfer(pt)
        , M_fine(ctx.M_f), M_coarse(ctx.M_c)
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
    const LinearTransferBase& transfer;
    const ManifoldTransfer<MatrixType>& point_transfer;
    const MatrixType& M_fine;
    const MatrixType& M_coarse;
};


namespace experimental
{

template <typename MatrixType, typename InverseMatrixType>
class MassProjectionTransportAdjoint : public VectorTransportBase
{
public:
    static constexpr const char* id = "MA";
    static constexpr bool requires_inverse = true;
    using Context = MatrixContext<MatrixType>;
    using InverseContext = InverseMatrixContext<InverseMatrixType>;

    /**
     * @brief Constructor for the adjoint mass-metric projection transport.
     * @param mtx Contains the fine and coarse mass matrices (M_f, M_c).
     * @param inv_mtx Contains the inverse of the coarse mass matrix (M_inv_c).
     * @param I The linear transfer operator (prolongation/restriction).
     * @param pt Manifold transfer operator.
     */
    MassProjectionTransportAdjoint(const Context& mtx,
                                   const InverseContext& inv_mtx,
                                   const LinearTransferBase& I,
                                   const ManifoldTransfer<MatrixType>& pt)
        : transfer(I), point_transfer(pt)
        , M_coarse(mtx.M_c), M_fine(mtx.M_f), M_inv_coarse(inv_mtx.M_inv_c)
    {}

    /**
     * @brief Prolongs a tangent vector using ambient interpolation and M-metric projection.
     * Computes the operator P_y^x:
     * P_y^x(v) = Pi_x ( I_H^h(v) )
     * where I_H^h is the standard linear prolongation operator and Pi_x is the
     * M-orthogonal projection onto the fine tangent space T_x S_h.
     * @param x_fine The point on the fine manifold (x).
     * @param y_coarse The point on the coarse manifold (y).
     * @param v_coarse The tangent vector on the coarse manifold to be prolonged.
     * @param dst The resulting prolonged tangent vector on the fine manifold.
     */
    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        AssertDimension(y_coarse.size(), transfer.n_coarse());
        AssertDimension(x_fine.size(), transfer.n_fine());

        // Tangent vector interpolated to fine ambient space: I_H^h(v)
        Vector<double> Iv(transfer.n_fine());
        transfer.to_fine_mesh(v_coarse, Iv);

        // Orthogonal projection I_H^h(v) -> T_x S_h in M-metric: Pi_x( I_H^h(v) )
        ellipsoid::mass::project_onto_tangent_space(x_fine, M_fine, Iv, dst);
    }

    /**
     * @brief Restricts a tangent vector using the adjoint of the M-metric projection transport.
     * * Computes the operator R_x^y = (P_y^x)^*:
     * R_x^y(v) = Pi_y ( M_H^{-1} (I_H^h)^T M_h v )
     * * This operator ensures the Galerkin condition holds in the tangent spaces:
     * <u, R_x^y v>_{M_H} = <P_y^x u, v>_{M_h}
     * * @param y_coarse The point on the coarse manifold (y).
     * @param x_fine The point on the fine manifold (x).
     * @param v_fine The tangent vector on the fine manifold to be restricted.
     * @param dst The resulting restricted tangent vector on the coarse manifold.
     */
    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& x_fine,
                                const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        // 1. Apply fine mass matrix: M_h * v
        Vector<double> Mh_v(transfer.n_fine());
        M_fine.vmult(Mh_v, v_fine);

        // 2. Apply transpose of prolongation: (I_H^h)^T * (M_h * v)
        Vector<double> I_Mh_v(transfer.n_coarse());
        transfer.to_coarse_mesh(Mh_v, I_Mh_v);

        // 3. Apply inverse coarse mass matrix: w = M_H^{-1} * (I_H^h)^T * M_h * v
        M_inv_coarse.vmult(dst, I_Mh_v);

        const double alpha = y_coarse * I_Mh_v;
        dst.add(-alpha, y_coarse);
    }

private:
    const LinearTransferBase& transfer;
    const ManifoldTransfer<MatrixType>& point_transfer;

    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    const InverseMatrixType& M_inv_coarse;
};


template <typename MatrixType, typename InverseMatrixType>
class DifferentialTransportAdjoint : public VectorTransportBase
{
public:
    using Context = MatrixContext<MatrixType>;
    using InverseContext = InverseMatrixContext<InverseMatrixType>;
    static constexpr const char* id = "DA";
    static constexpr bool requires_inverse = true;

    explicit DifferentialTransportAdjoint(const Context& ctx,
                                          const InverseContext& inv_ctx,
                                          const LinearTransferBase& I,
                                          const ManifoldTransfer<MatrixType>& pt)
        : transfer(I), point_transfer(pt)
        , M_fine(ctx.M_f), M_coarse(ctx.M_c), M_inv_fine(inv_ctx.M_inv_f)
    {}

    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // 1. Calculate beta = || I_h^H x ||_{M_H}
        // using the explicit norm: sqrt( (I_h^H x)^T * M_H * (I_h^H x) )
        Vector<double> Ix(transfer.n_coarse());
        transfer.to_coarse_mesh(x_fine, Ix); // Applies I_h^H = (I_H^h)^T

        Vector<double> MH_Ix(transfer.n_coarse());
        M_coarse.vmult(MH_Ix, Ix);

        // Safe, direct evaluation of the M_H norm
        double beta = std::sqrt(Ix * MH_Ix);

        // 2. Apply coarse mass matrix: M_H u
        Vector<double> MH_u(transfer.n_coarse());
        M_coarse.vmult(MH_u, v_coarse);

        // 3. Prolong: I_H^h (M_H u)
        Vector<double> I_MH_u(transfer.n_fine());
        transfer.to_fine_mesh(MH_u, I_MH_u);

        // 4. Apply fine inverse mass matrix and scale by 1/beta: 1/beta * M_h^{-1} I_H^h M_H u
        M_inv_fine.vmult(dst, I_MH_u);
        dst /= beta;


        // TODO: optional?
        // Vector<double> unprojected_dst = dst;
        // ellipsoid::mass::project_onto_tangent_space(x_fine, M_fine, unprojected_dst, dst);
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
        // TODO: way to signal that y_coarse and x_fine result from y = r(x)
        ellipsoid::mass::project_onto_tangent_space(y_coarse, M_coarse, D_ry, dst);
    }

private:
    const LinearTransferBase& transfer;
    const ManifoldTransfer<MatrixType>& point_transfer;

    const MatrixType& M_fine;
    const MatrixType& M_coarse;
    const InverseMatrixType& M_inv_fine;
};

} // namespace experimental


} // namespace gpe


#endif //GPE_GRID_OPERATORS_H