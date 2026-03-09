#ifndef GPE_GRID_OPERATORS_H
#define GPE_GRID_OPERATORS_H
#include "option_types.h"
#include "manifold.h"

#include <deal.II/numerics/vector_tools.h>

namespace gpe
{

// Linear interpolation/restriction for the ambient space R^n
template <int dim>
class LinearTransfer
{
public:
    LinearTransfer(const dealii::DoFHandler<dim>& dof_c,
                   const dealii::DoFHandler<dim>& dof_f,
                   const dealii::AffineConstraints<double>& aff_c,
                   const dealii::AffineConstraints<double>& aff_f)
        : dof_coarse(dof_c), dof_fine(dof_f), aff_coarse(aff_c), aff_fine(aff_f)
    {}

    void to_coarse_mesh(const Vector<double>& y_fine, Vector<double>& x_coarse) const
    {
        x_coarse.reinit(dof_coarse.n_dofs());
        x_coarse = 0.0;

        dealii::VectorTools::interpolate_to_coarser_mesh(dof_fine,
            y_fine, dof_coarse, aff_coarse, x_coarse);
    }

    void to_fine_mesh(const Vector<double>& x_coarse, Vector<double>& y_fine) const
    {
        y_fine.reinit(dof_fine.n_dofs());
        y_fine = 0.0;

        dealii::VectorTools::interpolate_to_finer_mesh(dof_coarse,
            x_coarse, dof_fine, aff_fine, y_fine);
    }

    unsigned n_coarse() const { return dof_coarse.n_dofs(); }
    unsigned n_fine() const { return dof_fine.n_dofs(); }

private:
    const dealii::DoFHandler<dim>& dof_coarse;
    const dealii::DoFHandler<dim>& dof_fine;
    const dealii::AffineConstraints<double>& aff_coarse;
    const dealii::AffineConstraints<double>& aff_fine;
};


// Metric-independent transfers for the fine/coarse manifolds S_h/S_H
template <int dim, typename MatrixType>
class ManifoldTransfer
{
public:
    ManifoldTransfer(const MatrixType& M_c, const MatrixType& M_f,
                     const LinearTransfer<dim>& I)
        : M_coarse(M_c), M_fine(M_f), transfer(I)
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
     * @param y_fine The base point $y \in \mathcal{S}_{M_h}$ on the fine grid.
     * @param x_coarse [out] The restricted point $r(y) \in \mathcal{S}_{M_H}$ on the coarse grid.
     */
    void restriction(const Vector<double>& y_fine, Vector<double>& x_coarse) const
    {
        transfer.to_coarse_mesh(y_fine, x_coarse);
        ellipsoid::retract_by_norm(M_coarse, x_coarse);
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
     * @param x_coarse The base point $x \in \mathcal{S}_{M_H}$ on the coarse grid.
     * @param y_fine [out] The prolonged point $p(x) \in \mathcal{S}_{M_h}$ on the fine grid.
     */
    void prolongation(const Vector<double>& x_coarse, Vector<double>& y_fine) const
    {
        transfer.to_fine_mesh(x_coarse, y_fine);
        ellipsoid::retract_by_norm(M_fine, y_fine);
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
     * @param y_fine The base point $y \in \mathcal{S}_{M_h}$ on the fine grid.
     * @param v The tangent vector $v \in T_y \mathcal{S}_{M_h}$.
     * @param dst The mapped tangent vector $\Drm r(y)[v] \in T_{r(y)} \mathcal{S}_{M_H}$.
     */
    void diff_restriction(const Vector<double>& y_fine, const Vector<double>& v, Vector<double>& dst) const
    {
        // Linear restriction of the base point: I_h^H(y)
        Vector<double> x_lin(transfer.n_coarse());
        transfer.to_coarse_mesh(y_fine, x_lin);

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
     * @param x_coarse The base point $x \in \mathcal{S}_{M_H}$ on the coarse grid.
     * @param v The tangent vector $v \in T_x \mathcal{S}_{M_H}$.
     * @param dst The mapped tangent vector $\Drm p(x)[v] \in T_{p(x)} \mathcal{S}_{M_h}$.
     */
    void diff_prolongation(const Vector<double>& x_coarse, const Vector<double>& v, Vector<double>& dst) const
    {
        // 1. Linear prolongation of the base point: I_H^h(x)
        Vector<double> y_lin(transfer.n_fine());
        transfer.to_fine_mesh(x_coarse, y_lin);

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
    const LinearTransfer<dim>& transfer;
};


template <int dim>
class VectorTransportBase
{
public:
    virtual ~VectorTransportBase() = default;

    /**
     * @brief Prolongs a tangent vector from the coarse grid to the fine grid.
     * @param y_fine The actual fine iterate (defines the target tangent space).
     * @param x_coarse The coarse iterate (defines the source tangent space).
     * @param v_coarse The vector in T_x S_H to be prolonged.
     * @param dst_fine [out] The prolonged vector, guaranteed to be in T_y S_h.
     */
    virtual void vector_prolongation(const Vector<double>& y_fine,
                                     const Vector<double>& x_coarse,
                                     const Vector<double>& v_coarse,
                                     Vector<double>& dst_fine) const = 0;

    /**
     * @brief Restricts a tangent vector from the fine grid to the coarse grid.
     * @param x_coarse The actual coarse iterate (defines the target tangent space).
     * @param y_fine The fine iterate (defines the source tangent space).
     * @param v_fine The vector in T_y S_h to be restricted.
     * @param dst_coarse [out] The restricted vector, guaranteed to be in T_x S_H.
     */
    virtual void vector_restriction(const Vector<double>& x_coarse,
                                    const Vector<double>& y_fine,
                                    const Vector<double>& v_fine,
                                    Vector<double>& dst_coarse) const = 0;
};

/**
 * @brief Computes the coarse grid correction term for a multilevel cycle.
 *
 * In Riemannian multilevel optimization, the coarse grid problem is modified so that
 * its initial gradient matches the restricted fine grid gradient (first-order coherence.)
 * This function computes the correction vector $w_k$, which acts as a tilt to the coarse
 * energy landscape.
 *
 * Mathematically, the correction term is defined as the difference between the actual
 * coarse gradient and the restricted fine gradient:
 * $$w_k = \nabla_M E_H(x_H) - \mathcal{T}_{h \to H}(\nabla_M E_h(y_h))$$
 *
 * @tparam dim The spatial dimension of the problem.
 * @param transport The vector transport strategy used to move tangent vectors between grids.
 * @param x_grad_coarse The gradient of the coarse energy evaluated at the coarse base point, $\nabla_M E_H(x_H)$.
 * @param y_grad_fine The gradient of the fine energy evaluated at the fine base point, $\nabla_M E_h(y_h)$.
 * @param x_coarse The target coarse base point $x_H \in \mathcal{S}_H$.
 * @param y_fine The source fine base point $y_h \in \mathcal{S}_h$.
 * @param dst [out] The computed correction vector $w_k \in T_{x_H} \mathcal{S}_H$.
 */
template <int dim>
void coarse_correction(const VectorTransportBase<dim>& transport,
                       const Vector<double>& x_grad_coarse,
                       const Vector<double>& y_grad_fine,
                       const Vector<double>& x_coarse,
                       const Vector<double>& y_fine,
                       Vector<double>& dst)
{
    Vector<double> y_grad_restr(x_grad_coarse.size());
    // 3-input interface for projection + differentiation transports
    transport.vector_restriction(x_coarse, y_fine, y_grad_fine, y_grad_restr);

    dst = x_grad_coarse;
    dst.add(-1.0, y_grad_restr);
}


/**
 * @brief Strategy for vector transport via ambient space transfer and orthogonal projection.
 * This class implements the `VectorTransportBase` interface using the standard projection
 * strategy. Tangent vectors are first transferred as standard Euclidean vectors in the
 * ambient space $\mathbb{R}^n$, and then forcefully projected onto the target tangent
 * space using the mass-metric orthogonal projector.
 *
 * This strategy is highly efficient as it does not require computing the exact differentials
 * of the manifold restriction/prolongation maps.
 *
 * @note Because the ambient interpolation transfers the vector directly to the target space
 * grid, this strategy only requires the *target* base point to define the final tangent space.
 * The *source* base points in the overridden methods are ignored.
 *
 * @tparam dim The spatial dimension of the problem.
 * @tparam MatrixType The matrix type defining the ambient space metric (typically the Mass matrix).
 */
template <int dim, typename MatrixType>
class ProjectionTransport : public VectorTransportBase<dim>
{
public:
    ProjectionTransport(const MatrixType& M_c, const MatrixType& M_f,
                        const LinearTransfer<dim>& I,
                        const ManifoldTransfer<dim, MatrixType>& pt)
        : M_coarse(M_c), M_fine(M_f)
        , transfer(I), point_transfer(pt)
    {}

    /**
     * @brief Prolongs a tangent vector using ambient interpolation and M-metric projection.
     * Computes:
     * $$\mathcal{T}_{H \to h}(v) = P_{T_y \mathcal{S}_h} (I_H^h v)$$
     * where $I_H^h$ is the standard linear prolongation operator.
     */
    void vector_prolongation(const Vector<double>& y_fine, const Vector<double>&,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Tangent vector interpolated to fine ambient space
        Vector<double> Iv(transfer.n_fine());
        transfer.to_fine_mesh(v_coarse, Iv);

        // Orthogonal projection I(v \in T_x S_H) -> T_p(x) S_h in M-metric
        ellipsoid::project_onto_tangent_space(y_fine, M_fine, Iv, dst);
    }

    /**
     * @brief Restricts a tangent vector using ambient restriction and M-metric projection.
     * Computes:
     * $$\mathcal{T}_{h \to H}(v) = P_{T_x \mathcal{S}_H} (I_h^H v)$$
     * where $I_h^H$ is the standard linear restriction operator.
     */
    void vector_restriction(const Vector<double>& x_coarse, const Vector<double>&,
                            const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        // Tangent vector restricted to coarse ambient space
        Vector<double> Iv(transfer.n_coarse());
        transfer.to_coarse_mesh(v_fine, Iv);

        // Orthogonal projection I(v \in T_y S_h) -> T_r(y) S_H
        ellipsoid::project_onto_tangent_space(x_coarse, M_coarse, Iv, dst);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const LinearTransfer<dim>& transfer;
    const ManifoldTransfer<dim, MatrixType>& point_transfer;
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
template <int dim, typename MatrixType>
class DifferentialTransport : public VectorTransportBase<dim>
{
public:
    DifferentialTransport(const ManifoldTransfer<dim, MatrixType>& pt,
                          const MatrixType& M_f, const MatrixType& M_c)
        : point_transfer(pt), M_fine(M_f), M_coarse(M_c)
    {}

    void vector_prolongation(const Vector<double>& x_coarse, const Vector<double>& v_coarse,
                             Vector<double>& dst) const
    {
        point_transfer.diff_prolongation(x_coarse, v_coarse, dst);
    }

    /**
     * @brief Prolongs a tangent vector by computing the differential of the prolongation map.
     * This two-step method computes the differential and then performs a corrective
     * vector transport to the fine target space:
     * $$\hat{v} = D p(x_H)[v_H] \quad \in T_{p(x_H)} \mathcal{S}_h$$
     * $$\mathcal{T}_{H \to h}(v_H) = P_{T_y \mathcal{S}_h} (\hat{v})$$
     */
    void vector_prolongation(const Vector<double>& y_fine, const Vector<double>& x_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst) const override
    {
        // Differential D_p(x): T_x S_H -> T_p(x) S_h
        Vector<double> D_px(point_transfer.n_fine());
        vector_prolongation(x_coarse, v_coarse, D_px);

        // Vector transport T_p(x) S_h -> T_y S_h
        ellipsoid::project_onto_tangent_space(y_fine, M_fine, D_px, dst);
    }

    void vector_restriction(const Vector<double>& y_fine, const Vector<double>& v_fine,
                            Vector<double>& dst) const
    {
        point_transfer.diff_restriction(y_fine, v_fine, dst);
    }

    /**
     * @brief Restricts a tangent vector by computing the differential of the restriction map.
     * This two-step method computes the differential and then performs a corrective
     * vector transport to the coarse target space:
     * $$\hat{v} = D r(y_h)[v_h] \quad \in T_{r(y_h)} \mathcal{S}_H$$
     * $$\mathcal{T}_{h \to H}(v_h) = P_{T_x \mathcal{S}_H} (\hat{v})$$
     */
    void vector_restriction(const Vector<double>& x_coarse, const Vector<double>& y_fine,
        const Vector<double>& v_fine, Vector<double>& dst) const override
    {
        // Differential D_r(y): T_y S_h -> T_r(y) S_H
        Vector<double> D_ry(point_transfer.n_coarse());
        vector_restriction(y_fine, v_fine, D_ry);

        // Vector transport T_r(y) S_H -> T_x S_H
        ellipsoid::project_onto_tangent_space(x_coarse, M_coarse, D_ry, dst);
    }

private:
    const ManifoldTransfer<dim, MatrixType>& point_transfer;
    const MatrixType& M_fine;
    const MatrixType& M_coarse;
};


// Strategy C: Transport with Galerkin condition (M-metric)
template <int dim, typename MatrixType>
class GalerkinTransport : public VectorTransportBase<dim>
{
public:
    GalerkinTransport(const ManifoldTransfer<dim, MatrixType>& pt)
        : point_transfer(pt)
    {}

    void vector_prolongation(const Vector<double>&, const Vector<double>&,
                             const Vector<double>&, Vector<double>&) const override
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    void vector_restriction(const Vector<double>&, const Vector<double>&,
                            const Vector<double>&, Vector<double>&) const override
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

private:
    const ManifoldTransfer<dim, MatrixType>& point_transfer;
};


// Strategy D: Transport through pseudo-inverse
template <int dim, typename MatrixType>
class PseudoInvTransport : public VectorTransportBase<dim>
{
public:
    PseudoInvTransport(const MatrixType& M_c, const MatrixType& M_f,
                       const LinearTransfer<dim>& I,
                       const ManifoldTransfer<dim, MatrixType>& pt)
        : M_coarse(M_c), M_fine(M_f)
        , transfer(I), point_transfer(pt)
    {}

    void vector_prolongation(const Vector<double>&, const Vector<double>&,
                             const Vector<double>&, Vector<double>&) const override
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    void vector_restriction(const Vector<double>&, const Vector<double>&,
                            const Vector<double>&, Vector<double>&) const override
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;

    const LinearTransfer<dim>& transfer;
    const ManifoldTransfer<dim, MatrixType>& point_transfer;
};

} // namespace gpe

#endif //GPE_GRID_OPERATORS_H