#ifndef GPE_GRID_OPERATORS_H
#define GPE_GRID_OPERATORS_H
#include "option_types.h"
#include "manifold.h"

#include <deal.II/numerics/vector_tools.h>

namespace gpe
{

enum class VectorTransportKind
{
    PROJECTION,       // composition of orthogonal projection and linear interpolation
    DIFF_ADJOINT_R,   // metric adjoint w.r.t. A/M-metric, starting from prolongation map  p -> Dp -> Dp* , Dr
    PINV_ADJOINT_R,   // as above, but taking the pseudo-inverse Dp+
    DIFF_ADJOINT_P,   // metric adjoint w.r.t. A/M-metric, starting from restriction map   r -> Dr -> Dr* , Dp
    PINV_ADJOINT_P,   // as above, but taking the pseudo-inverse Dr+
};

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

private:
    const MatrixType& M_coarse;
    const MatrixType& M_fine;
    const LinearTransfer<dim>& transfer;
};


template <int dim>
class VectorTransfer
{
public:
    VectorTransfer(const LinearTransfer<dim>& transfer, VectorTransportKind vtk)
        : m_transfer(transfer), m_vtk(vtk)
    {}

    template <typename MatrixType>
    void vector_prolongation(const Vector<double>& v)
    {
        Vector<double> Pv(transfer.n_fine());
        transfer.to_fine_mesh(v, Pv);

        switch (m_vtk) {
            case VectorTransportKind::PROJECTION:
                break;
            case VectorTransportKind::DIFF_ADJOINT_R:
                break;
            case VectorTransportKind::DIFF_ADJOINT_P:
                break;
            default:
                throw dealii::ExcNotImplemented("Unknown transport kind");
        }
    }

    template <typename MatrixType>
    void vector_restriction(const Vector<double>& v)
    {
        Vector<double> Rv(transfer.n_coarse());
        transfer.to_coarse_mesh(v, Rv);

        switch (m_vtk) {
            case VectorTransportKind::PROJECTION:
                break;
            case VectorTransportKind::DIFF_ADJOINT_R:
                break;
            case VectorTransportKind::DIFF_ADJOINT_P:
                break;
            default:
                throw dealii::ExcNotImplemented("Unknown transport kind");
        }
    }

private:
    const LinearTransfer<dim>& m_transfer;
    VectorTransferKind m_vtk;
};

} // namespace gpe

#endif //GPE_GRID_OPERATORS_H