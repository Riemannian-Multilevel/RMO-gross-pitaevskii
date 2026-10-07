//
// Created by alad on 9/9/26.
//

#ifndef RMO_GPE_INTERPOLATE_H
#define RMO_GPE_INTERPOLATE_H

#include <rmo/fe/interpolate.h>

/**
 * @file
 * @brief MassTransfer: grid transfer with the mass-weighted (L2-projection) restriction.
 */
namespace rmo::gpe
{

/**
 * @brief Grid transfer with the restriction \f$ I_h^H = M_H^{-1} (I_H^h)^\top M_h \f$, the L2 projection onto the
 * coarse space; the prolongation \f$ I_H^h \f$ is that of @p TransferType (e.g. fe::LinearTransferMG).
 */
template <int dim, typename TransferType, typename MatrixType, typename InverseMatrixType>
class MassTransfer : public LinearTransferBase
{
public:
    MassTransfer(const dealii::DoFHandler<dim>& dof_coarse,
                 const dealii::DoFHandler<dim>& dof_fine,
                 const dealii::AffineConstraints<double>& constraints_coarse,
                 const dealii::AffineConstraints<double>& constraints_fine,
                 const MatrixType& M_fine,
                 const InverseMatrixType& M_inv_coarse)
        : _transfer(TransferType(dof_coarse, dof_fine, constraints_coarse, constraints_fine))
        , _M_fine(M_fine)
        , _M_inv_coarse(M_inv_coarse)
    {}

    /** @brief \f$ v_H = M_H^{-1} (I_H^h)^\top M_h v_h \f$ */
    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        // 1. Multiply by fine mass matrix: M_h * v_h
        Vector<double> Mh_v(_transfer.n_fine());
        _M_fine.vmult(Mh_v, src_fine);

        // 2. Apply TRANSPOSE of prolongation: (I_H^h)^T * (M_h * v_h)
        Vector<double> Ih_Mh_v(_transfer.n_coarse());
        _transfer.Tfine(Mh_v, Ih_Mh_v);

        // 3. Apply inverse coarse mass matrix
        _M_inv_coarse.vmult(dst_coarse, Ih_Mh_v);
    }

    /** @brief \f$ v_h = I_H^h v_H \f$ */
    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        _transfer.to_fine_mesh(src_coarse, dst_fine);
    }

    /** @brief \f$ (I_H^h)^\top v_h \f$ */
    void Tfine(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        _transfer.Tfine(src_fine, dst_coarse);
    }

    /** @brief Transpose of the restriction, \f$ (I_h^H)^\top v_H = M_h I_H^h M_H^{-1} v_H \f$. */
    void Tcoarse(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        // 1. Multiply by inverse coarse mass matrix: M_H^{-1} * v_H
        Vector<double> MH_inv_v(_transfer.n_coarse());
        _M_inv_coarse.vmult(MH_inv_v, src_coarse);

        // 2. Apply FORWARD prolongation: I_H^h * (M_H^{-1} * v_H)
        Vector<double> TIh_MH_inv_v(_transfer.n_fine());
        _transfer.to_fine_mesh(MH_inv_v, TIh_MH_inv_v);

        // 3. Multiply by fine mass matrix
        _M_fine.vmult(dst_fine, TIh_MH_inv_v);
    }

    unsigned n_coarse() const override { return _transfer.n_coarse(); };
    unsigned n_fine() const override { return _transfer.n_fine(); };

private:
    TransferType _transfer;
    const MatrixType& _M_fine;
    const InverseMatrixType& _M_inv_coarse;
};

} // namespace rmo::gpe

#endif //RMO_GPE_INTERPOLATE_H
