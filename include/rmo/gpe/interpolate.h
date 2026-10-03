//
// Created by alad on 9/9/26.
//

#ifndef RMO_GPE_INTERPOLATE_H
#define RMO_GPE_INTERPOLATE_H

#include <rmo/fe/interpolate.h>

namespace rmo::gpe
{

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

    /**
     * @brief Computes mass-weighted restriction: I_h^H = M_H^{-1} * (I_H^h)^T * M_h
     */
    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        // 1. Multiply by fine mass matrix: M_h * v_h
        Vector<double> Mh_v(_transfer.n_fine());
        _M_fine.vmult(Mh_v, src_fine);

        // 2. Apply TRANSPOSE of prolongation: (I_H^h)^T * (M_h * v_h)
        Vector<double> Ih_Mh_v(_transfer.n_coarse());
        _transfer.Tfine(Mh_v, Ih_Mh_v); // FIXED: Was _transfer.to_coarse_mesh

        // 3. Apply inverse coarse mass matrix
        _M_inv_coarse.vmult(dst_coarse, Ih_Mh_v);
    }

    /**
     * @brief Prolongation remains unchanged: I_H^h
     */
    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        _transfer.to_fine_mesh(src_coarse, dst_fine);
    }

    /**
     * @brief Transpose of prolongation remains unchanged: (I_H^h)^T
     */
    void Tfine(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        _transfer.Tfine(src_fine, dst_coarse);
    }

    /**
     * @brief Transpose of mass-weighted restriction: (I_h^H)^T = M_h * I_H^h * M_H^{-1}
     */
    void Tcoarse(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        // 1. Multiply by inverse coarse mass matrix: M_H^{-1} * v_H
        Vector<double> MH_inv_v(_transfer.n_coarse());
        _M_inv_coarse.vmult(MH_inv_v, src_coarse);

        // 2. Apply FORWARD prolongation: I_H^h * (M_H^{-1} * v_H)
        Vector<double> TIh_MH_inv_v(_transfer.n_fine());
        _transfer.to_fine_mesh(MH_inv_v, TIh_MH_inv_v); // FIXED: Was _transfer.Tcoarse

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
