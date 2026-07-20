//
// Created by Ferdinand Vanmaele on 05.04.26.
//

#ifndef GPE_FE_INTERPOLATE_H
#define GPE_FE_INTERPOLATE_H

#include <gpe/lac.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace gpe
{

class LinearTransferBase
{
public:
    virtual ~LinearTransferBase() = default;

    // TODO: match vmult() interface (output&, const input&)
    virtual void to_coarse_mesh(const Vector<double>&, Vector<double>&) const = 0;

    virtual void to_fine_mesh(const Vector<double>&, Vector<double>&) const = 0;

    // Transpose of to_coarse_mesh() (for matrix implementations)
    virtual void Tcoarse(const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    // Transpose of to_fine_mesh()
    virtual void Tfine(const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    virtual unsigned n_coarse() const = 0;
    virtual unsigned n_fine() const = 0;
};


// Linear interpolation using deal.ii mesh interpolation
template <int dim>
class LinearTransfer : public LinearTransferBase
{
public:
    LinearTransfer(const dealii::DoFHandler<dim>& dof_c,
                   const dealii::DoFHandler<dim>& dof_f,
                   const dealii::AffineConstraints<double>& aff_c,
                   const dealii::AffineConstraints<double>& aff_f)
        : dof_coarse(dof_c), dof_fine(dof_f), aff_coarse(aff_c), aff_fine(aff_f)
    {}

    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        dst_coarse.reinit(dof_coarse.n_dofs());
        dst_coarse = 0.0;

        dealii::VectorTools::interpolate_to_coarser_mesh(dof_fine,
            src_fine, dof_coarse, aff_coarse, dst_coarse);
    }

    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        dst_fine.reinit(dof_fine.n_dofs());
        dst_fine = 0.0;

        dealii::VectorTools::interpolate_to_finer_mesh(dof_coarse,
            src_coarse, dof_fine, aff_fine, dst_fine);
    }

    unsigned n_coarse() const override { return dof_coarse.n_dofs(); }
    unsigned n_fine() const override { return dof_fine.n_dofs(); }

private:
    const dealii::DoFHandler<dim>& dof_coarse;
    const dealii::DoFHandler<dim>& dof_fine;
    const dealii::AffineConstraints<double>& aff_coarse;
    const dealii::AffineConstraints<double>& aff_fine;
};


/**
 * @brief Linear transfer built on deal.II's geometric global-coarsening
 * multigrid infrastructure (MGTwoLevelTransfer).
 *
 * Operator mapping (cf. the class documentation of MGTwoLevelTransferBase,
 * which distinguishes restriction of "right hand side vectors" from
 * interpolation of "solution vectors"):
 *  - to_fine_mesh()   = prolongate_and_add()  (FE embedding I_H^h)
 *  - to_coarse_mesh() = interpolate()         (solution/primal restriction,
 *                                              the FAS injection r)
 *  - Tfine()          = restrict_and_add()    (exact transpose (I_H^h)^T)
 */
template <int dim>
class LinearTransferMG : public LinearTransferBase
{
    using DVector = dealii::LinearAlgebra::distributed::Vector<double>;

public:
    LinearTransferMG(const dealii::DoFHandler<dim>& dof_coarse,
                     const dealii::DoFHandler<dim>& dof_fine,
                     const dealii::AffineConstraints<double>& constraints_coarse,
                     const dealii::AffineConstraints<double>& constraints_fine)
        : n_c(dof_coarse.n_dofs())
        , n_f(dof_fine.n_dofs())
        , constraints_c(constraints_coarse)
        , constraints_f(constraints_fine)
    {
        transfer.reinit(dof_fine, dof_coarse, constraints_fine, constraints_coarse);
    }

    /**
     * @brief Prolongates a vector from the coarse mesh to the fine mesh.
     * Evaluates $v_{fine} = I_H^h \cdot v_{coarse}$.
     */
    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        DVector src(n_c), dst(n_f);
        std::copy(src_coarse.begin(), src_coarse.end(), src.begin());

        transfer.prolongate_and_add(dst, src);

        dst_fine.reinit(n_f);
        std::copy(dst.begin(), dst.end(), dst_fine.begin());
        constraints_f.distribute(dst_fine);
    }

    /**
     * @brief Restricts a vector from the fine mesh to the coarse mesh via
     * solution interpolation (pointwise injection at coarse nodes for nested
     * Lagrange elements), matching the point-restriction map r.
     */
    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        DVector src(n_f), dst(n_c);
        std::copy(src_fine.begin(), src_fine.end(), src.begin());

        transfer.interpolate(dst, src);

        dst_coarse.reinit(n_c);
        std::copy(dst.begin(), dst.end(), dst_coarse.begin());
        constraints_c.distribute(dst_coarse);
    }

    /**
     * @brief Transpose of the prolongation, $(I_H^h)^T$ (multigrid residual
     * restriction). Exact adjoint of to_fine_mesh() w.r.t. the Euclidean
     * pairing by construction.
     */
    void Tfine(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        DVector src(n_f), dst(n_c);
        std::copy(src_fine.begin(), src_fine.end(), src.begin());

        transfer.restrict_and_add(dst, src);

        dst_coarse.reinit(n_c);
        std::copy(dst.begin(), dst.end(), dst_coarse.begin());
    }

    unsigned int n_coarse() const override { return n_c; }
    unsigned int n_fine() const override { return n_f; }

private:
    unsigned int n_c;
    unsigned int n_f;
    const dealii::AffineConstraints<double>& constraints_c;
    const dealii::AffineConstraints<double>& constraints_f;

    dealii::MGTwoLevelTransfer<dim, DVector> transfer;
};


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

} // namespace gpe

#endif //GPE_FE_INTERPOLATE_H
