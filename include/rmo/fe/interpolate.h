//
// Created by Ferdinand Vanmaele on 05.04.26.
//

#ifndef RMO_FE_INTERPOLATE_H
#define RMO_FE_INTERPOLATE_H

#include <rmo/ropt/interpolate.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

namespace rmo::fe
{

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


} // namespace rmo::fe
#endif //RMO_FE_INTERPOLATE_H
