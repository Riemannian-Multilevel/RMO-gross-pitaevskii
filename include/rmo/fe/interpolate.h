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

/**
 * @file
 * @brief Linear grid transfers between two nested meshes: LinearTransfer (deal.II mesh interpolation) and
 * LinearTransferMG (deal.II multigrid transfer, with the exact transpose Tfine()).
 */
namespace rmo::fe
{

/** @brief Grid transfer by deal.II's interpolation to coarser and finer meshes. */
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
 * @brief Grid transfer by deal.II's multigrid transfer (MGTwoLevelTransfer).
 *
 * - to_fine_mesh() = `prolongate_and_add()`: the embedding \f$ I_H^h \f$;
 * - to_coarse_mesh() = `interpolate()`: restriction of solution vectors (injection at the coarse nodes);
 * - Tfine() = `restrict_and_add()`: the exact transpose \f$ (I_H^h)^\top \f$.
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

    /** @brief \f$ v_h = I_H^h v_H \f$, then distributes the fine constraints. */
    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        DVector src(n_c), dst(n_f);
        std::copy(src_coarse.begin(), src_coarse.end(), src.begin());

        transfer.prolongate_and_add(dst, src);

        dst_fine.reinit(n_f);
        std::copy(dst.begin(), dst.end(), dst_fine.begin());
        constraints_f.distribute(dst_fine);
    }

    /** @brief Injection at the coarse nodes, then distributes the coarse constraints. */
    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        DVector src(n_f), dst(n_c);
        std::copy(src_fine.begin(), src_fine.end(), src.begin());

        transfer.interpolate(dst, src);

        dst_coarse.reinit(n_c);
        std::copy(dst.begin(), dst.end(), dst_coarse.begin());
        constraints_c.distribute(dst_coarse);
    }

    /** @brief \f$ (I_H^h)^\top v_h \f$, the transpose of to_fine_mesh(). */
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
