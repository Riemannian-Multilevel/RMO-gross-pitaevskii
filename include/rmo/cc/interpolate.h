//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_INTERPOLATE_H
#define RMO_CC_INTERPOLATE_H

#include <rmo/cc/grid.h>
#include <rmo/ropt/interpolate.h>

namespace rmo::cc
{

/**
 * @brief Factor-2 grid transfer between a coarse and fine @ref PixelGrid (\S 6.3.2),
 * requiring @f$ \mathrm{fine.rows()} = 2\cdot\mathrm{coarse.rows()}-1 @f$ and likewise
 * for columns (one global doubling in each dimension, as for a uniformly refined mesh).
 *
 * @ref to_coarse_mesh is pointwise injection @f$ J_h^H @f$; @ref to_fine_mesh is bilinear
 * interpolation @f$ B_H^h @f$; @ref Tfine is its exact transpose, @f$ 4 F_h^H @f$, since
 * @f$ F_h^H = \tfrac14 (B_H^h)^\top @f$.
 *
 * Under the doubling precondition, every fine index's coarse neighbor(s) are always in
 * range, so no boundary special-casing is needed beyond the row/column parity itself.
 */
class BernoulliGridTransfer : public LinearTransferBase
{
public:
    BernoulliGridTransfer(const PixelGrid& coarse, const PixelGrid& fine)
        : m_coarse(coarse), m_fine(fine)
    {
        AssertDimension(fine.rows(), 2 * coarse.rows() - 1);
        AssertDimension(fine.cols(), 2 * coarse.cols() - 1);
    }

    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        dst_coarse.reinit(m_coarse.n_dofs());
        for (unsigned int I = 0; I < m_coarse.rows(); I++) {
            for (unsigned int J = 0; J < m_coarse.cols(); J++) {
                dst_coarse[m_coarse.dof_index(I, J)] = src_fine[m_fine.dof_index(2 * I, 2 * J)];
            }
        }
    }

    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        dst_fine.reinit(m_fine.n_dofs());
        for (unsigned int i = 0; i < m_fine.rows(); i++) {
            for (unsigned int j = 0; j < m_fine.cols(); j++) {
                dst_fine[m_fine.dof_index(i, j)] = bilinear_at(src_coarse, i, j);
            }
        }
    }

    /** @brief Exact transpose of @ref to_fine_mesh (scatter form of the same stencil). */
    void Tfine(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        dst_coarse.reinit(m_coarse.n_dofs());
        dst_coarse = 0.0;

        for (unsigned int i = 0; i < m_fine.rows(); i++) {
            for (unsigned int j = 0; j < m_fine.cols(); j++) {
                scatter_bilinear(src_fine[m_fine.dof_index(i, j)], i, j, dst_coarse);
            }
        }
    }

    [[nodiscard]] unsigned int n_coarse() const override { return m_coarse.n_dofs(); }
    [[nodiscard]] unsigned int n_fine() const override { return m_fine.n_dofs(); }

private:
    // The three cases of standard bilinear interpolation, stated directly on (row, col)
    // parity: a fine index that is even in both coordinates coincides with a coarse point
    // (weight 1); even in one coordinate lies on a coarse edge midpoint (weight 1/2, 1/2);
    // odd in both lies at a coarse cell center (weight 1/4 each).
    [[nodiscard]] double bilinear_at(const Vector<double>& coarse, unsigned int i, unsigned int j) const
    {
        const unsigned int I0 = i / 2, J0 = j / 2;

        if (i % 2 == 0 && j % 2 == 0) {
            return coarse[m_coarse.dof_index(I0, J0)];
        }
        if (i % 2 == 1 && j % 2 == 0) {
            return 0.5 * (coarse[m_coarse.dof_index(I0, J0)] + coarse[m_coarse.dof_index(I0 + 1, J0)]);
        }
        if (i % 2 == 0 && j % 2 == 1) {
            return 0.5 * (coarse[m_coarse.dof_index(I0, J0)] + coarse[m_coarse.dof_index(I0, J0 + 1)]);
        }
        return 0.25 * (coarse[m_coarse.dof_index(I0, J0)]     + coarse[m_coarse.dof_index(I0 + 1, J0)]
                      + coarse[m_coarse.dof_index(I0, J0 + 1)] + coarse[m_coarse.dof_index(I0 + 1, J0 + 1)]);
    }

    void scatter_bilinear(double v, unsigned int i, unsigned int j, Vector<double>& dst_coarse) const
    {
        const unsigned int I0 = i / 2, J0 = j / 2;

        if (i % 2 == 0 && j % 2 == 0) {
            dst_coarse[m_coarse.dof_index(I0, J0)] += v;
        } else if (i % 2 == 1 && j % 2 == 0) {
            dst_coarse[m_coarse.dof_index(I0, J0)]     += 0.5 * v;
            dst_coarse[m_coarse.dof_index(I0 + 1, J0)] += 0.5 * v;
        } else if (i % 2 == 0 && j % 2 == 1) {
            dst_coarse[m_coarse.dof_index(I0, J0)]     += 0.5 * v;
            dst_coarse[m_coarse.dof_index(I0, J0 + 1)] += 0.5 * v;
        } else {
            dst_coarse[m_coarse.dof_index(I0, J0)]         += 0.25 * v;
            dst_coarse[m_coarse.dof_index(I0 + 1, J0)]     += 0.25 * v;
            dst_coarse[m_coarse.dof_index(I0, J0 + 1)]     += 0.25 * v;
            dst_coarse[m_coarse.dof_index(I0 + 1, J0 + 1)] += 0.25 * v;
        }
    }

    const PixelGrid& m_coarse;
    const PixelGrid& m_fine;
};

} // namespace rmo::cc

#endif //RMO_CC_INTERPOLATE_H
