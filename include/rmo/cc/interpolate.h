//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_INTERPOLATE_H
#define RMO_CC_INTERPOLATE_H

#include <rmo/cc/grid.h>
#include <rmo/ropt/interpolate.h>

#include <vector>

namespace rmo::cc
{

/**
 * @brief Factor-2 grid transfer between a coarse and fine @ref PixelGrid (\S 6.3.2),
 * requiring @f$ \mathrm{coarse.rows()} = \lceil\mathrm{fine.rows()}/2\rceil @f$ and likewise
 * for columns -- i.e. @f$ \mathrm{fine.rows()} \in \{2\cdot\mathrm{coarse.rows()}-1,\,
 * 2\cdot\mathrm{coarse.rows()}\} @f$, one global doubling in each dimension, matching the
 * reference's own @c r_injection (a plain @c [::2,::2] slice, which yields exactly this
 * many samples for either parity of @c fine).
 *
 * @ref to_coarse_mesh is pointwise injection @f$ J_h^H @f$; @ref to_fine_mesh is bilinear
 * interpolation @f$ B_H^h @f$; @ref Tfine is its exact transpose, @f$ 4 F_h^H @f$, since
 * @f$ F_h^H = \tfrac14 (B_H^h)^\top @f$.
 *
 * For an odd @c fine size (one exact mesh-refinement step, @c fine=2*coarse-1) every fine
 * index's coarse neighbor(s) are always in range. For an even @c fine size (the reference's
 * own image-pyramid convention, @c fine=2*coarse, as used for e.g. a 960x1280 fine / 240x320
 * composed-coarse pair) the last row/column's "+1" neighbor falls one past the coarse grid's
 * edge; @ref bilinear_at and @ref scatter_bilinear drop that term (zero-padded, matching the
 * reference's @c conv2d(..., padding=1) boundary convention) rather than renormalizing the
 * remaining weights -- verified against @c prolong_bilinear on a 2x2 coarse / 4x4 fine case
 * (test_interpolate.cc), including at the corner where only one of four terms survives.
 */
class BernoulliGridTransfer : public LinearTransferBase
{
public:
    BernoulliGridTransfer(const PixelGrid& coarse, const PixelGrid& fine)
        : m_coarse(coarse), m_fine(fine)
    {
        AssertDimension(coarse.rows(), (fine.rows() + 1) / 2);
        AssertDimension(coarse.cols(), (fine.cols() + 1) / 2);
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
    // (weight 1, I0/J0 always in range -- see the class documentation); even in one
    // coordinate lies on a coarse edge midpoint (weight 1/2, 1/2); odd in both lies at a
    // coarse cell center (weight 1/4 each). The "+1" neighbor(s) can fall one past the
    // coarse grid's last row/column when fine is even-sized (fine=2*coarse exactly); has_ir1
    // / has_jc1 gate those terms, dropped rather than renormalized (zero-padded, matching
    // the reference's conv2d boundary convention -- see the class documentation).
    [[nodiscard]] double bilinear_at(const Vector<double>& coarse, unsigned int i, unsigned int j) const
    {
        const unsigned int I0 = i / 2, J0 = j / 2;
        const bool has_i1 = (I0 + 1 < m_coarse.rows()), has_j1 = (J0 + 1 < m_coarse.cols());

        if (i % 2 == 0 && j % 2 == 0) {
            return coarse[m_coarse.dof_index(I0, J0)];
        }
        if (i % 2 == 1 && j % 2 == 0) {
            double s = coarse[m_coarse.dof_index(I0, J0)];
            if (has_i1) s += coarse[m_coarse.dof_index(I0 + 1, J0)];
            return 0.5 * s;
        }
        if (i % 2 == 0 && j % 2 == 1) {
            double s = coarse[m_coarse.dof_index(I0, J0)];
            if (has_j1) s += coarse[m_coarse.dof_index(I0, J0 + 1)];
            return 0.5 * s;
        }
        double s = coarse[m_coarse.dof_index(I0, J0)];
        if (has_i1) s += coarse[m_coarse.dof_index(I0 + 1, J0)];
        if (has_j1) s += coarse[m_coarse.dof_index(I0, J0 + 1)];
        if (has_i1 && has_j1) s += coarse[m_coarse.dof_index(I0 + 1, J0 + 1)];
        return 0.25 * s;
    }

    void scatter_bilinear(double v, unsigned int i, unsigned int j, Vector<double>& dst_coarse) const
    {
        const unsigned int I0 = i / 2, J0 = j / 2;
        const bool has_i1 = (I0 + 1 < m_coarse.rows()), has_j1 = (J0 + 1 < m_coarse.cols());

        if (i % 2 == 0 && j % 2 == 0) {
            dst_coarse[m_coarse.dof_index(I0, J0)] += v;
        } else if (i % 2 == 1 && j % 2 == 0) {
            dst_coarse[m_coarse.dof_index(I0, J0)] += 0.5 * v;
            if (has_i1) dst_coarse[m_coarse.dof_index(I0 + 1, J0)] += 0.5 * v;
        } else if (i % 2 == 0 && j % 2 == 1) {
            dst_coarse[m_coarse.dof_index(I0, J0)] += 0.5 * v;
            if (has_j1) dst_coarse[m_coarse.dof_index(I0, J0 + 1)] += 0.5 * v;
        } else {
            dst_coarse[m_coarse.dof_index(I0, J0)] += 0.25 * v;
            if (has_i1) dst_coarse[m_coarse.dof_index(I0 + 1, J0)] += 0.25 * v;
            if (has_j1) dst_coarse[m_coarse.dof_index(I0, J0 + 1)] += 0.25 * v;
            if (has_i1 && has_j1) dst_coarse[m_coarse.dof_index(I0 + 1, J0 + 1)] += 0.25 * v;
        }
    }

    const PixelGrid& m_coarse;
    const PixelGrid& m_fine;
};

/**
 * @brief Composes @c n one-factor-2 @ref BernoulliGridTransfer hops (finest first, coarsest
 * last) into a single factor-@f$2^n@f$ transfer, matching the reference's
 * @c operators.make_composed_ops on the geometric (metric-free) primitives it wraps:
 * @c to_coarse_mesh and @ref Tfine walk fine-to-coarse, applying each hop in order (the same
 * direction @c R_n composes in, since Option 1's @c R_op ignores its @c phi/psi arguments and
 * is exactly @ref BernoulliGridTransfer::Tfine per hop); @ref to_fine_mesh walks the hops in
 * reverse. Used for a coarse level that skips several discretizations at once (e.g. the
 * paper's 960x1280 -> 240x320 pair, two average-pool steps composed into one coarse-model
 * transition) without exposing an intermediate level to @ref rmo::FullApproximationScheme --
 * see @ref ComposedVectorTransport for the corresponding vector transport, which needs the
 * intermediate points @ref ComposedGridTransfer itself has no reason to keep around.
 */
class ComposedGridTransfer : public LinearTransferBase
{
public:
    explicit ComposedGridTransfer(std::vector<const BernoulliGridTransfer*> hops)
        : m_hops(std::move(hops))
    {
        Assert(!m_hops.empty(), dealii::ExcMessage("at least one hop required"));
    }

    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        Vector<double> curr = src_fine;
        for (const auto* hop : m_hops) {
            Vector<double> next;
            hop->to_coarse_mesh(curr, next);
            curr = std::move(next);
        }
        dst_coarse = std::move(curr);
    }

    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        Vector<double> curr = src_coarse;
        for (auto it = m_hops.rbegin(); it != m_hops.rend(); ++it) {
            Vector<double> next;
            (*it)->to_fine_mesh(curr, next);
            curr = std::move(next);
        }
        dst_fine = std::move(curr);
    }

    void Tfine(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        Vector<double> curr = src_fine;
        for (const auto* hop : m_hops) {
            Vector<double> next;
            hop->Tfine(curr, next);
            curr = std::move(next);
        }
        dst_coarse = std::move(curr);
    }

    [[nodiscard]] unsigned int n_coarse() const override { return m_hops.back()->n_coarse(); }
    [[nodiscard]] unsigned int n_fine()   const override { return m_hops.front()->n_fine(); }

    [[nodiscard]] const std::vector<const BernoulliGridTransfer*>& hops() const { return m_hops; }

private:
    std::vector<const BernoulliGridTransfer*> m_hops;
};

} // namespace rmo::cc

#endif //RMO_CC_INTERPOLATE_H
