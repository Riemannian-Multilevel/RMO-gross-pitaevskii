//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_GRID_H
#define RMO_CC_GRID_H

#include <rmo/lac.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace rmo::cc
{

/**
 * @brief An @f$ m \times l @f$ pixel grid (\S 6.3.1). Pixels are numbered in row-major
 * (raster) order, and that is the only order used throughout the `cc` module -- there is
 * no separate "DoF order" to convert to or from. An earlier version of this class built one
 * via a throwaway deal.II @c FE_Q(1) mesh, so the grid transfer operators of \S 6.3.2 could
 * reuse deal.II's own finite-element mesh-transfer infrastructure; that infrastructure is
 * no longer used there (@ref BernoulliGridTransfer implements injection/bilinear/full-weighting
 * directly, as plain index arithmetic -- see its own documentation), so the indirection,
 * and the deal.II mesh dependency it required, were removed. @ref to_dof_order and
 * @ref to_raster_order are kept as the identity so callers written against the earlier,
 * two-order version do not need to change.
 */
class PixelGrid
{
public:
    /**
     * @param rows Number of pixel rows (image height).
     * @param cols Number of pixel columns (image width).
     */
    PixelGrid(unsigned int rows, unsigned int cols)
        : m_rows(rows), m_cols(cols)
    {
        AssertThrow(rows > 1 && cols > 1, dealii::ExcMessage("grid must have at least 2x2 pixels"));
    }

    [[nodiscard]] unsigned int rows() const { return m_rows; }
    [[nodiscard]] unsigned int cols() const { return m_cols; }
    [[nodiscard]] unsigned int n_dofs() const { return m_rows * m_cols; }

    /** @brief Row-major index of the pixel at (row, col). */
    [[nodiscard]] unsigned int dof_index(unsigned int row, unsigned int col) const
    {
        AssertIndexRange(row, m_rows);
        AssertIndexRange(col, m_cols);
        return row * m_cols + col;
    }

    /** @brief Identity (see class documentation). */
    [[nodiscard]] Vector<double> to_dof_order(const Vector<double>& raster) const
    {
        AssertDimension(raster.size(), n_dofs());
        return raster;
    }

    /** @brief Identity (see class documentation). */
    [[nodiscard]] Vector<double> to_raster_order(const Vector<double>& dof_vector) const
    {
        AssertDimension(dof_vector.size(), n_dofs());
        return dof_vector;
    }

private:
    unsigned int m_rows, m_cols;
};


/**
 * @brief Assembles the forward finite-difference gradient operator
 * @f$ D = [D_1; D_2] \in \mathbb{R}^{2n \times n} @f$, stacking horizontal
 * (@f$ D_1 @f$) and vertical (@f$ D_2 @f$) differences (eq. (42), \S 6.3.1).
 *
 * Row @f$ i @f$ of @f$ D_1 @f$ and row @f$ i @f$ of @f$ D_2 @f$ (equivalently,
 * row @f$ n+i @f$ of @f$ D @f$) both use the same column indexing as the
 * identity, i.e. rows and columns alike follow @ref PixelGrid::dof_index.
 */
class ForwardDifference
{
public:
    explicit ForwardDifference(const PixelGrid& grid)
        : m_grid(grid)
    {
        const unsigned int n = grid.n_dofs();

        dealii::DynamicSparsityPattern dsp(2 * n, n);
        for (unsigned int row = 0; row < grid.rows(); row++) {
            for (unsigned int col = 0; col < grid.cols(); col++) {
                const unsigned int i = grid.dof_index(row, col);

                dsp.add(i, i);                                    // D1 row i, diagonal
                if (col + 1 < grid.cols()) {
                    dsp.add(i, grid.dof_index(row, col + 1));     // D1 row i, forward neighbor
                }
                dsp.add(n + i, i);                                 // D2 row n+i, diagonal
                if (row + 1 < grid.rows()) {
                    dsp.add(n + i, grid.dof_index(row + 1, col)); // D2 row n+i, forward neighbor
                }
            }
        }
        m_sparsity.copy_from(dsp);
        m_D.reinit(m_sparsity);

        for (unsigned int row = 0; row < grid.rows(); row++) {
            for (unsigned int col = 0; col < grid.cols(); col++) {
                const unsigned int i = grid.dof_index(row, col);

                if (col + 1 < grid.cols()) {
                    const unsigned int i_right = grid.dof_index(row, col + 1);
                    m_D.set(i, i, -1.0);
                    m_D.set(i, i_right, 1.0);
                }
                // else: last column, D1 row i stays zero (Neumann boundary)

                if (row + 1 < grid.rows()) {
                    const unsigned int i_down = grid.dof_index(row + 1, col);
                    m_D.set(n + i, i, -1.0);
                    m_D.set(n + i, i_down, 1.0);
                }
                // else: last row, D2 row n+i stays zero (Neumann boundary)
            }
        }
    }

    [[nodiscard]] const SparseMatrix<double>& matrix() const { return m_D; }
    [[nodiscard]] unsigned int n_rows() const { return 2 * m_grid.n_dofs(); }
    [[nodiscard]] unsigned int n_cols() const { return m_grid.n_dofs(); }

private:
    const PixelGrid& m_grid;
    SparsityPattern m_sparsity;
    SparseMatrix<double> m_D;
};

} // namespace rmo::cc

#endif //RMO_CC_GRID_H
