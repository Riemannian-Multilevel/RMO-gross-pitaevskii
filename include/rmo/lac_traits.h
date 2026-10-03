#ifndef RMO_LAC_TRAITS_H
#define RMO_LAC_TRAITS_H

#include <deal.II/lac/diagonal_matrix.h>

namespace rmo
{

/** @brief True for `dealii::DiagonalMatrix<VectorType>`, i.e. matrices stored as their diagonal. */
template <typename MatrixType>
inline constexpr bool is_diagonal_matrix_v = false;

template <typename VectorType>
inline constexpr bool is_diagonal_matrix_v<dealii::DiagonalMatrix<VectorType>> = true;

} // namespace rmo

#endif //RMO_LAC_TRAITS_H
