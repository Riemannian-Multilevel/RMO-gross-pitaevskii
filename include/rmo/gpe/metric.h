#ifndef RMO_GPE_METRIC_H
#define RMO_GPE_METRIC_H

#include <rmo/ropt/manifold.h>
#include <rmo/util/random.h>

namespace rmo::gpe
{

namespace metric::energy
{

/**
 * @brief Projects a vector @p v onto the tangent space at @p x using an energy-based metric.
 *
 * This computes the projection orthogonal to the gradient of the energy functional.
 *
 * Formula:
 * \f[
 * \Pi_x(v) = v - \frac{\langle x, v \rangle_M}{\langle x, M A^{-1} M x \rangle} A^{-1} M x
 * \f]
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @tparam InverseMatrixType A solver or operator class representing \f$ A^{-1} \f$.
 * @param[in] A_inv The inverse operator \f$ A^{-1} \f$.
 * @param[in] x The base point.
 * @param[in] M The mass matrix.
 * @param[in] v The vector to be projected.
 * @param[out] output The resulting projected vector.
 * @param ignore_positivity_constraint
 */
template <typename MatrixType, typename InverseMatrixType>
void project_onto_tangent_space(const InverseMatrixType& A_inv,
                                const Vector<double>& x, const MatrixType& M, const Vector<double>& v,
                                Vector<double>& output,
                                bool ignore_positivity_constraint = false)
{
    AssertDimension(x.size(), v.size());
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    Vector<double> Ainv_Mx(x.size());
    A_inv.vmult(Ainv_Mx, Mx);

    Vector<double> My(x.size());
    M.vmult(My, Ainv_Mx);  // M A_x^{-1} M x

    double denom = x * My;
    // Requirement for M, A positive definite on the tangent space of x
    if (!ignore_positivity_constraint) {
        AssertThrow(denom > 0, dealii::ExcInternalError("x' M A^{-1} M x <= 0"));
    }

    Vector<double> Mv(v.size());
    M.vmult(Mv, v);
    const double nom = x*Mv;

    output = v;
    output.add(-nom / denom, Ainv_Mx);
}

/**
 * @brief Projects the base point @p x onto its own tangent space using an energy-based metric.
 *
 * This overload handles the specific case where the vector to be projected is the base point
 * itself (i.e., \f$ v = x \f$).
 *
 * It computes the projection \f$ \Pi_x(x) \f$ relative to the metric induced by
 * the operator \f$ A^{-1} \f$ (specifically \f$ x^T M A^{-1} M x \f$).
 *
 * Formula:
 * \f[
 * \mathrm{output} = x - \frac{1}{\langle x, M A^{-1} M x \rangle} A^{-1} M x
 * \f]
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @tparam InverseMatrixType A solver or operator class representing \f$ A^{-1} \f$.
 * @param[in] A_inv The inverse operator \f$ A^{-1} \f$.
 * @param[in] x The base point.
 * @param[in] M The mass matrix.
 * @param[out] output The resulting projected vector.
 */
template <typename MatrixType, typename InverseMatrixType>
void project_onto_tangent_space(const InverseMatrixType& A_inv, const Vector<double>& x, const MatrixType& M,
                                Vector<double>& output)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    Vector<double> Ainv_Mx(x.size());
    A_inv.vmult(Ainv_Mx, Mx);

    Vector<double> My(x.size());
    M.vmult(My, Ainv_Mx);  // M A_x^{-1} M x

    const double denom = x * My;
    AssertThrow(denom > 0, dealii::ExcInternalError("x' M A^{-1} M x <= 0"));

    output = x;
    output.add(-1.0/denom, Ainv_Mx);
}

template <typename MatrixType, typename InverseMatrixType>
void random_tangent_vector(const InverseMatrixType& A_inv, const Vector<double>& x, const MatrixType& M,
                           Vector<double>& v,
                           const double mean = 0.0, const double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(v.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the energy-based metric
    energy::project_onto_tangent_space(A_inv, x, M, tmp, v);
}

} // namespace metric::energy


namespace metric::mass
{

/**
 * @brief Projects a vector @p v onto the tangent space at @p x with respect to the
 * mass-weighted inner product.
 *
 * This function computes the projection
 * \f[
 * \Pi_x(v) = v - \frac{\langle x, v \rangle_M}{\|x\|_M^2} x,
 * \f]
 * assuming the manifold is the sphere defined by the mass matrix @p M. If @p x is
 * already normalized with respect to @p M (i.e., \f$\|x\|_M = 1\f$), this simplifies
 * to \f$ \Pi_x(v) = v - (x^T M v) x \f$.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method (e.g., SparseMatrix).
 * @param[in] x The base point on the manifold (assumed to be normalized in the M-metric).
 * @param[in] M The mass matrix defining the inner product \f$ \langle u, w \rangle_M = u^T M w \f$.
 * @param[in] v The vector to be projected.
 * @param[out] output The resulting projected vector in the tangent space \f$ T_x \mathcal{M} \f$.
 */
template <typename MatrixType>
void project_onto_tangent_space(const Vector<double>& x, const MatrixType& M, const Vector<double>& v,
                                Vector<double>& output)
{
    Vector<double> Mv(x.size());
    M.vmult(Mv, v);
    const double xMv = x*Mv;

    output = v;
    output.add(-xMv, x);
}

template <typename MatrixType>
void random_tangent_vector(const Vector<double>& x, const MatrixType& M,
                           Vector<double>& v,
                           double mean = 0.0, double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(v.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the mass metric
    mass::project_onto_tangent_space(x, M, tmp, v);
}

} // namespace metric::mass


namespace metric::frobenius
{

template <typename MatrixType>
void project_onto_tangent_space(const Vector<double>& x, const MatrixType& M, const Vector<double>& v,
                                Vector<double>& output)
{
    AssertDimension(x.size(), v.size());

    // Compute M * phi
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    // Denominator: phi^T M^2 phi = (M * phi)^T (M * phi) by symmetry of M
    const double denom = Mx * Mx;

    AssertThrow(denom > 0.0, dealii::ExcInternalError("x' M^2 x <= 0"));

    // Numerator: phi^T M xi = (M * phi)^T xi by symmetry of M
    const double nom = Mx * v;

    // Output: xi - (nom / denom) * (M * phi)
    output = v;
    output.add(-nom / denom, Mx);
}

template <typename MatrixType>
void random_tangent_vector(const Vector<double>& x, const MatrixType& M,
                           Vector<double>& v,
                           const double mean = 0.0, const double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(v.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the energy-based metric
    frobenius::project_onto_tangent_space(x, M, tmp, v);
}

} // namespace metric::frobenius

} // namespace rmo::gpe

#endif //RMO_GPE_METRIC_H
