#ifndef RMO_GPE_METRIC_H
#define RMO_GPE_METRIC_H

#include <rmo/ropt/manifold.h>
#include <rmo/util/random.h>

/**
 * @file
 * @brief Projections onto the tangent space \f$ T_x \mathcal{S} = \{ v : x^\top M v = 0 \} \f$ of the unit-mass
 * sphere, orthogonal in the metric of a Riemannian gradient (energy-adaptive \f$ A_x \f$, mass \f$ M \f$, or
 * Euclidean, the F-metric), and random tangent vectors for tests.
 *
 * Arguments: the base point `x` on the sphere, the mass matrix `M`, the vector `v` to project and the result
 * `output`.
 */
namespace rmo::gpe
{

namespace metric::energy
{

/**
 * @brief \f$ A_x \f$-orthogonal projection onto \f$ T_x \mathcal{S} \f$, with @p A_inv \f$ = A_x^{-1} \f$:
 * \f[ \mathrm{output} = v - \frac{x^\top M v}{x^\top M A_x^{-1} M x} \, A_x^{-1} M x \f]
 * @p ignore_positivity_constraint skips the check \f$ x^\top M A_x^{-1} M x > 0 \f$.
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
 * @brief Projection of @p x itself (\f$ v = x \f$, using \f$ x^\top M x = 1 \f$): the energy-adaptive Riemannian
 * gradient, see kernels::grad_energy_adaptive().
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

/** @brief Random tangent vector: normal entries (@p mean, @p stddev), projected \f$ A_x \f$-orthogonally. */
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

/** @brief \f$ M \f$-orthogonal projection onto \f$ T_x \mathcal{S} \f$: \f$ \mathrm{output} = v - (x^\top M v) \, x \f$. */
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

/** @brief Random tangent vector: normal entries (@p mean, @p stddev), projected \f$ M \f$-orthogonally. */
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

/**
 * @brief Euclidean (F-metric) orthogonal projection onto \f$ T_x \mathcal{S} \f$:
 * \f$ \mathrm{output} = v - \frac{x^\top M v}{x^\top M^2 x} \, M x \f$.
 */
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

/** @brief Random tangent vector: normal entries (@p mean, @p stddev), projected Euclidean-orthogonally. */
template <typename MatrixType>
void random_tangent_vector(const Vector<double>& x, const MatrixType& M,
                           Vector<double>& v,
                           const double mean = 0.0, const double stddev = 1.0)
{
    // 1. generate random vector in ambient space
    Vector<double> tmp(v.size());
    normrnd(mean, stddev, tmp);

    // 2. project orthogonally onto tangent space at x, wrt. the Frobenius metric
    frobenius::project_onto_tangent_space(x, M, tmp, v);
}

} // namespace metric::frobenius

} // namespace rmo::gpe

#endif //RMO_GPE_METRIC_H
