#ifndef RMO_GPE_MANIFOLD_POSITIVE_H
#define RMO_GPE_MANIFOLD_POSITIVE_H

#include <rmo/gpe/manifold.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

/**
 * @file
 * @brief Variants of the positive unit-mass sphere for large domains (tex/pos_sphere_variants.tex), as ManifoldBase:
 * the floored density retraction (FlooredPosSphere), the e-retraction (ExpPosSphere) and the normalization
 * retraction with a positivity cap (NormPosSphere). The mass matrix must be diagonal (lumped).
 */
namespace rmo::gpe
{

/**
 * @brief Positive sphere whose density retraction clamps every component at @p floor (then normalizes), so that it is
 * defined for every step; the step limit (option A) only counts the nodes above 10 floor, or none (@p cap false).
 * The lifting is the density lifting, see ellipsoid_positive::retract_inv_by_density().
 */
template <typename MatrixType>
class FlooredPosSphere : public ManifoldBase
{
    static_assert(is_diagonal_matrix_v<MatrixType>, "the density retraction requires a diagonal (lumped) mass matrix");

public:
    FlooredPosSphere(const MatrixType& D, double floor, bool cap) : D(D), floor(floor), cap(cap) {}

    void set_floor(double floor_new) const { floor = floor_new; }
    [[nodiscard]] double get_floor() const { return floor; }

    void retract(const Vector<double>& z, Vector<double>& x, double factor) const override
    {
        for (unsigned i = 0; i < x.size(); ++i) {
            x[i] = std::sqrt(std::max(x[i] * x[i] + 2 * factor * x[i] * z[i], floor * floor));
        }
        ellipsoid::retract_by_norm(D, x);
    }

    void retract(const Vector<double>& z, const Vector<double>& x, Vector<double>& output, double factor) const override
    {
        output = x;
        retract(z, output, factor);
    }

    [[nodiscard]] double max_step(const Vector<double>& x, const Vector<double>& v) const override
    {
        double h_max = std::numeric_limits<double>::infinity();
        if (cap) {
            for (unsigned i = 0; i < x.size(); ++i) {
                if (v[i] < 0 && x[i] > 10 * floor) {
                    h_max = std::min(h_max, x[i] / (-2 * v[i]));
                }
            }
        }
        return h_max;
    }

    //! retract() of t v with its velocity: clamped components do not move (exact line search)
    void retract_velocity(const Vector<double>& x, const Vector<double>& v, double t,
                          Vector<double>& y, Vector<double>& y_dot) const override
    {
        y.reinit(x.size(), true);
        y_dot.reinit(x.size(), true);
        for (unsigned i = 0; i < x.size(); ++i) {
            const double q = x[i] * x[i] + 2 * t * x[i] * v[i];
            y[i] = std::sqrt(std::max(q, floor * floor));
            y_dot[i] = q > floor * floor ? x[i] * v[i] / y[i] : 0.0;
        }
        ellipsoid::normalize_with_velocity(D, y, y_dot);
    }

    [[nodiscard]] double constraint_inner(const Vector<double>& y, const Vector<double>& z) const override
    {
        return ellipsoid::mass_inner(D, y, z);
    }

    void retract_inv(Vector<double>& v, const Vector<double>& x) const override
    {
        ellipsoid_positive::retract_inv_by_density(v, x);
    }

    void retract_diff(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                      Vector<double>&) const override { throw dealii::ExcNotImplemented(); }

    /** @brief Differential of the density lifting, see ellipsoid_positive::retract_inv_diff_by_density(). */
    void retract_inv_diff(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                          Vector<double>& output) const override
    {
        ellipsoid_positive::retract_inv_diff_by_density(x, zeta, u, output);
    }

    /** @brief D-adjoint of retract_inv_diff(): the differential is diagonal, hence self-adjoint. */
    void retract_inv_diff_adjoint(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                                  Vector<double>& output) const override
    {
        ellipsoid_positive::retract_inv_diff_by_density(x, zeta, u, output);
    }

private:
    const MatrixType& D;
    mutable double floor;
    bool cap;
};


//! Step limit of ExpPosSphere: none, 1 / max_i |v_i| / x_i (a relative change exp(1) per component), or
//! 1 / max_i (v_i)_+ / x_i (growth by at most exp(1))
enum class ExpCap : std::uint8_t { NONE, TWO_SIDED, GROWTH };

/** @brief Positive sphere with the e-retraction, see ellipsoid_positive::retract_by_e(), with the step limit @p cap. */
template <typename MatrixType>
class ExpPosSphere : public ManifoldBase
{
public:
    ExpPosSphere(const MatrixType& D, ExpCap cap) : D(D), cap(cap) {}

    void retract(const Vector<double>& z, Vector<double>& x, double factor) const override
    {
        ellipsoid_positive::retract_by_e(D, z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x, Vector<double>& output, double factor) const override
    {
        output = x;
        retract(z, output, factor);
    }

    [[nodiscard]] double max_step(const Vector<double>& x, const Vector<double>& v) const override
    {
        if (cap == ExpCap::NONE) {
            return std::numeric_limits<double>::infinity();
        }
        double rate = 0;
        for (unsigned i = 0; i < x.size(); ++i) {
            rate = std::max(rate, (cap == ExpCap::GROWTH ? v[i] : std::abs(v[i])) / x[i]);
        }
        return 1 / rate;
    }

    void retract_velocity(const Vector<double>& x, const Vector<double>& v, double t,
                          Vector<double>& y, Vector<double>& y_dot) const override
    {
        ellipsoid_positive::retract_velocity_by_e(D, x, v, t, y, y_dot);
    }

    [[nodiscard]] double constraint_inner(const Vector<double>& y, const Vector<double>& z) const override
    {
        return ellipsoid::mass_inner(D, y, z);
    }

    void retract_inv(Vector<double>& v, const Vector<double>& x) const override
    {
        ellipsoid_positive::retract_inv_by_e(D, v, x);
    }

    void retract_diff(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                      Vector<double>&) const override { throw dealii::ExcNotImplemented(); }
    void retract_inv_diff(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                          Vector<double>&) const override { throw dealii::ExcNotImplemented(); }
    void retract_inv_diff_adjoint(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                                  Vector<double>&) const override { throw dealii::ExcNotImplemented(); }

private:
    const MatrixType& D;
    ExpCap cap;
};


/**
 * @brief Positive sphere with the normalization retraction (x + h v) / |x + h v|_D and the positivity cap
 * h <= theta min_{v_i < 0} x_i / |v_i| (twice the step limit of the density retraction): the unit sphere restricted to
 * the positive orthant, with the closed-form exact step of the ray (OracleBase::ray_step(), capped).
 */
template <typename MatrixType>
class NormPosSphere : public ManifoldBase
{
public:
    explicit NormPosSphere(const MatrixType& D) : D(D) {}

    void retract(const Vector<double>& z, Vector<double>& x, double factor) const override
    {
        ellipsoid::retract_by_norm(D, z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x, Vector<double>& output, double factor) const override
    {
        output = x;
        retract(z, output, factor);
    }

    [[nodiscard]] double max_step(const Vector<double>& x, const Vector<double>& v) const override
    {
        return 2 * ellipsoid_positive::max_step(x, v);
    }

    void retract_velocity(const Vector<double>& x, const Vector<double>& v, double t,
                          Vector<double>& y, Vector<double>& y_dot) const override
    {
        ellipsoid::retract_velocity_by_norm(D, x, v, t, y, y_dot);
    }

    [[nodiscard]] double constraint_inner(const Vector<double>& y, const Vector<double>& z) const override
    {
        return ellipsoid::mass_inner(D, y, z);
    }

    [[nodiscard]] bool retracts_by_normalization() const override { return true; }

    void retract_inv(Vector<double>& v, const Vector<double>& x) const override
    {
        ellipsoid::retract_inv_by_norm(D, v, x);
    }

    void retract_diff(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                      Vector<double>&) const override { throw dealii::ExcNotImplemented(); }
    void retract_inv_diff(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                          Vector<double>&) const override { throw dealii::ExcNotImplemented(); }
    void retract_inv_diff_adjoint(const Vector<double>&, const Vector<double>&, const Vector<double>&,
                                  Vector<double>&) const override { throw dealii::ExcNotImplemented(); }

private:
    const MatrixType& D;
};

} // namespace rmo::gpe

#endif //RMO_GPE_MANIFOLD_POSITIVE_H
