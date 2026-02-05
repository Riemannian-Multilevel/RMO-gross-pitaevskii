#ifndef GPE_FUNCTIONS_H
#define GPE_FUNCTIONS_H
#ifndef M_NORM_RESIDUAL
#define M_NORM_RESIDUAL 1
#endif

#include "lac.h"
#include <deal.II/base/function.h>

namespace gpe
{

// TODO: include other potentials
//       use deal.II function objects
template <int dim>
class Square
{
public:
    double operator()(const Point<dim>& p) const {
        typename Point<dim>::value_type out = 0.0;
        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    }
};

// Functions for GPE minimization
namespace energy
{

// MatrixType can be any object providing vmult() (TODO: requires?)
template <typename MatrixType>
double function_value(const Vector<double>& x, const MatrixType& A0, const MatrixType& Mpp)
{
    Vector<double> Bx(x.size());
    A0.vmult(Bx, x);
    Bx *= 0.5;

    Vector<double> Mpp_x(x.size());
    Mpp.vmult(Mpp_x, x);

    Bx.add(0.25, Mpp_x);
    return x * Bx;
}

// case x != v
template <typename MatrixType>
void project_onto_tangent_space(const Vector<double>& Ainv_Mx, const Vector<double>& x,
                                const MatrixType& M, const Vector<double>& v,
                                Vector<double>& output)
{
    AssertDimension(x.size(), v.size());
    AssertDimension(x.size(), Ainv_Mx.size());

    Vector<double> My(x.size());
    M.vmult(My, Ainv_Mx);  // M A_x^{-1} M x

    double denom = x * My;
    AssertThrow(denom > 0, dealii::ExcInternalError("x' M A^{-1} M x <= 0"));

    Vector<double> Mv(v.size());
    M.vmult(Mv, v);
    const double nom = x*Mv;

    output = v;
    output.add(-nom / denom, Ainv_Mx);
}

// case x == v
template <typename MatrixType>
void project_onto_tangent_space(const Vector<double>& Ainv_Mx, const Vector<double>& x,
                                const MatrixType& M,
                                Vector<double>& output)
{
    AssertDimension(x.size(), Ainv_Mx.size());

    Vector<double> My(x.size());
    M.vmult(My, Ainv_Mx);  // M A_x^{-1} M x

    const double denom = x * My;
    AssertThrow(denom > 0, dealii::ExcInternalError("x' M A^{-1} M x <= 0"));

    output = x;
    output.add(-1.0/denom, Ainv_Mx);
}

// Riemannian gradient in S^{n-1} with energy metric
template <typename MatrixType, typename InverseMatrixType>
void gradient(const InverseMatrixType& Ainv, const MatrixType& M, const Vector<double>& x, Vector<double>& output)
{
    // y <- A^{-1} Mx
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    // inner solve (linear system)
    Vector<double> y(x.size());
    Ainv.vmult(y, Mx);

    // \Pi_x(x): R^n -> T_x S^{n-1}
    project_onto_tangent_space(y, x, M, output);
}

// Retraction by normalization, with base point x and argument z
// Note: this overwrites the vector x
// The factor argument is an optimization for gradient descent (FMA operation)
template <typename MatrixType>
void retract_by_norm(const MatrixType& M, const Vector<double>& z, Vector<double>& x,
                     const double factor = 1.0)
{
    AssertThrow(factor != 0.0, dealii::ExcMessage("factor must be nonzero"));
    x.add(factor, z);           // x' <- x + h z

    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    x /= std::sqrt(x * Mx);     // x' <- x' / ||x'||_M
}

// Inverse retraction / lift by normalization, S -> TS
// Note: this overwrites the vector v
template <typename MatrixType>
void inverse_by_norm(const MatrixType& M, Vector<double>& v, const Vector<double>& x)
{
    Vector<double> Mv(v.size());
    M.vmult(Mv, v);

    const double xMv = x*Mv;
    AssertThrow(xMv > 0, dealii::ExcInternalError("x'Mv must be nonzero"));

    v /= xMv;
    v.add(-1.0, x);
}

template <typename MatrixType>
void retract_by_ortho(const MatrixType& M, const Vector<double>& z,
                      Vector<double>& x, const double factor = 1.0)
{
    AssertThrow(factor != 0.0, dealii::ExcMessage("factor must be non-zero"));
    Vector<double> Mz(x.size());
    M.vmult(Mz, z);

    double zMz = z*Mz;
    zMz *= factor;
    zMz *= factor;  // (hz) * M(hZ) = h^2 zMz
    AssertThrow(zMz < 1.0, ExcInternalError("z'Mz required < 1"));

    x *= std::sqrt(1-zMz);
    x.add(factor, z);
}

template <typename MatrixType>
void inverse_by_ortho(const MatrixType& M, Vector<double>& v, const Vector<double>& x)
{
    Vector<double> Mv(v.size());
    M.vmult(Mv, v);

    double xMv = x*Mv;
    v.add(-xMv, x);
}

// Exponential map
template <typename MatrixType>
void retract_by_exp(const MatrixType& M, const Vector<double>& z, Vector<double>& x,
                    const double factor = 1.0)
{
    AssertThrow(factor != 0.0, dealii::ExcMessage("factor must be non-zero"));
    Vector<double> Mz(x.size());
    M.vmult(Mz, z);

    double zMz = z*Mz;
    double z_Mnorm = std::sqrt(zMz);
    AssertThrow(z_Mnorm > 0.0, dealii::ExcInternalError("|z|_M must be positive"));

    //                  |hz|_M  =  h |z|_M
    //             hz / |hz|_M  =  z / |z|_M
    // sin(|hz|_M) hz / |hz|_M  =  sin(h|z|_M) z / |z|_M
    x *= std::cos(factor*z_Mnorm);
    x.add(std::sin(factor*z_Mnorm) / z_Mnorm, z);
}

// Logarithmic map
template <typename MatrixType>
void inverse_by_exp(const MatrixType& M, Vector<double>& v, const Vector<double>& x)
{
    Vector<double> Mv(v.size());
    M.vmult(Mv, v);

    double xMv = x*Mv;
    v.add(-xMv, x);

    double nom   = std::acos(xMv);
    double denom = std::sin(nom);

    AssertThrow(denom != 0, dealii::ExcInternalError("sin(arccos(x' M v)) must be non-zero"));
    v *= (nom/denom);
}

// Termination criteria for energy function minimization
// TODO: make this generic?
struct Property
{
    double mass{0};
    double lambda{0};
    double residual{0};
};

// TODO: x*Mx is only for debugging/diagnostic purposes
template <typename MatrixType>
Property residual(const Vector<double>& x,
                  const MatrixType& A0,
                  const MatrixType& Mpp,
                  const MatrixType& M, double beta)
{
    Property prop;
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    prop.mass = x * Mx;             // should be ~ 1 (energy constraint)

    Vector<double> Ax1(x.size()); // A0 x
    A0.vmult(Ax1, x);

    Vector<double> Ax2(x.size()); // Mpp x
    Mpp.vmult(Ax2, x);

    Ax1.add(beta, Ax2);             // (A0 + beta Mpp) x
    prop.lambda = x * Ax1;          // Rayleigh quotient (x'Ax / x'Mx)

    Vector<double> r(Ax1);
    r.add(-prop.lambda, Mx);        // r = A x - lambda M x

    // TODO: use enum for setting norm at runtime
    prop.residual = 0.0;
    if (M_NORM_RESIDUAL) {
        Vector<double> Mr(r.size());
        M.vmult(Mr, r);
        prop.residual = std::sqrt(r * Mr);
    }
    else {
        prop.residual = r.l2_norm();
    }
    return prop;
}

} // namespace energy

namespace coarse_model
{
// TODO: functions for Nash coarse model

} // namespace coarse_model

} // namespace gpe

#endif //GPE_FUNCTIONS_H