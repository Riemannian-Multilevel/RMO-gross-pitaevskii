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

// TODO: object that supplies vmult(), instead of SparseMatrix (matrix-free methods)
inline double
function_value(const Vector<double>& x, const SparseMatrix<double>& A0,
               const SparseMatrix<double>& Mpp)
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
inline void
project_onto_tangent_space(const Vector<double>& Ainv_Mx, const Vector<double>& x,
    const SparseMatrix<double>& M, const Vector<double>& v, Vector<double>& output)
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
inline void
project_onto_tangent_space(const Vector<double>& Ainv_Mx, const Vector<double>& x,
    const SparseMatrix<double>& M, Vector<double>& output)
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
// TODO: A = A0 + Mpp, use vmult() + vmult() instead of matrix copy
template <typename PreconditionType>
unsigned gradient(const SparseMatrix<double>& A, const SparseMatrix<double>& M,
                  const Vector<double>& x, Vector<double>& output,
                  const dealii::AffineConstraints<double>& constraints,
                  const PreconditionType& precondition,
                  SolverMethod solver, unsigned int max_inner, double tol_inner)
{
    // y <- A^{-1} Mx
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    // inner solve (linear system)
    Vector<double> y(x.size());
    auto solve_control = solve_sparse(A, Mx, y, solver,
        precondition, max_inner, tol_inner);

    // Apply boundary condition
    constraints.distribute(y);

    // \Pi_x(x): R^n -> T_x S^{n-1}
    project_onto_tangent_space(y, x, M, output);

    unsigned int last_step = solve_control.last_step();
    return last_step;  // TODO: return general status object
}

// Retraction by normalization, with base point x and argument z
// Note: this overwrites the vector x
inline void
retract_by_norm(const SparseMatrix<double>& M, const Vector<double>& z,
                Vector<double>& x, const double factor)
{
    // x <- x + h z
    x.add(factor, z);

    // x <- x / ||x||_M
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    x /= std::sqrt(x * Mx);
}

// TODO: add other retractions (orthographic, exponential)
inline void
retract_by_ortho(const SparseMatrix<double>&, const Vector<double>&,
                 Vector<double>&, const double)
{
    throw dealii::ExcNotImplemented();
}

inline void
retract_by_exp(const SparseMatrix<double>&, const Vector<double>&,
               Vector<double>&, const double)
{
    throw dealii::ExcNotImplemented();
}

// Termination criteria for energy function minimization
// TODO: make this generic?
struct Property
{
    double mass{0};
    double lambda{0};
    double residual{0};
};

inline Property
residual(const Vector<double>& x, const SparseMatrix<double>& A, const SparseMatrix<double>& M)
{
    Property prop;
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    prop.mass = x * Mx;      // should be ~ 1 (energy constraint)

    Vector<double> Ax(x.size());
    A.vmult(Ax, x);
    prop.lambda = x * Ax;   // Rayleigh quotient (x'Ax / x'Mx)

    Vector<double> r(Ax);
    r.add(-prop.lambda, Mx);        // r = A x - lambda M x

    // TODO: use enum for setting norm at runtime
    prop.residual = 0.0;

    if (M_NORM_RESIDUAL) {
        Vector<double> Mr(r.size());
        M.vmult(Mr, r);
        prop.residual = std::sqrt(r * Mr);
    } else {
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