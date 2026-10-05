/**
 * @file
 * @brief Checks that the lumped \f$ A_0 = S + M_{V,L} \f$ is an M-matrix for elements of degree 1 (2d Q1, 2d P1, 3d Q1).
 *
 * Then \f$ A_u = A_0 + \beta M_{\phi\phi}(u) \f$ has a non-negative inverse, and the energy-adaptive step
 * \f$ A_u^{-1} M u \f$ preserves positivity. \f$ A_0 \f$ is symmetric positive definite, so non-positive
 * off-diagonal entries suffice; the non-negative inverse is also checked directly on a few columns. Dirichlet rows
 * and columns are excluded. The consistent \f$ A_0 \f$, and elements of higher degree, give no M-matrix.
 */
#include "check.h"

#include <rmo/gpe/gpe.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <algorithm>
#include <string>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;
using namespace dealii;

namespace
{

//! Signs of the entries of a matrix in the unconstrained rows and columns, relative to max |A_ij|.
struct SignPattern
{
    unsigned n_positive_offdiag = 0;  ///< off-diagonal entries > 0
    double   max_positive       = 0;  ///< largest of them
    double   min_diagonal       = 1;  ///< smallest diagonal entry

    [[nodiscard]] std::string str() const
    {
        return std::to_string(n_positive_offdiag) + " positive off-diagonal entries (max " + sci(max_positive)
             + "), min diagonal " + sci(min_diagonal);
    }
};

SignPattern sign_pattern(const SparseMatrix<double>& A, const AffineConstraints<double>& constraints)
{
    SignPattern sp;
    const double scale = A.linfty_norm();

    for (unsigned i = 0; i < A.m(); ++i) {
        if (constraints.is_constrained(i)) {
            continue;
        }
        for (auto it = A.begin(i); it != A.end(i); ++it) {
            const unsigned j = it->column();
            if (j == i) {
                sp.min_diagonal = std::min(sp.min_diagonal, it->value() / scale);
            }
            else if (!constraints.is_constrained(j) && it->value() > 1e-14 * scale) {
                ++sp.n_positive_offdiag;
                sp.max_positive = std::max(sp.max_positive, it->value() / scale);
            }
        }
    }
    return sp;
}

//! Smallest entry of A^{-1} e_i over 5 unconstrained columns i, relative to the largest entry of the column.
double min_inverse_entry(const SparseMatrix<double>& A, const AffineConstraints<double>& constraints)
{
    const unsigned n = A.m();
    PreconditionJacobi<SparseMatrix<double>> jacobi;
    jacobi.initialize(A);
    double min_rel = 1.0;

    for (unsigned k = 1; k <= 5; ++k) {
        unsigned i = (k * n) / 6;
        while (constraints.is_constrained(i)) {
            ++i;
        }
        Vector<double> e(n), y(n);
        e[i] = 1.0;

        SolverControl control(10 * n, 1e-14);
        SolverCG<Vector<double>> cg(control);
        cg.solve(A, y, e, jacobi);

        for (unsigned j = 0; j < n; ++j) {
            if (!constraints.is_constrained(j)) {
                min_rel = std::min(min_rel, y[j] / y.linfty_norm());
            }
        }
    }
    return min_rel;
}

template <int dim>
GPE_Options make_options(MeshKind mesh, int degree)
{
    GPE_Options options{};
    options.dimension = dim;
    options.degree    = degree;
    options.radius    = 8.0;
    options.beta      = 100.0;
    options.order     = Ordering::DEFAULT;
    options.bc        = BoundaryCondition::DIRICHLET;
    options.mesh_kind = mesh;
    options.potential = Potential::SQUARE;
    return options;
}

template <int dim>
void check_lumped_A0(CheckReport& report, MeshKind mesh, unsigned n_levels, const std::string& name)
{
    const GrossPitaevskiiPackage<dim> package(make_options<dim>(mesh, 1), n_levels);
    const auto& constraints = package.get_constraints();
    const auto  lumped      = package.template system<GrossPitaevskiiLumpedSystem<dim>>(potential::Square<dim>());

    const SignPattern sp = sign_pattern(lumped.get_A0(), constraints);
    report.check(sp.n_positive_offdiag == 0 && sp.min_diagonal > 0, name + ": lumped A0 is a Z-matrix with positive diagonal",
                 sp.str());

    const double min_inv = min_inverse_entry(lumped.get_A0(), constraints);
    report.check(min_inv > -1e-10, name + ": lumped A0^{-1} is non-negative (5 columns)",
                 "min relative entry " + sci(min_inv));
}

} // namespace


int main()
{
    std::cerr.setstate(std::ios::failbit);  // silence the mesh statistics of GrossPitaevskiiPackage

    return run_tests([](CheckReport& report) {
        check_lumped_A0<2>(report, MeshKind::QUADRILATERAL, 5, "2d Q1");
        check_lumped_A0<2>(report, MeshKind::SIMPLEX, 4, "2d P1");
        check_lumped_A0<3>(report, MeshKind::QUADRILATERAL, 3, "3d Q1");
    });
}
