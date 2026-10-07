//
// Checks that A0 = S + M_{V,L} is an M-matrix with mass lumping. Then A_u = A0 + beta Mpp(u) has a
// non-negative inverse, and the energy-adaptive step A_u^{-1} M u preserves positivity.
//
// A0 is symmetric positive definite, so non-positive off-diagonal entries suffice; the non-negative
// inverse is also checked directly on a few columns. Dirichlet rows and columns are excluded.
//
// Required: degree 1 (2d Q1, 2d P1 on the simplex mesh, 3d Q1). Reported as INFO, since they are no
// M-matrices: consistent A0, Q2, P2-bubble, 3d P1 on the simplex mesh.
//
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

//! Discretization of one test case.
struct Config
{
    MeshKind    mesh;
    int         degree;
    unsigned    n_levels;
    std::string name;
    bool        required;  ///< false: lumped A0 is reported, not checked
};

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
void check_config(CheckReport& report, const Config& config)
{
    const GrossPitaevskiiPackage<dim> package(make_options<dim>(config.mesh, config.degree), config.n_levels);
    const auto& constraints = package.get_constraints();
    const potential::Square<dim> V;

    // The consistent potential term adds positive off-diagonal entries
    const auto consistent = package.template system<GrossPitaevskiiSystem<dim>>(V);
    report.info(config.name + ", consistent A0", sign_pattern(consistent.get_A0(), constraints).str());

    const auto lumped = package.template system<GrossPitaevskiiLumpedSystem<dim>>(V);
    const SignPattern sp = sign_pattern(lumped.get_A0(), constraints);
    if (!config.required) {
        report.info(config.name + ", lumped A0", sp.str());
        return;
    }
    report.check(sp.n_positive_offdiag == 0 && sp.min_diagonal > 0,
                 config.name + ": lumped A0 is a Z-matrix with positive diagonal", sp.str());

    const double min_inv = min_inverse_entry(lumped.get_A0(), constraints);
    report.check(min_inv > -1e-10, config.name + ": lumped A0^{-1} is non-negative (5 columns)",
                 "min relative entry " + sci(min_inv));
}

} // namespace


int main()
{
    std::cerr.setstate(std::ios::failbit);  // silence the mesh statistics of GrossPitaevskiiPackage

    return run_tests([](CheckReport& report) {
        check_config<2>(report, {MeshKind::QUADRILATERAL, 1, 5, "2d Q1 (squares)", true});
        check_config<2>(report, {MeshKind::SIMPLEX,       1, 4, "2d P1 (simplex mesh)", true});
        check_config<3>(report, {MeshKind::QUADRILATERAL, 1, 3, "3d Q1 (cubes)", true});

        check_config<2>(report, {MeshKind::QUADRILATERAL, 2, 4, "2d Q2 (squares)", false});
        check_config<2>(report, {MeshKind::SIMPLEX,       2, 3, "2d P2-bubble (simplex mesh)", false});
        check_config<3>(report, {MeshKind::SIMPLEX,       1, 3, "3d P1 (simplex mesh)", false});
    });
}
