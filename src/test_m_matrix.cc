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
#include <rmo/gpe/gpe.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace rmo;
using namespace rmo::gpe;
using namespace dealii;

namespace
{
unsigned n_failed = 0;

void check(bool ok, const std::string& name, const std::string& detail = "")
{
    std::cout << (ok ? "PASS  " : "FAIL  ") << name;
    if (!detail.empty()) {
        std::cout << "  (" << detail << ")";
    }
    std::cout << "\n";
    n_failed += !ok;
}

std::string fmt_double(double x)
{
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2) << x;
    return ss.str();
}

struct SignPattern
{
    unsigned n_positive_offdiag = 0;  ///< off-diagonal entries > 0
    double   max_positive       = 0;  ///< largest one, relative to max |A_ij|
    double   min_diagonal       = 0;  ///< smallest diagonal entry, relative to max |A_ij|
};

SignPattern sign_pattern(const SparseMatrix<double>& A, const AffineConstraints<double>& constraints)
{
    SignPattern sp;
    const double scale = A.linfty_norm();
    sp.min_diagonal = 1.0;

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

//! Smallest entry of A^{-1} e_i over a few unconstrained columns i, relative to the largest entry.
double min_inverse_entry(const SparseMatrix<double>& A, const AffineConstraints<double>& constraints)
{
    const unsigned n = A.m();
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
        PreconditionJacobi<SparseMatrix<double>> jacobi;
        jacobi.initialize(A);
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
void test_config(MeshKind mesh, int degree, unsigned n_levels, const std::string& name, bool required)
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

    GrossPitaevskiiPackage<dim> package(options, n_levels);
    const auto& constraints = package.get_constraints();
    const potential::Square<dim> V;

    // The consistent potential term adds positive off-diagonal entries
    const auto consistent = package.template system<GrossPitaevskiiSystem<dim>>(V);
    const SignPattern sp_c = sign_pattern(consistent.get_A0(), constraints);
    std::cout << "INFO  " << name << ", consistent A0: " << sp_c.n_positive_offdiag
              << " positive off-diagonal entries (max " << fmt_double(sp_c.max_positive) << ")\n";

    const auto lumped = package.template system<GrossPitaevskiiLumpedSystem<dim>>(V);
    const SignPattern sp = sign_pattern(lumped.get_A0(), constraints);
    const std::string detail = std::to_string(sp.n_positive_offdiag) + " positive off-diagonal entries (max "
                             + fmt_double(sp.max_positive) + "), min diagonal " + fmt_double(sp.min_diagonal);

    if (!required) {
        std::cout << "INFO  " << name << ", lumped A0: " << detail << "\n";
        return;
    }
    check(sp.n_positive_offdiag == 0 && sp.min_diagonal > 0, name + ": lumped A0 is a Z-matrix with positive diagonal", detail);

    const double min_inv = min_inverse_entry(lumped.get_A0(), constraints);
    check(min_inv > -1e-10, name + ": lumped A0^{-1} is non-negative (5 columns)", "min relative entry " + fmt_double(min_inv));
}

} // namespace


int main()
{
    std::cerr.setstate(std::ios::failbit);  // silence mesh statistics printed by GrossPitaevskiiPackage

    try {
        test_config<2>(MeshKind::QUADRILATERAL, 1, 5, "2d Q1 (squares)", true);
        test_config<2>(MeshKind::SIMPLEX,       1, 4, "2d P1 (simplex mesh)", true);
        test_config<3>(MeshKind::QUADRILATERAL, 1, 3, "3d Q1 (cubes)", true);

        test_config<2>(MeshKind::QUADRILATERAL, 2, 4, "2d Q2 (squares)", false);
        test_config<2>(MeshKind::SIMPLEX,       2, 3, "2d P2-bubble (simplex mesh)", false);
        test_config<3>(MeshKind::SIMPLEX,       1, 3, "3d P1 (simplex mesh)", false);
    }
    catch (const std::exception& exc) {
        std::cerr.clear();
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }

    std::cout << (n_failed ? std::to_string(n_failed) + " check(s) failed" : "all checks passed") << std::endl;
    return n_failed ? 1 : 0;
}
