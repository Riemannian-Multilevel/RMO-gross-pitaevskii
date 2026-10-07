//
// Checks the matrix form of the grid transfers for Q1 elements on uniformly refined squares
// (coarse mesh 2x2 cells, 9 DoFs; fine mesh 4x4 cells, 25 DoFs):
//
//   to_fine_mesh() (P) evaluates the coarse function at the fine nodes. A fine node is a coarse node,
//   an edge midpoint or a cell centre, so each row of P is [1], [1/2, 1/2] or [1/4, 1/4, 1/4, 1/4].
//
//   to_coarse_mesh() (R) evaluates the fine function at the coarse nodes. Each coarse node is a fine
//   node, so each row of R is [1] (injection); in particular, R is not P^T.
//
//   Tfine() is P^T. Its row sums are up to 2^dim, those of R are 1, so the transports must not mix up
//   the two restrictions.
//
#include "check.h"

#include <rmo/fe/interpolate.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace rmo;
using namespace rmo::test;
using namespace dealii;

namespace
{
constexpr int    dim = 2;
constexpr double tol = 1e-12;

//! Q1 DoFs on [-1, 1]^2, refined once (coarse) and twice (fine), without constraints.
struct NestedMeshes
{
    NestedMeshes()
    {
        GridGenerator::hyper_cube(tria_coarse, -1.0, 1.0);
        GridGenerator::hyper_cube(tria_fine, -1.0, 1.0);
        tria_coarse.refine_global(1);
        tria_fine.refine_global(2);

        for (auto* dofs : {&dofs_coarse, &dofs_fine}) {
            dofs->distribute_dofs(fe);
            dofs->distribute_mg_dofs();  // required by LinearTransferMG
        }
        constraints_coarse.close();
        constraints_fine.close();
    }

    // distribute_mg_dofs() requires this mesh smoothing flag
    Triangulation<dim> tria_coarse{Triangulation<dim>::limit_level_difference_at_vertices};
    Triangulation<dim> tria_fine{Triangulation<dim>::limit_level_difference_at_vertices};
    const FE_Q<dim> fe{1};
    DoFHandler<dim> dofs_coarse{tria_coarse};
    DoFHandler<dim> dofs_fine{tria_fine};
    AffineConstraints<double> constraints_coarse, constraints_fine;
};

using TransferMethod = void (LinearTransferBase::*)(const Vector<double>&, Vector<double>&) const;

//! Matrix of a method of @p transfer, with column j the image of the unit vector e_j.
FullMatrix<double> matrix_of(const LinearTransferBase& transfer, TransferMethod method, unsigned n_src, unsigned n_dst)
{
    FullMatrix<double> A(n_dst, n_src);
    Vector<double> e_j(n_src), col(n_dst);

    for (unsigned j = 0; j < n_src; j++) {
        e_j    = 0.0;
        e_j[j] = 1.0;
        (transfer.*method)(e_j, col);

        for (unsigned i = 0; i < n_dst; i++) {
            A(i, j) = col[i];
        }
    }
    return A;
}

FullMatrix<double> transpose(const FullMatrix<double>& A)
{
    FullMatrix<double> At(A.n(), A.m());
    At.copy_transposed(A);
    return At;
}

//! Largest entrywise difference of two matrices of equal size.
double max_diff(const FullMatrix<double>& A, const FullMatrix<double>& B)
{
    AssertDimension(A.m(), B.m());
    AssertDimension(A.n(), B.n());

    double diff = 0.0;
    for (unsigned i = 0; i < A.m(); i++) {
        for (unsigned j = 0; j < A.n(); j++) {
            diff = std::max(diff, std::abs(A(i, j) - B(i, j)));
        }
    }
    return diff;
}

double max_row_sum(const FullMatrix<double>& A)
{
    double max_sum = 0.0;
    for (unsigned i = 0; i < A.m(); i++) {
        double sum = 0.0;
        for (unsigned j = 0; j < A.n(); j++) {
            sum += A(i, j);
        }
        max_sum = std::max(max_sum, sum);
    }
    return max_sum;
}

//! True if each row of @p A has @p n_nonzero entries equal to 1 / n_nonzero, for one of the given counts.
bool rows_are_averages(const FullMatrix<double>& A, const std::vector<unsigned>& n_nonzero)
{
    for (unsigned i = 0; i < A.m(); i++) {
        std::vector<double> row;
        for (unsigned j = 0; j < A.n(); j++) {
            if (std::abs(A(i, j)) > tol) {
                row.push_back(A(i, j));
            }
        }
        const bool is_average = std::ranges::find(n_nonzero, row.size()) != n_nonzero.end()
            && std::ranges::all_of(row, [&](double a) { return std::abs(a - 1.0 / row.size()) < tol; });
        if (!is_average) {
            return false;
        }
    }
    return true;
}

bool is_bilinear_interpolation(const FullMatrix<double>& P) { return rows_are_averages(P, {1, 2, 4}); }
bool is_injection(const FullMatrix<double>& R) { return rows_are_averages(R, {1}); }

} // namespace


int main()
{
    return run_tests([](CheckReport& report) {
        const NestedMeshes m;
        const fe::LinearTransfer<dim>   transfer(m.dofs_coarse, m.dofs_fine, m.constraints_coarse, m.constraints_fine);
        const fe::LinearTransferMG<dim> transfer_mg(m.dofs_coarse, m.dofs_fine, m.constraints_coarse, m.constraints_fine);
        const unsigned n_c = m.dofs_coarse.n_dofs();
        const unsigned n_f = m.dofs_fine.n_dofs();

        const auto P    = matrix_of(transfer,    &LinearTransferBase::to_fine_mesh,   n_c, n_f);
        const auto R    = matrix_of(transfer,    &LinearTransferBase::to_coarse_mesh, n_f, n_c);
        const auto P_mg = matrix_of(transfer_mg, &LinearTransferBase::to_fine_mesh,   n_c, n_f);
        const auto R_mg = matrix_of(transfer_mg, &LinearTransferBase::to_coarse_mesh, n_f, n_c);
        const auto T_mg = matrix_of(transfer_mg, &LinearTransferBase::Tfine,          n_f, n_c);
        const auto P_t  = transpose(P);

        report.check(is_bilinear_interpolation(P), "LinearTransfer::to_fine_mesh is bilinear interpolation");
        report.check(is_bilinear_interpolation(P_mg), "LinearTransferMG::to_fine_mesh is bilinear interpolation");
        report.check(max_diff(P, P_mg) < tol, "both prolongations agree", "max diff " + sci(max_diff(P, P_mg)));

        report.check(is_injection(R), "LinearTransfer::to_coarse_mesh is injection");
        report.check(is_injection(R_mg), "LinearTransferMG::to_coarse_mesh is injection");
        report.check(max_diff(R, P_t) > 0.1, "to_coarse_mesh is not P^T");

        report.check(max_diff(T_mg, P_t) < tol, "LinearTransferMG::Tfine is P^T", "max diff " + sci(max_diff(T_mg, P_t)));
        report.info("largest row sums", "to_coarse_mesh " + std::to_string(max_row_sum(R))
                                      + ", Tfine " + std::to_string(max_row_sum(T_mg)));
    });
}
