//
// Verifies that the grid transfer operators have the matrix form the solver assumes.
//
// Both transfers ultimately evaluate the source finite element function at the support points of
// the target mesh, which is what fixes their matrix form. For Q1 elements on uniformly refined
// quadrilateral grids:
//
//   Prolongation P (coarse -> fine): evaluate the coarse bilinear function at the fine nodes.
//     fine node coincides with a coarse node  -> row = [1]                     (injection)
//     fine node is the midpoint of a coarse edge -> row = [0.5, 0.5]           (linear average)
//     fine node is the centre of a coarse cell   -> row = [0.25, 0.25, ... ]   (bilinear average)
//   These are the only three cases, so P is standard bilinear interpolation.
//
//   Restriction R (fine -> coarse): evaluate the fine bilinear function at the coarse nodes. The
//   meshes are nested, so every coarse node is also a fine node and the evaluation simply returns
//   the fine value living there. Each row therefore has a single entry equal to 1: injection.
//   In particular R is neither P^T (full weighting / Galerkin restriction) nor a row-normalised
//   P^T (half weighting) -- both of those have several nonzeros per row.
//
// That is why fe::LinearTransferBase carries two different restrictions. to_coarse_mesh() is the
// injection above (primal, magnitude preserving), while Tfine() is the adjoint P^T (dual, row sums
// 2^dim on interior nodes). The transports pick one or the other, so this test pins both down --
// mixing them up rescales a restricted vector by 2^dim.
//
// Setup: coarse mesh 1 uniform refinement (2x2 cells, 3x3 = 9 DoFs),
//        fine mesh   2 uniform refinements (4x4 cells, 5x5 = 25 DoFs).
//
#include <rmo/fe/interpolate.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include <vector>

using namespace rmo;
using namespace dealii;

namespace
{
constexpr int    dim = 2;
constexpr double tol = 1e-12;

//! Matrix of a transfer operator, obtained by applying it to every canonical unit vector.
template <typename Apply>
FullMatrix<double> transfer_matrix(Apply&& apply, unsigned n_src, unsigned n_dst)
{
    FullMatrix<double> M(n_dst, n_src);
    Vector<double> e_j(n_src), col(n_dst);

    for (unsigned j = 0; j < n_src; j++) {
        e_j    = 0.0;
        e_j[j] = 1.0;
        apply(e_j, col);

        for (unsigned i = 0; i < n_dst; i++) {
            M(i, j) = col[i];
        }
    }
    return M;
}

//! Nonzero entries of one row.
std::vector<double> row_entries(const FullMatrix<double>& M, unsigned i)
{
    std::vector<double> nz;
    for (unsigned j = 0; j < M.n(); j++) {
        if (std::abs(M(i, j)) > tol) {
            nz.push_back(M(i, j));
        }
    }
    return nz;
}

bool all_close(const std::vector<double>& v, double value)
{
    return std::ranges::all_of(v, [value](double x) { return std::abs(x - value) < tol; });
}

//! Every row is injection, an edge average, or a cell average.
void check_bilinear_prolongation(const FullMatrix<double>& P, const std::string& name)
{
    for (unsigned i = 0; i < P.m(); i++) {
        const auto nz = row_entries(P, i);
        const bool ok = (nz.size() == 1 && all_close(nz, 1.0))
                     || (nz.size() == 2 && all_close(nz, 0.5))
                     || (nz.size() == 4 && all_close(nz, 0.25));

        AssertThrow(ok, ExcMessage(name + ": row " + std::to_string(i) + " has " +
            std::to_string(nz.size()) + " nonzeros and is neither injection, edge nor cell average"));
    }
    std::cout << "  PASS  " << name << " is bilinear interpolation\n";
}

//! Every row has a single entry equal to one.
void check_injection(const FullMatrix<double>& R, const std::string& name)
{
    for (unsigned i = 0; i < R.m(); i++) {
        const auto nz = row_entries(R, i);
        AssertThrow(nz.size() == 1 && all_close(nz, 1.0), ExcMessage(
            name + ": row " + std::to_string(i) + " is not injection"));
    }
    std::cout << "  PASS  " << name << " is injection\n";
}

//! Maximum entrywise difference between A and B^T.
double max_diff_transpose(const FullMatrix<double>& A, const FullMatrix<double>& B)
{
    AssertDimension(A.m(), B.n());
    AssertDimension(A.n(), B.m());

    double diff = 0.0;
    for (unsigned i = 0; i < A.m(); i++) {
        for (unsigned j = 0; j < A.n(); j++) {
            diff = std::max(diff, std::abs(A(i, j) - B(j, i)));
        }
    }
    return diff;
}
} // namespace


int main(int argc, char* argv[])
{
    try {
        // Coarse: 1 refinement (3x3 DoFs).  Fine: 2 refinements (5x5 DoFs).
        Triangulation<dim> tria_coarse, tria_fine;
        GridGenerator::hyper_cube(tria_coarse, -1.0, 1.0);
        GridGenerator::hyper_cube(tria_fine,   -1.0, 1.0);
        tria_coarse.refine_global(1);
        tria_fine.refine_global(2);

        const FE_Q<dim> fe(1);
        DoFHandler<dim> dof_coarse(tria_coarse), dof_fine(tria_fine);
        dof_coarse.distribute_dofs(fe);
        dof_fine.distribute_dofs(fe);
        dof_fine.distribute_mg_dofs();
        dof_coarse.distribute_mg_dofs();

        // No boundary conditions: the plain operators are what is being characterised here
        AffineConstraints<double> constr_coarse, constr_fine;
        constr_coarse.close();
        constr_fine.close();

        const unsigned n_c = dof_coarse.n_dofs();
        const unsigned n_f = dof_fine.n_dofs();
        std::cout << "coarse DoFs: " << n_c << ", fine DoFs: " << n_f << "\n\n";

        const fe::LinearTransfer<dim>   transfer(dof_coarse, dof_fine, constr_coarse, constr_fine);
        const fe::LinearTransferMG<dim> transfer_mg(dof_coarse, dof_fine, constr_coarse, constr_fine);

        auto matrix_of = [&](const fe::LinearTransferBase& t, auto method, unsigned ns, unsigned nd) {
            return transfer_matrix([&](const Vector<double>& src, Vector<double>& dst) {
                (t.*method)(src, dst);
            }, ns, nd);
        };

        const auto P    = matrix_of(transfer,    &fe::LinearTransferBase::to_fine_mesh,   n_c, n_f);
        const auto R    = matrix_of(transfer,    &fe::LinearTransferBase::to_coarse_mesh, n_f, n_c);
        const auto P_mg = matrix_of(transfer_mg, &fe::LinearTransferBase::to_fine_mesh,   n_c, n_f);
        const auto R_mg = matrix_of(transfer_mg, &fe::LinearTransferBase::to_coarse_mesh, n_f, n_c);
        const auto T_mg = matrix_of(transfer_mg, &fe::LinearTransferBase::Tfine,          n_f, n_c);

        // 1. Both prolongations are the canonical bilinear interpolation, and agree
        check_bilinear_prolongation(P,    "LinearTransfer::to_fine_mesh");
        check_bilinear_prolongation(P_mg, "LinearTransferMG::to_fine_mesh");

        double diff = 0.0;
        for (unsigned i = 0; i < n_f; i++) {
            for (unsigned j = 0; j < n_c; j++) {
                diff = std::max(diff, std::abs(P(i, j) - P_mg(i, j)));
            }
        }
        AssertThrow(diff < tol, ExcMessage(
            "LinearTransfer and LinearTransferMG disagree on the prolongation (max diff "
            + std::to_string(diff) + ")"));
        std::cout << "  PASS  both prolongations agree\n";

        // 2. to_coarse_mesh() is injection in both implementations, so it is not the adjoint
        check_injection(R,    "LinearTransfer::to_coarse_mesh");
        check_injection(R_mg, "LinearTransferMG::to_coarse_mesh");

        AssertThrow(max_diff_transpose(R, P) > 0.1, ExcMessage(
            "to_coarse_mesh() equals P^T, but it is documented as pointwise injection"));
        std::cout << "  PASS  to_coarse_mesh is not P^T\n";

        // 3. Tfine() is the adjoint of the prolongation
        const double diff_T = max_diff_transpose(T_mg, P);
        AssertThrow(diff_T < tol, ExcMessage(
            "LinearTransferMG::Tfine is not the transpose of to_fine_mesh (max diff "
            + std::to_string(diff_T) + ")"));
        std::cout << "  PASS  LinearTransferMG::Tfine == P^T\n";

        // 4. The two restrictions differ in scale by 2^dim on interior rows, which is why the
        //    transports must not be mixed up
        double row_sum_R = 0.0, row_sum_T = 0.0;
        for (unsigned j = 0; j < n_f; j++) {
            row_sum_R += R(4, j);
            row_sum_T += T_mg(4, j);
        }
        std::cout << "\n  interior row sums: to_coarse_mesh " << row_sum_R
                  << ", Tfine " << row_sum_T << " (ratio " << row_sum_T / row_sum_R << ")\n";
    }
    catch (std::exception& e) {
        std::cerr << "\nFAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\nAll transfer operators have the expected form.\n";
    return 0;
}
