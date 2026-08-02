//
// Created by Ferdinand Vanmaele on 28.06.26.
//
// interpolation_operator_analysis.cc
//
// Determines what form VectorTools::interpolate_to_different_mesh takes
// for Q1 (bilinear) finite elements on uniformly refined quadrilateral grids
// in 2D.
//
// Setup:
//   Coarse mesh: 1 uniform refinement → 2×2 = 4 cells, 3×3 = 9 DoFs
//   Fine mesh:   2 uniform refinements → 4×4 = 16 cells, 5×5 = 25 DoFs
//
// The function works by evaluating the source FE function at the target
// mesh's nodal support points. We compute the full transfer matrices P
// (coarse→fine, 25×9) and R (fine→coarse, 9×25) by applying the operator
// to all canonical unit vectors, then classify each row by its nonzero
// pattern:
//
//   Injection:        nnz = 1, value = 1
//   Edge average:     nnz = 2, values = 0.5  (linear interpolation)
//   Cell-center avg:  nnz = 4, values = 0.25 (bilinear interpolation)
//
// We also check whether R = P^T (which would make it full-weighting /
// Galerkin restriction) or R = row_normalize(P^T).
//
// Compile with deal.II ≥ 9.4, e.g.:
//   mkdir build && cd build
//   cmake -DDEAL_II_DIR=/path/to/dealii ..
//   make

#include <deal.II/base/logstream.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

// ─────────────────────────────────────────────────────────────────────────────
// Compute the matrix of VectorTools::interpolate_to_different_mesh by
// applying it to every canonical unit vector e_j of the source space.
// The (i,j) entry of the resulting matrix is the i-th target DoF value
// when the source vector is the j-th unit vector.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<double>>
compute_transfer_matrix(const DoFHandler<2> &src, const DoFHandler<2> &tgt)
{
  const unsigned int ns = src.n_dofs();
  const unsigned int nt = tgt.n_dofs();

  std::vector<std::vector<double>> M(nt, std::vector<double>(ns, 0.0));

  Vector<double> e_j(ns), col(nt);
  for (unsigned int j = 0; j < ns; ++j)
    {
      e_j    = 0.0;
      e_j[j] = 1.0;
      VectorTools::interpolate_to_different_mesh(src, e_j, tgt, col);
      for (unsigned int i = 0; i < nt; ++i)
        M[i][j] = col[i];
    }
  return M;
}

// ─────────────────────────────────────────────────────────────────────────────
// Print a matrix with coordinates labeling rows and columns
// ─────────────────────────────────────────────────────────────────────────────
void
print_matrix(const std::vector<std::vector<double>> &M,
             const std::string                       &label,
             const std::vector<Point<2>>             &row_pts,
             const std::vector<Point<2>>             &col_pts)
{
  const int nrow = static_cast<int>(M.size());
  const int ncol = static_cast<int>(M[0].size());

  std::cout << "\n" << label << "  [" << nrow << " × " << ncol << "]\n";

  // Column header: x-coordinates
  std::cout << std::setw(18) << " ";
  for (int j = 0; j < ncol; ++j)
    std::cout << std::setw(7) << std::fixed << std::setprecision(2)
              << col_pts[j][0];
  std::cout << "\n";
  // Column header: y-coordinates
  std::cout << std::setw(18) << "col (x,y):";
  for (int j = 0; j < ncol; ++j)
    std::cout << std::setw(7) << std::fixed << std::setprecision(2)
              << col_pts[j][1];
  std::cout << "\n";

  for (int i = 0; i < nrow; ++i)
    {
      // Row label
      std::cout << "row (" << std::fixed << std::setprecision(2)
                << row_pts[i][0] << "," << row_pts[i][1] << "): ";
      for (int j = 0; j < ncol; ++j)
        {
          double v = (std::abs(M[i][j]) < 1e-12) ? 0.0 : M[i][j];
          std::cout << std::setw(7) << std::fixed << std::setprecision(3) << v;
        }
      std::cout << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Classify each row by its nonzero pattern
// ─────────────────────────────────────────────────────────────────────────────
void
analyze_matrix(const std::vector<std::vector<double>> &M,
               const std::string                       &label,
               const std::vector<Point<2>>             &row_pts)
{
  const int    nrow = static_cast<int>(M.size());
  const double tol  = 1e-10;

  int n_inject = 0; // nnz=1, val=1        → injection (coincident node)
  int n_edge   = 0; // nnz=2, vals=0.5     → linear edge average
  int n_center = 0; // nnz=4, vals=0.25    → bilinear cell-center average
  int n_other  = 0;

  std::cout << "\n=== Row-by-row analysis of " << label << " ===\n";

  for (int i = 0; i < nrow; ++i)
    {
      int    nnz    = 0;
      double rsum   = 0.0;
      double minval = 1e10, maxval = -1e10;

      for (double v : M[i])
        {
          rsum += v;
          if (std::abs(v) > tol)
            {
              ++nnz;
              minval = std::min(minval, v);
              maxval = std::max(maxval, v);
            }
        }
      if (nnz == 0)
        minval = maxval = 0.0;

      const bool uniform = (nnz > 0) && (maxval - minval < tol);

      if (nnz == 1 && std::abs(maxval - 1.0) < tol)
        ++n_inject;
      else if (nnz == 2 && uniform && std::abs(maxval - 0.5) < tol)
        ++n_edge;
      else if (nnz == 4 && uniform && std::abs(maxval - 0.25) < tol)
        ++n_center;
      else
        {
          ++n_other;
          std::cout << "  Row " << std::setw(2) << i
                    << " at (" << row_pts[i][0] << "," << row_pts[i][1] << ")"
                    << ": nnz=" << nnz << ", sum=" << rsum
                    << ", vals ∈ [" << minval << ", " << maxval << "]\n";
        }
    }

  std::cout << "  nnz=1 val=1.00   → injection (coinciding node):        "
            << n_inject << "\n";
  std::cout << "  nnz=2 val=0.50   → linear interpolation (edge):        "
            << n_edge << "\n";
  std::cout << "  nnz=4 val=0.25   → bilinear interpolation (cell ctr):  "
            << n_center << "\n";
  std::cout << "  other pattern:                                           "
            << n_other << "\n";

  // Partition-of-unity check (all row sums = 1)
  bool pou = true;
  for (const auto &row : M)
    {
      double s = 0.0;
      for (double v : row)
        s += v;
      if (std::abs(s - 1.0) > 1e-10)
        {
          pou = false;
          break;
        }
    }
  std::cout << "  Partition of unity (all row sums = 1): " << (pou ? "YES" : "NO")
            << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Check whether R ≈ P^T or R ≈ row_normalize(P^T)
// ─────────────────────────────────────────────────────────────────────────────
void
compare_R_with_Pt(const std::vector<std::vector<double>> &P,
                  const std::vector<std::vector<double>> &R)
{
  // P is (nf × nc),  R is (nc × nf),  P^T is (nc × nf)
  const int nf = static_cast<int>(P.size());
  const int nc = static_cast<int>(P[0].size());

  // ── exact match R == P^T ──────────────────────────────────────────────────
  double max_diff = 0.0;
  for (int i = 0; i < nc; ++i)
    for (int j = 0; j < nf; ++j)
      max_diff = std::max(max_diff, std::abs(R[i][j] - P[j][i]));

  std::cout << "\n=== R vs. P^T ===\n";
  std::cout << "  max |R[i,j] - P[j,i]| = " << max_diff << "\n";
  if (max_diff < 1e-10)
    {
      std::cout << "  → R = P^T exactly (Galerkin / full-weighting restriction)\n";
      return;
    }
  std::cout << "  → R ≠ P^T\n";

  // ── match after row-normalizing P^T: R[i][j] == P^T[i][j] / sum(P^T[i]) ──
  bool is_row_norm_Pt = true;
  for (int i = 0; i < nc && is_row_norm_Pt; ++i)
    {
      double pt_row_sum = 0.0;
      for (int j = 0; j < nf; ++j)
        pt_row_sum += P[j][i]; // P^T[i][j] = P[j][i]

      for (int j = 0; j < nf; ++j)
        if (std::abs(R[i][j] - P[j][i] / pt_row_sum) > 1e-10)
          {
            is_row_norm_Pt = false;
            break;
          }
    }
  std::cout << "  R = row_normalize(P^T)? " << (is_row_norm_Pt ? "YES" : "NO")
            << "\n";

  // ── check if R is injection (each row: one entry = 1) ────────────────────
  bool is_inj = true;
  for (int i = 0; i < nc && is_inj; ++i)
    {
      int    nnz = 0;
      double mx  = 0.0;
      for (int j = 0; j < nf; ++j)
        if (std::abs(R[i][j]) > 1e-10)
          {
            ++nnz;
            mx = std::max(mx, R[i][j]);
          }
      if (nnz != 1 || std::abs(mx - 1.0) > 1e-10)
        is_inj = false;
    }
  std::cout << "  R is pure injection?     " << (is_inj ? "YES" : "NO") << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
int
main()
{
  deallog.depth_console(0); // suppress deal.II internal output

  // ── Build meshes ───────────────────────────────────────────────────────────
  //   Coarse: refine_global(1) → 2×2 cells, 3×3 = 9 Q1 nodes
  //   Fine:   refine_global(2) → 4×4 cells, 5×5 = 25 Q1 nodes
  Triangulation<2> coarse_tria, fine_tria;
  GridGenerator::hyper_cube(coarse_tria, 0.0, 1.0);
  GridGenerator::hyper_cube(fine_tria, 0.0, 1.0);
  coarse_tria.refine_global(1);
  fine_tria.refine_global(2);

  FE_Q<2>       fe(1); // Q1 bilinear elements
  DoFHandler<2> coarse_dof(coarse_tria), fine_dof(fine_tria);
  coarse_dof.distribute_dofs(fe);
  fine_dof.distribute_dofs(fe);

  std::cout << "Coarse mesh: " << coarse_tria.n_active_cells()
            << " cells,  " << coarse_dof.n_dofs() << " DoFs (Q1)\n";
  std::cout << "Fine mesh:   " << fine_tria.n_active_cells()
            << " cells,  " << fine_dof.n_dofs() << " DoFs (Q1)\n";

  // ── Collect DoF support point coordinates ─────────────────────────────────
  const MappingQ1<2> mapping;

  std::vector<Point<2>> coarse_pts(coarse_dof.n_dofs());
  std::vector<Point<2>> fine_pts(fine_dof.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, coarse_dof, coarse_pts);
  DoFTools::map_dofs_to_support_points(mapping, fine_dof, fine_pts);

  std::cout << "\nCoarse DoF support points:\n";
  for (unsigned int i = 0; i < coarse_pts.size(); ++i)
    std::cout << "  DoF " << std::setw(2) << i << ": ("
              << coarse_pts[i][0] << ", " << coarse_pts[i][1] << ")\n";

  std::cout << "\nFine DoF support points:\n";
  for (unsigned int i = 0; i < fine_pts.size(); ++i)
    std::cout << "  DoF " << std::setw(2) << i << ": ("
              << fine_pts[i][0] << ", " << fine_pts[i][1] << ")\n";

  // ── Compute transfer matrices ─────────────────────────────────────────────
  std::cout << "\nComputing P (coarse → fine, " << fine_dof.n_dofs()
            << "×" << coarse_dof.n_dofs() << ")..." << std::flush;
  auto P = compute_transfer_matrix(coarse_dof, fine_dof);
  std::cout << " done.\n";

  std::cout << "Computing R (fine → coarse, " << coarse_dof.n_dofs()
            << "×" << fine_dof.n_dofs() << ")..." << std::flush;
  auto R = compute_transfer_matrix(fine_dof, coarse_dof);
  std::cout << " done.\n";

  // ── Print both matrices with coordinate labels ────────────────────────────
  print_matrix(P, "P (prolongation: coarse → fine)", fine_pts, coarse_pts);
  print_matrix(R, "R (restriction:  fine → coarse)", coarse_pts, fine_pts);

  // ── Row-by-row classification ─────────────────────────────────────────────
  analyze_matrix(P, "P (prolongation)", fine_pts);
  analyze_matrix(R, "R (restriction)", coarse_pts);

  // ── Compare R with P^T ────────────────────────────────────────────────────
  compare_R_with_Pt(P, R);

  // ── Interpretation ────────────────────────────────────────────────────────
  std::cout << R"(
=== Summary ===

VectorTools::interpolate_to_different_mesh evaluates the source FE function
at each support point of the target mesh. For Q1 elements:

  Prolongation P (coarse → fine):
    evaluates coarse bilinear FE function at fine nodes.
    • Fine node coincides with coarse node     → row = [1] (injection)
    • Fine node is midpoint of coarse edge     → row = [0.5, 0.5] (linear avg)
    • Fine node is center of coarse cell       → row = [0.25, 0.25, 0.25, 0.25]
    This is STANDARD BILINEAR INTERPOLATION (the canonical FEM prolongation).

  Restriction R (fine → coarse):
    evaluates fine bilinear FE function at coarse nodes.
    For nested meshes, coarse nodes are a SUBSET of fine nodes, so the fine
    FE function is evaluated at its own support points → it simply returns
    the existing fine DoF value.
    → Each row has exactly one nonzero entry equal to 1: INJECTION.
    → R ≠ P^T  (not full-weighting / not Galerkin restriction)
    → R ≠ row_normalize(P^T)  (not half-weighting)
)";

  return 0;
}