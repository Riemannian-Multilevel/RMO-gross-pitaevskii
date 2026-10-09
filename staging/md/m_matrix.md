# Design: checking the irreducible M-matrix property (`proto/m-matrix`)

Branch `proto/m-matrix`, 1 commit on top of `efc7292` (state of 2026-10-07); it also marks
`test_m_matrix` as obsolete:

| Commit | Content |
|---|---|
| `ce558ac` | `staging/main_m_matrix.cc`, ctests, `tex/taussky_theorem.tex` |

## 1. Motivation

- Positivity of the discrete ground state and of the energy-adaptive iterates rests on
  A_u = S + M_{V,L} + β M_{φφ,L}(u) being an M-matrix with a positive inverse: the step
  u ↦ A_u⁻¹ M_L u then maps non-negative u ≠ 0 to strictly positive vectors (Hauck, Liang,
  Peterseim, Numer. Math. 158, 2026).
- Lumping only adds non-negative values to the diagonal, so the question reduces to the stiffness
  matrix S on a given mesh: non-positive off-diagonal entries, plus irreducibility for a
  *strictly* positive inverse.
- `test_m_matrix` checked the sign pattern and a few columns of A₀⁻¹ for fixed meshes, but not
  irreducibility, and the mesh was not selectable.

## 2. Theory (`tex/taussky_theorem.tex`)

- **Taussky (1949):** an irreducibly diagonally dominant matrix (irreducible, weakly dominant in
  every row, strictly in at least one) is non-singular. Proof via the index set where |x_i| is
  maximal, which is closed under the edges of the matrix graph.
- **Corollary:** a Z-matrix with positive diagonal that is irreducibly diagonally dominant is a
  non-singular M-matrix with A⁻¹ > 0 entrywise (Jacobi splitting, ρ(B) < 1 by Taussky, Neumann
  series, (I + B)ⁿ⁻¹ > 0 for irreducible B ≥ 0).
- **Finite elements:** row sums of S vanish (partition of unity), so unconstrained rows are
  weakly dominant and rows coupled to Dirichlet nodes strictly dominant. S is a Z-matrix on
  non-obtuse meshes (2D P1: opposite angles sum ≤ π; Q1: aspect ratio ≤ √2; 3D: dihedral angles
  ≤ π/2). With Neumann boundary no row is strictly dominant, and S is singular (constants).
- The document has 12 references, verified against Crossref, publisher pages and zbMATH; DOIs
  are linked.

## 3. `staging/main_m_matrix.cc`

- Assembles `--matrix stiffness` (S, with the package's quadrature via `get_quadrature()`) or
  `--matrix A0` (S + M_V; lumped with `--mass-lumping`) on the mesh of `GrossPitaevskiiPackage`,
  for every level of `--levels`/`--multilevel`, with the mesh options of the solvers.
- Analysis on the **unconstrained DoFs only** (Dirichlet rows hold just their diagonal and would
  appear as isolated vertices); entries with |a_ij| ≤ `--tol`·max|a| count as zero, because
  right angles give couplings that are zero up to rounding but stored in the sparsity pattern.
- Checks, i.e. the hypotheses of the corollary:

| Check | Hypothesis |
|---|---|
| max \|a_ij − a_ji\| ≤ tol | symmetric: the matrix graph is undirected |
| min a_ii > 0 | positive diagonal |
| no a_ij > tol (i ≠ j) | Z-matrix |
| a_ii ≥ Σ_{j≠i} \|a_ij\| in every row | weakly diagonally dominant |
| strict inequality in some row | strictly dominant row |
| one component (`boost::connected_components`) | irreducible |
| `--inverse-columns k`: min of k columns of A⁻¹ | the conclusion A⁻¹ > 0, numerically |

- Boost Graph Library (header-only, in deal.II's Boost): an undirected `adjacency_list` with an
  edge for every coupling above the tolerance; `--print-components` lists component sizes.
- Exit code 0 if all hypotheses hold on every level, 1 otherwise, 2 on errors; usable as ctest.
- ctests `m_matrix_q1_2d`, `m_matrix_p1_2d`, `m_matrix_q1_3d` check the lumped A₀, whose
  off-diagonal entries and graph are those of S.

## 4. Results (Dirichlet, radius 8)

| Case | Result |
|---|---|
| 2D Q1, S, level 5 | all hypotheses hold (56 of 225 rows strictly dominant, 1 component) |
| 2D P1 simplex, S, level 4 | holds; min relative entry of A⁻¹ 7.8e-4 |
| 3D Q1, S, level 3 | holds |
| 2D P1 simplex, lumped A₀, levels 2–5 | holds on every level |
| 3D P1 simplex, lumped A₀ | fails: 268 positive off-diagonal entries (obtuse dihedral angles) |
| 2D Q1, consistent A₀ | fails: 1464 positive off-diagonal entries (consistent potential term) |
| 2D Q1, S, Neumann | fails only the strict row: S is singular |

- Q2 and P2-bubble are no M-matrices even with lumping; positivity can only be expected for
  degree 1.
- Mesh quality is not the reason for higher CG counts on the simplex mesh: at equal DoFs the
  condition numbers of A₀ are within a factor 1.4 of Q1.

## 5. Relation to positivity in practice

- The M-matrix property is necessary but not sufficient for non-negative iterates: inner solves
  with the default `--tol-inner-res 1e-2` leave negative entries of order 1e-5. With lumping and
  `--tol-inner-res` ≤ 1e-8 the iterates are non-negative up to rounding (measured).
- The theory covers the energy-adaptive step with step size ≤ 1, not the mass or Frobenius
  gradients, nor the coarse corrections of FAS.

## 6. Open issues

- Remove `test_m_matrix` (marked obsolete; still built and run as ctest `m_matrix`).
- Non-symmetric matrices would need strongly connected components (Tarjan) instead of a
  single-component check; the program only reports the symmetry error.
- Report the cells with obtuse angles for positive off-diagonal entries, to explain failures
  such as the 3D simplex mesh directly.
- The tolerance `--tol` decides which couplings are edges; a value near rounding (1e-14) is
  right for exact zeros, larger values make the graph artificially disconnected.
- Optional GraphBLAS variant (`GrB_select`/`GrB_reduce`, BFS-based components): not installed;
  identical reports would cross-check both implementations.
