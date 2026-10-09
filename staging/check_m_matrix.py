#!/usr/bin/env python3
"""Irreducible M-matrix check (Taussky's theorem) for the matrices exported by main_matrix.

Checks whether the stiffness matrix S, the potential-weighted mass matrix M_V, or A0 = S + M_V
(main_matrix --matrix-market exports S and M_V) is an irreducible non-singular M-matrix. By Taussky's
theorem (tex/taussky_theorem.tex), a symmetric matrix with

  - a positive diagonal and non-positive off-diagonal entries (Z-matrix),
  - weak diagonal dominance in every row and strict dominance in at least one row,
  - a connected matrix graph (vertices: DoFs, edges: non-zero off-diagonal entries a_ij),

is a non-singular M-matrix, and its inverse is strictly positive. Entries with |a_ij| <= tol max|a|
are treated as zero, so that rounding errors (e.g. the vanishing face couplings of Q1 in 3D) do not
create spurious positive entries or edges.

Constrained DoFs are left out of the analysis. The matrices are assembled with the (Dirichlet)
constraints, which deal.II eliminates: the row and column of a constrained DoF are zero except for a
positive diagonal entry, and the couplings of free DoFs to constrained ones are dropped. Constrained
DoFs are thus recognized as the rows of S without non-zero off-diagonal entries (not of M_V, which is
diagonal when lumped; its constrained diagonal entries are arbitrary). They are not unknowns of
the discrete problem (their value is fixed to zero), so the matrix to analyze is the block A_FF of
the free DoFs. Kept in the graph, each constrained DoF would be an isolated vertex, and the graph
would be disconnected for a trivial reason. Note that strict dominance comes from the boundary: the
rows of S sum to zero, except for the free DoFs next to the boundary, which lost the couplings to the
constrained DoFs. Without constraints (Neumann), S is singular, and only A0 with V > 0 can pass.

The output follows main_m_matrix, which this script replaces. Exit code: 0 if all hypotheses hold
for every matrix, 1 otherwise.

Example:
    main_matrix --dimension 2 --boundary dirichlet --levels 5 --mass-lumping --matrix-market
    python3 staging/check_m_matrix.py --matrix A0 matrix_lvl5_dim2
"""

import argparse
import sys

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components


def constrained_dofs(A):
    """Rows of a matrix assembled with constraints that have no non-zero off-diagonal entry."""
    off_diagonal = (A - sp.diags(A.diagonal())).tocsr()
    off_diagonal.eliminate_zeros()
    return np.diff(off_diagonal.indptr) == 0


def load(base, matrix):
    """(matrix S, M_V or A0 = S + M_V of main_matrix files <base>_S.mtx, <base>_MV.mtx, mask of free DoFs)"""
    S = sio.mmread(f"{base}_S.mtx").tocsr()
    free = ~constrained_dofs(S)
    if matrix == "S":
        return S, free
    M_V = sio.mmread(f"{base}_MV.mtx").tocsr()
    return (M_V if matrix == "MV" else (S + M_V).tocsr()), free


def check(A, tol, print_components):
    """Prints whether A (the block A_FF of the free DoFs) satisfies the hypotheses of Taussky's theorem;
    returns True if all hold."""
    print(f"  {A.shape[0]} unconstrained DoFs")

    scale = abs(A).max()
    eps = tol * scale
    symmetry_error = abs(A - A.T).max() / scale
    diagonal = A.diagonal()
    off_diagonal = (A - sp.diags(diagonal)).tocsr()
    positive = off_diagonal.data[off_diagonal.data > eps]
    margin = diagonal - abs(off_diagonal).sum(axis=1).A1  # a_ii - sum_{j != i} |a_ij|
    n_components, component = connected_components(abs(off_diagonal) > eps, directed=False)
    sizes = np.sort(np.bincount(component))[::-1]

    def sci(x):
        return f"{x:.2e}"

    hypotheses = [
        ("symmetric", symmetry_error <= tol, f"max |a_ij - a_ji| = {sci(symmetry_error)}"),
        ("positive diagonal", diagonal.min() / scale > tol, f"min a_ii = {sci(diagonal.min() / scale)}"),
        ("non-positive off-diagonal entries (Z-matrix)", len(positive) == 0,
         f"{len(positive)} positive, max {sci(positive.max(initial=0.0) / scale)}"),
        ("weakly diagonally dominant in every row", not (margin < -eps).any(),
         f"{(margin < -eps).sum()} rows not dominant"),
        ("strictly diagonally dominant in at least one row", (margin > eps).any(),
         f"{(margin > eps).sum()} of {A.shape[0]} rows"),
        ("irreducible (connected matrix graph)", n_components == 1, f"{n_components} connected component(s)"),
    ]
    for name, ok, detail in hypotheses:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({detail})")
    if print_components and n_components > 1:
        print("        component sizes: " + " ".join(map(str, sizes[:20])) + (" ..." if n_components > 20 else ""))

    all_ok = all(ok for _, ok, _ in hypotheses)
    print("  => irreducible non-singular M-matrix (Taussky); its inverse is strictly positive" if all_ok
          else "  => hypotheses of Taussky's theorem not satisfied")
    return all_ok


def min_inverse_entry(A, free, n_columns):
    """Smallest entry of A_FF^{-1} e_i over n_columns free columns i, relative to the largest entry."""
    lu = spla.splu(A[free][:, free].tocsc())
    free_index = np.cumsum(free) - 1  # position of a free DoF in A_FF
    n = A.shape[0]
    min_rel = 1.0
    for k in range(1, n_columns + 1):
        # evenly spaced columns, moved to the next free DoF (as in main_m_matrix)
        i = next((j for j in range(k * n // (n_columns + 1), n) if free[j]), None)
        if i is None:
            continue
        e = np.zeros(lu.shape[0])
        e[free_index[i]] = 1.0
        y = lu.solve(e)
        min_rel = min(min_rel, y.min() / abs(y).max())
    return min_rel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bases", nargs="+", metavar="base",
                        help="base name of the main_matrix files, e.g. matrix_lvl5_dim2 for matrix_lvl5_dim2_S.mtx")
    parser.add_argument("--matrix", choices=["S", "MV", "A0"], default="S",
                        help="matrix to check: S, M_V or A0 = S + M_V (default: S)")
    parser.add_argument("--tol", type=float, default=1e-14,
                        help="relative threshold: entries with |a_ij| <= tol max|a| are treated as zero")
    parser.add_argument("--print-components", action="store_true",
                        help="print the sizes of the connected components if there are several")
    parser.add_argument("--inverse-columns", type=int, default=0,
                        help="if all hypotheses hold, also check positivity of this many columns of A^{-1} (0: skip)")
    args = parser.parse_args()

    all_ok = True
    for base in args.bases:
        A, free = load(base, args.matrix)
        print(f"{base}: matrix {args.matrix}, {A.shape[0]} DoFs")
        ok = check(A[free][:, free], args.tol, args.print_components)
        if ok and args.inverse_columns > 0:  # A_FF may be singular otherwise
            print(f"  INFO  min relative entry of {args.inverse_columns} columns of A^{{-1}}: "
                  f"{min_inverse_entry(A, free, args.inverse_columns):.2e}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
