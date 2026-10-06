#!/usr/bin/env python3
"""Verification of Gross-Pitaevskii ground states for consistent and lumped mass matrices.

Solves the discrete ground state problem with main_coarse and checks the final iterate x against the
matrices of the discretization exported by main_matrix, and against known exact solutions. Both
programs run with the same mesh options, so their DoF numbering agrees. Per potential, mass variant
(consistent, lumped) and level, the script runs

  main_matrix   M, S and M_V in Matrix Market format (M and M_V lumped with --mass-lumping);
                the linear part of the operator is A0 = S + M_V
  main_coarse   single-level gradient descent (--metric none) per beta, with --output-bin

Constrained (Dirichlet) DoFs are the rows of S without non-zero off-diagonal entries: deal.II keeps
only the diagonal of constrained rows when it assembles the matrices with constraints.

Checks per solution (required checks fail the run; the rest are only reported):

  converged    final residual <= --tol-residual
  normalized   x^T M x = 1
  energy       E = 1/2 x^T A0 x + beta/4 sum_i m_i x_i^4 matches main_coarse (lumped, or beta = 0)
  residual     M-norm of A(x) x - lambda M x is small (same cases as energy)
  positive     min x_i >= -tol on unconstrained DoFs (required for lumped only)
  symmetric    x invariant under x_d -> -x_d and coordinate swaps (all potentials are symmetric)
  ritz         E >= E_exact for zero, Dirichlet, beta = 0, consistent (exact quadrature)
  reference    nodal error and energy error, where a reference is known (reported):
                 zero (Dirichlet, beta = 0):   prod_d cos(pi x_d / 2R) / R^{dim/2}, E = dim/2 (pi/2R)^2
                 square (beta = 0):            pi^{-dim/4} exp(-|x|^2/2), E = dim/2
                 optical_lattice (beta = 0):   separable, 1D eigenfunction computed with scipy
                 zero (Neumann, any beta):     constant |Omega|^{-1/2}, E = beta / (4 |Omega|)

Across levels: observed convergence rates of the energy (self-differences), of the reference error,
and of the gap between consistent and lumped energies (expected: 2, i.e. O(h^2)).
Choose levels that resolve the solution and the potential: h = 2R / 2^(level-1) <= 1 for zero and
square, but h <= 0.125 for the optical lattice (period 2, amplitude 100; level >= 8 at radius 8). On
coarser meshes the lumped potential term, which samples V at the nodes, aliases the lattice, and the
rates are not yet asymptotic.

Output in --out-dir: results.csv, rates.csv, report.md, and SVG plots in plots/ (2D only): solutions
and consistent-lumped differences per level, a cut along x_2 = 0, and energy convergence.
The exit code is 1 if a required check fails.

Example:
    python3 staging/verify_solutions.py --levels 5 6 7 --betas 0 100 10000 --jobs 8
"""

import argparse
import itertools
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_solution import load_coords, load_solution  # noqa: E402

VARIANTS = ("consistent", "lumped")


# --------------------------------------------------------------------------------------------------
# Running the programs
# --------------------------------------------------------------------------------------------------
def mesh_args(args, potential, variant, level):
    a = ["--dimension", str(args.dimension), "--mesh", args.mesh, "--boundary", args.boundary,
         "--radius", str(args.radius), "--potential", potential, "--levels", str(level)]
    return a + (["--mass-lumping"] if variant == "lumped" else [])


def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed in {cwd}:\n{result.stderr[-2000:]}")
    return result.stdout


def run_case(args, potential, variant, level, betas):
    """Exports the matrices and solves for every beta; returns the case directory."""
    case_dir = Path(args.out_dir) / potential / variant / f"lvl{level}"
    case_dir.mkdir(parents=True, exist_ok=True)
    base = mesh_args(args, potential, variant, level)

    run([args.build_dir / "main_matrix", *base, "--matrix-market"], case_dir)
    for beta in betas:
        stdout = run([args.build_dir / "main_coarse", *base, "--beta", str(beta), "--metric", "none",
                      "--initial", args.initial, "--max-iter", str(args.max_iter),
                      "--tol-residual", str(args.tol_residual), "--tol-inner-res", str(args.tol_inner_res),
                      "--max-inner", "5000", f"--output-bin=b{beta}"], case_dir)
        (case_dir / f"b{beta}.txt").write_text(stdout)
    return case_dir


def parse_table(text):
    """Rows of the convergence table printed by main_coarse, as a list of dicts."""
    lines = [l for l in text.splitlines() if l.startswith("|")]
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[1:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0][:1].isdigit():
            rows.append(dict(zip(header, cells)))
    return rows


# --------------------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------------------
def lattice_1d(radius, nu=100.0, n=4000):
    """Ground state of -u'' + (x^2/2 + nu sin^2(pi x/2)) u on [-R, R] (Dirichlet), L2-normalized."""
    x = np.linspace(-radius, radius, n + 1)
    h = x[1] - x[0]
    xi = x[1:-1]
    V = 0.5 * xi**2 + nu * np.sin(0.5 * np.pi * xi) ** 2
    T = sp.diags([-np.ones(n - 2), 2 * np.ones(n - 1), -np.ones(n - 2)], [-1, 0, 1]) / h**2
    lam, vec = spla.eigsh(T + sp.diags(V), k=1, sigma=0.0, which="LM")
    u = np.concatenate([[0.0], np.abs(vec[:, 0]), [0.0]])
    u /= math.sqrt(np.trapezoid(u**2, x))
    return lam[0], x, u


def reference(args, potential, beta, coords):
    """(values at the coordinates, energy) of a known solution, or None."""
    R, dim = args.radius, args.dimension
    if potential == "zero" and args.boundary == "neumann":
        volume = (2 * R) ** dim
        return np.full(len(coords), volume**-0.5), beta / (4 * volume)
    if beta != 0 or args.boundary != "dirichlet":
        return None
    if potential == "zero":
        u = np.prod(np.cos(0.5 * np.pi * coords / R), axis=1) / R ** (dim / 2)
        return u, dim / 2 * (np.pi / (2 * R)) ** 2
    if potential == "square":
        return np.pi ** (-dim / 4) * np.exp(-0.5 * np.sum(coords**2, axis=1)), dim / 2
    if potential == "optical_lattice":
        lam, x, u1 = lattice_1d(R)
        return np.prod([np.interp(coords[:, d], x, u1) for d in range(dim)], axis=0), dim / 2 * lam
    return None


# --------------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------------
def symmetry_error(coords, x):
    """Max |x(p) - x(Tp)| / max |x| over reflections x_d -> -x_d and swaps of two coordinates."""
    tree = cKDTree(coords)
    scale = np.abs(x).max()
    dim = coords.shape[1]
    maps = [np.diag([-1.0 if e == d else 1.0 for e in range(dim)]) for d in range(dim)]
    for d, e in itertools.combinations(range(dim), 2):
        P = np.eye(dim)
        P[[d, e]] = P[[e, d]]
        maps.append(P)
    error = 0.0
    for T in maps:
        dist, idx = tree.query(coords @ T.T)
        if dist.max() > 1e-8 * max(1.0, np.abs(coords).max()):
            return math.nan  # mesh is not symmetric under T
        error = max(error, np.abs(x - x[idx]).max() / scale)
    return error


def final_iterate(case_dir, beta):
    """(coordinates, last iterate of main_coarse) for one beta, with the sign fixed to sum(x) >= 0."""
    iterates = sorted(case_dir.glob(f"b{beta}_iter*.bin"), key=lambda p: int(p.stem.split("iter")[-1]))
    x = load_solution(iterates[-1])
    return load_coords(case_dir / f"b{beta}_coords.bin"), (x if x.sum() >= 0 else -x)


def constrained_dofs(A):
    """Rows of a matrix assembled with constraints that have no non-zero off-diagonal entry."""
    off_diagonal = (A - sp.diags(A.diagonal())).tocsr()
    off_diagonal.eliminate_zeros()
    return np.diff(off_diagonal.indptr) == 0


def check_case(args, potential, variant, level, beta, case_dir):
    """Checks for one solution; returns a dict of measured values and booleans."""
    rows = parse_table((case_dir / f"b{beta}.txt").read_text())
    coords, x = final_iterate(case_dir, beta)

    mtx = f"matrix_lvl{level}_dim{args.dimension}"
    M, S, M_V = (sio.mmread(case_dir / f"{mtx}_{name}.mtx").tocsr() for name in ("M", "S", "MV"))
    if S.shape[0] != len(x):
        raise ValueError(f"{case_dir}: {S.shape[0]} DoFs in {mtx}_S.mtx, but {len(x)} in the solution")
    A0 = S + M_V
    constrained = constrained_dofs(S)

    r = {"potential": potential, "variant": variant, "level": level, "beta": beta, "n_dofs": len(x),
         "iterations": len(rows) - 1, "residual": float(rows[-1]["residual"]), "energy": float(rows[-1]["energy"])}
    r["converged"] = r["residual"] <= args.tol_residual
    r["mass_error"] = abs(x @ (M @ x) - 1.0)
    r["normalized"] = r["mass_error"] <= 1e-8

    # Energy and residual from the matrices, where the nonlinear term is known: beta sum_i m_i x_i^4 with the
    # lumped weights m_i = M_ii, or zero for beta = 0
    if variant == "lumped" or beta == 0:
        m = M.diagonal() if variant == "lumped" else np.zeros(len(x))
        Ax = A0 @ x + beta * m * x**3
        energy = 0.5 * x @ (A0 @ x) + 0.25 * beta * m @ x**4
        Mx = M @ x
        res = Ax - (x @ Ax) / (x @ Mx) * Mx
        r["energy_error"] = abs(energy - r["energy"]) / max(abs(r["energy"]), 1e-300)
        r["energy_matches"] = r["energy_error"] <= 1e-8
        r["residual_recomputed"] = math.sqrt(res @ (M @ res))  # M-norm, as in GrossPitaevskiiFunctional::residual
        r["residual_small"] = r["residual_recomputed"] <= 10 * args.tol_residual
    r["min_value"] = x[~constrained].min()
    r["positive"] = r["min_value"] >= -args.pos_tol
    r["symmetry_error"] = symmetry_error(coords, x)
    r["symmetric"] = not r["symmetry_error"] > 1e-6

    ref = reference(args, potential, beta, coords)
    if ref is not None:
        u_ref, e_ref = ref
        r["reference_error"] = np.abs(x - u_ref).max() / np.abs(u_ref).max()
        r["reference_energy"] = e_ref
        r["energy_error_vs_reference"] = abs(r["energy"] - e_ref)
        if potential == "zero" and args.boundary == "dirichlet" and variant == "consistent":
            r["ritz"] = r["energy"] >= e_ref - 1e-12
    return r


REQUIRED = ["converged", "normalized", "energy_matches", "residual_small", "symmetric", "ritz"]


def failures(r):
    names = [c for c in REQUIRED if c in r and r[c] is False]
    if r["variant"] == "lumped" and not r["positive"]:
        names.append("positive")
    return names


def rates(df):
    """Observed convergence rates between consecutive levels, per potential, beta and variant."""
    out = []

    def add(pot, beta, quantity, err):
        """log2(err_a / err_b) for consecutive levels a, b of a level-indexed series of errors."""
        err = err.dropna()
        for a, b in zip(err.index, err.index[1:]):
            out.append({"potential": pot, "beta": beta, "quantity": quantity, "levels": f"{a}-{b}",
                        "rate": math.log2(err[a] / err[b]) if err[b] > 0 else math.nan})

    for (pot, beta), group in df.groupby(["potential", "beta"]):
        by_variant = {v: group[group["variant"] == v].set_index("level").sort_index() for v in VARIANTS}
        for v, g in by_variant.items():
            add(pot, beta, f"energy ({v})", g["energy"].diff().abs())  # |E_l - E_(l-1)|, indexed by l
            if "reference_error" in g:
                add(pot, beta, f"reference error ({v})", g["reference_error"])
        add(pot, beta, "consistent - lumped energy",
            (by_variant["consistent"]["energy"] - by_variant["lumped"]["energy"]).abs())
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------------------
def plot_all(args, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    plot_dir = Path(args.out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    levels = args.levels

    for potential, beta in itertools.product(args.potentials, args.betas):
        fig, axes = plt.subplots(3, len(levels), figsize=(3.2 * len(levels), 9), squeeze=False)
        for j, level in enumerate(levels):
            data = {v: final_iterate(Path(args.out_dir) / potential / v / f"lvl{level}", beta) for v in VARIANTS}
            coords = data["consistent"][0]
            tri = mtri.Triangulation(coords[:, 0], coords[:, 1])
            for i, (title, values) in enumerate([("consistent", data["consistent"][1]), ("lumped", data["lumped"][1]),
                                                 ("|consistent - lumped|", np.abs(data["consistent"][1] - data["lumped"][1]))]):
                ax = axes[i, j]
                image = ax.tripcolor(tri, values, shading="gouraud", cmap="viridis")
                fig.colorbar(image, ax=ax, shrink=0.8)
                ax.set_title(f"{title}, level {level}", fontsize=9)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"{potential}, beta = {beta}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{potential}_b{beta}_solutions.svg")
        plt.close(fig)

        # Cut along x_2 = 0 on the finest level, with reference where known
        fig, ax = plt.subplots(figsize=(6, 4))
        for v in VARIANTS:
            coords, x = final_iterate(Path(args.out_dir) / potential / v / f"lvl{levels[-1]}", beta)
            on_cut = np.abs(coords[:, 1]) < 1e-12
            order = np.argsort(coords[on_cut, 0])
            ax.plot(coords[on_cut, 0][order], x[on_cut][order], ".-", label=v, markersize=3)
        ref = reference(args, potential, beta, np.column_stack([np.linspace(-args.radius, args.radius, 801),
                                                                np.zeros((801, args.dimension - 1))]))
        if ref is not None:
            ax.plot(np.linspace(-args.radius, args.radius, 801), ref[0], "k--", label="reference", linewidth=1)
        ax.set_xlabel("x_1 (x_2 = 0)")
        ax.set_ylabel("u")
        ax.set_title(f"{potential}, beta = {beta}, level {levels[-1]}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{potential}_b{beta}_cut.svg")
        plt.close(fig)

    # Energy convergence: |E_L - E_ref| (reference energy, or finest level)
    for potential in args.potentials:
        fig, ax = plt.subplots(figsize=(6, 4))
        for beta, v in itertools.product(args.betas, VARIANTS):
            sel = df[(df["potential"] == potential) & (df["beta"] == beta) & (df["variant"] == v)].sort_values("level")
            if "reference_energy" in sel and sel["reference_energy"].notna().all():
                err, label = (sel["energy"] - sel["reference_energy"]).abs(), f"beta={beta} {v} (vs reference)"
            else:
                err, label = (sel["energy"] - sel["energy"].iloc[-1]).abs().iloc[:-1], f"beta={beta} {v} (vs finest)"
            levels_sel = sel["level"].iloc[: len(err)]
            if (err > 0).any():
                ax.semilogy(levels_sel, err, "o-", label=label)
        ax.set_xlabel("level")
        ax.set_ylabel("energy error")
        ax.set_title(f"{potential}: energy convergence")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{potential}_energy.svg")
        plt.close(fig)


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def markdown_table(df, floatfmt):
    """Markdown table of a DataFrame, without the optional tabulate dependency of DataFrame.to_markdown()."""
    def cell(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(args, df, df_rates):
    out = Path(args.out_dir)
    df.to_csv(out / "results.csv", index=False)
    df_rates.to_csv(out / "rates.csv", index=False)

    lines = ["# Verification of mass-lumped and consistent solutions", "",
             f"Mesh: {args.dimension}d {args.mesh}, boundary {args.boundary}, radius {args.radius}, "
             f"levels {args.levels}, initial value {args.initial}", ""]
    failed = [(r, failures(r)) for _, r in df.iterrows() if failures(r)]
    lines += [f"**{len(df) - len(failed)} of {len(df)} cases pass all required checks.**", ""]
    if failed:
        lines += ["| potential | variant | level | beta | failed checks |", "|---|---|---|---|---|"]
        lines += [f"| {r['potential']} | {r['variant']} | {r['level']} | {r['beta']} | {', '.join(f)} |" for r, f in failed]
        lines.append("")
    cols = ["potential", "variant", "level", "beta", "iterations", "residual", "energy", "mass_error",
            "min_value", "symmetry_error"] + [c for c in ("reference_error", "energy_error_vs_reference") if c in df]
    lines += ["## Results", "", markdown_table(df[cols], ".3e"), "",
              "## Convergence rates (expected about 2)", "",
              markdown_table(df_rates, ".2f") if len(df_rates) else "(need at least 3 levels)", ""]
    (out / "report.md").write_text("\n".join(lines))
    return len(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=Path(__file__).resolve().parent.parent / "cmake-build-debug-deal.ii")
    parser.add_argument("--out-dir", default="verify_out")
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7],
                        help="levels (h = 2R / 2^(level-1)); resolve the solution, e.g. h <= 1 for radius 8")
    parser.add_argument("--potentials", nargs="+", default=["zero", "square", "optical_lattice"])
    parser.add_argument("--betas", type=float, nargs="+", default=[0, 100, 10000])
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--mesh", default="quadrilateral", choices=["quadrilateral", "simplex"])
    parser.add_argument("--boundary", default="dirichlet", choices=["dirichlet", "neumann"])
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--initial", default="random", choices=["constant", "cosine", "random"],
                        help="starting value; cosine is the exact discrete solution for zero (Dirichlet, beta = 0)")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol-residual", type=float, default=1e-9)
    parser.add_argument("--tol-inner-res", type=float, default=1e-10,
                        help="relative inner solver tolerance; small values keep lumped iterates non-negative")
    parser.add_argument("--pos-tol", type=float, default=1e-10)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.betas = [int(b) if float(b).is_integer() else b for b in args.betas]

    cases = list(itertools.product(args.potentials, VARIANTS, args.levels))
    with ThreadPoolExecutor(args.jobs) as pool:
        dirs = dict(zip(cases, pool.map(lambda c: run_case(args, *c, args.betas), cases)))

    results = [check_case(args, pot, var, lvl, beta, dirs[(pot, var, lvl)])
               for (pot, var, lvl), beta in itertools.product(cases, args.betas)]
    df = pd.DataFrame(results)
    df_rates = rates(df)
    if not args.no_plots and args.dimension == 2:
        plot_all(args, df)
    n_failed = write_report(args, df, df_rates)
    print(f"{len(df) - n_failed} of {len(df)} cases pass; report in {Path(args.out_dir) / 'report.md'}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
