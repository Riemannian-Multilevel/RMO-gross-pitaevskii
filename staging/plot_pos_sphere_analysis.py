#!/usr/bin/env python3
"""Figures and tables of tex/pos_sphere_variants.tex from the output of staging/pos_sphere_analysis.sh.

Usage: staging/plot_pos_sphere_analysis.py <data dir> <output dir>
"""
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pos_sphere_plot import data_and_out_dirs, read_csv, read_rows, status, write_table

DISCS = ["Q1", "P1", "P2", "Q2"]
RADII = [4, 6, 8, 10]
METRICS = ["ENERGY_ADAPTIVE", "MASS", "FROBENIUS"]
METRIC_NAMES = {"ENERGY_ADAPTIVE": "energy-adaptive, $\\tau=1$", "ENERGY_ADAPTIVE_ARMIJO": "energy-adaptive, Armijo",
                "MASS": "mass, Armijo", "FROBENIUS": "Frobenius, Armijo"}
CASES = ["unit", "positive_theta0.99", "positive_clip_theta0.99", "positive_floor_theta0.99", "positive_floor-nocap",
         "positive_clip-floor_theta0.99", "positive_floor-cont"]
SHORT = {"unit": "unit", "positive_theta0.99": "A", "positive_clip_theta0.99": "clip",
         "positive_floor_theta0.99": "floor+A", "positive_floor-nocap": "floor",
         "positive_clip-floor_theta0.99": "clip+floor", "positive_floor-cont": "cont."}


def disc_name(r):
    return f"{'Q' if r['mesh'] == 'QUADRILATERAL' else 'P'}{r['degree']}"


def metric_key(r):
    """Metric of a run; the energy-adaptive metric with Armijo is kept apart from the fixed step."""
    if r["metric"] == "ENERGY_ADAPTIVE" and r["line_search"] == "1":
        return "ENERGY_ADAPTIVE_ARMIJO"
    return r["metric"]


def read_matrix(data):
    """Rows of summary.csv by (discretization, radius, metric) and case."""
    runs = defaultdict(dict)
    for f in sorted((data / "matrix").glob("*/summary.csv")):
        for r in read_rows(f):
            runs[(disc_name(r), int(r["radius"]), metric_key(r))][r["case"]] = r
    return runs


# ---------------------------------------------------------------------------------------------------------------
# Overview: iterations relative to the unit sphere
# ---------------------------------------------------------------------------------------------------------------
def fig_overview(runs, out, metrics, name, figsize):
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, layout="constrained", sharey=True, squeeze=False)
    axes = axes[0]
    rows = [(d, rad) for d in DISCS for rad in RADII]
    norm = LogNorm(vmin=0.5, vmax=20)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    for ax, metric in zip(axes, metrics):
        cases_shown = [c for c in CASES if any(c in runs[(d, rad, metric)] for d, rad in rows)]
        ratio = np.full((len(rows), len(cases_shown)), np.nan)
        for i, (d, rad) in enumerate(rows):
            cases = runs[(d, rad, metric)]
            unit = cases["unit"]
            base = int(unit["iterations"]) if status(unit) == "converged" else 1000
            for j, c in enumerate(cases_shown):
                if c not in cases:
                    continue
                r = cases[c]
                st = status(r)
                if st == "converged":
                    ratio[i, j] = int(r["iterations"]) / base
                    ax.text(j, i, r["iterations"], ha="center", va="center", fontsize=7)
                elif st == "stopped":
                    ax.text(j, i, f"{float(r['residual']):.0e}".replace("e-0", "e-"), ha="center", va="center",
                            fontsize=6, color="#444444")
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#dddddd", zorder=0))
                else:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#b2182b", zorder=0))
                    ax.text(j, i, "undef.", ha="center", va="center", fontsize=6, color="white")
        ax.imshow(ratio, cmap=cmap, norm=norm, aspect="auto", zorder=1, alpha=0.75)
        ax.set_xticks(range(len(cases_shown)), [SHORT[c] for c in cases_shown], rotation=45, ha="right")
        ax.set_yticks(range(len(rows)), [f"{d}, r{rad}" for d, rad in rows], fontsize=7)
        for k in range(1, len(DISCS)):
            ax.axhline(k * len(RADII) - 0.5, color="k", lw=0.6)
        ax.set_title(METRIC_NAMES[metric])
        ax.set_xlim(-0.5, len(cases_shown) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, shrink=0.6, label="iterations / iterations of the unit sphere")
    fig.savefig(out / name)
    plt.close(fig)


def table_summary(runs, out):
    lines = [r"\begin{tabular}{@{}l" + "c" * len(CASES) + r"@{}}", r"\toprule",
             "metric & " + " & ".join(SHORT[c] for c in CASES) + r" \\", r"\midrule"]
    for metric in METRICS:
        cells = []
        for c in CASES:
            counts = defaultdict(int)
            for d in DISCS:
                for rad in RADII:
                    cases = runs[(d, rad, metric)]
                    if c not in cases:
                        continue
                    r = cases[c]
                    counts[status(r)] += 1
            if not counts:
                cells.append("--")
                continue
            cells.append(f"{counts['converged']}/{counts['stopped']}/{counts['failed']}")
        lines.append(METRIC_NAMES[metric] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(out / "tab_analysis_summary.tex", lines)


def table_discretizations(runs, out):
    """DoFs, positive off-diagonal entries of A_0 (number; largest ratio to the diagonal over the radii, since the
    diagonal contains the potential term) and smallest entry of the ground state per radius."""
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"discretization & DoFs & \multicolumn{2}{c}{positive off-diagonals} & "
             r"\multicolumn{4}{c}{$\min_i x^*_i$ (unit sphere), radius} \\",
             r"\cmidrule(lr){3-4}\cmidrule(l){5-8}",
             r" & & number & max.\ ratio & 4 & 6 & 8 & 10 \\", r"\midrule"]
    for d in DISCS:
        units = [runs[(d, rad, "ENERGY_ADAPTIVE")]["unit"] for rad in RADII]
        ratio = max(float(r["max_pos_ratio"]) for r in units)
        mins = [f"${float(r['min_value']):.1e}$".replace("e-0", "e-").replace("e", r"\mathrm{e}") for r in units]
        lines.append(f"{d} & {units[0]['n_dofs']} & {units[0]['n_pos_offdiag']} & {ratio:.3f} & "
                     + " & ".join(mins) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(out / "tab_analysis_disc.tex", lines)


# ---------------------------------------------------------------------------------------------------------------
# Floor: residual floor and continuation
# ---------------------------------------------------------------------------------------------------------------
def fig_floor(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    ax = axes[0]
    for d, color, name in [("q1", "#1f77b4", "Q1"), ("p1", "#d62728", "P1")]:
        eps, res = [], []
        for f in (data / "floor").glob(f"{d}_f*/summary.csv"):
            for r in read_rows(f):
                if r["case"] == "positive_floor-nocap":
                    eps.append(float(r["floor"]))
                    res.append(float(r["residual"]))
        order = np.argsort(eps)
        eps, res = np.array(eps)[order], np.array(res)[order]
        ax.loglog(eps, res, "o-", color=color, label=name)
        stalled = res > 1e-9
        c = np.median(res[stalled] / eps[stalled])
        fit = eps >= 1e-15
        ax.loglog(eps[fit], c * eps[fit], ":", color=color, label=rf"{name}: $r\approx{c:.0f}\,\varepsilon$")
    ax.axhline(1e-9, color="gray", ls="--", lw=0.8)
    ax.text(1e-49, 1.5e-9, "tolerance", fontsize=7, color="gray")
    ax.set_xlabel(r"floor $\varepsilon$")
    ax.set_ylabel("final residual (1000 iterations or converged)")
    ax.set_title("fixed floor, no cap: residual floor (radius 10)")
    ax.legend(fontsize=7)

    ax = axes[1]
    unit = read_csv(data / "factor" / "q1_c1e-3" / "iter_unit.csv")
    ax.semilogy(unit["k"], unit["residual"], color="k", label="unit sphere, residual")
    fl = read_csv(data / "floor" / "q1_f1e-12" / "iter_positive_floor-nocap.csv")
    ax.semilogy(fl["k"], fl["residual"], color="#7f7f7f", ls="--", label=r"fixed floor $10^{-12}$, residual")
    for c, color in [("1e-3", "#1f77b4"), ("1e-5", "#2ca02c")]:
        it = read_csv(data / "factor" / f"q1_c{c}" / "iter_positive_floor-cont.csv")
        e = int(c.split("e")[1])
        ax.semilogy(it["k"], it["residual"], color=color, label=rf"continuation $c=10^{{{e}}}$, residual")
        ax.semilogy(it["k"], it["floor"], color=color, ls=":", label=rf"continuation $c=10^{{{e}}}$, floor $\varepsilon_k$")
    ax.set_xlabel("iteration $k$")
    ax.set_title(r"continuation $\varepsilon_k=\min(\varepsilon_{k-1},c\,r_k)$, Q1, radius 10")
    ax.legend(fontsize=6.5)
    fig.savefig(out / "fig_analysis_floor.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Sign of the discrete ground state
# ---------------------------------------------------------------------------------------------------------------
def fig_ground_state(data, out):
    fig, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
    for d, color in [("q1", "#1f77b4"), ("p2", "#9467bd"), ("q2", "#d62728")]:
        f = read_csv(data / "matrix" / f"{d}_r10_ea" / "fields.csv")
        diag = np.isclose(f["px"], f["py"])
        order = np.argsort(f["px"][diag])
        s = f["px"][diag][order] * math.sqrt(2)
        x = f["x_unit"][diag][order]
        name = d.upper()
        ax.semilogy(s, np.abs(x), color=color, lw=1.0, label=name)
        neg = x < 0
        ax.semilogy(s[neg], np.abs(x[neg]), "x", color=color, ms=3.5)
    ax.set_xlabel("position along the diagonal")
    ax.set_ylabel(r"$|x^*_i|$ (crosses: $x^*_i<0$)")
    ax.set_title("ground state on the unit sphere, radius 10, energy-adaptive")
    ax.set_ylim(1e-18, 1)
    ax.legend(fontsize=7)
    fig.savefig(out / "fig_analysis_ground_state.pdf")
    plt.close(fig)


def main():
    data, out = data_and_out_dirs(__doc__)
    runs = read_matrix(data)
    fig_overview(runs, out, ["ENERGY_ADAPTIVE"], "fig_analysis_overview_ea.pdf", (6.3, 4.6))
    fig_overview(runs, out, ["ENERGY_ADAPTIVE_ARMIJO"], "fig_analysis_overview_ea_armijo.pdf", (6.3, 4.6))
    fig_overview(runs, out, ["MASS"], "fig_analysis_overview_mass.pdf", (5.2, 4.6))
    fig_overview(runs, out, ["FROBENIUS"], "fig_analysis_overview_frobenius.pdf", (5.2, 4.6))
    table_summary(runs, out)
    table_discretizations(runs, out)
    fig_floor(data, out)
    fig_ground_state(data, out)
    n = sum(len(v) for v in runs.values())
    print(f"{n} runs in the matrix")


if __name__ == "__main__":
    main()
