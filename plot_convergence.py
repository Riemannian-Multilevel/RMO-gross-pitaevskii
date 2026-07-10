#!/usr/bin/env python3
"""Generates figures 7a/7b and 8a-8d from FAS/EARGD convergence logs
(dealii org-mode tables written by cycle_finalize()/gradient_descent(), see
include/gpe/ropt/{descent,solver}.h and include/gpe/main/fas.h).

Expected input files (default --data-dir is the current directory):
    sl_b<beta>_l<level>_optical_lattice.org
    ml_mass_<operator>_b<beta>_l<level>_depth<N>_optical_lattice.org

Figure 7 (a: vs. iteration, b: vs. elapsed time) compares transport operator
variants at a fixed depth (4). Figure 8 (a/c: energy error, b/d: residual;
a/b vs. iteration, c/d vs. elapsed time) compares single-level EARGD against
multilevel EARGD at increasing depth, for one fixed transport operator
(--operator, default "proj").

Examples
--------
Run from the directory containing the .org files:
    python3 ../plot_convergence.py

Or point at them explicitly, and pick a different operator for figure 8:
    python3 plot_convergence.py --data-dir cmake-build-release-deal.ii \\
        --operator adj1 --out-dir figures
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import orgparse
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from orgparse.extra import Table

COARSE_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def load_org_table(path):
    """Parses a dealii org-mode convergence table (a single '| col | col |
    ... |' table, no separator row, so every parsed row before the header
    is data) into a DataFrame. All columns are coerced to numeric except
    "coarse" (a "*"/blank marker column), with unparseable cells (e.g. the
    leading "-" in the rate columns) becoming NaN.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"convergence log not found: {path}")

    doc = orgparse.load(path)
    header, *data = Table(doc.body.splitlines()).rows

    df = pd.DataFrame(data, columns=header)
    for col in df.columns:
        if col == "coarse":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def reference_energy(sl_path):
    """The reference ground-state energy, taken as the final (best-converged,
    residual ~1e-14) row of the single-level run."""
    df = load_org_table(sl_path)
    return df["energy"].iloc[-1]


def multi_series_xmax(series, xcol, ycol, ymin):
    """The x-axis upper bound to use so the slowest of `series` to reach
    `ymin` sits near the plot's right edge, instead of every series' long
    near-flat tail below the visible y-range padding the axis out further --
    used for figure 7, which (unlike figure 8) has no single reference series
    to anchor on.
    """
    xmax = 0.0
    for path, _key, _label in series:
        df = load_org_table(path)
        below = df[df[ycol] <= ymin]
        x = below[xcol].iloc[0] if not below.empty else df[xcol].iloc[-1]
        xmax = max(xmax, x)
    return xmax


def sl_convergence_point(sl_path, e_ref, residual_ymin):
    """The row of the single-level ("EARGD") run where its residual first
    drops to `residual_ymin`, used to derive:
      - the matching energy-error cutoff for the energy plots (8a, 8c), so
        both pairs of figure 8 plots are cut off at the "same" point of
        convergence rather than an arbitrarily chosen energy value;
      - the x-axis upper bound for all four figure 8 plots, so the "EARGD"
        reference line's convergence point sits near the right edge instead
        of the multilevel curves finishing early and leaving it stranded in
        the middle of a long near-flat tail that falls below the visible
        y-range anyway.
    Returns a dict with "iter", "elapsed" and "energy_ymin".
    """
    df = load_org_table(sl_path)
    below = df[df["residual"] <= residual_ymin]
    row = below.iloc[0] if not below.empty else df.iloc[-1]
    return {
        "iter": row["iter"],
        "elapsed": row["elapsed"],
        "energy_ymin": abs(row["energy"] - e_ref),
    }


def build_figure(series, xcol, ycol, ylabel, xlabel, out_path, energy_ref=None, logy=True, ymin=None, xmax=None,
                 coarse_steps=False, dpi=200):
    """series: list of (path, key, legend_label). key must be unique per
    series (used for grouping/coloring); legend_label is the text shown in
    the legend and may repeat across series (e.g. two operators sharing a
    "Version" label) without merging their data -- grouping is done by key,
    the legend text is relabeled afterwards.

    ymin, when given, fixes the lower y-axis bound (the upper bound stays
    auto-scaled) -- e.g. to cut off a long near-machine-precision tail that
    isn't otherwise informative, while several plots sharing the same ymin
    remain directly comparable at the bottom of the scale.

    xmax, when given, fixes the right x-axis bound (with a small margin) --
    e.g. so a slowly-converging reference series isn't padded out by a long
    invisible tail below `ymin`, leaving its last visible point stranded well
    short of the plot's right edge.

    coarse_steps, when true, overlays a marker (a different shape per series,
    from COARSE_MARKERS, in the series' own line color) on every row where
    the "coarse" column is "*", i.e. where a coarse-grid correction was taken
    that cycle -- so the effect of coarse steps on convergence is visible
    directly on the curve instead of only in the raw table.
    """
    frames = []
    for path, key, _label in series:
        df = load_org_table(path)
        if energy_ref is not None:
            df["_y"] = np.abs(df["energy"] - energy_ref)
        else:
            df["_y"] = df[ycol]
        df["_key"] = key
        cols = [xcol, "_y", "_key"] + (["coarse"] if coarse_steps else [])
        frames.append(df[cols])
    combined = pd.concat(frames, ignore_index=True)

    key_order = [key for _, key, _label in series]
    key_to_label = {key: label for _, key, label in series}

    plt.figure(figsize=(7, 5))
    palette = dict(zip(key_order, sns.color_palette(n_colors=len(key_order))))
    ax = sns.lineplot(data=combined, x=xcol, y="_y", hue="_key", style="_key",
                      hue_order=key_order, style_order=key_order, palette=palette)
    if logy:
        ax.set_yscale("log")
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if xmax is not None:
        ax.set_xlim(right=xmax * 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    legend_labels = [key_to_label[k] for k in labels]

    if coarse_steps:
        for i, key in enumerate(key_order):
            marker = COARSE_MARKERS[i % len(COARSE_MARKERS)]
            sub = combined[(combined["_key"] == key) & (combined["coarse"] == "*")]
            if sub.empty:
                continue
            ax.scatter(sub[xcol], sub["_y"], marker=marker, color=palette[key],
                      edgecolor="black", linewidths=0.5, s=45, zorder=5)
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="lightgray",
                              markeredgecolor="black", markeredgewidth=0.5, markersize=7))
        legend_labels.append("coarse step")

    ax.legend(handles, legend_labels, title=None)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=".", help="directory containing the .org convergence logs")
    parser.add_argument("--out-dir", default=".", help="directory to write figures into")
    parser.add_argument("--beta", default="1000", help="beta token in filenames (default: 1000)")
    parser.add_argument("--level", default="11", help="finest-level token in filenames (default: 11)")
    parser.add_argument("--operator", default="adj2",
                        help="transport operator used for the depth2/3/4 series in figure 8 "
                             "(one of: adj1, adj2, diff, diff_gal, proj, proj_gal; default: adj2)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"], help="output figure format")
    parser.add_argument("--residual-ymin", type=float, default=1e-8,
                        help="lower y-axis bound for the figure 8 residual plots (8b, 8d); the "
                             "matching cutoff for the figure 8 energy-error plots (8a, 8c) is "
                             "derived automatically from where the single-level run's residual "
                             "crosses this value (default: 1e-8)")
    parser.add_argument("--fig7-residual-ymin", type=float, default=1e-6,
                        help="lower y-axis bound for the figure 7 residual plots (7a, 7b) "
                             "(default: 1e-6)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="output figure resolution in dots per inch (default: 200, "
                             "higher than matplotlib's default of 100)")
    parser.add_argument("--coarse-steps", action="store_true",
                        help="mark cycles where a coarse-grid correction was taken (the org "
                             "table's \"coarse\" column) with a per-series marker on all plots")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format

    def sl_path():
        return data_dir / f"sl_b{args.beta}_l{args.level}_optical_lattice.org"

    def ml_path(operator, depth):
        return data_dir / f"ml_mass_{operator}_b{args.beta}_l{args.level}_depth{depth}_optical_lattice.org"

    ylabel_residual = r"$\|\mathrm{res}(\phi_k)\|_M$"
    ylabel_energy   = r"$|E^{GP}(\phi_k) - E_{ref}|$"
    xlabel_iter     = r"Iteration $k$"
    xlabel_elapsed  = "Elapsed time (s)"

    # -------------------------------------------------------------------
    # Figure 7: transport-operator comparison at fixed depth 4.
    # -------------------------------------------------------------------
    fig7_series = [
        (ml_path("adj2", 4), "adj2", "Version III"),
        (ml_path("proj", 4), "proj", "Version IV"),
        (ml_path("adj1", 4), "adj1", "Version V"),
    ]

    fig7_xmax_iter = multi_series_xmax(fig7_series, "iter", "residual", args.fig7_residual_ymin)
    fig7_xmax_elapsed = multi_series_xmax(fig7_series, "elapsed", "residual", args.fig7_residual_ymin)

    build_figure(fig7_series, "iter", "residual", ylabel_residual, xlabel_iter,
                out_dir / f"fig7a.{ext}", ymin=args.fig7_residual_ymin, xmax=fig7_xmax_iter,
                coarse_steps=args.coarse_steps, dpi=args.dpi)
    build_figure(fig7_series, "elapsed", "residual", ylabel_residual, xlabel_elapsed,
                out_dir / f"fig7b.{ext}", ymin=args.fig7_residual_ymin, xmax=fig7_xmax_elapsed,
                coarse_steps=args.coarse_steps, dpi=args.dpi)

    # -------------------------------------------------------------------
    # Figure 8: single-level vs. multilevel depth scaling, fixed operator.
    # -------------------------------------------------------------------
    e_ref = reference_energy(sl_path())
    print(f"reference energy (from {sl_path().name}, last row): {e_ref:.16e}")

    cutoff = sl_convergence_point(sl_path(), e_ref, args.residual_ymin)
    energy_ymin = cutoff["energy_ymin"]
    print(f"energy-error cutoff matching residual ymin {args.residual_ymin:.1e}: {energy_ymin:.6e}")
    print(f"EARGD reaches residual ymin at iter={cutoff['iter']:.0f}, elapsed={cutoff['elapsed']:.3g}s")

    fig8_series = [
        (sl_path(),                  "sl",     "EARGD"),
        (ml_path(args.operator, 2),  "depth2", "2-level EARGD"),
        (ml_path(args.operator, 3),  "depth3", "3-level EARGD"),
        (ml_path(args.operator, 4),  "depth4", "4-level EARGD"),
    ]

    build_figure(fig8_series, "iter", "residual", ylabel_energy, xlabel_iter,
                out_dir / f"fig8a.{ext}", energy_ref=e_ref, ymin=energy_ymin, xmax=cutoff["iter"],
                coarse_steps=args.coarse_steps, dpi=args.dpi)
    build_figure(fig8_series, "iter", "residual", ylabel_residual, xlabel_iter,
                out_dir / f"fig8b.{ext}", ymin=args.residual_ymin, xmax=cutoff["iter"],
                coarse_steps=args.coarse_steps, dpi=args.dpi)
    build_figure(fig8_series, "elapsed", "residual", ylabel_energy, xlabel_elapsed,
                out_dir / f"fig8c.{ext}", energy_ref=e_ref, ymin=energy_ymin, xmax=cutoff["elapsed"],
                coarse_steps=args.coarse_steps, dpi=args.dpi)
    build_figure(fig8_series, "elapsed", "residual", ylabel_residual, xlabel_elapsed,
                out_dir / f"fig8d.{ext}", ymin=args.residual_ymin, xmax=cutoff["elapsed"],
                coarse_steps=args.coarse_steps, dpi=args.dpi)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
