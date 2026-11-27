#!/usr/bin/env python3
"""Generates Figures 7-9 of the manuscript (Gross--Pitaevskii, optical
lattice) from FAS/EARGD convergence logs (dealii org-mode tables written by
cycle_finalize()/gradient_descent(), see include/gpe/ropt/{descent,solver}.h
and include/gpe/main/fas.h).

Expected input files (default --data-dir is the current directory):
    sl_b<beta>_l<level>_<suffix>.org
    ml_<metric>_<operator>_b<beta>_l<level>_depth<N>_<suffix>.org

where <metric> is the --metric token (mass or frob; default: mass).

Figure 7 (residual vs. iteration | CPU time) compares transport operator
variants at a fixed depth (4). Figures 8 (vs. iteration) and 9 (vs. CPU time)
show energy error | residual for single-level EARGD against multilevel EARGD
at increasing depth (--depths), for one fixed transport operator (--operator,
default "adj2", i.e. Version III).

With --format pdf (default) or tex, each figure is written as an \\input-able
pgfplots tikzpicture fig<N>_<metric>.tex in the style of the manuscript's
results_gp/optical_lattice/plot_GP_b1000_ol_*.tex (colors cpl1/2/4/6 and
unia-purple, macros \\res and \\GP from the manuscript preamble; filled markers
for coarse steps), plus, for pdf, a standalone-compiled fig<N>_<metric>.pdf
(requires pdflatex). With --format png, the old seaborn output is written
instead (one file per panel: fig7a/b, fig8a-d).

Examples
--------
Run from the directory containing the .org files:
    python3 ../plot/plot_convergence.py

Or point at them explicitly, and plot the Frobenius family instead of mass:
    python3 plot/plot_convergence.py --data-dir cmake-build-release-deal.ii \\
        --metric frob --operator adj1 --out-dir figures
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import orgparse
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from orgparse.extra import Table

COARSE_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def logs_exist(figure, series):
    """True if the logs of all `series` exist; otherwise warns that `figure` is skipped."""
    missing = [Path(path).name for path, *_ in series if not Path(path).exists()]
    if missing:
        print(f"warning: {figure} skipped, missing: {', '.join(missing)}", file=sys.stderr)
    return not missing


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

    # A --simple-table log holds the rows of all levels: keep the finest level
    if "level" in df.columns:
        df = df[df["level"] == df["level"].max()].reset_index(drop=True)
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


# ---------------------------------------------------------------------------
# pgfplots backend, matching the format of Figures 7-9 in the manuscript
# (results_gp/optical_lattice/plot_GP_b1000_ol_*.tex).
# ---------------------------------------------------------------------------

# Per-series line style: (color, mark, extra line options). Colors are the
# \definecolor names from the manuscript preamble (see TIKZ_PREAMBLE).
TIKZ_STYLES = {
    # figure 7: transport operators
    "adj2":   ("unia-purple", "square*",   ""),
    "proj":   ("cpl1",        "diamond*",  ""),
    "adj1":   ("cpl4",        "triangle*", ""),
    "diff":   ("cpl2",        "*",         ""),
    # figures 8/9: depth scaling
    "sl":     ("cpl6",        "x",         ""),
    "depth2": ("cpl4",        "diamond*",  ""),
    "depth3": ("cpl2",        "triangle*", ""),
    "depth4": ("unia-purple", "square*",   "mark size=1.8pt, "),
    "depth5": ("cpl1",        "pentagon*", ""),
}

TIKZ_PREAMBLE = r"""\documentclass[border=1pt]{standalone}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{calc}
\definecolor{unia-purple}{RGB}{173, 0, 124}
\definecolor{cpl1}{rgb}{0.8889,0.4356,0.2781}
\definecolor{cpl2}{rgb}{0.0,0.6056,0.9787}
\definecolor{cpl4}{rgb}{0.2422,0.6433,0.3044}
\definecolor{cpl6}{rgb}{0.2, 0.2, 0.2}
\DeclareMathOperator{\res}{res}
\newcommand{\GP}{{\rm GP}}
\setlength{\textwidth}{155mm}  % as in the manuscript preamble
"""

TIKZ_FONT8 = r"{\fontsize{8 pt}{10.4 pt}\selectfont}"


def series_xy(path, xcol, ycol, energy_ref=None):
    """(x, y, coarse) arrays for one convergence log; y is the energy error
    |E - E_ref| when energy_ref is given, else column ycol. Non-positive y
    (e.g. the reference run's own last row, error exactly 0) is dropped since
    it cannot be shown on a log axis."""
    df = load_org_table(path)
    y = np.abs(df["energy"] - energy_ref) if energy_ref is not None else df[ycol]
    keep = y > 0
    coarse = df["coarse"].astype(str).str.strip() == "*"
    return np.asarray(df[xcol][keep]), np.asarray(y[keep]), np.asarray(coarse[keep])


def tikz_table(xs, ys):
    rows = "".join(f"{x:.6g} {y:.6g} \\\\\n" for x, y in zip(xs, ys))
    return "    table[row sep={\\\\}]\n    { \\\\\n" + rows + "};\n"


def tikz_axis(series, xcol, ycol, xlabel, ylabel, *, name, at, energy_ref=None, ymin=None, xmax=None,
              coarse_steps=True, mark_repeat=None, legend_name=None):
    """One pgfplots axis in the style of the manuscript's Figures 7-9.
    series: list of (path, key, legend_label); key selects TIKZ_STYLES."""
    xmin = -0.01 * xmax if xmax is not None else None
    opts = [
        f"name={name}, at={{{at}}}",
        "width={0.48\\textwidth}, height={0.32\\textwidth}",
        f"xlabel={{{xlabel}}}, xlabel style={{font=\\small}}",
        f"ylabel={{{ylabel}}}, ylabel style={{yshift=-1mm, font=\\small}}",
        f"xticklabel style={{font={TIKZ_FONT8}}}, yticklabel style={{font={TIKZ_FONT8}}}",
        "xtick align={inside}, ytick align={inside}",
        "xmajorgrids={true}, ymajorgrids={true}",
        "x grid style={color=black, draw opacity={0.1}, line width={0.5}, solid}",
        "y grid style={color=black, draw opacity={0.1}, line width={0.5}, solid}",
        "scaled y ticks={false}, ymode={log}, log basis y={10}",
    ]
    if xmax is not None:
        opts.append(f"xmin={{{xmin:.6g}}}, xmax={{{xmax * 1.02:.6g}}}")
    if ymin is not None:
        # half a decade of padding below the cutoff, so curves ending at the
        # tolerance stay visible instead of being clipped by the frame
        opts.append(f"ymin={{{ymin / 3:.6g}}}")
    if legend_name is not None:
        opts += [
            f"legend to name={legend_name}, legend columns={{-1}}, legend cell align={{left}}",
            # the manuscript's 0.5cm column sep fits 4 entries; tighten
            # beyond that so the legend row stays within \textwidth
            "legend style={/tikz/every even column/.append style="
            f"{{column sep={0.5 if len(series) <= 4 else 0.25}cm}}, "
            f"draw=black, fill=white, font={TIKZ_FONT8}, cells={{anchor={{center}}}}}}",
        ]

    out = ["\\begin{axis}[\n    " + ",\n    ".join(opts) + "\n    ]\n"]
    repeat = f", mark repeat={mark_repeat}" if mark_repeat else ""
    for path, key, label in series:
        color, mark, extra = TIKZ_STYLES[key]
        xs, ys, coarse = series_xy(path, xcol, ycol, energy_ref)
        if xmax is not None:
            # drop the tail well beyond the right edge (keep one point past
            # it so the clipped line still runs into the frame)
            n = int(np.searchsorted(xs, xmax * 1.02, side="right")) + 1
            xs, ys, coarse = xs[:n], ys[:n], coarse[:n]
        fill = "" if mark == "x" else ", mark options = {fill = white}"
        out.append(f"\\addplot[color={color}, line width={{1}}, solid, {extra}mark={mark}{fill}{repeat}]\n")
        out.append(tikz_table(xs, ys))
        if coarse_steps and coarse.any():
            out.append(f"\\addplot[only marks, color={color}, line width={{1}}, solid, {extra}mark={mark}, "
                       f"mark options = {{fill = {color}}}, forget plot]\n")
            out.append(tikz_table(xs[coarse], ys[coarse]))
        if legend_name is not None:
            out.append(f"\\addlegendentry{{{label}}}\n")
    out.append("\\end{axis}\n")
    return "".join(out)


def tikz_figure(left, right, legend_name):
    """Two axes side by side with one shared legend row centered above."""
    return (
        "\\begin{tikzpicture}\n" + left + right +
        f"\\node[anchor=south] at ($(ax1.north west)!0.5!(ax2.north east) + (0, 2mm)$) "
        f"{{\\pgfplotslegendfromname{{{legend_name}}}}};\n"
        "\\end{tikzpicture}\n"
    )


def write_tikz(tikz, out_tex, compile_pdf):
    """Writes the \\input-able tikzpicture to out_tex and, if compile_pdf,
    renders it to a .pdf next to it via a standalone wrapper and pdflatex."""
    out_tex.write_text(tikz)
    print(f"wrote {out_tex}")
    if not compile_pdf:
        return
    with tempfile.TemporaryDirectory() as tmp:
        wrapper = Path(tmp) / "fig.tex"
        wrapper.write_text(TIKZ_PREAMBLE + "\\begin{document}\n" + tikz + "\\end{document}\n")
        proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", wrapper.name],
                              cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"pdflatex failed for {out_tex}:\n{proc.stdout[-3000:]}")
        out_pdf = out_tex.with_suffix(".pdf")
        shutil.copyfile(Path(tmp) / "fig.pdf", out_pdf)
    print(f"wrote {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=".", help="directory containing the .org convergence logs")
    parser.add_argument("--out-dir", default=".", help="directory to write figures into")
    parser.add_argument("--beta", default="1000", help="beta token in filenames (default: 1000)")
    parser.add_argument("--level", default="11", help="finest-level token in filenames (default: 11)")
    parser.add_argument("--suffix", default="optical_lattice",
                        help="filename suffix, e.g. optical_lattice_lumped (default: optical_lattice)")
    parser.add_argument("--metric", default="mass",
                        help="metric token in the multilevel filenames, i.e. the family "
                             "ml_<metric>_*.org (typically mass or frob; default: mass). Does "
                             "not affect the single-level reference sl_*.org.")
    parser.add_argument("--operator", default="adj2",
                        help="transport operator used for the depth2/3/4 series in figure 8 "
                             "(one of: adj1, adj2, diff, diff_gal, proj, proj_gal; default: adj2)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "tex", "png"],
                        help="output format: pdf (pgfplots .tex in the manuscript's Figure 7-9 "
                             "format, compiled with pdflatex), tex (the pgfplots .tex only), or png "
                             "(matplotlib/seaborn, one file per panel)")
    parser.add_argument("--depths", default="2,3,4,5",
                        help="comma-separated multilevel depths for figures 8/9 (default: 2,3,4,5)")
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
        return data_dir / f"sl_b{args.beta}_l{args.level}_{args.suffix}.org"

    def ml_path(operator, depth):
        return data_dir / f"ml_{args.metric}_{operator}_b{args.beta}_l{args.level}_depth{depth}_{args.suffix}.org"

    if ext == "png":
        main_matplotlib(args, out_dir, sl_path(), ml_path)
        return

    compile_pdf = ext == "pdf"
    ylabel_residual = r"$\|\res(\phi_k)\|_{M}$"
    ylabel_energy   = r"$E^{\GP}(\phi_k) - E_{\rm ref}$"
    xlabel_iter     = "iterations"
    xlabel_elapsed  = "CPU time [s]"

    # -------------------------------------------------------------------
    # Figure 7: transport-operator comparison at fixed depth 4; residual
    # vs. iterations (left) and CPU time (right). As in the manuscript, no
    # coarse-step markers and a marker on every iterate.
    # -------------------------------------------------------------------
    fig7_series = [
        (ml_path("adj2", 4), "adj2", "Version III"),
        (ml_path("proj", 4), "proj", "Version IV"),
        (ml_path("adj1", 4), "adj1", "Version V"),
    ]
    made = 0
    if logs_exist("figure 7", fig7_series):
        ymin7 = args.fig7_residual_ymin
        left = tikz_axis(fig7_series, "iter", "residual", xlabel_iter, ylabel_residual, name="ax1", at="(0, 0)",
                         ymin=ymin7, xmax=multi_series_xmax(fig7_series, "iter", "residual", ymin7),
                         coarse_steps=args.coarse_steps, legend_name="fig7legend")
        right = tikz_axis(fig7_series, "elapsed", "residual", xlabel_elapsed, ylabel_residual, name="ax2",
                          at="(0.48\\textwidth, 0)", ymin=ymin7,
                          xmax=multi_series_xmax(fig7_series, "elapsed", "residual", ymin7),
                          coarse_steps=args.coarse_steps)
        write_tikz(tikz_figure(left, right, "fig7legend"), out_dir / f"fig7_{args.metric}.tex", compile_pdf)
        made += 1

    # -------------------------------------------------------------------
    # Figures 8/9: single-level vs. multilevel depth scaling, fixed
    # operator; energy error (left) and residual (right) vs. iterations
    # (figure 8) and CPU time (figure 9). Filled markers: coarse steps.
    # -------------------------------------------------------------------
    depths = [int(d) for d in args.depths.split(",")]
    fig89_series = [(sl_path(), "sl", "EARGD")] + [
        (ml_path(args.operator, d), f"depth{d}", f"{d}-level EARGD") for d in depths]
    if logs_exist("figures 8 and 9", fig89_series):
        e_ref = reference_energy(sl_path())
        print(f"reference energy (from {sl_path().name}, last row): {e_ref:.16e}")
        cutoff = sl_convergence_point(sl_path(), e_ref, args.residual_ymin)
        print(f"EARGD reaches residual ymin at iter={cutoff['iter']:.0f}, elapsed={cutoff['elapsed']:.3g}s")

        for fig, xcol, xlabel in [(8, "iter", xlabel_iter), (9, "elapsed", xlabel_elapsed)]:
            legend = f"fig{fig}legend"
            left = tikz_axis(fig89_series, xcol, "residual", xlabel, ylabel_energy, name="ax1", at="(0, 0)",
                             energy_ref=e_ref, ymin=cutoff["energy_ymin"], xmax=cutoff[xcol],
                             mark_repeat=3, legend_name=legend)
            right = tikz_axis(fig89_series, xcol, "residual", xlabel, ylabel_residual, name="ax2",
                              at="(0.48\\textwidth, 0)", ymin=args.residual_ymin, xmax=cutoff[xcol], mark_repeat=3)
            write_tikz(tikz_figure(left, right, legend), out_dir / f"fig{fig}_{args.metric}.tex", compile_pdf)
        made += 1

    if not made:
        sys.exit("error: no figure could be made")

def main_matplotlib(args, out_dir, sl_path, ml_path):
    """Original seaborn output: one PNG per panel (fig7a/b, fig8a-d)."""
    ext = args.format

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

    made = 0
    if logs_exist("figure 7", fig7_series):
        fig7_xmax_iter = multi_series_xmax(fig7_series, "iter", "residual", args.fig7_residual_ymin)
        fig7_xmax_elapsed = multi_series_xmax(fig7_series, "elapsed", "residual", args.fig7_residual_ymin)

        build_figure(fig7_series, "iter", "residual", ylabel_residual, xlabel_iter,
                    out_dir / f"fig7a_{args.metric}.{ext}", ymin=args.fig7_residual_ymin, xmax=fig7_xmax_iter,
                    coarse_steps=args.coarse_steps, dpi=args.dpi)
        build_figure(fig7_series, "elapsed", "residual", ylabel_residual, xlabel_elapsed,
                    out_dir / f"fig7b_{args.metric}.{ext}", ymin=args.fig7_residual_ymin, xmax=fig7_xmax_elapsed,
                    coarse_steps=args.coarse_steps, dpi=args.dpi)
        made += 1

    # -------------------------------------------------------------------
    # Figure 8: single-level vs. multilevel depth scaling, fixed operator.
    # -------------------------------------------------------------------
    fig8_series = [
        (sl_path,                  "sl",     "EARGD"),
        *[(ml_path(args.operator, int(d)), f"depth{d}", f"{d}-level EARGD") for d in args.depths.split(",")],
    ]
    if logs_exist("figure 8", fig8_series):
        e_ref = reference_energy(sl_path)
        print(f"reference energy (from {sl_path.name}, last row): {e_ref:.16e}")

        cutoff = sl_convergence_point(sl_path, e_ref, args.residual_ymin)
        energy_ymin = cutoff["energy_ymin"]
        print(f"energy-error cutoff matching residual ymin {args.residual_ymin:.1e}: {energy_ymin:.6e}")
        print(f"EARGD reaches residual ymin at iter={cutoff['iter']:.0f}, elapsed={cutoff['elapsed']:.3g}s")

        build_figure(fig8_series, "iter", "residual", ylabel_energy, xlabel_iter,
                    out_dir / f"fig8a_{args.metric}.{ext}", energy_ref=e_ref, ymin=energy_ymin, xmax=cutoff["iter"],
                    coarse_steps=args.coarse_steps, dpi=args.dpi)
        build_figure(fig8_series, "iter", "residual", ylabel_residual, xlabel_iter,
                    out_dir / f"fig8b_{args.metric}.{ext}", ymin=args.residual_ymin, xmax=cutoff["iter"],
                    coarse_steps=args.coarse_steps, dpi=args.dpi)
        build_figure(fig8_series, "elapsed", "residual", ylabel_energy, xlabel_elapsed,
                    out_dir / f"fig8c_{args.metric}.{ext}", energy_ref=e_ref, ymin=energy_ymin, xmax=cutoff["elapsed"],
                    coarse_steps=args.coarse_steps, dpi=args.dpi)
        build_figure(fig8_series, "elapsed", "residual", ylabel_residual, xlabel_elapsed,
                    out_dir / f"fig8d_{args.metric}.{ext}", ymin=args.residual_ymin, xmax=cutoff["elapsed"],
                    coarse_steps=args.coarse_steps, dpi=args.dpi)
        made += 1

    if not made:
        sys.exit("error: no figure could be made")

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
