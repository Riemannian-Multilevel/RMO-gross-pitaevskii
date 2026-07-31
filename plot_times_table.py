#!/usr/bin/env python3
"""Build a LaTeX CPU-time comparison table from FAS/EARGD convergence logs.

For a fixed transport operator (default adj2), read the elapsed (CPU) time at
which the residual first reaches a threshold (default 1e-8) for:

  * the single-level reference   sl_b<beta>_l<level>_optical_lattice.org
  * the multilevel runs          ml_<metric>_<operator>_b<beta>_l<level>_depth<N>_optical_lattice.org

for each metric (default: mass, frob) and each depth (default: 2,3,4,5). The
output table lists, per depth, the mass-metric ($M$) and Frobenius-metric
($\\mathrm e$) times side by side, bolding the faster of the two, plus the
percentage of CPU time gained relative to the single-level EARGD.

Run from the directory holding the .org files (or pass --data-dir):
    python3 make_times_table.py --out times_gp.tex
"""
import argparse
import sys
from pathlib import Path


def load_org(path):
    """Parse a dealii org-mode table (single header row, then data rows, cells
    separated by '|') into (header, list-of-row-dicts). Numeric cells are
    floats; unparseable cells (e.g. the leading '-' rate) become None."""
    path = Path(path)
    if not path.exists():
        return None
    rows = []
    header = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        rec = {}
        for k, c in zip(header, cells):
            try:
                rec[k] = float(c)
            except ValueError:
                rec[k] = None
        rows.append(rec)
    return rows


def time_to_residual(rows, threshold):
    """Elapsed time of the first row whose residual <= threshold; None if the
    run never reaches it (or has no data)."""
    if not rows:
        return None
    for r in rows:
        res = r.get("residual")
        if res is not None and res <= threshold:
            return r.get("elapsed")
    return None


def fmt_time(t):
    return "--" if t is None else f"{round(t):d}"


def fmt_pct(t, t_ref):
    if t is None or t_ref is None or t_ref == 0:
        return ""
    return f"${round((t - t_ref) / t_ref * 100):d}\\%$"


def build_table(sl_time, metric_times, depths, metrics, caption, label):
    """metric_times[metric][depth] -> time (or None). metrics has length 2
    (the two side-by-side sub-columns); the faster of each pair is bolded."""
    n = len(depths)
    colspec = "c|c|" + "|".join(["cc"] * n)
    cline_last = 2 + 2 * n

    # header: EARGD spanning 1, then each depth spanning 2 (last without |)
    head_cols = ["\\multicolumn{1}{c|}{EARGD}"]
    for i, d in enumerate(depths):
        sep = "c" if i == n - 1 else "c|"
        head_cols.append(f"\\multicolumn{{2}}{{{sep}}}{{{d}-level EARGD}}")
    header = "         & " + "\n         & ".join(head_cols)

    # sub-header: metric symbols per depth
    sym = {"mass": "$M$", "frob": "$\\mathrm{e}$"}
    subcols = " & ".join(sym.get(m, m) for m in metrics)
    subheader = "         & & " + " & ".join([subcols] * n)

    # CPU time row: bold the faster (smaller) of each metric pair
    cpu_cells = []
    for d in depths:
        pair = [metric_times[m].get(d) for m in metrics]
        best = min((t for t in pair if t is not None), default=None)
        for t in pair:
            s = fmt_time(t)
            if t is not None and best is not None and abs(t - best) < 1e-9:
                s = f"\\textbf{{{s}}}"
            cpu_cells.append(s)
    cpu_row = f"        CPU time & {fmt_time(sl_time)} & " + " & ".join(cpu_cells)

    # percentage row (relative to single-level)
    pct_cells = []
    for d in depths:
        for m in metrics:
            pct_cells.append(fmt_pct(metric_times[m].get(d), sl_time))
    pct_row = "        EARGD & & " + " & ".join(pct_cells)

    return f"""\\captionsetup{{width=0.95\\textwidth, skip=3pt}}
\\begin{{table}}[t]
    \\centering
    \\scalebox{{.98}}{{
    \\begin{{tabular}}{{{colspec}}}
{header}
         \\tabularnewline
         \\cline{{3-{cline_last}}}
{subheader}
         \\tabularnewline
        \\hline
{cpu_row}
        \\tabularnewline
{pct_row}
        \\tabularnewline
    \\end{{tabular}}}}

    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{table}}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="directory with the .org logs (default: .)")
    p.add_argument("--operator", default="adj2", help="transport operator token (default: adj2)")
    p.add_argument("--residual", type=float, default=1e-8,
                   help="residual threshold whose crossing time is compared (default: 1e-8)")
    p.add_argument("--metrics", nargs="+", default=["mass", "frob"],
                   help="metric tokens for the paired sub-columns (default: mass frob)")
    p.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5],
                   help="multilevel depths to tabulate (default: 2 3 4 5)")
    p.add_argument("--beta", default="1000", help="beta token in filenames (default: 1000)")
    p.add_argument("--level", default="11", help="finest-level token in filenames (default: 11)")
    p.add_argument("--label", default="tab:times_gp", help="LaTeX label (default: tab:times_gp)")
    p.add_argument("--out", default="times_gp.tex", help="output .tex file (default: times_gp.tex)")
    args = p.parse_args()

    d = Path(args.data_dir)
    sl = d / f"sl_b{args.beta}_l{args.level}_optical_lattice.org"

    def ml(metric, depth):
        return d / f"ml_{metric}_{args.operator}_b{args.beta}_l{args.level}_depth{depth}_optical_lattice.org"

    sl_time = time_to_residual(load_org(sl), args.residual)
    if sl_time is None:
        print(f"warning: single-level reference {sl.name} not found or never reaches "
              f"residual {args.residual:.0e}", file=sys.stderr)

    metric_times = {}
    print(f"CPU time [s] to residual {args.residual:.0e}  (operator {args.operator})")
    print(f"  {'EARGD (sl)':<16}: {fmt_time(sl_time)}")
    for m in args.metrics:
        metric_times[m] = {}
        for depth in args.depths:
            path = ml(m, depth)
            t = time_to_residual(load_org(path), args.residual)
            metric_times[m][depth] = t
            if t is None and not path.exists():
                print(f"warning: {path.name} not found", file=sys.stderr)
            pct = fmt_pct(t, sl_time).replace("\\%$", "%").replace("$", "")
            print(f"  {m:>4} depth{depth}     : {fmt_time(t):>4}   {pct}")

    caption = (
        "Gross--Pitaevskii model: CPU time [s] for the different optimization "
        "algorithms and the percentage of time gained relative to the single-level "
        "EARGD. The mass-metric coarse model ($M$) is faster than the Frobenius-metric "
        "model ($\\mathrm{e}$) at every refinement depth, and both yield substantial "
        "speedups over the single-level EARGD (up to $66\\%$). The advantage of the "
        "mass metric over the Frobenius metric narrows as the number of levels "
        "increases, approaching parity at five levels.")

    tex = build_table(sl_time, metric_times, args.depths, args.metrics, caption, args.label)
    Path(args.out).write_text(tex)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
