#!/usr/bin/env python3
"""CPU-time overviews of the positive-sphere studies, in the format of fig_analysis_overview_*.pdf
(staging/plot_pos_sphere_analysis.py), from the output of staging/pos_sphere_cpu.sh.

Usage: staging/plot_pos_sphere_cpu.py <study> <data dir> <output dir>

<study> is one of
  analysis  staging/pos_sphere_analysis.sh (MATRIX_ONLY=1): fig_analysis_overview_{ea,ea_armijo,mass,frobenius}_cpu.pdf
  econn     staging/pos_sphere_econn_cpu.sh and pos_sphere_econn_exact_cpu.sh:
            fig_econn_overview_{ea,ea_armijo,mass,frobenius,ea_exact,mass_exact,frobenius_exact}_cpu.pdf
  exact     staging/pos_sphere_exact_cpu.sh: fig_density_exact_overview_<metric>_<rule>_cpu.pdf

<data dir> contains rep1, rep2, ... with matrix/<disc>_r<radius>_<panel>/summary.csv each. The CPU time of a case is
the median of `elapsed` over the repeats; status and iterations are taken from rep1 (a warning is printed if the
iterations differ between repeats). Cell text: CPU time in ms of converged cases; colour: CPU time relative to the
reference case of the panel (the unit sphere). Where the reference did not converge, its CPU time until it stopped is
the reference, and the row label is marked with a dagger. Also writes cpu_<study>.csv with all medians.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pos_sphere_plot import read_rows, status

DISCS = ["Q1", "P1", "P2", "Q2"]
RADII = [4, 6, 8, 10]

ANALYSIS_CASES = ["unit", "positive_theta0.99", "positive_clip_theta0.99", "positive_floor_theta0.99",
                  "positive_floor-nocap", "positive_clip-floor_theta0.99", "positive_floor-cont"]
DENSITY_SHORT = {"unit": "unit", "positive_theta0.99": "A", "positive_clip_theta0.99": "clip",
                 "positive_floor_theta0.99": "floor+A", "positive_floor-nocap": "floor",
                 "positive_clip-floor_theta0.99": "clip+floor", "positive_floor-cont": "cont.",
                 "positive_floor-cont-cap_theta0.99": "cont.+A", "positive_norm-cap_theta0.99": "norm+cap"}
ECONN_CASES = ["unit", "positive_e", "positive_e-cap_theta0.99", "positive_e-cap-up_theta0.99",
               "positive_clip-e-cap-up_theta0.99"]
ECONN_SHORT = {"unit": "unit", "positive_e": "e", "positive_e-cap_theta0.99": "e-cap",
               "positive_e-cap-up_theta0.99": "e-cap-up", "positive_clip-e-cap-up_theta0.99": "clip-e-cap-up"}
EXACT_CASES = ["unit", "positive_norm-cap_theta0.99", "positive_theta0.99", "positive_clip_theta0.99",
               "positive_floor_theta0.99", "positive_clip-floor_theta0.99", "positive_floor-cont-cap_theta0.99"]

# Title suffix of the panels with the exact line search: which columns use the closed form
EXACT_NOTE = "\n(unit, norm+cap: closed-form ray step; density: bracketing and zoom)"
EXACT_NOTE_E = "\n(unit: closed-form ray step; e: bracketing and zoom)"

# Panels per study: run-directory suffix -> (title, output file, figure size)
STUDIES = {
    "analysis": (ANALYSIS_CASES, DENSITY_SHORT, {
        "ea": ("energy-adaptive, $\\tau=1$", "fig_analysis_overview_ea_cpu.pdf", (6.3, 4.6)),
        "ea-armijo": ("energy-adaptive, Armijo", "fig_analysis_overview_ea_armijo_cpu.pdf", (6.3, 4.6)),
        "mass": ("mass, Armijo", "fig_analysis_overview_mass_cpu.pdf", (5.2, 4.6)),
        "frobenius": ("Frobenius, Armijo", "fig_analysis_overview_frobenius_cpu.pdf", (5.2, 4.6)),
    }),
    "econn": (ECONN_CASES, ECONN_SHORT, {
        "ea": ("energy-adaptive, $\\tau=1$", "fig_econn_overview_ea_cpu.pdf", (5.4, 4.6)),
        "ea-armijo": ("energy-adaptive, Armijo", "fig_econn_overview_ea_armijo_cpu.pdf", (5.4, 4.6)),
        "mass": ("mass, Armijo", "fig_econn_overview_mass_cpu.pdf", (4.8, 4.6)),
        "frobenius": ("Frobenius, Armijo", "fig_econn_overview_frobenius_cpu.pdf", (4.8, 4.6)),
        "ea-exact": ("energy-adaptive, exact" + EXACT_NOTE_E, "fig_econn_overview_ea_exact_cpu.pdf", (5.4, 4.6)),
        "mass-exact": ("mass, exact" + EXACT_NOTE_E, "fig_econn_overview_mass_exact_cpu.pdf", (4.8, 4.6)),
        "frobenius-exact": ("Frobenius, exact" + EXACT_NOTE_E, "fig_econn_overview_frobenius_exact_cpu.pdf", (4.8, 4.6)),
    }),
    "exact": (EXACT_CASES, DENSITY_SHORT, {
        f"{m}-{rule}": (f"{title}, {rule_title}", f"fig_density_exact_overview_{m}_{rule}_cpu.pdf",
                        (5.6, 4.6) if m == "ea" else (4.6, 4.6))
        for m, title in [("ea", "energy-adaptive"), ("mass", "mass"), ("frobenius", "Frobenius")]
        for rule, rule_title in [("exact", "exact" + EXACT_NOTE), ("armijo", "Armijo"), ("step1", "$\\tau=1$")]
        if m == "ea" or rule != "step1"
    }),
}


def disc_name(r):
    return f"{'Q' if r['mesh'] == 'QUADRILATERAL' else 'P'}{r['degree']}"


def read_runs(data):
    """{(disc, radius, panel): {case: (row of rep1, median CPU time)}}."""
    reps = sorted(p for p in data.glob("rep*") if p.is_dir())
    if not reps:
        sys.exit(f"no rep* directories in {data}")
    times = defaultdict(list)
    iterations = defaultdict(set)
    first = {}
    for rep in reps:
        seen = set()  # a case that occurs in several run directories of a panel (the unit sphere) counts once
        for f in sorted((rep / "matrix").glob("*/summary.csv")):
            # Run directories <disc>_r<radius>_<panel>[+<suffix>]: runs with a suffix add cases to the same panel
            panel = f.parent.name.split("_", 2)[2].split("+")[0]
            for r in read_rows(f):
                key = (disc_name(r), int(r["radius"]), panel, r["case"])
                if key in seen:
                    continue
                seen.add(key)
                times[key].append(float(r["elapsed"]))
                iterations[key].add(r["iterations"])
                if rep == reps[0]:
                    first[key] = r
    for key, its in iterations.items():
        if len(its) > 1:
            print(f"warning: iterations differ between repeats for {key}: {sorted(its)}")
    runs = defaultdict(dict)
    for key, r in first.items():
        d, rad, panel, case = key
        runs[(d, rad, panel)][case] = (r, statistics.median(times[key]))
    return runs, len(reps)


def fmt_ms(t):
    ms = 1e3 * t
    return f"{ms:.2g}" if ms < 10 else f"{ms:.0f}"


def fig_overview(runs, out, cases_all, short, panel, title, name, figsize):
    rows = [(d, rad) for d in DISCS for rad in RADII]
    cases = [c for c in cases_all if any(c in runs[(d, rad, panel)] for d, rad in rows)]
    if not cases:
        return False
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    norm = LogNorm(vmin=0.5, vmax=20)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    ratio = np.full((len(rows), len(cases)), np.nan)
    labels = []
    for i, (d, rad) in enumerate(rows):
        entry = runs[(d, rad, panel)]
        ref_row, ref_time = entry[cases[0]]
        mark = "" if status(ref_row) == "converged" else r"$^\dagger$"
        labels.append(f"{d}, r{rad}{mark}")
        for j, c in enumerate(cases):
            if c not in entry:
                continue
            r, t = entry[c]
            st = status(r)
            if st == "converged":
                ratio[i, j] = t / ref_time
                ax.text(j, i, fmt_ms(t), ha="center", va="center", fontsize=7)
            elif st == "stopped":
                ax.text(j, i, f"{float(r['residual']):.0e}".replace("e-0", "e-"), ha="center", va="center",
                        fontsize=6, color="#444444")
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#dddddd", zorder=0))
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#b2182b", zorder=0))
                ax.text(j, i, "undef.", ha="center", va="center", fontsize=6, color="white")
    ax.imshow(ratio, cmap=cmap, norm=norm, aspect="auto", zorder=1, alpha=0.75)
    ax.set_xticks(range(len(cases)), [short[c] for c in cases], rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), labels, fontsize=7)
    for k in range(1, len(DISCS)):
        ax.axhline(k * len(RADII) - 0.5, color="k", lw=0.6)
    ax.set_title(title)
    ax.set_xlim(-0.5, len(cases) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.6, label="CPU time / CPU time of the unit sphere")
    fig.savefig(out / name)
    plt.close(fig)
    return True


def write_csv(runs, out, study):
    with open(out / f"cpu_{study}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["disc", "radius", "panel", "case", "status", "iterations", "residual", "energy", "cpu_ms"])
        for (d, rad, panel), entry in sorted(runs.items()):
            for case, (r, t) in entry.items():
                w.writerow([d, rad, panel, case, status(r), r["iterations"], r["residual"], r["energy"],
                            f"{1e3 * t:.4g}"])


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in STUDIES:
        sys.exit(__doc__)
    study, data, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    cases, short, panels = STUDIES[study]
    runs, n_reps = read_runs(data)
    for panel, (title, name, figsize) in panels.items():
        if fig_overview(runs, out, cases, short, panel, title, name, figsize):
            print(f"{name}")
    write_csv(runs, out, study)
    print(f"{sum(len(v) for v in runs.values())} cases, median over {n_reps} repeats")


if __name__ == "__main__":
    main()
