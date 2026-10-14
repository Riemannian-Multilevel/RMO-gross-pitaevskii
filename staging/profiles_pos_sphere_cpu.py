#!/usr/bin/env python3
"""Performance profiles (Dolan and More 2002) of all versions (retraction, variant, metric, step rule) of the CPU-time
studies, over the problems discretization x radius, and a global table.

Usage: staging/profiles_pos_sphere_cpu.py <output dir> <cpu_*.csv>...

The CSV files are those of staging/plot_pos_sphere_cpu.py (analysis, econn, exact). A version that occurs in several
files (the unit sphere, and the density variants with Armijo or step 1 in both analysis and exact) is taken from the
first file that has it, so the order of the arguments sets the priority. Problems: case 1 = Q1, P1 x radius 4, 6, 8,
10 (8 problems, Z-matrices); case 2 = Q1, P1, P2, Q2 x radius (16 problems). Cost: CPU time if converged, else
infinity. For problem p and version s, r_ps = t_ps / min_s t_ps over all versions, and rho_s(tau) is the fraction of
problems with r_ps <= tau. The unit sphere solves the problem without the constraint x >= 0 (marked with a dagger).

Writes profiles_case<c>_{ea,mass,frobenius}.pdf (per metric, the best version per retraction and step rule, with the
ratios to the best version of that metric) and profiles_case<c>_best.pdf (the best version per metric and retraction,
with the ratios to the best version overall), c = 1, 2, profiles.csv and tab_profiles.tex (all versions, ratios to
the best version overall).
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from pos_sphere_plot import plt

PANELS = {  # run-directory suffix -> (metric, step rule)
    "ea": ("EA", "1"), "ea-step1": ("EA", "1"), "ea-armijo": ("EA", "Armijo"), "ea-exact": ("EA", "exact"),
    "mass": ("mass", "Armijo"), "mass-armijo": ("mass", "Armijo"), "mass-exact": ("mass", "exact"),
    "frobenius": ("Frobenius", "Armijo"), "frobenius-armijo": ("Frobenius", "Armijo"),
    "frobenius-exact": ("Frobenius", "exact"),
}
CASES = {  # case -> (retraction family, short name)
    "unit": ("unit", "unit"),
    "positive_theta0.99": ("m", "A"), "positive_clip_theta0.99": ("m", "clip"),
    "positive_floor_theta0.99": ("m", "floor+A"), "positive_floor-nocap": ("m", "floor"),
    "positive_clip-floor_theta0.99": ("m", "clip+floor"), "positive_floor-cont": ("m", "cont."),
    "positive_floor-cont-cap_theta0.99": ("m", "cont.+A"),
    "positive_e": ("e", "e"), "positive_e-cap_theta0.99": ("e", "e-cap"),
    "positive_e-cap-up_theta0.99": ("e", "e-cap-up"), "positive_clip-e-cap-up_theta0.99": ("e", "clip-e-cap-up"),
    "positive_norm-cap_theta0.99": ("norm", "norm+cap"),
}
DISCS = ["Q1", "P1", "P2", "Q2"]
RADII = [4, 6, 8, 10]
PROBLEMS = {1: [(d, r) for d in ["Q1", "P1"] for r in RADII], 2: [(d, r) for d in DISCS for r in RADII]}
METRICS = ["EA", "mass", "Frobenius"]
METRIC_FILE = {"EA": "ea", "mass": "mass", "Frobenius": "frobenius"}
RULES = ["1", "Armijo", "exact"]
FAMILY_COLOR = {"unit": "#555555", "norm": "#2ca02c", "m": "#1f77b4", "e": "#d62728"}
RULE_STYLE = {"1": ":", "Armijo": "--", "exact": "-"}
TAUS = [1, 1.25, 2, 4, 10]
TABLE_TAUS = [1, 2, 10]  # columns of tab_profiles.tex


def label(v):
    metric, rule, family, name = v
    dagger = r"$^\dagger$" if family == "unit" else ""
    return f"{metric}, {rule if rule != '1' else 'τ=1'}, {name}{dagger}"


def read(files):
    """{version: {(disc, radius): cpu or inf}}, versions (metric, rule, family, name) in order of first occurrence."""
    cost = {}
    for f in files:
        seen = defaultdict(dict)
        for r in csv.DictReader(open(f)):
            if r["panel"] not in PANELS or r["case"] not in CASES:
                continue
            v = PANELS[r["panel"]] + CASES[r["case"]]
            seen[v][(r["disc"], int(r["radius"]))] = float(r["cpu_ms"]) if r["status"] == "converged" else math.inf
        for v, c in seen.items():
            cost.setdefault(v, c)
    return cost


def ratios(cost, problems):
    """{version: array of r_ps over the problems}."""
    best = {p: min(c.get(p, math.inf) for c in cost.values()) for p in problems}
    return {v: np.array([c.get(p, math.inf) / best[p] if math.isfinite(best[p]) else math.inf for p in problems])
            for v, c in cost.items()}


def rho(r, tau):
    return float(np.mean(r <= tau))


def representatives(r, key):
    """The version with the largest area under rho (log tau in [1, 1000]) per value of key(version)."""
    taus = np.logspace(0, 3, 301)
    area = {v: np.trapezoid([rho(rv, t) for t in taus], np.log10(taus)) for v, rv in r.items()}
    best = {}
    for v in r:
        k = key(v)
        if area[v] > 0 and (k not in best or area[v] > area[best[k]]):
            best[k] = v
    return list(best.values())


def plot_profile(ax, r, versions, title):
    taus = np.logspace(0, 3, 601)
    for v in sorted(versions, key=lambda v: -rho(r[v], 1000)):
        metric, rule, family, name = v
        ax.step(taus, [rho(r[v], t) for t in taus], where="post", color=FAMILY_COLOR[family],
                ls=RULE_STYLE[rule], lw=1.2, label=label(v))
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\tau$ (CPU time / best CPU time)")
    ax.set_title(title)
    ax.legend(fontsize=5.5, loc="lower right")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cost = read(sys.argv[2:])
    rows = []
    for case, problems in PROBLEMS.items():
        r = ratios(cost, problems)
        n = len(problems)
        problems_title = ("Q1, P1" if case == 1 else "Q1, P1, P2, Q2") + f" x radius 4, 6, 8, 10 ({n} problems)"
        panels = []
        for metric in METRICS:
            # Within a metric, the ratios are to the best version of that metric
            r_metric = ratios({v: c for v, c in cost.items() if v[0] == metric}, problems)
            panels.append((METRIC_FILE[metric], r_metric, representatives(r_metric, lambda v: (v[1], v[2])),
                           f"{metric}: best variant per retraction and rule\n(ratio to the best {metric} version)"))
        panels.append(("best", r, representatives(r, lambda v: (v[0], v[2])),
                       "best version per metric and retraction\n(ratio to the best version)"))
        for name, r_panel, versions, title in panels:
            fig, ax = plt.subplots(figsize=(4.6, 4.0), layout="constrained")
            plot_profile(ax, r_panel, versions, f"Case {case}: {problems_title}\n{title}")
            ax.set_ylabel(r"$\rho(\tau)$: fraction of problems")
            fig.savefig(out / f"profiles_case{case}_{name}.pdf")
            plt.close(fig)
        for v, rv in r.items():
            solved = rv[np.isfinite(rv)]
            rows.append((case, v, int(np.isfinite(rv).sum()), n, [rho(rv, t) for t in TAUS],
                         float(np.exp(np.mean(np.log(solved)))) if len(solved) else math.inf))
    # Table: all versions, both cases
    by_version = defaultdict(dict)
    for case, v, solved, n, rhos, gmean in rows:
        by_version[v][case] = (solved, n, rhos, gmean)
    order = sorted(by_version, key=lambda v: (-by_version[v][2][0], -by_version[v][2][2][0],
                                              by_version[v][2][3], METRICS.index(v[0]), RULES.index(v[1])))
    with open(out / "profiles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "rule", "family", "variant", "case", "solved", "problems"]
                   + [f"rho_{t}" for t in TAUS] + ["geomean_ratio_solved"])
        for v in order:
            for case in (1, 2):
                solved, n, rhos, gmean = by_version[v][case]
                w.writerow(list(v) + [case, solved, n] + [f"{x:.3f}" for x in rhos] + [f"{gmean:.3f}"])
    fmt = lambda x: f"{x:.2f}".lstrip("0") if x < 1 else "1"
    nt = len(TABLE_TAUS)
    cols = [TAUS.index(t) for t in TABLE_TAUS]
    lines = [r"\begin{longtable}{@{}lll" + ("r" * (nt + 2)) * 2 + "@{}}",
             r"\toprule",
             r" & & & \multicolumn{" + str(nt + 2) + r"}{c}{case 1 (Q1, P1; 8)} & \multicolumn{"
             + str(nt + 2) + r"}{c}{case 2 (all; 16)}\\",
             r"\cmidrule(lr){4-" + str(nt + 5) + r"}\cmidrule(l){" + str(nt + 6) + "-" + str(2 * nt + 7) + "}",
             "metric & rule & version & solved & " + " & ".join(rf"$\rho({t})$" for t in TABLE_TAUS)
             + r" & g.m. & solved & " + " & ".join(rf"$\rho({t})$" for t in TABLE_TAUS) + r" & g.m.\\",
             r"\midrule", r"\endhead"]
    for v in order:
        metric, rule, family, name = v
        cells = [metric, rule if rule != "1" else r"$\tau=1$",
                 ({"unit": "unit", "norm": "", "m": "m: ", "e": "e: "}[family] + (name if family != "unit" else ""))
                 + (r"$^\dagger$" if family == "unit" else "")]
        for case in (1, 2):
            solved, n, rhos, gmean = by_version[v][case]
            cells += [str(solved)] + [fmt(rhos[c]) for c in cols] + [f"{gmean:.2f}" if math.isfinite(gmean) else "--"]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (out / "tab_profiles.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(by_version)} versions; profiles_case{{1,2}}_{{ea,mass,frobenius,best}}.pdf, profiles.csv, "
          "tab_profiles.tex")
    for v in order[:12]:
        print(f"  {label(v):40s} case 1: {by_version[v][1][0]}/8 rho(1)={by_version[v][1][2][0]:.2f} "
              f"case 2: {by_version[v][2][0]}/16 rho(1)={by_version[v][2][2][0]:.2f} g.m. {by_version[v][2][3]:.2f}")


if __name__ == "__main__":
    main()
