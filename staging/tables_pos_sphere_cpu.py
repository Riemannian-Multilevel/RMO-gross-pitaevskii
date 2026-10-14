#!/usr/bin/env python3
"""LaTeX tables of the CPU-time studies (tex/pos_sphere_cpu.tex, tex/exact_line_search.tex).

Usage: staging/tables_pos_sphere_cpu.py <output dir> <cpu_analysis.csv> <cpu_econn.csv> <cpu_exact.csv>

Writes
  tab_cpu_summary_<study>.tex   converged cases (of 16) and the median CPU time relative to the unit sphere over the
                                cases where both converged, per case (rows) and panel (columns);
  tab_cpu_rows_<metric>_<rule>.tex  per discretization and radius (rows), for each version with the metric and step
                                rule (unit sphere, norm-cap, density variants, e variants): CPU time in milliseconds to
                                the tolerance; the fastest positive version in bold. Not converged: L (1000 iterations)
                                or Z (the line search returned step 0), CPU time until the stop, final residual. F:
                                the retraction became undefined.
For the exact rule the unit sphere uses the closed-form ray step (caption).
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DISCS = ["Q1", "P1", "P2", "Q2"]
RADII = [4, 6, 8, 10]
SHORT = {"unit": "unit", "positive_theta0.99": "A", "positive_clip_theta0.99": "clip",
         "positive_floor_theta0.99": "floor+A", "positive_floor-nocap": "floor",
         "positive_clip-floor_theta0.99": "clip+floor", "positive_floor-cont": "cont.",
         "positive_floor-cont-cap_theta0.99": "cont.+A", "positive_e": "e", "positive_e-cap_theta0.99": "e-cap",
         "positive_e-cap-up_theta0.99": "e-cap-up", "positive_clip-e-cap-up_theta0.99": "clip-e-cap-up",
         "positive_norm-cap_theta0.99": "norm+cap"}
NORM = ["positive_norm-cap_theta0.99"]
DENSITY = ["positive_theta0.99", "positive_clip_theta0.99", "positive_floor_theta0.99", "positive_floor-nocap",
           "positive_clip-floor_theta0.99", "positive_floor-cont", "positive_floor-cont-cap_theta0.99"]
ECONN = ["positive_e", "positive_e-cap_theta0.99", "positive_e-cap-up_theta0.99", "positive_clip-e-cap-up_theta0.99"]
PANEL_TITLE = {"ea": r"EA, $\tau=1$", "ea-step1": r"EA, $\tau=1$", "ea-armijo": "EA, Armijo", "ea-exact": "EA, exact",
               "mass": "mass, Armijo", "mass-armijo": "mass, Armijo", "mass-exact": "mass, exact",
               "frobenius": "Frobenius, Armijo", "frobenius-armijo": "Frobenius, Armijo",
               "frobenius-exact": "Frobenius, exact"}
# Panel name per (metric, rule) in each study
PANEL = {
    "analysis": {("ea", "step1"): "ea", ("ea", "armijo"): "ea-armijo", ("mass", "armijo"): "mass",
                 ("frobenius", "armijo"): "frobenius"},
    "econn": {("ea", "step1"): "ea", ("ea", "armijo"): "ea-armijo", ("mass", "armijo"): "mass",
              ("frobenius", "armijo"): "frobenius", ("ea", "exact"): "ea-exact", ("mass", "exact"): "mass-exact",
              ("frobenius", "exact"): "frobenius-exact"},
    "exact": {("ea", "step1"): "ea-step1", ("ea", "armijo"): "ea-armijo", ("ea", "exact"): "ea-exact",
              ("mass", "armijo"): "mass-armijo", ("mass", "exact"): "mass-exact",
              ("frobenius", "armijo"): "frobenius-armijo", ("frobenius", "exact"): "frobenius-exact"},
}


def load(path):
    t = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        t[(r["panel"], r["disc"], int(r["radius"]))][r["case"]] = r
    return t


def write(path, lines):
    Path(path).write_text("\n".join(lines) + "\n")


def summary(data, panels, cases, path):
    lines = [r"\begin{tabular}{@{}l" + "c" * len(panels) + "@{}}", r"\toprule",
             "variant & " + " & ".join(PANEL_TITLE[p] for p in panels) + r"\\", r"\midrule"]
    for c in cases:
        cells = []
        for p in panels:
            n, rat = 0, []
            present = False
            for d in DISCS:
                for rad in RADII:
                    e = data.get((p, d, rad), {})
                    if c not in e:
                        continue
                    present = True
                    r = e[c]
                    if r["status"] == "converged":
                        n += 1
                        u = e.get("unit")
                        if u and u["status"] == "converged":
                            rat.append(float(r["cpu_ms"]) / float(u["cpu_ms"]))
            if not present:
                cells.append("--")
            else:
                cells.append(f"{n}" + (f" ({statistics.median(rat):.2f})" if rat and c != "unit" else ""))
        if any(x != "--" for x in cells):
            lines.append(SHORT[c] + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


MAX_ITER = 1000  # COMMON in staging/pos_sphere_lib.sh


def fmt_ms(ms):
    return f"{ms:.2g}" if ms < 10 else f"{ms:.0f}"


def stop_reason(r):
    """L: the iteration limit; Z: the line search returned step 0 before it."""
    return "L" if int(r["iterations"]) >= MAX_ITER else "Z"


def cell(r, bold=False):
    """Converged: CPU time in ms. Not converged: reason (L, Z), CPU time in ms until the stop, final residual.
    Failed (retraction undefined): F."""
    if r is None:
        return ""
    s = fmt_ms(float(r["cpu_ms"]))
    if r["status"] == "converged":
        return rf"\textbf{{{s}}}" if bold else s
    if r["status"] == "failed":
        return r"\textsf{F}"
    m, e = f"{float(r['residual']):.0e}".split("e")
    return rf"\textsf{{{stop_reason(r)}}}\,{s}\,{{\tiny${m}\mathrm{{e}}{int(e)}$}}"


def rows(sources, metric, rule, path):
    """sources: [(data, study)] in priority order; columns: unit, density variants, e variants present."""
    cols = []
    for c in ["unit"] + NORM + DENSITY + ECONN:
        for data, study in sources:
            p = PANEL[study].get((metric, rule))
            if p and any(c in data.get((p, d, r), {}) for d in DISCS for r in RADII):
                cols.append((c, data, p))
                break
    if len(cols) <= 1:
        return False
    n_norm = sum(1 for c, _, _ in cols if c in NORM)
    n_dens = sum(1 for c, _, _ in cols if c in DENSITY)
    n_e = sum(1 for c, _, _ in cols if c in ECONN)
    groups = [(n_norm, "norm."), (n_dens, "density (m)"), (n_e, "exponential (e)")]
    groups = [(k, t) for k, t in groups if k]
    lines = [r"\begin{tabular}{@{}lr|r|" + "|".join("r" * k for k, _ in groups) + "@{}}", r"\toprule",
             r" & & & " + " & ".join(r"\multicolumn{" + str(k) + "}{c" + ("|" if i < len(groups) - 1 else "")
                                     + "}{" + t + "}" for i, (k, t) in enumerate(groups)) + r"\\",
             "disc. & $R$ & " + " & ".join(SHORT[c] for c, _, _ in cols) + r"\\", r"\midrule"]
    for d in DISCS:
        for rad in RADII:
            rs = [data.get((p, d, rad), {}).get(c) for c, data, p in cols]
            conv = [float(r["cpu_ms"]) if r and r["status"] == "converged" and c != "unit" else float("inf")
                    for r, (c, _, _) in zip(rs, cols)]
            best = min(conv)
            cells = [cell(r, bold=(conv[i] == best and best < float("inf"))) for i, r in enumerate(rs)]
            lines.append(f"{d} & {rad} & " + " & ".join(cells) + r"\\")
        if d != DISCS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)
    return True


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    analysis, econn, exact = load(sys.argv[2]), load(sys.argv[3]), load(sys.argv[4])
    summary(analysis, ["ea", "ea-armijo", "mass", "frobenius"], ["unit"] + DENSITY,
            out / "tab_cpu_summary_analysis.tex")
    summary(econn, ["ea", "ea-armijo", "ea-exact", "mass", "mass-exact", "frobenius", "frobenius-exact"],
            ["unit"] + ECONN, out / "tab_cpu_summary_econn.tex")
    summary(exact, ["ea-step1", "ea-armijo", "ea-exact", "mass-armijo", "mass-exact", "frobenius-armijo",
                    "frobenius-exact"], ["unit"] + NORM + DENSITY, out / "tab_cpu_summary_exact.tex")
    written = []
    for metric in ["ea", "mass", "frobenius"]:
        for rule in ["step1", "armijo", "exact"]:
            # Armijo and step 1: density from analysis (all variants), then exact (cont.+A); exact: exact, econn
            sources = [(analysis, "analysis"), (exact, "exact"), (econn, "econn")]
            if rows(sources, metric, rule, out / f"tab_cpu_rows_{metric}_{rule}.tex"):
                written.append(f"tab_cpu_rows_{metric}_{rule}.tex")
    print("tab_cpu_summary_{analysis,econn,exact}.tex, " + ", ".join(written))


if __name__ == "__main__":
    main()
