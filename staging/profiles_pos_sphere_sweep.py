#!/usr/bin/env python3
"""Performance profile over all metrics (as profiles_case<c>_all.pdf of staging/profiles_pos_sphere_cpu.py) of the
parameter sweep staging/pos_sphere_sweep_cpu.sh, over the problems discretization x beta x potential x radius.

Usage: staging/profiles_pos_sphere_sweep.py <data dir> <output dir> <name> [max iterations, default 1000]

<data dir> is the output of staging/pos_sphere_cpu.sh with the sweep: rep1, rep2, ... with
matrix/b<beta>-<potential>/<disc>_r<radius>_<panel>/summary.csv each. The CPU time of a case is the median of
`elapsed` over the repeats; status and iterations are taken from rep1 (a warning is printed if the iterations differ
between repeats). Cost: CPU time if converged, else infinity. Writes cpu_sweep_<name>.csv (all medians),
profiles_sweep_<name>_all.pdf and one profile per metric, profiles_sweep_<name>_{ea,mass,frobenius}.pdf (ratios to the
best version of all metrics).
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from pos_sphere_plot import read_rows, status
from profiles_pos_sphere_cpu import CASES, METRIC_FILE, PANELS, plot_all

POTENTIAL_TITLE = {"square": "square", "optical_lattice": "optical lattice"}


def disc_name(r):
    return f"{'Q' if r['mesh'] == 'QUADRILATERAL' else 'P'}{r['degree']}"


def read_runs(data):
    """{(disc, beta, potential, radius, panel, case): (row of rep1, median CPU time in ms)}."""
    reps = sorted(p for p in data.glob("rep*") if p.is_dir())
    if not reps:
        sys.exit(f"no rep* directories in {data}")
    times, iterations, first = defaultdict(list), defaultdict(set), {}
    levels = {}  # disc -> (level, number of DoFs); the mesh does not depend on the radius
    for rep in reps:
        for f in sorted((rep / "matrix").glob("b*-*/*/summary.csv")):
            beta, potential = f.parent.parent.name[1:].split("-", 1)
            panel = f.parent.name.split("_", 2)[2]
            for r in read_rows(f):
                key = (disc_name(r), int(beta), potential, int(r["radius"]), panel, r["case"])
                levels[disc_name(r)] = (r["level"], r["n_dofs"])
                times[key].append(1e3 * float(r["elapsed"]))
                iterations[key].add(r["iterations"])
                if rep == reps[0]:
                    first[key] = r
    for key, its in iterations.items():
        if len(its) > 1:
            print(f"warning: iterations differ between repeats for {key}: {sorted(its)}")
    incomplete = [k for k in first if len(times[k]) != len(reps)]
    if incomplete:
        print(f"warning: {len(incomplete)} cases not in every repeat, e.g. {incomplete[0]}")
    return {k: (r, statistics.median(times[k])) for k, r in first.items()}, len(reps), levels


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    data, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    max_iter = int(sys.argv[4]) if len(sys.argv) == 5 else 1000
    out.mkdir(parents=True, exist_ok=True)
    runs, n_reps, levels = read_runs(data)
    with open(out / f"cpu_sweep_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["disc", "beta", "potential", "radius", "panel", "case", "status", "iterations", "residual",
                    "energy", "cpu_ms"])
        for key, (r, t) in sorted(runs.items()):
            w.writerow(list(key) + [status(r), r["iterations"], r["residual"], r["energy"], f"{t:.4g}"])
    # Versions (metric, rule, family, name) -> {problem: cost}; the unit sphere occurs in every run of a panel and
    # is taken once per problem (from the first run directory, as in staging/plot_pos_sphere_cpu.py)
    cost = defaultdict(dict)
    for (disc, beta, potential, radius, panel, case), (r, t) in sorted(runs.items()):
        if panel not in PANELS or case not in CASES:
            continue
        v = PANELS[panel] + CASES[case]
        problem = (disc, beta, potential, radius)
        cost[v].setdefault(problem, t if status(r) == "converged" else math.inf)
    problems = sorted({k[:4] for k in runs})
    discs = sorted({p[0] for p in problems}, key=lambda d: (d[1], d[0] != "Q"))
    betas = sorted({p[1] for p in problems})
    potentials = sorted({p[2] for p in problems}, key=lambda p: p != "square")
    radii = sorted({p[3] for p in problems})
    title = (f"{', '.join(discs)} x β {', '.join(map(str, betas))} x V "
             f"{', '.join(POTENTIAL_TITLE.get(p, p) for p in potentials)}\n"
             f"x radius {', '.join(map(str, radii))}, level {', '.join(f'{d} {levels[d][0]} ({levels[d][1]} DoFs)' for d in discs)},\n"
             f"at most {max_iter} iterations "
             f"({len(problems)} problems, median of {n_reps} runs)\n"
             "all metrics, best version per metric, rule and retraction\n"
             "(n: normalization, m: density, e: e-retraction; exact step only for m and e)")
    plot_all(out / f"profiles_sweep_{name}_all.pdf", dict(cost), problems, title)
    # One profile per metric, with the ratios to the best version of all metrics (comparable between the metrics)
    for metric, short in METRIC_FILE.items():
        plot_all(out / f"profiles_sweep_{name}_{short}.pdf", dict(cost), problems,
                 title.replace("all metrics, best version per metric, rule and retraction",
                               f"{metric}: best version per rule and retraction (ratios to the best of all metrics)"),
                 metric=metric)
    solved = {v: sum(math.isfinite(x) for x in c.values()) for v, c in cost.items()}
    print(f"{len(runs)} cases, {len(problems)} problems, {len(cost)} versions, median over {n_reps} repeats; "
          f"profiles_sweep_{name}_{{all,ea,mass,frobenius}}.pdf, cpu_sweep_{name}.csv")
    unsolved = [p for p in problems if not any(math.isfinite(c.get(p, math.inf)) for c in cost.values())]
    if unsolved:
        print(f"  {len(unsolved)} problems solved by no version: {unsolved}")
    for v in sorted(solved, key=lambda v: -solved[v])[:8]:
        print(f"  {v}: {solved[v]}/{len(problems)} solved")


if __name__ == "__main__":
    main()
