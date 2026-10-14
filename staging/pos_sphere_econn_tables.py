"""Markdown tables of staging/pos_sphere_econn.sh: python3 staging/pos_sphere_econn_tables.py <output dir>

Cells: iterations to residual 1e-9; F<k>: retraction undefined after k iterations; S (r): stopped (1000 iterations or a
stalled line search) at residual r; dE: |E - E*| > 1e-8 for E* the energy of the unit sphere with the energy-adaptive
metric and Armijo steps; neg: an iterate with a nonpositive component. The verdict counts the converged runs, + those
at the ground-state energy (|E - E*| < 1e-10) with residual below 1e-8 (the KKT floor of the non-Z discretizations).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
CASES = [("unit", "norm"), ("unit_ortho_theta0.99", "ortho"), ("unit_exp", "exp"), ("positive_theta0.99", "m"),
         ("positive_e", "e"), ("positive_e-cap_theta0.99", "e-cap"), ("positive_e-cap-up_theta0.99", "e-cap-up"),
         ("positive_clip-e", "clip-e"), ("positive_clip-e-cap_theta0.99", "clip-e-cap"),
         ("positive_clip-e-cap-up_theta0.99", "clip-e-cap-up")]
RULES = ["energy_adaptive_step1", "energy_adaptive_armijo", "energy_adaptive_bb_max", "mass_armijo",
         "mass_bb_monotone", "mass_bb_max", "frobenius_armijo", "frobenius_bb_monotone", "frobenius_bb_max"]


def load(group):
    return {f.parent.name: {r["case"]: r for r in csv.DictReader(open(f))}
            for f in sorted((root / group).glob("*/summary.csv"))}


matrix_runs = load("matrix")
# Reference energy per discretization and radius
REF = {tuple(name.split("_")[:2]): float(rows["unit"]["energy"])
       for name, rows in matrix_runs.items() if name.endswith("energy_adaptive_armijo")}


def disc_radius(name):
    parts = name.split("_")
    return (parts[0], parts[1]) if parts[0] in ("q1", "q2", "p1", "p2") else ("q1", parts[0])


def cell(r, ref):
    if r is None:
        return ""
    if r["failed"] == "1":
        s = f"F{r['iterations']}"
    elif r["converged"] == "1":
        s = r["iterations"]
    else:
        s = f"S ({float(r['residual']):.0e})"
    if r["converged"] == "1" and abs(float(r["energy"]) - ref) > 1e-8:
        s += f" dE{abs(float(r['energy']) - ref):.0e}"
    if float(r["min_value"]) <= 0:
        s += " neg"
    return s


def table(runs, cases, order):
    print("| run | " + " | ".join(n for _, n in cases) + " |")
    print("|---|" + "---|" * len(cases))
    for name in sorted(runs, key=order):
        ref = REF[disc_radius(name)]
        print(f"| {name} | " + " | ".join(cell(runs[name].get(c), ref) for c, _ in cases) + " |")


def verdict():
    count = defaultdict(lambda: [0, 0, 0])
    for name, rows in matrix_runs.items():
        d, r, *rest = name.split("_")
        for case, row in rows.items():
            c = count[("_".join(rest), case)]
            converged = row["converged"] == "1"
            floor = (not converged and float(row["residual"]) < 1e-8
                     and abs(float(row["energy"]) - REF[(d, r)]) < 1e-10)
            c[0] += converged
            c[1] += floor
            c[2] += 1
    print("| metric, step rule | " + " | ".join(n for _, n in CASES) + " |")
    print("|---|" + "---|" * len(CASES))
    for rule in RULES:
        cells = []
        for case, _ in CASES:
            c = count.get((rule, case))
            cells.append("" if c is None else f"{c[0]}" + (f"+{c[1]}" if c[1] else "") + f"/{c[2]}")
        print(f"| {rule} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    verdict()
    print()
    table(matrix_runs, CASES, lambda n: (n.split("_")[0], int(n.split("_")[1][1:]), n))
    print()
    table(load("fixed"), CASES, lambda n: (n.rsplit("_t", 1)[0], float(n.rsplit("_t", 1)[1])))
    print()
    table(load("long"), [c for c in CASES if c[0].startswith("positive_e")], lambda n: n)
