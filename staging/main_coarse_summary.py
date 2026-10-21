#!/usr/bin/env python3
"""One run of main_coarse (the unit-sphere FAS, or single level with one level), written as pos_sphere_ml writes its
runs: <out>/summary.csv (appended, the columns of pos_sphere_ml) and <out>/iter.csv (the rows of every level), so that
the unit sphere enters the multilevel benchmark (md/positive_multilevel_benchmark.md, step 6) as a reference.

Usage: staging/main_coarse_summary.py --binary <main_coarse> <main_coarse options> --out <dir>

main_coarse prints the table of the finest level to stdout and the tables of the coarser levels to stderr, each at the
end of its cycle, after the line `level: <l>` that FullApproximationScheme::cycle() prints at its start. The FAS
fallbacks (`falling back to gradient step`; not `not a descent direction`, which the Armijo line search of the coarse
steps prints too) are counted on the level whose cycle is open: `level: <l>` opens a cycle, a table closes the cycle
of its level. Unit-sphere settings in the summary: smoother and coarse retraction `unit` (normalization), coarse
gradient `unit-<metric of the model>` (main_coarse: energy-adaptive coarse descent), point transfer: the transport
(`--transport`); `failed`, `skipped` are 0 (control none), `wall` is empty (main_coarse times only the solve, as CPU
time: the column `elapsed` of the last row of the finest level).
"""
import argparse
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HEADER = ("case,mesh,degree,levels,n_levels,level,n_dofs,metric,smoother,step_rule,coarse_retraction,point_transfer,"
          "coarse_grad,coarse_max_iter,floor,kappa,eps,coarse_every,control,backoff,ratio_eta,tol_inner_res,"
          "tol_residual,max_iter,beta,potential,radius,status,iterations,residual,energy,min_value,cpu,wall,attempted,"
          "accepted,rejected,fallback,failed,skipped,attempted_mid,accepted_mid,rejected_mid,fallback_mid,failed_mid,"
          "skipped_mid,coarse_iter,rho_n,rho_nan,rho_min,rho_median").split(",")
DEFAULTS = {"kappa": "0.8", "eps": "2e-08", "coarse-every": "2", "correction-control": "none", "backoff": "1",
            "ratio-eta": "0.1", "tol-inner-res": "0.01", "tol-residual": "0.0001", "max-iter": "25",
            "metric-coarse": "mass", "transport": "mass", "mesh": "quadrilateral", "degree": "1"}
COARSE_MAX_ITER = 4  # fixed in main_coarse
PLATEAU = 20         # as pos_sphere_ml: energy and residual unchanged over 20 rows of the finest level


def given(args, name):
    """Value of --name (or --name=value) in the main_coarse arguments, None if not given."""
    for k, a in enumerate(args):
        if a == f"--{name}" and k + 1 < len(args):
            return args[k + 1]
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


def option(args, name) -> str:
    """Value of --name, else the default of main_coarse; exits if neither exists."""
    v = given(args, name)
    if v is None:
        v = DEFAULTS.get(name)
    if v is None:
        sys.exit(f"main_coarse_summary.py: --{name} is required")
    return v


def flag(args, name):
    return any(a == f"--{name}" or a.startswith(f"--{name}=") for a in args)


def parse_tables(text):
    """Tables of a text (lines starting with '|'), in order: a list of lists of row dicts; a header line starts a
    table."""
    tables, head = [], None
    for line in text.splitlines():
        if not line.startswith("|"):
            head = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if "iter" in cells and "level" in cells and not re.fullmatch(r"[-+0-9.e]+", cells[0]):
            head = cells
            tables.append([])
            continue
        if head and len(cells) == len(head):
            tables[-1].append(dict(zip(head, cells)))
    return tables


def fallbacks_per_level(err):
    """FAS fallbacks per level from stderr: `level: <l>` opens a cycle, the table of level l closes it."""
    counts, stack, in_table = {}, [], False
    for line in err.splitlines():
        m = re.match(r"level: (\d+)", line)
        if m:
            stack.append(int(m[1]))
            in_table = False
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if "iter" in cells and "level" in cells:  # a header: a new table, which closes the cycle of its level
                in_table = True
                if stack:
                    stack.pop()
        else:
            in_table = False
            if "falling back to gradient step" in line and stack:
                counts[stack[-1]] = counts.get(stack[-1], 0) + 1
    return counts


def num(v, spec=".6e"):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", required=True)
    parser.add_argument("--out", required=True)
    known, args = parser.parse_known_args()
    out = Path(known.out)
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, DEAL_II_NUM_THREADS=os.environ.get("DEAL_II_NUM_THREADS", "1"))
    p = subprocess.run([known.binary, *args], capture_output=True, text=True, env=env)
    sys.stderr.write(p.stderr)

    levels = [int(l) for l in (given(args, "multilevel") or option(args, "levels")).split(",") if l]
    hi, lo = max(levels), min(levels)
    ml = len(levels) > 1
    fine_tables = parse_tables(p.stdout)
    fine = fine_tables[-1] if fine_tables else []
    coarse = [row for t in parse_tables(p.stderr) for row in t]
    error = p.returncode != 0 or not fine

    def f(row, key):
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return math.nan

    # Status as pos_sphere_ml
    tol = float(option(args, "tol-residual"))
    max_iter = int(option(args, "max-iter"))
    last = fine[-1] if fine else {}
    residual = f(last, "residual")
    run, flat = 0, False
    for a, b in zip(fine, fine[1:]):
        run = run + 1 if a["energy"] == b["energy"] and a["residual"] == b["residual"] else 0
        flat = flat or run >= PLATEAU
    iterations = int(last["iter"]) if last else 0
    if error:
        status = "error"
    elif residual < tol:
        status = "converged"
    elif flat or (iterations > 0 and f(last, "step") == 0.0):
        status = "stalled"
    elif iterations >= max_iter:
        status = "max_iter"
    else:
        status = "stopped"

    fb = fallbacks_per_level(p.stderr)
    accepted = sum(1 for r in fine if r.get("coarse") == "*" and f(r, "step") > 0)
    rejected = sum(1 for r in fine if r.get("coarse") == "*" and f(r, "step") == 0)
    mid = [r for r in coarse if lo < int(r["level"]) < hi]
    accepted_mid = sum(1 for r in mid if r.get("coarse") == "*" and f(r, "step") > 0)
    rejected_mid = sum(1 for r in mid if r.get("coarse") == "*" and f(r, "step") == 0)
    fallback, fallback_mid = fb.get(hi, 0), sum(n for l, n in fb.items() if lo < l < hi)
    coarse_iter = sum(1 for r in coarse if int(r["iter"]) > 0)
    rho = sorted(f(r, "ratio") for r in fine if r.get("coarse") == "*" and f(r, "step") > 0 and "ratio" in r)
    rho_fin = [v for v in rho if math.isfinite(v)]

    vertices = [int(m) for m in re.findall(r"Number of vertices: (\d+)", p.stderr)]
    mesh = option(args, "mesh").upper()
    control = option(args, "correction-control").upper()
    counts = (lambda v: str(v)) if ml else (lambda v: "")
    coarse_setting = (lambda v: v) if ml else (lambda v: "")
    row = {
        "case": "ml" if ml else "sl", "mesh": mesh, "degree": option(args, "degree"),
        "levels": "-".join(map(str, sorted(levels))), "n_levels": len(levels), "level": hi,
        "n_dofs": vertices[-1] if vertices else "", "metric": "ENERGY_ADAPTIVE", "smoother": "unit",
        "step_rule": "exact" if flag(args, "ls-exact") else ("armijo" if flag(args, "line-search") else "fixed"),
        "coarse_retraction": coarse_setting("unit"), "point_transfer": coarse_setting(option(args, "transport")),
        "coarse_grad": coarse_setting("unit-" + option(args, "metric-coarse")),
        "coarse_max_iter": coarse_setting(str(COARSE_MAX_ITER)), "floor": "",
        "kappa": coarse_setting(option(args, "kappa")), "eps": coarse_setting(option(args, "eps")),
        "coarse_every": coarse_setting(option(args, "coarse-every")), "control": coarse_setting(control),
        "backoff": coarse_setting(option(args, "backoff")), "ratio_eta": coarse_setting(option(args, "ratio-eta")),
        "tol_inner_res": option(args, "tol-inner-res"), "tol_residual": option(args, "tol-residual"),
        "max_iter": max_iter, "beta": option(args, "beta"), "potential": option(args, "potential").upper(),
        "radius": option(args, "radius"), "status": status, "iterations": iterations, "residual": num(residual),
        "energy": last.get("energy", ""), "min_value": num(f(last, "min_value")), "cpu": num(f(last, "elapsed"), ".6f"),
        "wall": "", "attempted": counts(accepted + rejected + fallback), "accepted": counts(accepted),
        "rejected": counts(rejected), "fallback": counts(fallback), "failed": counts(0), "skipped": counts(0),
        "attempted_mid": counts(accepted_mid + rejected_mid + fallback_mid), "accepted_mid": counts(accepted_mid),
        "rejected_mid": counts(rejected_mid), "fallback_mid": counts(fallback_mid), "failed_mid": counts(0),
        "skipped_mid": counts(0), "coarse_iter": counts(coarse_iter),
        "rho_n": counts(len(rho_fin)), "rho_nan": counts(len(rho) - len(rho_fin)),
        "rho_min": num(min(rho_fin)) if rho_fin else "", "rho_median": num(statistics.median(rho_fin)) if rho_fin else "",
    }
    summary = out / "summary.csv"
    new = not summary.exists()
    with open(summary, "a") as csv:
        if new:
            csv.write(",".join(HEADER) + "\n")
        csv.write(",".join(str(row[k]) for k in HEADER) + "\n")
    with open(out / "iter.csv", "w") as csv:
        csv.write("iter,level,coarse,step,residual,energy,min_value,elapsed,grad_norm,grad_restr_norm,ratio\n")
        for r in coarse + fine:  # coarse rows first (stderr), then the finest level (stdout)
            csv.write(",".join([r["iter"], r["level"], "1" if r.get("coarse") == "*" else "0", r.get("step", ""),
                                r.get("residual", ""), r.get("energy", ""), r.get("min_value", ""),
                                r.get("elapsed", ""), r.get("grad_norm", ""), r.get("grad_restr_norm", ""),
                                r.get("ratio", "")]) + "\n")
    print(f"{row['case']} {row['levels']} unit, beta {row['beta']} {row['potential']} R {row['radius']}: {status} after "
          f"{iterations} iterations, residual {residual:.3e}, CPU {row['cpu']} s")
    return 0 if not error else 1


if __name__ == "__main__":
    sys.exit(main())
