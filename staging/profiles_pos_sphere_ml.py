#!/usr/bin/env python3
"""Performance profiles (Dolan and More 2002) and tables of the staged multilevel benchmark
(md/positive_multilevel_benchmark.md, section 3, stage C, and step 6.3).

Usage: staging/profiles_pos_sphere_ml.py <output dir> <stage 0 dir> [<stage C dir> <state.json>]

Versions: single level (SL) and FAS on n levels per smoother, the baseline (stage 0) and, with stage C and the state of
stage B, the tuned configuration (marked *; the shared one if it was accepted); the unit sphere (main_coarse) as a
reference. Problems: Q1 and P1 x the 24 problems at one size (4225 or 16641 DoFs on the finest level); level counts are
only compared within one size. Cost: CPU time (median over the repeats) or iterations if converged, else infinity;
r_ps = t_ps / min_s t_ps over the versions of the profile, rho_s(tau) the fraction of problems with r_ps <= tau.

Writes, per size:
- profile_<size>_<n>lv_{cpu,iter}.pdf: SL and FAS on n levels (baseline, tuned) of every smoother, the unit sphere;
- profile_<size>_<smoother>_{cpu,iter}.pdf: one smoother, SL against FAS on 2, 3, 4 levels (baseline, tuned);
- pair_<size>_<smoother>_<n>lv_{tuned,sl}.pdf: two-version profiles (Gould and Scott 2016: with more than two versions
  only the best is ranked): tuned against baseline, and the better FAS configuration against SL;
and tables.csv, tables.md: per version solved, rho(1), rho(2), the area under rho on log tau in [0, log 4] (normalized
to 1), the geometric means of the CPU and iteration ratios to SL of the same smoother over the problems solved by both,
and the corrections (attempted, accepted, rejected, fallback; finest level, summed over the problems).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ml_stages import SMOOTHERS, aggregate, load_runs, tuned_configs  # noqa: E402
from pos_sphere_plot import plt  # noqa: E402
from profiles_pos_sphere_cpu import ratios, rho  # noqa: E402

COLORS = {"ea-armijo-floor": "#08519c", "ea-armijo-clip": "#a50f15", "ea-exact-floor": "#31a354",
          "mass-exact-floor": "#756bb1", "unit": "#555555"}
LEVEL_STYLE = {1: ":", 2: "-", 3: "--", 4: "-."}


def versions(data, tuned):
    """{label: (smoother, n, config)} of the runs present; tuned: {smoother: config id}."""
    out = {}
    for s in [*SMOOTHERS, "unit"]:
        for n in (1, 2, 3, 4):
            if ((data.config == s) & (data.n == n)).any():
                out[f"{s} {'SL' if n == 1 else f'FAS {n}lv'}"] = (s, n, s)
            t = tuned.get(s)
            if n > 1 and t and t != s and ((data.config == t) & (data.n == n)).any():
                out[f"{s}* FAS {n}lv"] = (s, n, t)
    return out


def cost(data, size, v, measure):
    s, n, cid = v
    d = data[(data.config == cid) & (data.n == n) & (data["size"] == size)]
    return {(r.disc, r.problem): (getattr(r, measure) if r.solved else math.inf) for r in d.itertuples()}


def plot(path, costs, problems, title, measure):
    r = ratios(costs, problems)
    finite = [x for v in r.values() for x in v if math.isfinite(x)]
    tau_max = next(t for t in [2, 5, 10, 20, 50, 100, 1000] if t >= 1.2 * max(finite, default=1))
    taus = np.logspace(0, math.log10(tau_max), 401)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    for label in costs:
        s = label.split()[0].rstrip("*")
        n = 1 if label.endswith("SL") else int(label.split()[-1][0])
        ax.step(taus, [rho(r[label], t) for t in taus], where="post", color=COLORS.get(s, "k"),
                ls=LEVEL_STYLE[n], lw=2.0 if "*" in label else 1.2, alpha=0.6 if s == "unit" else 1.0, label=label)
    ax.set_xscale("log")
    ax.set_xlim(1, tau_max)
    ticks = [t for t in [1, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000] if t <= tau_max]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_ylim(0, 1.02)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_xlabel(r"$\tau$ (" + ("CPU time" if measure == "cpu" else "iterations") + " / best)")
    ax.set_ylabel(r"$\rho(\tau)$")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(path)
    plt.close(fig)


def table_rows(data, size, vs, problems):
    rows = []
    costs = {label: cost(data, size, v, "cpu") for label, v in vs.items()}
    r = ratios(costs, problems)
    taus = np.logspace(0, math.log10(4), 201)
    for label, (s, n, cid) in vs.items():
        d = data[(data.config == cid) & (data.n == n) & (data["size"] == size)]
        sl = data[(data.config == s) & (data.n == 1) & (data["size"] == size)]
        m = d.merge(sl, on=["disc", "problem"], suffixes=("", "_sl"))
        both = m[m.solved & m.solved_sl]
        gm = lambda x: float(np.exp(np.mean(np.log(x)))) if len(x) else math.nan
        rows.append({
            "size": size, "version": label, "smoother": s, "levels": n, "config": cid, "problems": len(d),
            "solved": int(d.solved.sum()), "rho1": rho(r[label], 1), "rho2": rho(r[label], 2),
            "area_1_4": float(np.trapezoid([rho(r[label], t) for t in taus], np.log10(taus)) / math.log10(4)),
            "cpu_vs_sl": gm(both.cpu / both.cpu_sl) if n > 1 else 1.0,
            "iter_vs_sl": gm(both.iterations / both.iterations_sl) if n > 1 else 1.0,
            **{k: int(pd.to_numeric(d[k], errors="coerce").fillna(0).sum()) if n > 1 else 0
               for k in ("attempted", "accepted", "rejected", "fallback")},
        })
    return rows


def main():
    if len(sys.argv) not in (3, 5):
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    frames = [load_runs(sys.argv[2])]
    tuned = {}
    if len(sys.argv) == 5:
        frames.append(load_runs(sys.argv[3]))
        tuned = tuned_configs(json.loads(Path(sys.argv[4]).read_text()))
    data = aggregate(pd.concat(frames, ignore_index=True))
    rows = []
    for size in sorted(data["size"].unique()):
        ds = data[data["size"] == size]
        problems = sorted({(r.disc, r.problem) for r in ds.itertuples()})
        vs = versions(ds, tuned)
        rows += table_rows(ds, size, vs, problems)
        for measure in ("cpu", "iterations"):
            tag = "cpu" if measure == "cpu" else "iter"
            for n in (2, 3, 4):
                sel = {k: v for k, v in vs.items() if v[1] in (1, n)}
                if any(v[1] == n for v in sel.values()):
                    plot(out / f"profile_{size}_{n}lv_{tag}.pdf",
                         {k: cost(ds, size, v, measure) for k, v in sel.items()}, problems,
                         f"{size} DoFs: single level and FAS on {n} levels ({tag})", measure)
            for s in [*SMOOTHERS, "unit"]:
                sel = {k: v for k, v in vs.items() if v[0] == s}
                if len(sel) > 1:
                    plot(out / f"profile_{size}_{s}_{tag}.pdf", {k: cost(ds, size, v, measure) for k, v in sel.items()},
                         problems, f"{size} DoFs, {s}: level counts ({tag})", measure)
        for s in SMOOTHERS:
            for n in (2, 3, 4):
                base, tun, sl = f"{s} FAS {n}lv", f"{s}* FAS {n}lv", f"{s} SL"
                if base in vs and tun in vs:
                    plot(out / f"pair_{size}_{s}_{n}lv_tuned.pdf",
                         {k: cost(ds, size, vs[k], "cpu") for k in (base, tun)}, problems,
                         f"{size} DoFs, {s}, {n} levels: tuned against baseline (CPU)", "cpu")
                fas = tun if tun in vs else base
                if fas in vs and sl in vs:
                    plot(out / f"pair_{size}_{s}_{n}lv_sl.pdf",
                         {k: cost(ds, size, vs[k], "cpu") for k in (sl, fas)}, problems,
                         f"{size} DoFs, {s}, {n} levels: FAS against single level (CPU)", "cpu")
    table = pd.DataFrame(rows)
    table.to_csv(out / "tables.csv", index=False)
    with open(out / "tables.md", "w") as f:
        for size, t in table.groupby("size"):
            f.write(f"## {size} DoFs\n\n")
            cols = ["version", "solved", "rho1", "rho2", "area_1_4", "cpu_vs_sl", "iter_vs_sl", "attempted", "accepted",
                    "rejected", "fallback"]
            f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
            for r in t[cols].itertuples(index=False):
                f.write("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |\n")
            f.write("\n")
    print(f"{len(table)} versions; tables and profiles in {out}")


if __name__ == "__main__":
    main()
