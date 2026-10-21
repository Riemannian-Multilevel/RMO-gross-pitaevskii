#!/usr/bin/env python3
"""Selection of the staged multilevel benchmark (md/positive_multilevel_benchmark.md, section 3 and step 6.2).

Usage:
  ml_stages.py noise <stage 0 dir> <state.json>             # run-to-run spread, delta
  ml_stages.py screen <stage 0 dir> <stage A dir> <state.json>  # stage A: promoted values, near-misses (screen.csv)
  ml_stages.py next <state.json> <config file>              # stage B: candidates of the next round; exit 3 when done
  ml_stages.py decide <state.json> [<round dir>]            # stage B: apply the rule, update state and traces
  ml_stages.py stagec <state.json> <config file>            # stage C: the tuned configurations (shared if accepted)
  ml_stages.py selftest                                     # the selection on synthetic data with known effects

A stage directory holds the merged runs of pos_sphere_ml.sh (runs.csv, with the run directory
<config>/<disc>-<levels>/<problem> first), directly or in rep<k>/ (staging/pos_sphere_cpu.sh). Per run the CPU time
is the median over the repeats; status and iterations are those of the first repeat (checked equal over the repeats).

Promotion rule (section 3), candidate against reference on the same runs (Q1 and P1, the 24 problems, 2 and 3 levels,
4225 DoFs): it loses no problem, or at most as many as it wins, and the geometric mean of the CPU-time ratios over the
runs solved by both is at most 1 - delta; delta = max(5 %, twice the 90 % quantile of the run-to-run spread of stage 0).

Stage B per smoother: greedy rounds over the promoted values (each candidate is the current configuration with one
value added, or swapped for another value of its factor; the best that passes against the current one is kept, the
others are dropped), then one round of the near-misses of stage A, the 2 x 2 check of the two kept values with the
largest gains (a value whose removal costs less than delta is removed), and the tuned configuration on 4 levels
against the baseline on 4 levels. Then the shared configuration: per factor the value kept by at least half of the
smoothers (none on a tie), run on every smoother whose tuned configuration differs, accepted if every smoother loses
no problem and is within delta of its own. Traces: trace_<smoother>.csv next to state.json.
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SMOOTHERS = ["ea-armijo-floor", "ea-armijo-clip", "ea-exact-floor", "mass-exact-floor"]
FACTORS = ["pt", "cg", "cr", "cmi", "inner", "kappa", "ctrl"]  # order of the values in a configuration id
SCREEN_COUNTS = (2, 3)
SIZE = 4225


# --- configurations -------------------------------------------------------------------------------------------------

def config_id(smoother, values):
    values = sorted(values, key=lambda v: FACTORS.index(v.split("=")[0]))
    return "__".join([smoother, *values])


def parse_config(cid):
    words = cid.split("__")
    return words[0], words[1:]


def factor(value):
    return value.split("=")[0]


# --- data -----------------------------------------------------------------------------------------------------------

def load_runs(stage_dir):
    """All runs of a stage directory (runs.csv, or rep<k>/runs.csv), with the run directory split up."""
    stage_dir = Path(stage_dir)
    files = sorted(stage_dir.glob("rep*/runs.csv")) or sorted(stage_dir.glob("runs.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["rep"] = f.parent.name if f.parent != stage_dir else "rep1"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    parts = df["run"].str.split("/", expand=True)
    df["config"], df["case_dir"], df["problem"] = parts[0], parts[1], parts[2]
    df["disc"] = df["case_dir"].str.split("-").str[0]
    df["n"] = df["n_levels"].astype(int)
    df["size"] = df["n_dofs"].astype(int)
    return df


def aggregate(df):
    """One row per run (config, disc, n, size, problem): median CPU over the repeats, first repeat otherwise."""
    if df.empty:
        return df
    keys = ["config", "disc", "n", "size", "problem"]
    df = df.sort_values("rep")
    it = df.groupby(keys)["iterations"].agg(["min", "max"])
    bad = it[it["min"] != it["max"]]
    if len(bad):
        print(f"warning: iterations differ over the repeats in {len(bad)} runs, e.g. {bad.index[0]}", file=sys.stderr)
    first = df.groupby(keys).first().reset_index()
    first["cpu"] = df.groupby(keys)["cpu"].median().values
    first["solved"] = first["status"] == "converged"
    return first


def load_sources(state):
    return aggregate(pd.concat([load_runs(d) for d in state["sources"]], ignore_index=True))


def select(data, cid, counts, size=SIZE):
    return data[(data.config == cid) & data.n.isin(counts) & (data["size"] == size)]


def measured(data, cid, counts, size=SIZE):
    d = select(data, cid, counts, size)
    return all(((d.n == n) & (d.disc == disc)).any() for n in counts for disc in ("q1", "p1"))


# --- the rule -------------------------------------------------------------------------------------------------------

def gmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x)))) if len(x) else math.nan


def compare(data, ref, cand, delta, counts=SCREEN_COUNTS, size=SIZE):
    """The promotion rule, candidate against reference on the same runs."""
    keys = ["disc", "n", "problem"]
    a, b = select(data, ref, counts, size), select(data, cand, counts, size)
    m = a.merge(b, on=keys, suffixes=("_ref", "_cand"))
    both = m[m.solved_ref & m.solved_cand]
    ratio = both.cpu_cand / both.cpu_ref
    r = {
        "ref": ref, "cand": cand, "runs": len(m), "solved_ref": int(m.solved_ref.sum()),
        "solved_cand": int(m.solved_cand.sum()), "lost": int((m.solved_ref & ~m.solved_cand).sum()),
        "won": int((~m.solved_ref & m.solved_cand).sum()), "cpu": gmean(ratio),
        "iter": gmean(both.iterations_cand / both.iterations_ref),
    }
    for n in (2, 3, 4):
        r[f"cpu_{n}lv"] = gmean(ratio[both.n == n]) if n in counts else math.nan
    for disc in ("q1", "p1"):
        r[f"cpu_{disc}"] = gmean(ratio[both.disc == disc])
    try:
        from scipy.stats import wilcoxon
        x = np.log(ratio.to_numpy())
        r["wilcoxon_p"] = float(wilcoxon(x).pvalue) if len(x) > 5 and np.any(x != 0) else math.nan
    except ImportError:
        r["wilcoxon_p"] = math.nan
    r["passes"] = bool((r["lost"] == 0 or r["lost"] <= r["won"]) and r["cpu"] <= 1 - delta)
    return r


# --- state ----------------------------------------------------------------------------------------------------------

def read_state(path):
    return json.loads(Path(path).read_text())


def write_state(path, state):
    Path(path).write_text(json.dumps(state, indent=1))


TRACE_COLUMNS = ["round", "phase", "candidate", "ref", "cand", "runs", "solved_ref", "solved_cand", "lost", "won", "cpu",
                 "cpu_2lv", "cpu_3lv", "cpu_4lv", "cpu_q1", "cpu_p1", "iter", "wilcoxon_p", "passes", "interaction",
                 "decision"]


def trace(state_path, smoother, row):
    path = Path(state_path).parent / f"trace_{smoother}.csv"
    df = pd.DataFrame([row]).reindex(columns=TRACE_COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def cmd_noise(stage0, state_path):
    df = load_runs(stage0)
    keys = ["config", "disc", "n", "size", "problem"]
    g = df[df.status == "converged"].groupby(keys)["cpu"]
    spread = (g.max() / g.min() - 1)[g.count() > 1]
    q50, q90 = float(spread.quantile(0.5)), float(spread.quantile(0.9))
    delta = max(0.05, 2 * q90)
    state = read_state(state_path) if Path(state_path).exists() else {}
    state.update({"delta": delta, "spread_median": q50, "spread_q90": q90, "spread_runs": int(len(spread)),
                  "sources": [str(stage0)]})
    write_state(state_path, state)
    print(f"run-to-run spread over {len(spread)} runs: median {q50:.2%}, 90 % quantile {q90:.2%}; delta {delta:.2%}")


def cmd_screen(stage0, stageA, state_path):
    state = read_state(state_path)
    delta = state["delta"]
    data = aggregate(pd.concat([load_runs(stage0), load_runs(stageA)], ignore_index=True))
    rows = []
    state["smoothers"] = {}
    for s in SMOOTHERS:
        promoted, near = [], []
        for cid in sorted(c for c in data.config.unique() if c.startswith(s + "__") and "ctrl=" not in c):
            _, values = parse_config(cid)
            r = compare(data, s, cid, delta)
            r.update(smoother=s, value=values[0])
            r["decision"] = "promoted" if r["passes"] else (
                "near-miss" if (r["lost"] == 0 or r["lost"] <= r["won"]) and 1 - delta < r["cpu"] < 1 else "")
            (promoted if r["passes"] else near if r["decision"] else []).append(values[0])
            rows.append(r)
        state["smoothers"][s] = {"current": [], "promoted": promoted, "near_miss": near, "dropped": [], "gains": {},
                                 "phase": "greedy", "round": 1, "pending": [], "levels4": None}
    screen = pd.DataFrame(rows)
    screen.to_csv(Path(state_path).parent / "screen.csv", index=False)
    write_state(state_path, state)
    cols = ["smoother", "value", "solved_cand", "lost", "won", "cpu", "cpu_2lv", "cpu_3lv", "iter", "decision"]
    print(screen[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # RATIO diagnostic (not a candidate): decrease ratios of the corrections
    diag = load_runs(stageA)
    diag = diag[diag.config.str.contains("ctrl=ratio")]
    if len(diag):
        print("RATIO diagnostic:")
        print(diag.groupby("config").agg(runs=("run", "size"), solved=("status", lambda x: (x == "converged").sum()),
                                         rho_median=("rho_median", "median"), rho_min=("rho_min", "min"),
                                         failed=("failed", "sum"), skipped=("skipped", "sum")).to_string())


# --- stage B --------------------------------------------------------------------------------------------------------

def candidates(sm, smoother):
    """Configurations of the next round of a smoother, by phase."""
    cur = sm["current"]
    used = {factor(v): v for v in cur}
    if sm["phase"] in ("greedy", "near_miss"):
        pool = sm["promoted"] if sm["phase"] == "greedy" else sm["near_miss"]
        out = []
        for v in pool:
            if v in cur or v in sm["dropped"]:
                continue
            new = [w for w in cur if factor(w) != factor(v)] + [v]
            out.append((config_id(smoother, new), v))
        return out
    if sm["phase"] == "check2x2":
        top = sorted(cur, key=lambda v: -sm["gains"].get(v, 0))[:2]
        if len(top) < 2:
            return []
        a, b = top
        return [(config_id(smoother, [w for w in cur if w != a]), f"-{a}"),
                (config_id(smoother, [w for w in cur if w != b]), f"-{b}"),
                (config_id(smoother, [w for w in cur if w not in (a, b)]), f"-{a},-{b}")]
    return []


def cmd_next(state_path, config_file):
    state = read_state(state_path)
    data = load_sources(state)
    lines = []
    done = True
    for s, sm in state["smoothers"].items():
        while sm["phase"] in ("greedy", "near_miss", "check2x2"):
            cands = candidates(sm, s)
            if cands:
                break
            sm["phase"] = {"greedy": "near_miss", "near_miss": "check2x2", "check2x2": "levels4"}[sm["phase"]]
        if sm["phase"] in ("greedy", "near_miss", "check2x2"):
            sm["pending"] = [list(c) for c in cands]
            lines += [f"{cid} 2,3" for cid, _ in cands if not measured(data, cid, SCREEN_COUNTS)]
            done = False
        elif sm["phase"] == "levels4" and not sm["current"]:
            sm["phase"] = "done"  # the baseline: nothing to compare on 4 levels
        elif sm["phase"] == "levels4":
            cid = config_id(s, sm["current"])
            sm["pending"] = [[cid, "4 levels"]]
            if not measured(data, cid, (4,)):
                lines.append(f"{cid} 4")
            done = False
    if done and state.get("shared_phase") is None:
        shared = shared_candidate(state)
        state["shared_phase"] = "pending" if shared is not None else "none"
        state["shared"] = shared
        if shared is not None:
            lines += [f"{config_id(s, shared)} 2,3" for s in state["smoothers"]
                      if not measured(data, config_id(s, shared), SCREEN_COUNTS)]
            done = False
    elif state.get("shared_phase") == "pending":
        done = False
    Path(config_file).write_text("".join(f"{line}\n" for line in sorted(set(lines))))
    write_state(state_path, state)
    print(f"{len(set(lines))} configurations to run" + ("" if not done else "; stage B done"))
    return 3 if done else 0


def shared_candidate(state):
    """Per factor the value kept by at least half of the smoothers (none on a tie between two values); None if every
    tuned configuration is already that one."""
    tuned = [sm["current"] for sm in state["smoothers"].values()]
    count = {}
    for values in tuned:
        for v in values:
            count[v] = count.get(v, 0) + 1
    shared = []
    for f in FACTORS:
        votes = sorted(((c, v) for v, c in count.items() if factor(v) == f), reverse=True)
        if votes and 2 * votes[0][0] >= len(tuned) and (len(votes) == 1 or votes[1][0] < votes[0][0]):
            shared.append(votes[0][1])
    if all(sorted(values) == sorted(shared) for values in tuned):
        return None
    return shared


def cmd_decide(state_path, round_dir=None):
    state = read_state(state_path)
    if round_dir and Path(round_dir).exists() and str(round_dir) not in state["sources"]:
        state["sources"].append(str(round_dir))
    data = load_sources(state)
    delta = state["delta"]
    for s, sm in state["smoothers"].items():
        if not sm["pending"]:
            continue
        cur = config_id(s, sm["current"])
        if sm["phase"] in ("greedy", "near_miss"):
            results = []
            for cid, v in sm["pending"]:
                r = compare(data, cur, cid, delta)
                if r["runs"] == 0:
                    print(f"warning: no runs of {cid} (or of {cur}); counted as failed", file=sys.stderr)
                results.append((r, v))
            passing = [(r, v) for r, v in results if r["passes"]]
            best = min(passing, key=lambda x: x[0]["cpu"]) if passing else None
            for r, v in results:
                if best and v == best[1]:
                    decision = "kept"
                elif r["passes"]:
                    decision = "passed, not best"
                elif sm["phase"] == "greedy" and sm["round"] > 1:
                    decision = "failed in combination"
                else:
                    decision = "failed"
                trace(state_path, s, {"round": sm["round"], "phase": sm["phase"], "candidate": v, **r,
                                      "decision": decision})
                if decision.startswith("failed"):
                    sm["dropped"].append(v)
            if best:
                r, v = best
                sm["current"] = [w for w in sm["current"] if factor(w) != factor(v)] + [v]
                sm["gains"][v] = 1 - r["cpu"]
                sm["round"] += 1
            else:
                sm["phase"] = "check2x2" if sm["phase"] == "near_miss" else "near_miss"
        elif sm["phase"] == "check2x2":
            (cid_a, a), (cid_b, b), (cid_ab, _) = sm["pending"]
            ra, rb, rab = (compare(data, cid, cur, delta) for cid in (cid_a, cid_b, cid_ab))  # cur against without
            sel = lambda c: select(data, c, SCREEN_COUNTS).set_index(["disc", "n", "problem"])["cpu"]
            lc = np.log(pd.concat([sel(cur), sel(cid_a), sel(cid_b), sel(cid_ab)], axis=1).dropna())
            interaction = float(np.exp((lc.iloc[:, 0] - lc.iloc[:, 1] - lc.iloc[:, 2] + lc.iloc[:, 3]).mean()))
            removed = []
            if not rab["passes"]:
                removed = [a[1:], b[1:].split(",")[0]]
            elif not ra["passes"] or not rb["passes"]:
                removed = [a[1:]] if (not ra["passes"] and (rb["passes"] or ra["cpu"] >= rb["cpu"])) else [b[1:]]
            for r, label in ((ra, a), (rb, b), (rab, f"{a},{b}")):
                trace(state_path, s, {"round": sm["round"], "phase": "check2x2", "candidate": label, **r,
                                      "interaction": interaction,
                                      "decision": "removed" if any(x in label for x in removed) and not r["passes"]
                                      else "needed"})
            sm["current"] = [w for w in sm["current"] if w not in removed]
            sm["phase"] = "levels4"
        elif sm["phase"] == "levels4":
            r = compare(data, s, cur, delta, counts=(4,))
            sm["levels4"] = {"passes": r["passes"], "cpu": r["cpu"], "lost": r["lost"], "won": r["won"]}
            trace(state_path, s, {"round": sm["round"], "phase": "levels4", "candidate": cur, **r,
                                  "decision": "passes on 4 levels" if r["passes"] else "fails on 4 levels"})
            sm["phase"] = "done"
        sm["pending"] = []
    if state.get("shared_phase") == "pending" and all(sm["phase"] == "done" for sm in state["smoothers"].values()):
        shared = state["shared"]
        ok = True
        for s, sm in state["smoothers"].items():
            r = compare(data, config_id(s, sm["current"]), config_id(s, shared), delta)
            within = (r["lost"] == 0 or r["lost"] <= r["won"]) and r["cpu"] <= 1 + delta
            ok = ok and within
            trace(state_path, s, {"round": sm["round"], "phase": "shared", "candidate": config_id(s, shared), **r,
                                  "decision": "within delta" if within else "outside delta"})
        state["shared_phase"] = "accepted" if ok else "rejected"
    write_state(state_path, state)
    for s, sm in state["smoothers"].items():
        print(f"{s}: {sm['phase']}, current {config_id(s, sm['current'])}")
    if state.get("shared_phase") in ("accepted", "rejected"):
        print(f"shared configuration {state['shared']}: {state['shared_phase']}")


def tuned_configs(state):
    """{smoother: tuned configuration id}: the shared configuration if it was accepted, else each smoother's own."""
    shared = state.get("shared") if state.get("shared_phase") == "accepted" else None
    return {s: config_id(s, shared if shared is not None else sm["current"]) for s, sm in state["smoothers"].items()}


def cmd_stagec(state_path, config_file):
    """Stage C: the tuned configurations on 2, 3, 4 levels, and that of EA Armijo floor at 16641 DoFs (the size block,
    out of sample); a tuned configuration equal to the baseline is not rerun (stage 0 has it)."""
    tuned = tuned_configs(read_state(state_path))
    lines = [f"{cid} 2,3,4" for s, cid in tuned.items() if cid != s]
    if tuned.get("ea-armijo-floor", "ea-armijo-floor") != "ea-armijo-floor":
        lines.append(f"{tuned['ea-armijo-floor']} 2,3,4 16641")
    Path(config_file).write_text("".join(f"{line}\n" for line in lines))
    print(f"{len(lines)} configurations for stage C:", *lines, sep="\n  ")


# --- selftest -------------------------------------------------------------------------------------------------------

EFFECTS = {"cg=mass": 0.80, "kappa=0.5": 0.90, "cmi=8": 0.92, "cr=plain": 1.02, "pt=log": 0.97, "inner=default": 1.03}


def synthetic_cpu(smoother, values, base, rng):
    f = np.prod([EFFECTS.get(v, 1.0) for v in values if not (smoother != SMOOTHERS[0] and v == "kappa=0.5")])
    if "cg=mass" in values and "cmi=8" in values:  # cmi=8 is redundant with cg=mass (an interaction)
        f /= EFFECTS["cmi=8"]
    return base * f * float(np.exp(rng.normal(0, 0.004)))


def simulate(out, configs, reps, seed):
    """runs.csv in out/rep<k> for the configurations [(config id, counts, size)], with the effects of EFFECTS."""
    problems = [f"b{b}-{v}-r{r}" for b in (100, 1000, 10000) for v in ("square", "optical_lattice") for r in (4, 6, 8, 10)]
    base = {(d, p): 1.0 + 0.1 * k for k, (d, p) in enumerate((d, p) for d in ("q1", "p1") for p in problems)}
    for k in range(1, reps + 1):
        rng = np.random.default_rng(seed + k)
        rows = []
        for cid, counts, size in configs:
            smoother, values = parse_config(cid)
            for d in ("q1", "p1"):
                for n in counts:
                    for p in problems:
                        cpu = synthetic_cpu(smoother, values, base[(d, p)] * (1 + 0.05 * n), rng)
                        lost = "cr=plain" in values and p == problems[0]
                        rows.append({"run": f"{cid}/{d}-lv{n}/{p}", "n_levels": n, "n_dofs": size,
                                     "status": "max_iter" if lost else "converged", "iterations": 100, "cpu": cpu,
                                     "rho_median": math.nan, "rho_min": math.nan, "failed": 0, "skipped": 0})
        (out / f"rep{k}").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / f"rep{k}" / "runs.csv", index=False)


def cmd_selftest():
    global SMOOTHERS
    SMOOTHERS = ["ea-armijo-floor", "ea-exact-floor"]  # the second without the effect of kappa=0.5
    tmp = Path(tempfile.mkdtemp())
    try:
        s = "ea-armijo-floor"
        simulate(tmp / "stage0", [(t, (2, 3, 4), SIZE) for t in SMOOTHERS], 3, 0)
        values = list(EFFECTS)
        simulate(tmp / "stageA", [(f"{t}__{v}", (2, 3), SIZE) for t in SMOOTHERS for v in values], 1, 10)
        state = tmp / "state.json"
        cmd_noise(tmp / "stage0", state)
        cmd_screen(tmp / "stage0", tmp / "stageA", state)
        st = read_state(state)["smoothers"][s]
        assert sorted(st["promoted"]) == ["cg=mass", "cmi=8", "kappa=0.5"], st["promoted"]
        assert st["near_miss"] == ["pt=log"], st["near_miss"]
        for r in range(1, 12):
            cfg = tmp / f"round{r}.txt"
            if cmd_next(state, cfg) == 3:
                break
            lines = [line.split() for line in cfg.read_text().splitlines()]
            if lines:
                simulate(tmp / f"round{r}", [(c, tuple(int(x) for x in n.split(",")), SIZE) for c, n in lines], 3,
                         100 + 10 * r)
            cmd_decide(state, tmp / f"round{r}")
        st = read_state(state)["smoothers"][s]
        assert st["phase"] == "done", st["phase"]
        assert sorted(st["current"]) == ["cg=mass", "kappa=0.5"], st["current"]
        assert "cmi=8" in st["dropped"] and "pt=log" in st["dropped"], st["dropped"]
        assert st["levels4"]["passes"], st["levels4"]
        tr = pd.read_csv(tmp / f"trace_{s}.csv")
        assert (tr[tr.candidate == "cmi=8"].decision == "failed in combination").any()
        full = read_state(state)
        assert full["smoothers"]["ea-exact-floor"]["current"] == ["cg=mass"], full["smoothers"]["ea-exact-floor"]
        assert full["shared"] == ["cg=mass", "kappa=0.5"] and full["shared_phase"] == "accepted", full["shared"]
        print("selftest passed: tuned", config_id(s, st["current"]), "and ea-exact-floor__cg=mass; shared",
              full["shared"], full["shared_phase"])
    finally:
        shutil.rmtree(tmp)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "noise" and len(args) == 2:
        cmd_noise(*args)
    elif cmd == "screen" and len(args) == 3:
        cmd_screen(*args)
    elif cmd == "next" and len(args) == 2:
        return cmd_next(*args)
    elif cmd == "decide" and len(args) in (1, 2):
        cmd_decide(*args)
    elif cmd == "stagec" and len(args) == 2:
        cmd_stagec(*args)
    elif cmd == "selftest":
        cmd_selftest()
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
