#!/usr/bin/env python3
"""Runs the study and evaluates it. From the build directory:

    python3 ../study.py             all stages: a new study, its median tables and figures
    python3 ../study.py study       a new study only
    python3 ../study.py median      median tables of the latest study (or of --out runs/<study>)
    python3 ../study.py figures     median tables and figures of the latest study (or of --out runs/<study>)

  study    REPS repetitions of all runs in OUT/rep<k>/, with the source state (commit.txt, dirty.txt,
           dirty.patch), the parameters (params.txt) and the binary used (main_coarse). Every study gets a
           new directory runs/l<level>_<potential>_<date>_<time>_<commit>, with -dirty after the commit if
           tracked files have uncommitted changes; runs/latest links to the last one.
  median   plot/median_tables.py: tables with the median CPU time over the repetitions in OUT/
  figures  plot/plot_convergence.py: Figures 7-9 per beta and metric in OUT/figures/b<beta>/; a figure
           without its tables is skipped

Parameters are options, e.g.
    python3 ../study.py --beta-range 100 1000 10000 --iter-range 20 40 60 --top-level 10 --reps 1
    python3 ../study.py figures --plot-args='--format png --coarse-steps'
Pass the same parameters to median and figures.
"""
import argparse
import datetime
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent  # the source tree

# --transport per transport token of the file names, for the metric mass; for frob, proj -> frobenius and
# the others get the ending _frobenius
TRANSPORTS = {"proj": "mass", "diff": "differential", "adj1": "adjoint_restriction", "adj2": "adjoint_differential"}
METRICS = {"mass": "mass", "frob": "frobenius"}


def fail(message):
    sys.exit(f"error: {message}")


def run(argv, stdout=None):
    """Prints the command like `set -x`, runs it and stops the study if it fails."""
    argv = [str(a) for a in argv]
    print("+", shlex.join(argv), flush=True)
    if subprocess.run(argv, stdout=stdout).returncode != 0:
        fail(f"{shlex.join(argv)} failed")


def git(*args):
    """Output of a git command in the source tree ("" if it fails)."""
    return subprocess.run(["git", "-C", SRC, *args], capture_output=True, text=True).stdout


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", choices=["all", "study", "median", "figures"], default="all")
    p.add_argument("--out", type=Path, help="study directory; default: a new one for all/study, else runs/latest")
    p.add_argument("--argv0", default="./main_coarse")
    p.add_argument("--reps", default=3, type=int, help="repetitions of every run")
    p.add_argument("--beta-range", nargs="+", default=["1000"])
    p.add_argument("--iter-range", nargs="+", default=["150"], help="--max-iter of the multilevel runs, one per beta")
    p.add_argument("--metrics", nargs="+", choices=METRICS, default=["mass", "frob"])
    p.add_argument("--transports", nargs="+", choices=TRANSPORTS, default=["proj", "diff", "adj1", "adj2"])
    p.add_argument("--depths", nargs="+", default=[2, 3, 4, 5], type=int, help="numbers of levels")
    p.add_argument("--kappa", default="0.5")
    p.add_argument("--eps", default="2e-8")
    p.add_argument("--tol-res", default="1e-8")
    p.add_argument("--top-level", default=11, type=int, help="finest level")
    p.add_argument("--radius", default="11")
    p.add_argument("--boundary", default="dirichlet")
    p.add_argument("--potential", default="optical_lattice", help="also the ending of the file names")
    p.add_argument("--line-search", default=1, type=int, choices=[0, 1])
    p.add_argument("--simple-table", default=1, type=int, choices=[0, 1])
    p.add_argument("--plot-args", default="", help="options of plot_convergence.py, e.g. --plot-args='--format tex'")
    a = p.parse_args()

    if a.reps < 1:
        p.error("--reps must be a positive integer")
    if len(a.beta_range) != len(a.iter_range):
        p.error("--beta-range and --iter-range differ in length")
    return a


def study_dir(a):
    """A new directory runs/l<level>_<potential>_<date>_<time>_<commit>[-dirty]."""
    commit = git("rev-parse", "--short", "HEAD").strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        commit += "-dirty"
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path(f"runs/l{a.top_level}_{a.potential}_{now}_{commit}")


def record_source(a, out):
    """The parameters and the source state of the study. dirty.patch holds the changes of tracked files only;
    untracked files are listed in dirty.txt."""
    with open(out / "params.txt", "w") as f:
        for name, value in vars(a).items():
            if isinstance(value, list):
                value = " ".join(map(str, value))
            print(f"{name}={value}", file=f)
    (out / "commit.txt").write_text(git("rev-parse", "HEAD"))
    (out / "dirty.txt").write_text(git("status", "--short"))
    (out / "dirty.patch").write_text(git("diff"))


def transport_option(metric, transport):
    """--transport for the tokens of the file names: for frob, proj -> frobenius and the others get the
    ending _frobenius."""
    if metric == "frob":
        return "frobenius" if transport == "proj" else TRANSPORTS[transport] + "_frobenius"
    return TRANSPORTS[transport]


def runs(a, binary):
    """The runs of one repetition, in the order of the study: (file name, command)."""
    common = [binary, "--potential", a.potential, "--boundary", a.boundary, "--radius", a.radius]
    if a.line_search:
        common.append("--line-search")
    if a.simple_table:
        common.append("--simple-table")
    L = a.top_level

    for beta, iters in zip(a.beta_range, a.iter_range):
        # Single level reference, solved to a residual of 1e-14
        yield (f"sl_b{beta}_l{L}_{a.potential}.org",
               common + ["--levels", L, "--metric", "none", "--max-iter", 1000, "--tol-residual", "1e-14",
                         "--beta", beta])

        # Multilevel
        for depth in a.depths:
            levels = ",".join(str(level) for level in range(L, L - depth, -1))  # e.g. 11,10,9 for depth 3
            for metric in a.metrics:
                for transport in a.transports:
                    yield (f"ml_{metric}_{transport}_b{beta}_l{L}_depth{depth}_{a.potential}.org",
                           common + ["--kappa", a.kappa, "--eps", a.eps, "--multilevel", levels,
                                     "--max-iter", iters, "--tol-residual", a.tol_res, "--beta", beta,
                                     "--metric", METRICS[metric], "--transport", transport_option(metric, transport)])


def link_latest(out):
    latest = Path("runs/latest")
    latest.parent.mkdir(exist_ok=True)
    latest.unlink(missing_ok=True)
    latest.symlink_to(out.resolve())


def study(a, out):
    """REPS repetitions of all runs, in the new directory out (out/rep<k>/)."""
    if out.exists():
        fail(f"{out} exists: choose another --out")

    # Build the binary first, so that the recorded source state describes it
    run(["cmake", "--build", Path(a.argv0).parent, "--target", "main_coarse"])
    out.mkdir(parents=True)
    record_source(a, out)

    # The runs use a copy of the binary: rebuilding main_coarse during the study does not affect them
    binary = out / "main_coarse"
    shutil.copy2(a.argv0, binary)

    total = a.reps * len(list(runs(a, binary)))
    count = 0
    for rep in range(1, a.reps + 1):
        rep_dir = out / f"rep{rep}"
        rep_dir.mkdir()
        for name, argv in runs(a, binary):
            count += 1
            print(f"[{count}/{total}]", flush=True)
            with open(rep_dir / name, "w") as f:
                run(argv, stdout=f)

    # Marks the study as complete for median and figures
    (out / "study.done").touch()
    link_latest(out)


def median(out):
    if not (out / "study.done").exists():
        fail(f"no complete study in {out}: run {sys.argv[0]} study")
    run(["python3", SRC / "plot/median_tables.py", out, *sorted(out.glob("rep*"))])


def figures(a, out):
    for beta in a.beta_range:
        for metric in a.metrics:
            run(["python3", SRC / "plot/plot_convergence.py", "--data-dir", out,
                 "--out-dir", out / "figures" / f"b{beta}", "--metric", metric, "--beta", beta,
                 "--level", a.top_level, "--suffix", a.potential,
                 "--depths", ",".join(map(str, a.depths)), *shlex.split(a.plot_args)])


def main():
    a = parse_args()
    new_study = a.stage in ("all", "study")
    if a.out is None:
        a.out = study_dir(a) if new_study else Path("runs/latest")
    out = a.out

    if new_study:
        study(a, out)
    if a.stage in ("all", "median", "figures"):
        median(out)  # figures need the median tables
    if a.stage in ("all", "figures"):
        figures(a, out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("interrupted")
