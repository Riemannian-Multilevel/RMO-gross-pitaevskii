# README: staged multilevel benchmark on the positive sphere

Plan and design: `md/positive_multilevel_benchmark.md` (drafts), section 3 (stages) and steps 5–7; matrix:
`tex/ml_benchmark_matrix.tex`; theory: `tex/drafts/positive_multilevel.tex`. Scripts in `staging/`:

| file | role |
|---|---|
| `pos_sphere_ml` (target, `staging/pos_sphere_ml.cc`) | one run: FAS (two or more levels) or single level (one level); writes `summary.csv`, `iter.csv` to `--out` |
| `pos_sphere_ml.h` | shared by `pos_sphere_ml` and `main_coarse_lumping`: positive-sphere options, transfer prototypes, level hierarchy |
| `main_coarse_lumping` (target, `staging/main_coarse_lumping.cc`) | one run with the options of `pos_sphere_ml` (no `--out`) and the outputs of `main_coarse`: finest-level table on stdout, coarse tables and cycle log on stderr, `--output-bin`, `--output-vtk`; not used by the stages |
| `pos_sphere_ml.sh` | the runs of one stage (`STAGE`, `CONFIGS`, `BLOCKS`, `SMOKE`), on `pos_sphere_lib.sh` |
| `pos_sphere_cpu.sh` | repeats of a stage into `rep<k>/`, `WAIT_LOAD` |
| `main_coarse_summary.py` | one unit-sphere run of `main_coarse`, written like `pos_sphere_ml` (the reference block) |
| `ml_stages.py` | selection: `noise`, `screen`, `next`, `decide`, `stagec`, `selftest` |
| `pos_sphere_ml_stageB.sh` | the rounds of stage B |
| `profiles_pos_sphere_ml.py` | performance profiles and tables |

## Setup

- Release build of `pos_sphere_ml` and `main_coarse`; pass the build directory as an **absolute** path (the commands
  in `commands.txt` use it as given).
- One deal.II thread (`pos_sphere_cpu.sh` exports `DEAL_II_NUM_THREADS=1`), at most 4 jobs on this machine (5
  performance cores), `WAIT_LOAD=25` (waits while the 1-minute load average is above 25; see `pos_sphere_cpu.md`).
- Python 3 with numpy, pandas, matplotlib (scipy optional: Wilcoxon p-values).
- Output outside the repositories, e.g. `O=~/runs/ml_v1`, `B=$PWD/build/mac-release`.

## Configurations

A configuration id is `<smoother>[__<factor>=<value>...]`, values in the order `pt, cg, cr, cmi, inner, kappa, ctrl`:

| smoother | options |
|---|---|
| `ea-armijo-floor` | `--smoother floor` (energy-adaptive, Armijo, floored density retraction with cap A) |
| `ea-armijo-clip` | `--smoother clip-e-cap-up` (clipped EA direction, e-retraction with growth cap) |
| `ea-exact-floor` | `--smoother floor --ls-exact` |
| `mass-exact-floor` | `--smoother floor --metric-smooth mass --ls-exact` |

| factor value | options |
|---|---|
| `pt=log`, `pt=restriction` | `--point-transfer log-threshold` / `restriction` (restriction-based transfer, density full weighting; prototype) |
| `cg=<g>` | `--coarse-grad <g>` (`model-shift`, `fine`, `mass`, `frobenius`; baseline `model`) |
| `cr=plain` | `--coarse-retraction plain` |
| `cmi=<n>` | `--coarse-max-iter <n>` (baseline 4) |
| `inner=default` | the inner-solver defaults instead of `--tol-inner-res 1e-6 --tol-inner 1e-12 --max-inner 5000` |
| `kappa=<k>` | `--kappa <k>` (baseline 0.8) |
| `ctrl=ratio` | `--correction-control ratio` (diagnostic: decrease ratio per correction) |

Sizes: 4225 DoFs (Q1 fine level 7, P1 fine level 6) and 16641 (Q1 8, P1 7); `n` levels are the `n` finest down from
the fine level. Level counts are compared only at the same size. Run directories:
`<stage out>/rep<k>/<config>/<disc>-<levels>/b<beta>-<potential>-r<radius>`, merged into `rep<k>/runs.csv`.

## Running the stages

```sh
cd step-1
# 0. checks: dry run (counts), selftest, smoke run of a stage (2 problems)
DRY_RUN=1 STAGE=0 staging/pos_sphere_ml.sh $B $O/dry0          # 1152 runs
python3 staging/ml_stages.py selftest
SMOKE=1 STAGE=0 staging/pos_sphere_cpu.sh staging/pos_sphere_ml.sh $B $O/smoke0 1 4

# Stage 0: baseline (FAS 2-4 levels x 4 smoothers, single level, size block, unit sphere), 3 repeats
STAGE=0 WAIT_LOAD=25 staging/pos_sphere_cpu.sh staging/pos_sphere_ml.sh $B $O/stage0 3 4
python3 staging/ml_stages.py noise $O/stage0 $O/state.json      # run-to-run spread -> delta

# Stage A: one factor value at a time from the baseline, 2 and 3 levels, 1 repeat
STAGE=A WAIT_LOAD=25 staging/pos_sphere_cpu.sh staging/pos_sphere_ml.sh $B $O/stageA 1 4
python3 staging/ml_stages.py screen $O/stage0 $O/stageA $O/state.json   # screen.csv, promoted values

# Stage B: greedy combination per smoother (rounds in $O/B/round<r>, traces $O/trace_<smoother>.csv)
WAIT_LOAD=25 staging/pos_sphere_ml_stageB.sh $B $O

# Stage C: the tuned configurations, 3 repeats; profiles
python3 staging/ml_stages.py stagec $O/state.json $O/stageC.txt
STAGE=C CONFIGS=$O/stageC.txt WAIT_LOAD=25 staging/pos_sphere_cpu.sh staging/pos_sphere_ml.sh $B $O/stageC 3 4
python3 staging/profiles_pos_sphere_ml.py $O/profiles $O/stage0 $O/stageC $O/state.json
```

Custom runs: `STAGE=B CONFIGS=<file>` runs any list of configurations, one per line
`<config id> <level counts> [<size>]`, e.g. `ea-armijo-floor__pt=restriction 2,3` or `ea-armijo-floor 1` (single level);
`BLOCKS="fas sl"` restricts stage 0 to some of its blocks (`fas sl size unit`). A single run by hand:
`build/mac-release/pos_sphere_ml --mass-lumping --dimension 2 --max-iter 1000 --tol-residual 1e-9 --line-search
--ls-max-iter 40 --ls-beta 0.5 --ls-min 1e-12 --ls-alpha 1 --tol-inner-res 1e-6 --tol-inner 1e-12 --max-inner 5000
--mesh quadrilateral --degree 1 --beta 1000 --potential square --radius 6 --multilevel 5,6,7 --out /tmp/run`.

## Outputs

- `summary.csv` per run (merged: `runs.csv`, first column `run`): configuration, problem, `status` (`converged`,
  `stalled`, `max_iter`, `stopped`, `error`), iterations, residual, energy, minimum, `cpu` (solve only, median over
  the repeats in the analysis), `wall`, and for FAS the corrections of the finest level and of the middle levels
  (`attempted`, `accepted`, `rejected`, `fallback`, `failed`, `skipped`), `coarse_iter`, and with `ctrl=ratio` the
  decrease ratios (`rho_n`, `rho_nan`, `rho_min`, `rho_median`).
- `iter.csv` per run: every row of every level.
- `state.json`, `screen.csv`, `trace_<smoother>.csv` (selection), `profiles/` (PDFs, `tables.csv`, `tables.md`).

## Pitfalls

- Do not edit `pos_sphere_ml.sh` or `pos_sphere_lib.sh` while a stage runs: bash reads scripts incrementally,
  the running script then fails at its end (the runs finish, `merge_tree` does not run; merge by hand with
  `bash -c 'source staging/pos_sphere_lib.sh; merge_tree <stage out>/rep<k>'`).
- A stage output directory is replaced on the next run of the same stage (only if it contains `commands.txt`).
- Counts come from `CorrectionStats`, never from stderr: the coarse Armijo steps print `not a descent direction`
  too.
- Iterations must agree over the repeats (deterministic); `ml_stages.py` warns otherwise.
- `pt=restriction` needs a nonnegative interpolation (Q1, P1) and lumped masses; squares underflow
  below about 1e-154 (not reached with the floor 1e-12).
