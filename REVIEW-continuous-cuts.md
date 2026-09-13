# Review: `main` since 154f2a0 and the continuous-cuts port (branch `continuous-cuts`, 47844ec)

Date: 2026-09-13. Scope: the 27 commits on `main` after 154f2a0 (up to 5aa4076) and the
15 `cc:` commits on `continuous-cuts` (3ef4b4b..47844ec), reviewed against the paper
(arXiv:2607.09517, §3, §6.3, §6.4) and its reference implementation
`RMO-continuous-cuts` (python package `bernoulli_multilevel`).

Every number below was measured on this machine from the committed sources; the
commands are in §5. Nothing in `src/` or `include/` is changed by this branch; the only
additions are this file and `test/cc/compare_reference.py`.

## 0. Summary

1. **The port's "no speedup" result is caused by two lines in `src/main_cc_coarse.cc`,
   not by the vector transport.** With `options_coarse.ls.alpha = 0.01` (line 104) and
   `options_fas.coarse_every = 1` (line 117) the driver stalls at E = −94.01 while
   single-level RGD reaches −112.23. Restoring the reference's settings (Armijo from
   α = 1 on the coarse level, no two consecutive coarse steps) gives E = −109.62 after
   **one** fine iteration; single-level RGD needs 16 iterations to get there. The Python
   reference on the same synthetic image does the same thing (E = −108.5 after one
   iteration). `doc/plan_continuous_cuts.tex` §"main_cc_coarse.cc" attributes the failure
   to the DC gain of R = 4F and to gradient concentration at the interface; neither
   experiment in §4 supports that.
2. **The verification strategy could not have found this.** Golden values check
   primitives (`exp`, `lifting`, one objective evaluation, stencils). Nothing compares the
   *algorithm* (trigger → coarse solve → prolongation → line search) with the reference on
   the same problem. `test/cc/compare_reference.py` (this branch) does that; running it took
   five seconds and shows the reference accelerating on exactly the synthetic problem the
   tex declares hopeless.
3. **The port cannot run the paper's experiment.** Grid transfer requires
   `fine = 2·coarse − 1` per dimension; the paper's image is 960×1280 → 240×320
   (two pooling steps, even sizes). No operator composition, no image input, no Options
   2/3/5, no 3- or 4-level hierarchy, no L-BFGS-B reference energy. The tex's Table 1
   "every module … implemented and tested" is true only of the tex's own table.
4. **Fidelity issues that do not change the speedup but do change the numbers**: the
   ε convention (paper eq. (42) uses ε², the reference code uses ε unsquared; the paper's
   figures were produced with the code), the μ gate in eq. (16) implemented three
   different ways (paper / reference / port), coarse ρ built by a boundary-shrinking
   full-weighting instead of the reference's block average.
5. **`main` since 154f2a0**: the two argument-order bugs are fixed; the port of the
   `--potential-expr`, ARPACK and king/min_deg features duplicates the fork's branches
   (which are now unmergeable); `test_arpack` still aborts in Debug on the
   `add_component` zero-weight assert; `main` carries a commit titled "WIP".
6. Answer to "why rewrite working Python in C++": the tex's stated goal is to show the
   `rmo::ropt` framework is problem-independent by hosting the paper's second problem.
   That goal is legitimate. The work order was wrong: port first, compare with the
   reference never. The result is 3 000 lines of C++ plus a 684-line design note whose
   central conclusion is refuted by a two-line parameter change.

## 1. Environment caveat

`deal-ii 9.7.1-2` on this machine was built against boost 1.91; `boost-libs` is now
1.92 (installed 2026-08-20). `libdeal_II.so` has `NEEDED libboost_iostreams.so.1.91.0`
(not found), `/usr/lib/cmake/deal.II/deal.IITargets.cmake` links the 1.91 paths, and
`deal.II/base/config.h:695` static-asserts `BOOST_VERSION == 1.91`. Nothing linked against
deal.II runs, including the installed `gpe-coarse`. **`deal-ii` (and `gpe-multilevel`)
need a rebuild against boost 1.92.**

For this review the build used a scratch shim (symlinks `libboost_*.so.1.91.0 →
1.92.0`, a copied `config.h` with the minor version patched, `LD_LIBRARY_PATH` to the
shim). Boost ABI is not guaranteed across minor versions; the `cc` module touches
only `Vector`/`SparseMatrix`/`Timer`, the GP tests touch FE code. All 10 ctest entries
pass under the shim (Release). Treat GP-side timings as unreliable; the `cc` numbers
are deterministic given the problem and were cross-checked against the Python reference
to 6 digits at iteration 1 (§4.1).

## 2. `main` 154f2a0..5aa4076 (27 commits)

| Commit | Content | Assessment |
|---|---|---|
| ef79df6, 4de68e3 | `apply_metric(u, grad_tilt)`, `update_parameters(w_proj, phi)` | Same two fixes as fork branch `fix/coarse-residual-tilt`. Correct. Still no test exercising `GrossPitaevskiiCoarseResidual` with a non-zero tilt. |
| 226a906 `--potential-expr`, 83de6f0 ARPACK test, cfae1be king/min_deg, 7db3583 gradient test | Feature-for-feature the fork's `feat/dealii-features` and `test/real-assertions` | Fork branches `fix/*`, `test/*`, `feat/*` are now duplicates on moved paths (`include/gpe/…` → `include/rmo/{gpe,fe,ropt}`); delete them. `build/packaging` and `docs/architecture` would need a rebase; `README.md` line 3 still links a non-existent `ARCHITECTURE.md` (pre-existing since 8d50026). |
| `include/rmo/lac.h:97` | `Assert(weight != 0)` in `LinearCombination::add_component` kept | `test_arpack` (β = 0) aborts in every Debug build: `weight must be non-zero … Abort() was called` (reproduced). The tex lists it under "pre-existing test failures" and leaves it. The fork's version skips zero-weight components. |
| `CMakeLists.txt:115` | `add_test(descent test_descent COMMAND test_descent)` | Legacy signature: runs `test_descent` with the literal arguments `COMMAND test_descent`. Harmless because `main()` ignores `argv`; wrong. |
| `include/rmo/ropt/descent.h:90-93` | Armijo accepts a step that **failed** sufficient decrease once `alpha < ls.min`, and returns `ls.min` instead of the α it applied | `solver.h:54` then prints "Step rejected by line search" although `x` was overwritten. With CLI defaults (α 1, β 0.6, min 0.1) this fires on the sixth backtrack. Pre-existing; now load-bearing for FAS (§4.2: 2 464 such messages in one run). |
| `include/rmo/ropt/fas.h:106` | `std::cerr << "level: "` unconditional | 301 lines of noise per 300-iteration run. Also `goto fine_step` into an `else` block (legal, unreadable) and `x_hist` copying every fine iterate. |
| 5aa4076 | "WIP: refactor for additional problems", 2007+/1986− | A commit titled WIP on `main`. |
| `include/rmo/util/random.h:12` | `std::random_device` seed, no override | Every test draws different points on every run; a failure cannot be reproduced. |
| 3ef4b4b `test_coherence.cc` | First real two-mesh coarse-oracle test (Prop. 3.2 on GP, mass metric) | Good. Mass metric only; Frobenius and energy-adaptive coarse oracles remain untested on a real transfer chain. |
| 654b6d1 observer pattern, f4dd26e `FullApproximationScheme<Functional>`, c421fcc namespaces, 26ec434 transfer split | Refactors | Reasonable structure. The `TiltOracle`/`CoarseOracle` concepts require a `SolverOptions` constructor argument that two of the five oracles ignore (§3.6). |

ctest on this machine (Release, shim): 10/10 pass. The tex reports `operator` failing on
`distribute_mg_dofs()`; it passes with deal.II 9.7.1 here. `arpack` passes in Release
(the assert is compiled out) and aborts in Debug.

## 3. The continuous-cuts port

### 3.1 What is correct

Stated first so the rest is not read as "everything is wrong".

- `cc/metric.h`, `cc/manifold.h`: Table 9 formulas, stable `sigmoid`/`logit`,
  retraction/lifting inverse pair, differentials. Verified against `manifold.py` to 1e-9.
- `cc/oracle_coarse.h`: q(z), Dq(z)[v] and the Riemannian gradient
  `diag(z(1−z))·g − w` are right; the derivation that the metric adjoint of DL_ψ(z) is the
  identity is right (⟨DL[u], w⟩_ψ = Σ u_i w_i /(z_i(1−z_i)) = ⟨u, w⟩_z). The Prop. 3.2
  coherence test is the right test and it is what caught the earlier `diag(z(1−z))(g − w)`
  bug.
- `cc/interpolate.h`, `cc/transport.h`: injection, bilinear, `Tfine` = B^T, and the
  Option 1/4 (R, P) equivalence all match `primitives.py`/`operators.py` coefficient for
  coefficient (I re-derived the corner value 6.75 by hand).
- `cc/cc.h`: ambient gradient `ρ + α D^T diag(ω) Dφ` matches torch autograd to 1e-6.
- Single-level RGD (`main_cc.cc`) reproduces the reference's single-level trajectory to
  six digits at iteration 1 (130.192658 both) and to 5e-7 relative at iteration 300
  under the same ε convention (§4.1). The fine-level port is faithful.

### 3.2 The two lines that produce "no speedup" — severity: high

`src/main_cc_coarse.cc`:

```cpp
options_coarse.ls.alpha  = 0.01;  // line 104
options_fas.coarse_every = 1;     // line 117
```

- **α = 0.01.** Reference `optimizer.py:8` starts Armijo at α = 1 on every level and
  never damps. With α = 0.01 and backtracking only, the 10 coarse iterations take ten
  steps of exactly 0.01 (coarse table in the run's stderr: step column `1.0000e-02` ×10,
  energy 45.85 → 28.30, residual not decreasing). The coarse model is not minimised, the
  prolonged correction is tiny, and the fine-level Armijo on it backtracks to
  `ls.min = 1e-12` — 2 464 "Step rejected by line search" lines in the baseline run
  (`descent.h:90` then applies the 1e-12 step anyway, §2).
- **coarse_every = 1.** Paper §6.4: "consecutive coarse corrections are not permitted –
  every coarse correction is preceded by a fine-level gradient step (cf. (17))".
  Reference `multilevel.py:148` enforces it with `lockouts`. With `coarse_every = 1` the
  fine level never takes a gradient step once the trigger fires, so a rejected coarse
  step is followed by another rejected coarse step, forever: 300/300 rows flagged `*`,
  E frozen at −94.01 from iteration 50 on, residual frozen at 2.492.
- The tex justifies α = 0.01 by "the coarse model is unbounded below as z → boundary".
  True (q has the term −wᵀ logit z), but the reference lives with it, clamping iterates
  at 1e-7 (`optimizer.py:15,49`); the port already clamps at `MIN_WEIGHT = 1e-12`
  (`cc/metric.h:29`). With α = 1 nothing ran to the boundary in any run of §4.

Measured (§4.2): α = 1 alone → E = −109.62 after iteration 1 (then stalls because of
`coarse_every = 1`); `coarse_every = 2` alone → no stall but coarse steps nearly useless
(E = 164.63 after iteration 1, 15 coarse steps, slower than single-level);
both → E = −109.62 after iteration 1, −111.95 after 10 (single-level: −94.33 after 10,
reaches −109.62 at iteration 16 and −111.95 at iteration 23), identical final energy,
3 coarse corrections. That is the reference's behaviour.

The tex's 60 lines on why R = 4F's DC gain drives the ratio ‖Rg‖/‖g‖ from 2 to 220
describe the stalled state: `x` does not move after iteration ~50, the gradient is
whatever the stall left, and the ratio is measured at a point the algorithm should never
have stayed at. In the fixed runs the ratio is ~1e5 at iteration 3 (‖Rg‖ evaluated in the
Fisher–Rao metric at ψ next to the boundary) and the trigger fires every time it is
evaluated — as in the reference, where `c_check` also fires every time and only the μ
gate (§3.4) stops corrections after iteration 1.

### 3.3 Parameters do not reproduce the paper's experiment — severity: medium

- **ε.** Paper eq. (42): `sqrt((D₁φ)² + (D₂φ)² + ε²)`. Reference `objective.py:14`:
  `sqrt(gx² + gy² + eps)` — unsquared. §6.3.4: "All numerical experiments for the binary
  continuous cuts problem were carried out in Python", so Figures 13–15 were produced with
  effective ε_h = √1e-4 = 1e-2 and ε_H = √1e-3 ≈ 3.2e-2. The port uses the paper's formula
  with the paper's numbers (`main_cc.cc:30`, `main_cc_coarse.cc:79-80`): smoothing 100×
  sharper than the paper's runs. The tex noticed the mismatch for the golden test only
  (`test_oracle.cc:186`, passes √eps) and did not carry it to the drivers.
  Measured effect (§4.2, `epsref` rows): changes the minimum (−110.63 vs −112.23) and
  nothing about the speedup. The reference behaves the same in both conventions (§4.1).
  Worth fixing for fidelity, not the cause of anything.
- **μ gate in eq. (16)**: `‖R g‖_ψ ≥ max(η ‖g‖_φ, μ)` with η = 0.6, μ = 0.5 for
  Options 1/4/5 (Fig. 15 caption). Three implementations:
  paper — μ bounds the *coarse* norm; reference `multilevel.py:104-106` —
  `gnorm_c_threshold = 0.5` on the *Euclidean* norm of the *fine* Riemannian gradient;
  port `ropt/condition.h:37` / `cc/condition.h:29` — fine *Fisher–Rao* norm > `eps`.
  Measured: paper's form gives 4 coarse steps instead of 3, final energies identical.
- **`grid_scale = 2`** (`cc/condition.h`, `main_cc_coarse.cc:124`) is copied from
  `operators.py:166-197`. It is not in the paper; the tex presents it as the reference's
  "compensation". It is a reference-side deviation and should be labelled as one.
- **Coarse ρ.** Reference `problem.py:77`: `avg_pool2d(kernel 2, stride 2, ceil_mode)` —
  a block mean that never shrinks at the border. Port `main_cc_coarse.cc:76`:
  `0.25·Tfine(ρ)`, stencil sums 0.5625 at corners, 0.75 on edges, 1 inside (checked
  numerically): the coarse data term is biased towards 0 along the whole border. Measured
  with the reference (`--rho-coarse fullweight`): negligible on this image (first coarse
  step −106.94 vs −106.96). Still wrong in principle; the same border shrink sits in the
  FULL_WEIGHTING point restriction of both codes (`restrict_fw_normalized`, kernel/16 with
  zero padding).

### 3.4 The port cannot run the paper's problem — severity: high for the stated purpose

- `cc/interpolate.h:32-33` asserts `fine = 2·coarse − 1` per dimension. The reference's
  conv2d kernels take any size (checked: 41→21, 42→21, 960→480). The paper's image is
  960×1280 and the coarse level is 240×320: **two** pooling steps
  (`compare_variants.py:69`, `n_pools = 2`), composed by `operators.py:117
  make_composed_ops`. The port has no composition, no even sizes, no image I/O
  (`problem.py`), no 3/4-level hierarchy (`levels_2_3_4.py`), no Options 2/3/5
  (`linear_solvers.py` for Option 3), no L-BFGS-B reference energy (`ground_truth.py`)
  for the relative-error plots of Figures 13–14.
- Consequently the only problem the port can run is a 41×41 synthetic disk, where the
  multilevel benefit is a few iterations and CPU time is irrelevant. The tex's
  "per-problem tuning beyond this module's scope" is not what is missing.

### 3.5 Verification method — severity: medium

- Golden values compare primitives. The algorithm was never compared with the reference
  on the same input. `test/cc/compare_reference.py` does it in five seconds (§4.1). Had it
  existed, §3.2 would have been a five-minute diff of two energy columns.
- `test/cc/derive_golden_values.py` does not run without scikit-image
  (`ModuleNotFoundError: No module named 'skimage'`, reproduced), because
  `bernoulli_multilevel/__init__.py` imports `problem.py`. The script's docstring claims
  the opposite. The golden constants are therefore not regenerable as documented;
  `compare_reference.py` installs a stub module for `skimage` instead.
- The tex claims `main_cc.cc` "converges in 147 iterations (residual 1e-6)". As committed
  it runs all 300 iterations; the residual plateaus at 1.446e-5 from iteration ~146
  (`tol_residual = 1e-6` never reached). The energy statement (182.7 → −112.2) is right.
- Tests draw unseeded random points (`util/random.h:12`) from `unifrnd(0.2, 0.8)`, so the
  boundary regime — where `MIN_WEIGHT = 1e-12` differs from the reference's `1e-8`
  (`manifold.py:10`) and `1e-7` (`optimizer.py:15,49`), and where retraction and lifting
  stop being inverses — is never exercised.

### 3.6 Framework friction — severity: low, but it is where the time went

The tex documents three "framework assumption" bugs (retraction hitting the boundary,
two oracle instances losing each other's `update`, coarse model running to the boundary)
and fixes each locally. The pattern is the finding:

- `FullApproximationScheme::cycle` (`fas.h:182-196`) constructs two oracles over one
  functional and relies on shared mutable state on the functional. `cc` had to move its
  base point into `ContinuousCutsFunctional::get_phi()` to satisfy that. The `OracleBase`
  comment (`oracle_base.h:33-39`) now states the requirement; the requirement is the
  problem.
- `ContinuousCutsCoarseOracle::gradient` (`oracle_coarse.h:109-122`) recovers the ambient
  gradient by multiplying by `weight(z_arg)` (`FisherRaoOracle::gradient`) and dividing by
  `weight(phi_cached)` (`apply_metric`) — correct only when the two coincide; three
  elementwise passes to avoid one accessor call, justified in the tex as "no downcast
  needed".
- Dead `SolverOptions` constructor parameters (`oracle.h:70`, `oracle_coarse.h:83`) to
  satisfy `TiltOracle`/`CoarseOracle`; `PixelGrid::to_dof_order` kept as the identity "for
  callers" (`grid.h:53-64`); `FISHER_RAO` in `MetricKind` but not parseable
  (`option.h:33-37`); `BernoulliManifold::retract_inv_diff_adjoint` implements the
  Euclidean adjoint that the tex itself says is the wrong object for this manifold
  (`manifold.h:148-155`) — the interface invites the next user to call it.
- `armijo_line_search` accepting failed steps below `ls.min` (§2) is what turned the
  α = 0.01 choice into a hard stall instead of a visible "line search failed".

## 4. Measurements

Problem in all rows: 41×41 synthetic disk (`cc/synthetic.h`), coarse 21×21, α = 0.1/0.4,
uniform φ₀ = 0.5, 300 fine iterations, Option 1 (injection + geometric bilinear), η = 0.6,
10 coarse iterations per correction. "paper ε" = ε² under the root with ε = 1e-4/1e-3
(port default); "ref ε" = ε unsquared (what the reference and hence the paper's figures
use).

### 4.1 Python reference on the port's problem (`test/cc/compare_reference.py`)

| run | E₀ | E after it 1 (ML) | coarse steps | ML reaches SL(300) energy at it | SL(300) |
|---|---|---|---|---|---|
| ref ε, avg-pool ρ (defaults) | 184.381 | **−106.96** | 1 | 87 | −110.631127 |
| paper ε (`--eps-mode paper`) | 182.717 | **−108.55** | 1 | 76 | −112.225363 |
| ref ε, `--gnorm-threshold 0` | 184.381 | −106.96 | 150 | 129 | −110.631127 |
| ref ε, `--rho-coarse fullweight` | 184.381 | −106.94 | 1 | 87 | −110.631127 |

Single-level after 12 iterations: −102.03; multilevel after 1: −106.96. On this problem
the reference makes exactly one coarse correction (the Euclidean gradient norm drops to
0.42 < μ = 0.5 after it) and that correction is worth more than 12 single-level
iterations.
Without the μ gate it corrects every other iteration, converges *slower* and costs 10×
the CPU: the gate matters, the transport does not.

### 4.2 C++ port, parameter isolation (temporary edits of the two drivers, sources restored)

| variant (`src/main_cc_coarse.cc`) | E after it 1 | E after it 10 | E(300) | res(300) | coarse steps | ML ≤ SL(300) at it |
|---|---|---|---|---|---|---|
| committed (α_c = 0.01, every = 1, paper ε) | 164.63 | 17.81 | **−94.011** | 2.49 | 300 | never |
| `coarse_every = 2` | 164.63 | −60.10 | −112.225 | 1.4e-5 | 15 | 164 |
| `ls.alpha = 1.0` (coarse) | **−109.62** | −110.31 | −110.312 | 0.71 | 300 | never |
| both | **−109.62** | −111.95 | −112.225 | 1.4e-5 | 3 | 149 |
| both + ref ε | −108.02 | −110.38 | −110.631 | 1.4e-5 | 3 | 160 |
| ref ε only | 165.45 | 18.61 | −94.474 | 2.35 | 300 | never |
| both + μ on coarse norm (eq. 16) | −109.62 | −111.87 | −112.225 | 1.4e-5 | 4 | 150 |

Single-level (`main_cc`, paper ε): it 1 130.19, it 10 −94.33, it 20 −111.64, E(300)
−112.225422, residual plateau 1.446e-5 from it ~146. Python single-level, paper ε:
it 1 130.19, E(300) −112.225363.

Reading: rows 1/3/6 stall (every fine step is a coarse step, most of them rejected by
Armijo and applied at 1e-12 anyway); rows 2/4/5/7 converge; only rows 4/5/7 — the
reference's settings — show the reference's one-iteration jump. Neither ε (row 6 vs 1,
row 5 vs 4) nor the μ form (row 7 vs 4) matters for the qualitative behaviour.

### 4.3 Tests

Release, boost shim, `OMP_NUM_THREADS=1`: `arpack cc_interpolate cc_manifold cc_oracle
cc_oracle_coarse coherence descent gradient matrix_free operator` — 10/10 pass, 4.9 s.
Debug `test_arpack`: `weight must be non-zero`, `Abort() was called`.

## 5. Reproduction

```sh
# build (after rebuilding deal-ii against the installed boost; otherwise see §1)
git submodule update --init fmt
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
(cd build && OMP_NUM_THREADS=1 ctest)

# baseline drivers (fine-level table on stdout, coarse cycles + "Step rejected" on stderr)
(cd build && ./main_cc > sl.org 2> sl.err && ./main_cc_coarse > ml.org 2> ml.err)
grep -c "Step rejected" ml.err          # 2464 in the committed state

# variants: edit src/main_cc_coarse.cc lines 104 (ls.alpha = 1.0) and 117 (coarse_every = 2),
# rebuild main_cc_coarse, rerun; for "ref eps" also lines 79-80 (3.1623e-2 / 1e-2) and
# src/main_cc.cc line 30 (1e-2)

# reference on the same problem (torch only; scikit-image is stubbed)
python3 test/cc/compare_reference.py ../RMO-continuous-cuts --every 50
python3 test/cc/compare_reference.py ../RMO-continuous-cuts --every 50 --eps-mode paper
python3 test/cc/compare_reference.py ../RMO-continuous-cuts --every 1 --iters 12
```

## 6. Recommendations, in order

1. `src/main_cc_coarse.cc`: `options_coarse.ls.alpha = 1.0`, `options_fas.coarse_every = 2`.
   Delete the paragraph in the tex that explains the stall by R's DC gain; replace with
   the table in §4.2.
2. Keep `test/cc/compare_reference.py` next to `derive_golden_values.py` and run both
   whenever `cc/` or the reference changes. Fix the docstring of `derive_golden_values.py`
   or stub `skimage` there too.
3. Decide the ε convention explicitly (`ε_paper = √ε_ref`) and use it in both drivers;
   record it in the tex's parameter table.
4. Implement eq. (16) as written (μ bounds the coarse norm) or document which of the three
   variants is meant; drop `grid_scale` or label it a reference-side deviation.
5. `armijo_line_search`: return 0 when no step satisfied the condition, never apply a
   sub-`ls.min` step; make "Step rejected" mean rejected.
6. Generalise `BernoulliGridTransfer` to even sizes and compose transfers for
   `n_pools > 1`; add image input; then run the 960×1280 cow. Until that exists the
   module does not reproduce §6.3 of the paper and the README should say so.
7. `util/random.h`: seed from `GPE_TEST_SEED` (or similar) and print the seed.
8. `lac.h`: skip zero-weight components in `add_component` so `test_arpack` runs in Debug;
   remove the `level:` print in `fas.h`; fix `add_test(descent …)`.
9. Fork housekeeping: delete `fix/coarse-residual-tilt`, `test/real-assertions`,
   `feat/dealii-features` on `pepopepopepo/RMO-gross-pitaevskii` (superseded upstream);
   rebase `build/packaging` and `docs/architecture` onto current `main` if still wanted.

## 7. CPU-time comparison, single- vs. two-level (addendum, 2026-09-13)

Recommendation 1's fix (§3.2) restores the reference's *iteration-count* behaviour: one
fine iteration with a coarse correction reaches an energy single-level needs 16 iterations
for. That is a claim about iterations, not CPU time -- a coarse correction also runs up to
`options_coarse.max_iter` coarse-level Armijo iterations on a quarter-size grid plus two
grid transfers, which is not free (the paper/reference value is 10 per call, Sec. 6.3.4;
`main_cc_coarse.cc` now uses 5, see §7.1). `src/main_cc_bench.cc` (new) measures CPU time
directly: both drivers already report
cumulative CPU time per iterate (`CycleInfo::elapsed`, a `dealii::Timer` restarted once per
run and read with `cpu_time()`, not paused during a coarse correction); the benchmark
records that history for both solvers at four problem sizes (41, 81, 161, 321 fine pixels
per side, each one exact coarse/fine doubling apart as `interpolate.h` requires) and
reports, for two convergence targets, the CPU time each solver needs to first reach it.

Methodology: 3 repeats per (size, solver), fastest kept; single-threaded
(`DEAL_II_NUM_THREADS=1` -- deal.II's own thread pool, not OpenMP, parallelizes
`Vector`/`SparseMatrix` operations above an internal size threshold, which otherwise
inflates `cpu_time()` non-deterministically by summing time across worker threads: an
early run without it measured 640% CPU utilization and gave inconsistent numbers). Two
targets: "90% converged" (energy within 10% of single-level's own 300-iteration energy
drop -- a scale-independent milestone reached well before either solver's residual
plateau, §3.5) and "fully converged" (matches single-level's own 300-iteration energy
outright).

Numbers below are with `options_coarse.max_iter = 5` (§7.1); the original run used the
paper/reference's 10 and is kept there for comparison.

| n_fine | n_dofs  | target | SL it | SL cpu (s) | ML it | ML cpu (s) | ML coarse corr. | speedup (SL/ML) |
|---|---|---|---|---|---|---|---|---|
| 41  | 1,681   | 90%  | 9   | 1.29e-3 | 1   | 1.70e-3 | 4 | 0.76 |
| 41  | 1,681   | full | 161 | 2.19e-2 | 161 | 2.73e-2 | 4 | 0.80 |
| 81  | 6,561   | 90%  | 9   | 4.80e-3 | 1   | 6.18e-3 | 2 | 0.78 |
| 81  | 6,561   | full | 161 | 8.13e-2 | 158 | 1.07e-1 | 2 | 0.76 |
| 161 | 25,921  | 90%  | 9   | 1.98e-2 | 1   | 2.38e-2 | 2 | 0.83 |
| 161 | 25,921  | full | 161 | 3.35e-1 | 157 | 4.38e-1 | 2 | 0.76 |
| 321 | 103,041 | 90%  | 9   | 8.51e-2 | 1   | 9.11e-2 | 5 | 0.93 |
| 321 | 103,041 | full | 161 | 1.42e0  | 161 | 2.26e0  | 5 | 0.63 |

Reading: at every size and both targets, single-level reaches the same energy in *less*
CPU time than the two-level driver, despite needing between 9x (90% target) and
effectively the same number (full target -- both solvers plateau around iteration
~158-161, the residual floor from §3.5) of outer iterations. The one-iteration jump
recommendation 1 restored is real -- one fine iteration with a correction reaches the 90%
target that single-level needs 9 iterations for -- but that one iteration is not cheap: it
runs the coarse solver for up to `options_coarse.max_iter` Armijo iterations on a grid a
quarter the size, plus two grid transfers, which on this problem costs about as much as the
9-16 fine iterations it replaces. At the full-convergence target the picture is worse, not
better: both solvers still need essentially the same ~160 fine-level steps to grind the
residual down to its plateau, so the coarse corrections taken along the way (2-5, depending
on size) are pure overhead once the trigger's μ gate is satisfied and no further correction
fires.

This does not contradict recommendation 1 -- restoring the reference's parameters was
still necessary to make the port behave like the reference algorithmically (§3.2), and the
iteration-count claim (Fig. 15's motivation) is now reproduced correctly. But it does mean
the port's two-level driver has not demonstrated a *CPU-time* speedup on any synthetic
problem size tested here, up to 321x321. Two explanations, not mutually exclusive: (a) the
fine-level solver is cheap per iteration on this problem (a forward difference and an
elementwise divide, no linear solve), so there is little slow, expensive work for a coarse
correction to substitute for -- unlike a linear-solver-based multigrid method, where a
fine-level smoothing step is the expensive part and a coarse solve is comparatively cheap;
(b) the coarse solve's own cost (fixed iterations, independent of problem size, at a
quarter the resolution) does not shrink relative to a fine iteration's cost as the problem
grows, since both scale the same way with resolution here -- consistent with the roughly
constant cost ratio (90% target) across all four sizes tested, both at 10 and at 5 coarse
iterations (§7.1). Reaching the paper's own 960x1280 image (§3.4) would test this at a size
375x larger than the biggest one here; extrapolating the current per-size trend gives no
reason to expect the ratio to improve from problem size alone.

### 7.1 Halving the coarse iteration budget (10 -> 5)

Since the coarse solve's fixed iteration count was one of the two suspected causes of the
CPU-time gap above, `main_cc_coarse.cc` and `main_cc_bench.cc` were changed from the
paper/reference's `options_coarse.max_iter = 10` to `5`, to see whether that closes it.

| n_fine | target | speedup at max_iter=10 | speedup at max_iter=5 |
|---|---|---|---|
| 41  | 90%  | 0.74 | 0.76 |
| 41  | full | 0.74 | 0.80 |
| 81  | 90%  | 0.80 | 0.78 |
| 81  | full | 0.79 | 0.76 |
| 161 | 90%  | 0.86 | 0.83 |
| 161 | full | 0.76 | 0.76 |
| 321 | 90%  | 0.88 | 0.93 |
| 321 | full | 0.55 | 0.63 |

It does not: halving the coarse budget moves each ratio by a few points in either
direction, well within the noise of the two other confounds it shares with iteration count
-- the per-correction overhead outside the coarse Armijo loop itself (grid transfers,
coarse gradient/oracle setup) and the fact that final energy and coarse-correction count
are unchanged (the coarse solves were already converging, or hitting the mu-gated cutoff,
well inside 5 iterations on this problem; the extra 5 iterations at `max_iter=10` were
mostly wasted, not what was making the correction expensive). The dominant cost is the
per-correction fixed overhead and the fine level's own cheapness (reading (a) above), not
the coarse iteration count -- so reducing it further is unlikely to close the gap either;
it would need either a cheaper coarse iteration (impossible while the coarse problem's
per-DOF cost matches the fine level's) or a fine-level problem expensive enough that 9-16
of its iterations cost more than one coarse correction, which this synthetic disk is not.

Reproduction:
```sh
ninja -C build main_cc_bench
DEAL_II_NUM_THREADS=1 ./build/main_cc_bench
```

## 8. The paper's own cow-image problem (addendum, 2026-09-13, recommendation 6)

§3.4 found the port could not run the paper's actual 960x1280 -> 240x320 problem at all:
`BernoulliGridTransfer` required `fine = 2*coarse-1` (one exact mesh-refinement step), while
the reference's own transfer needs two composed factor-2 average-pool steps on an
even-sized grid (`fine = 2*coarse`, no shared boundary point). That gap is closed:

- `BernoulliGridTransfer` generalized to `coarse = ceil(fine/2)` for either parity of
  `fine`, matching the reference's own `r_injection` (`phi[::2,::2]`) exactly; the
  even-size boundary case (a "+1" neighbor one past the coarse grid's last row/column)
  drops that term rather than renormalizing, matching the reference's zero-padded
  `conv2d`. Verified against `primitives.py` on a 2x2 coarse / 4x4 fine grid.
- `ComposedGridTransfer` / `ComposedVectorTransport` (new) chain two single-hop transfers
  through an internal 480x640 grid that is never exposed to `FullApproximationScheme` --
  matching the reference's `operators.make_composed_ops`, including that `P_inj_bil`'s
  metric weighting is applied at *each* intermediate grid, not just the two endpoints
  (`R_inj_bil` ignores its `phi`/`psi` arguments entirely, so composing it is just chained
  `Tfine`). Verified against `make_composed_ops(r_inj_bil, R_inj_bil, P_inj_bil, 2)` on an
  8x8 -> 4x4 -> 2x2 grid, including a case where several fine points get exactly zero from
  the composed boundary truncation.
- `test/cc/prepare_cow_data.py` (new) replicates `problem.py`'s preprocessing (grayscale,
  2x bilinear enlarge, Gaussian blur sigma=1, foreground/background seed-rectangle means,
  two composed average-pool steps for the coarse level) using skimage directly, checked
  against the reference's actual `avg_pool2d` call on the real image (max abs diff 1.2e-7,
  float32-pooling rounding). The C++ side has no JPEG decoder, resizer, or blur of its own;
  those stay delegated to skimage (the code path the paper's own figures were produced
  with) rather than risking a fresh reimplementation silently diverging from it -- what
  runs in C++ is the actual optimization under test, on real data (`src/main_cc_cow.cc`,
  new, loading the two precomputed rasters via `cc::load_raster`, new in `cc/rawio.h`).

### 8.1 Correctness: matches the reference to 6 digits at the checked iterations

`src/main_cc_cow.cc`'s single-level run on the real 960x1280 rho matches
`bernoulli_multilevel.single_level.solve_single_level` on the identical array at every
checked iteration (`alpha=0.1`, eps unsquared/squared conventions equated as in \S3.3):

| it | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| reference | 37965.978081 | 20493.208684 | 5555.309023 | -14429.485188 | -37795.208389 | -52970.249212 |
| this port | 37966 | 20493.2 | 5555.31 | -14429.5 | -37795.2 | -52970.2 |

`E0 = 56983.4` also matches `create_cc_objective`'s value on the same array exactly. This
is the first time this port's fine-level code has been checked against the reference at
the paper's actual resolution rather than the 41x41 synthetic disk; it passes.

### 8.2 CPU time: a real early speedup, and a real late-stage loss

Same methodology as §7 (best CPU time, single-threaded, `DEAL_II_NUM_THREADS=1`), one
problem (this one), 300 fine iterations, coarse solver `max_iter=5` (§7.1), `coarse_every=2`,
`grid_scale=4` (`2^n_pools`, `n_pools=2` -- not the synthetic disk's 2):

| target | SL it | SL cpu (s) | ML it | ML cpu (s) | speedup (SL/ML) |
|---|---|---|---|---|---|
| 90% converged | 19 | 2.39 | 1 | 0.76 | **3.16** |
| fully converged (=SL(300), E=-63143) | 300 | 45.9 | 300 | 145.6 | 0.32 |

(SL final E = -63143; ML final E = -63119, 145 of 150 eligible iterations triggered a
coarse correction.) This is the first target, on any problem size tried in this review,
where the two-level driver is actually faster: reaching 90% of the total energy drop takes
a third of the CPU time single-level needs, matching the paper's own headline claim (Fig.
13's early, large jump). It is also the first case where the *late*-run picture is a clean
loss rather than a wash: by 300 iterations the two-level driver has spent over 3x the
CPU time of single-level and reached a *worse* final energy.

The cause is visible in the trigger norms themselves (first 20 fine iterations, printed by
`main_cc_cow.cc`): `grad_norm` (fine, Fisher-Rao) drops into the tens by iteration 7 while
`grid_scale * grad_restr_norm` (coarse, scaled) stays in the hundreds of thousands through
iteration 20 -- a ratio of ~5000x at iteration 3 and still ~9000x at iteration 20. Eq. (16)'s
trigger, `norm_coarse_scaled >= max(kappa*norm_level, mu)`, is satisfied by a wide margin on
essentially every eligible iteration once this gap opens, regardless of how converged the
fine level already is: 145 of the 150 iterations coarse_every's lockout allows fire a
correction, each one costing a 5-iteration coarse solve on a 240x320 grid plus two composed
grid transfers. On the 41x41 synthetic disk this same ratio was the *original, retracted*
explanation for the (actually parameter-bug-caused) stall in §3.2 -- retracted there because
the measured ratio was an artifact of a frozen, stalled trajectory, not a per-step property
of the restriction operator. Here, at the paper's actual scale, with both parameter bugs
already fixed and the trajectory demonstrably not stalled (§8.1), the same large,
persistent ratio is real and is not an artifact: `mu=0.5` and `grid_scale=4`, tuned by
observation on a 1,681-pixel problem, do not gate anything on a 1,228,800-pixel one. Whether
that is fixable by scaling `mu` with problem size, a larger `grid_scale`, or a different
gate entirely (§3.3's three variants) is future work -- out of scope for a test-and-benchmark
pass -- but the mechanism is now measured, not hypothesized.

Reproduction:
```sh
python3 test/cc/prepare_cow_data.py /path/to/RMO-continuous-cuts   # writes test/cc/data/*.bin
ninja -C build main_cc_cow
DEAL_II_NUM_THREADS=1 ./build/main_cc_cow   # ~3 minutes (SL 46s, ML 146s)
```

## 9. A minimum-fine-gradient-norm floor, and 3-/4-level hierarchies (addendum, 2026-09-13)

\S8.2 found the cow image's coarse-correction trigger fires on 145 of 150 eligible
iterations because the scaled restricted gradient stays 5000-9000x the fine gradient's own
(steadily decaying) norm for the whole run, so eq. (16)'s `max(eta*||g||, mu)` gate never
shuts off on its own. GPE's `DefaultCoarseCondition` (`ropt/condition.h`) has always had an
extra term for exactly this situation -- `norm_level > options.eps`, wired to GPE's `--eps`
CLI flag (`option.h`, default `1e-4`) -- that the paper's own eq. (16) does not include and
that `cc::ScaledCoarseCondition`'s rewrite to implement eq. (16) literally (commit
`77e1531`) accordingly dropped. Re-added as an explicit, opt-in parameter rather than
reusing `options.eps` (already repurposed there as `mu`): `ScaledCoarseCondition(grid_scale,
min_fine_norm = 0.0)`, gating `trigger()` on `norm_level > min_fine_norm` before eq. (16)'s
own comparison. Default `0.0` keeps every existing result in \S3-\S8 unchanged.

### 9.1 Effect on the 2-level cow driver

`main_cc_cow.cc` now constructs its condition with `min_fine_norm = 10.0` -- past the
halfway point of the observed 139 -> 6.5 decay (crossed between iterations 121 and 141 in
\S8.2's ungated run). Same problem, same 300 iterations, single-threaded:

| | corrections | final E | cpu(300) | 90%-converged speedup | fully-converged speedup |
|---|---|---|---|---|---|
| ungated (\S8.2) | 145 | -63119 | 145.6 s | 3.16 | 0.317 |
| `min_fine_norm=10` | 59 | **-63147** | 85.9 s | 3.20 (unaffected) | **0.615** |

Corrections drop by more than half, CPU time for the full run drops by 41%, and -- unlike
\S8.2 -- the two-level driver now reaches a *better* final energy than single-level's
-63143, not a worse one: the corrections the floor removes were not merely wasted CPU, they
were actively net-negative by this point (each one perturbs `phi` away from the fine
level's own descent trajectory; past the point where the coarse model still resolves
something the fine gradient doesn't, that perturbation has nothing to correct). The
90%-converged speedup is exactly unchanged, as expected -- the floor only removes late
corrections, and 90% convergence is reached at iteration 1. The fully-converged gap nearly
halves (0.317 -> 0.615) but does not close: the 59 corrections that still fire before
`norm_level` crosses 10 remain expensive relative to what they buy late in the run.

### 9.2 3- and 4-level hierarchies

`src/main_cc_cow_levels.cc` (new) replicates `examples/levels_2_3_4.py`'s own "best
configurations" (RMO-continuous-cuts) for a 3-level (240x320 / 480x640 / 960x1280) and
4-level (120x160 / 240x320 / 480x640 / 960x1280) hierarchy, both `min_fine_norm=10`, sharing
one single-level run as baseline. Unlike \S8's 2-level driver, every adjacent pair here is a
single, un-composed factor-2 step (`cumulative_to_delta_pools([2,1]) == [1,1]`,
`[3,2,1] == [1,1,1]` in the reference), so this needed no `ComposedGridTransfer` -- plain
`BernoulliGridTransfer` at each transition and `FullApproximationScheme`'s ordinary N-level
recursion, `grid_scale=2` (`2^1`) throughout rather than the 2-level driver's 4 (`2^2`).

| | corrections | final E | cpu(300) | 90%-converged (it, cpu, speedup) | fully-converged (it, cpu, speedup) |
|---|---|---|---|---|---|
| SL (shared baseline) | -- | -63143 | 47.5 s | it=19, 2.48 s | it=300, 47.5 s |
| 2-level, `min_fine_norm=10` (\S9.1) | 59 | -63147 | 85.9 s | it=1, 0.77 s, **3.20** | it=262, 76.2 s, 0.615 |
| 3-level | 31 | -63150 | 142.6 s | it=1, 4.02 s, 0.618 | it=253, 130.8 s, 0.363 |
| 4-level | 35 | -63148 | 118.5 s | it=1, 2.70 s, 0.922 | it=216, 97.0 s, 0.490 |

Reading: going deeper does not help CPU time on this problem, at either target. The
90%-converged speedup -- the two-level driver's headline win -- disappears once a genuine
intermediate level is inserted: single-level is still faster to 90% for both 3- and
4-level (0.62x and 0.92x). The cause is structural, not a tuning accident: a 3+ level
hierarchy's first "1 fine iteration" is not cheap the way the composed 2-level's is,
because `FullApproximationScheme`'s recursion actually *solves* at every intermediate level
in between -- checking eq. (16) again and potentially firing its own nested correction --
rather than treating the whole coarse chain as one geometric jump to a single bottom-level
solve. That per-level solve cost is exactly what the composed transfer in \S8 was designed
to skip. All three multilevel configurations do reach a *better* final energy than
single-level (-63147/-63150/-63148 vs -63143) -- deeper hierarchies keep resolving
something the shallower ones miss -- but none recovers single-level's CPU-time lead by 300
iterations, and the deepest (4-level) is not the best on either count (3-level's final
energy is best; 4-level's cpu is lower than 3-level's, presumably because its per-level
`max_iter` schedule, 10/4/3 vs 3-level's 10/4, spends less on each intermediate solve).

Net conclusion of \S8-\S9: on this problem, the composed 2-level design (one geometric jump
straight to a single coarse solve, no intermediate optimization) is the only configuration
tested that is ever faster than single-level in CPU time, and only up to the early
90%-converged target; a minimum-fine-gradient-norm floor recovers a large fraction, but not
all, of what late-stage over-triggering was costing it; and adding genuine intermediate
levels trades that early win away for a better final answer, never for less CPU time.

Reproduction:
```sh
python3 test/cc/prepare_cow_data.py /path/to/RMO-continuous-cuts   # writes rho at all 4 levels
ninja -C build main_cc_cow main_cc_cow_levels
DEAL_II_NUM_THREADS=1 ./build/main_cc_cow          # ~2.2 minutes (SL 47s, gated ML 86s)
DEAL_II_NUM_THREADS=1 ./build/main_cc_cow_levels   # ~5.2 minutes (SL 47s, 3-level 143s, 4-level 119s)
```
