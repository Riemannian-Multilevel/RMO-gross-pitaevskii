# CPU time of the positive-sphere retractions and step rules

Date: 2026-10-14. LaTeX version with all comparison tables and the performance profiles: `tex/pos_sphere_cpu.tex`
(drafts). Code: `proto/pos-sphere` at `21b3ee3` plus the working-tree changes listed in section 7. Runs:
`staging/pos_sphere_cpu.sh` (sequential, one deal.II thread, 3 repeats) with the studies
`staging/pos_sphere_analysis.sh` (IV), `staging/pos_sphere_econn_cpu.sh` (II) and `staging/pos_sphere_exact_cpu.sh`
(III); figures and `cpu_<study>.csv` from `staging/plot_pos_sphere_cpu.py`. Output: `build_cpu/`. Theory:
`tex/exact_line_search.tex`.

## Summary

- **CPU time follows the iterations wherever the step costs the same.** With the energy-adaptive (EA) metric the
  inner solve dominates every step, so CPU/unit ≈ iterations/unit for all density and e variants (IV, II). The
  exceptions are the e-retraction with the mass and Frobenius metrics (an e-step costs about 1.27 unit steps: `log`
  and `exp` per node) and the exact line search (several derivative evaluations per step, each with an assembly).
- **Measure CPU time with one deal.II thread.** With the default thread count the CPU time includes waiting worker
  threads (3× variation between identical runs, 3–10× the single-thread time). With one thread, repeats agree within
  3 % (median) and 6 % (90th percentile).
- **Parallel runs: at most 4 jobs on this machine** (one fewer than its 5 performance cores). 4 jobs stay within the
  repeat noise at 3.6× the throughput; from 5 jobs on, runs land on efficiency cores and CPU times grow by up to 2.5×.
- **IV (density retraction, as `fig_analysis_overview_*`):** EA: floor (no cap) is the fastest positive variant
  (0.88× the unit sphere), cont. 0.99×, clip+floor 1.8×, clip 2.1×; A fails on radius ≥ 6. Mass and Frobenius with
  Armijo: the unit sphere stalls in 15 of 16 cases; A and floor+A converge in 7–10 of 16, floor and cont. in 1.
- **II (e-retraction):** EA: clip-e-cap-up converges in 13 of 16 (all but Q2 at radius ≥ 6) at 1.07× the unit
  sphere, as robust as clip+floor of the density retraction and 1.7× cheaper; e, e-cap and e-cap-up converge in 4–8.
  Mass and Frobenius: e-cap and e-cap-up converge in 8 of 16 (Q1 and P1 at radius ≥ 6, P2 radius 6, Q2 radius 4 for
  mass), against 10 for floor+A of the density retraction, which is also faster on most of these rows.
- **III (exact line search):** on the unit sphere the closed-form exact step makes the mass and Frobenius metrics
  converge in 15 of 16 cases (Armijo: 1 of 16). On the positive sphere it raises the convergence of the mass and
  Frobenius metrics (floor+A and cont.+A: 12 of 16, against 10 and 9 with Armijo), mainly at radius 4, but where both
  converge Armijo is faster (the cap binds in most steps, and each derivative evaluation costs an assembly). With EA
  the exact step converges in the same cases as τ = 1 and Armijo, with about 10 % fewer iterations but 10–15 % more
  CPU time (median; up to 3× on Q1, where a solve is cheap).
- **New variant cont.+A** (continuation with the cap of floor+A): with EA it converges in 13 of 16 for every step rule,
  as fast as cont. without cap; it is the only continuation that admits an exact step.

## 1. Setup

- **Problem:** lumped mass, Neumann boundary, `--potential square` (V = |x|²), β = 100, constant start
  (thresholded at 1e-9, normalized).
- **Discretizations** (`DISC` in `pos_sphere_lib.sh`): Q1 and P1 at level 6 (Z-matrices), Q2 and P2 with bubbles at
  level 5 (not Z-matrices); radius 4, 6, 8, 10.
- **Stopping:** residual ‖D⁻¹(A(x)x − λDx)‖_D ≤ 1e-9, or 1000 iterations, or a stalled line search (step 0).
- **Inner solves:** only the energy-adaptive (EA) metric solves A(x)z = Dx. CG without preconditioner
  (`--solver cg --precond none`), absolute tolerance min(`--tol-inner-res` · res_k, 1e-2) = min(1e-2 · res_k, 1e-2)
  on the Euclidean residual of the system, at most `--max-inner` = 500 iterations. The mass and Frobenius metrics
  only apply D⁻¹, which is exact (lumped).
- **Step rules:**
  - fixed step τ = 1 (`--step-size 1`);
  - Armijo: α₀ = 1, β = 0.5, σ = 0.2, at most 40 trials, minimum step 1e-12 (`LINE_SEARCH` in
    `pos_sphere_lib.sh`, σ is the default of `--ls-sigma`);
  - exact (`--line-search --ls-exact`): first local minimizer of φ(t) = f(R_x(tη)) on (0, θ·max_step], relative
    tolerance 1e-10 on φ′ and on the bracket (`--ls-exact-tol`), at most 40 evaluations of φ′ (`--ls-max-iter`),
    bracket started at the previous step (1 in the first iteration). On the unit sphere the closed-form step of the
    normalization ray (global minimizer, a quartic; `GrossPitaevskiiFunctional::ray_step()`).
- **Fraction to the boundary:** θ = 0.99 for every cap (`--thetas 0.99`).

## 2. Step-size strategies

Every strategy combines with a step rule in one of three ways: a **cap** limits the step (the start of the Armijo
search, the end of the exact bracket); a **direction** modification changes η before the search; a **lower bound** is
part of the retraction.

| retraction | name (case) | kind | definition | studies |
|---|---|---|---|---|
| normalization | unit (`unit`) | – | (x + tη)/‖x + tη‖_D on the unit sphere, no cap | IV, II, III |
| density | A (`positive_theta0.99`) | cap | t ≤ θ·t_max, t_max = min_{η_i<0} x_i/(2\|η_i\|) | IV, III |
| | clip (`positive_clip_theta0.99`) | direction | EA only: grad_i ← min(grad_i, x_i), i.e. η_i ≥ −x_i; with A | IV, III |
| | floor+A (`positive_floor_theta0.99`) | both | x_i' = sqrt(max(x_i² + 2t x_iη_i, ε²)), normalized, ε = 1e-12; A over the nodes x_i > 10ε | IV, III |
| | floor (`positive_floor-nocap`) | lower bound | the clamped retraction without cap | IV |
| | clip+floor (`positive_clip-floor_theta0.99`) | all | clip, floor and A | IV, III |
| | cont. (`positive_floor-cont`) | lower bound | floor with ε_k = max(1e-150, min(ε_{k−1}, c·res_k)), c = 1e-3, no cap | IV |
| | cont.+A (`positive_floor-cont-cap_theta0.99`) | both | cont. with the cap of floor+A (new) | III |
| e | e (`positive_e`) | lower bound | x_i' ∝ x_i·exp(tη_i/x_i), normalized; entries below `DBL_MIN` raised to it | II |
| | e-cap (`positive_e-cap_theta0.99`) | cap | t ≤ θ/max_i(\|η_i\|/x_i) | II |
| | e-cap-up (`positive_e-cap-up_theta0.99`) | cap | t ≤ θ/max_i((η_i)₊/x_i) | II |
| | clip-e-cap-up (`positive_clip-e-cap-up_theta0.99`) | all | EA clip with e-cap-up | II |
| | clip-e, clip-e-cap | direction | EA clip with e, e-cap (`pos_sphere_econn.md`) | – |
| | e-clip(ε), e-cont | lower bound | max(R_x(tη), ε) normalized; with ε_k as cont. (Python prototype only) | – |
| | KL | cap | t ≤ t_δ with KL(p(t_δ)‖p) = δ (`exact_line_search.tex`, not tested) | – |

**The variants without cap have no exact step.** Along the clamped density curve the energy can decrease towards
t → ∞ (the clamped nodes stop, the others keep growing); the bracket of the exact search then doubles to the evaluation
limit, the step reaches 1e15 and the iteration ends in NaN (Q1, radius 10, mass metric). III therefore uses floor+A
and cont.+A instead of floor and cont. With a cap, under Assumption 2.1 of `exact_line_search.tex` (Z-matrix,
connected), the exact step along the density path lies in (0, t_max) and needs no floor in exact arithmetic.

## 3. CPU time

**Measure:** the column `elapsed` of `summary.csv`, the CPU time of `GradientDescent::cycle` (deal.II
`Timer::cpu_time`): gradients and inner solves, line search, retraction, assembly of the nonlinear term and the
residual of every iterate; not the setup or the output. The figures show the median over 3 repeats. Iteration counts
agree between repeats in every case.

**One deal.II thread (`DEAL_II_NUM_THREADS=1`).** With the default thread count the process CPU time includes the
worker threads: in a test (Q1, level 6, radius 10, EA) it varied by a factor of 3 between identical runs and was 3 to
10 times the single-threaded time. With one thread, repeats agree: max/min over the 3 repeats has median 1.03,
90th percentile 1.06 and maximum 1.16 (IV, 384 cases).

**Parallel runs** (one deal.II thread per run; Apple M3 Pro, 5 performance and 6 efficiency cores). The IV matrix
(64 program runs) once with J jobs, against the sequential median (`build_cpu/par_compare.log`):

| jobs | wall time | CPU / sequential: median | 10–90 % | max | cases outside ±10 % | ratio variant/unit: 5–95 % of its deviation |
|---|---|---|---|---|---|---|
| 1 (one repeat) | 266 s | – | repeat spread 1.03, 1.06 | 1.16 | – | 0.94–0.97 to 1.02–1.03 |
| 4 | 73 s | 1.04 | 1.01–1.06 | 1.29 | 3 % | 0.96 to 1.05 |
| 5 | 60 s | 1.06 | 1.03–1.12 | 2.34 | 17 % | 0.90 to 1.09 |
| 8 | 46 s | 1.33 | 1.06–1.49 | 2.45 | 85 % | 0.50 to 1.25 |
| 11 | 41 s | 1.46 | 1.10–1.73 | 2.48 | 90 % | 0.65 to 1.47 |

Iteration counts are the same for every J. Up to 4 jobs the CPU times are within the repeat noise (+4 % median), at
3.6 times the throughput; from 5 jobs on, runs spill onto the efficiency cores, which take up to 2.5 times the CPU time
for the same work, and the deviations are no longer noise. **Recommendation:** at most one job fewer than the number
of performance cores (here 4), with the median of 3 repeats; macOS has no CPU affinity to pin runs to performance
cores. The studies below were run sequentially.

## 4. IV: density retraction (the existing overviews, in CPU time)

Figures: `fig_analysis_overview_{ea,ea_armijo,mass,frobenius}_cpu.pdf` (as `fig_analysis_overview_*.pdf`, cell text in
ms). Medians of CPU/unit over the cases where both converge (`build_cpu/summarize.py`):

| case | EA, τ = 1: converged, CPU/unit | EA, Armijo | mass, Armijo: converged | Frobenius, Armijo |
|---|---|---|---|---|
| unit | 16, 1 | 16, 1 | 1 | 1 |
| A | 4, 1.02 | 4, 1.03 | 7 (6 failed) | 8 (5 failed) |
| clip | 13, 2.14 | 13, 2.22 | – | – |
| floor+A | 6, 1.06 | 6, 1.06 | 10 | 10 |
| floor | 13, 0.88 | 13, 0.90 | 1 | 1 |
| clip+floor | 13, 1.83 | 13, 1.77 | – | – |
| cont. | 13, 0.99 | 13, 1.00 | 1 | 1 |

- **EA:** CPU/unit is within 10 % of the iteration ratio of `pos_sphere_variants.tex` for every variant; the ranking
  does not change. The 3 cases where floor, clip+floor and cont. do not converge are Q2 at radius ≥ 6 (negative ground
  state, KKT floor).
- **Mass and Frobenius:** the unit sphere stalls (Remark rem:posvar:stagnation); its CPU time until the stall is the
  colour reference of the figures (rows marked †). A and floor+A converge on Q1 and P1 at radius ≥ 6 and (mass) on Q2
  at radius 4, Frobenius also on P1 at radius 4; floor+A also on P2 at radius ≥ 6, where A fails (except Frobenius,
  radius 6).

## 5. II: e-retraction

Figures: `fig_econn_overview_{ea,ea_armijo,mass,frobenius}_cpu.pdf`.

| case | EA, τ = 1: converged, CPU/unit | EA, Armijo | mass, Armijo: converged | Frobenius, Armijo |
|---|---|---|---|---|
| unit | 16, 1 | 16, 1 | 1 | 1 |
| e | 4, 0.91 | 7, 0.91 | 1 | 1 |
| e-cap | 7, 1.06 | 7, 1.05 | 8 | 8 |
| e-cap-up | 8, 1.06 | 8, 1.06 | 8 | 8 |
| clip-e-cap-up | 13, 1.07 | 13, 1.07 | – | – |

- **EA:** without the clip, every e variant fails at radius ≥ 8 (the inexact solve gives w_i < 0 in the tails, see
  `pos_sphere_econn.md`); with it, clip-e-cap-up converges wherever the ground state is positive. Against the density
  retraction (IV) on the same rows, in ms: Q1 radius 10: unit 2.5, floor 2.2, cont. 2.7, clip+floor 4.7,
  clip-e-cap-up 3.1; P1: 18.2, 15.8, 17.9, 37.0, 19.4; P2: 49.1, 42.1, 49.5, 95.8, 47.5.
- **Mass and Frobenius:** e-cap and e-cap-up converge on Q1 and P1 at radius ≥ 6, on P2 at radius 6, and on Q2 at
  radius 4 (mass) or P1 at radius 4 (Frobenius); the density retraction with floor+A additionally converges on P2 at radius 8 and 10. Where both
  converge, the density retraction is faster on Q1 (mass, radius 10: floor+A 7.5 ms, e-cap 12.5, e-cap-up 17.1) and
  mixed on P1 (radius 10: 25.9 against 58.2 and 43.2; radius 8: 36.6 against 41.1 and 31.5).

## 6. III: exact line search on the density retraction

Figures: `fig_density_exact_overview_<metric>_<rule>_cpu.pdf` for metric ea, mass, frobenius and rule exact, armijo,
and step1 (EA only). Columns: unit (exact: closed-form ray step), A, clip and clip+floor (EA only), floor+A,
cont.+A. Converged cases (of 16), and the median CPU/unit where both converge:

| case | EA exact | EA Armijo | EA τ = 1 | mass exact | mass Armijo | Frobenius exact | Frobenius Armijo |
|---|---|---|---|---|---|---|---|
| unit | 16 | 16 | 16 | **15** | 1 | **15** | 1 |
| A | 4, 1.15 | 4, 1.03 | 4, 1.03 | 9 | 7 | 10 | 8 |
| clip | 13, 2.26 | 13, 2.20 | 13, 2.16 | – | – | – | – |
| floor+A | 6, 1.19 | 6, 1.05 | 6, 1.06 | 12 | 10 | 12 | 10 |
| clip+floor | 13, 1.63 | 13, 1.76 | 13, 1.76 | – | – | – | – |
| cont.+A | 13, 1.21 | 13, 1.09 | 13, 1.08 | 12 | 9 | 12 | 9 |

- **Unit sphere, mass and Frobenius:** the closed-form exact step converges in 129–1000 iterations (all but P2 at
  radius 4, which reaches 1000 iterations at residual 4e-6), in 8–300 ms. The Armijo search stalls at about 1e-7 since
  the excited tail is energetically invisible (Remark rem:posvar:stagnation); the exact step, the global minimizer
  along the ray, is never too long for the stiff tail modes. The final iterates have negative entries at the noise
  level (down to −1.5e-10) on Q1, P1, P2, and the negative ground-state tail of Q2 (down to −4.4e-6).
- **Positive sphere, mass and Frobenius:** the exact step converges on Q1 and P1 at radius 4 (and Q2 at radius 4 for
  Frobenius), where Armijo stalls, and for the mass metric on P2 at radius 6 also with cont.+A. On the rows where both
  converge, Armijo is mostly 1.3–2× faster (mass, floor+A: Q1 radius 10 7.4 against 9.4 ms; P1 radius 10 25.9 against
  49.0 ms; cont.+A up to 5×, P1 radius 6: 40.5 against 193.6 ms). The cap binds in most exact
  steps (Q1 radius 10, mass, A: 146 of 148): the minimizer along the density path lies within 1 % of t_max, as the
  barrier of Lemma 3.6 grows only like (t_max − t)^(−1/2), and the search pays several derivative evaluations (each an
  assembly of the nonlinear term and two products with A) to find that out.
- **EA:** the exact step needs 10–12 % fewer iterations than the unit sphere with A, floor+A and cont.+A (τ = 1: 3–7 %
  more), but CPU/unit rises from 1.03–1.08 to 1.15–1.21 (median); on Q1, where a solve is cheap, cont.+A takes up to 3×
  the time of τ = 1 (radius 10: 8.1 against 2.7 ms). The convergence pattern is that of τ = 1: full EA steps are already
  near-optimal along the density path.
- **Variants without cap** (floor, cont.) have no exact step and are left out (section 2).

## 7. Code changes

| file | change |
|---|---|
| `include/rmo/option_types.h`, `include/rmo/option.h` | `DescentOptions::ls.exact`, `ls.exact_tol`; options `--ls-exact` (requires `--line-search`), `--ls-exact-tol` |
| `include/rmo/ropt/manifold.h` | `ManifoldBase::retract_velocity()` (default: not implemented), `constraint_inner()` (default 0), `retracts_by_normalization()` (default false) |
| `include/rmo/ropt/oracle.h` | `OracleBase::ray_step()` (default: std::nullopt) |
| `include/rmo/ropt/descent.h` | `exact_line_search()`: closed form on the normalization ray, otherwise bracket by doubling and Illinois on φ′ = ∇f(y)ᵀẏ − λ(y)·yᵀMẏ; the bracket also stops where φ′ is flat to the tolerance |
| `include/rmo/ropt/solver.h` | `cycle_smooth()` calls it with `--ls-exact`; `GradientDescent` starts the bracket at the previous step |
| `include/rmo/gpe/gpe.h`, `include/rmo/gpe/oracle.h` | `GrossPitaevskiiFunctional::ray_step()` for lumped systems (quartic; sign changes on a geometric grid of [1e-14, 1e14], bisection), forwarded by `GrossPitaevskiiOracle` |
| `include/rmo/gpe/manifold.h` | `ellipsoid::mass_inner()`, `normalize_with_velocity()`, `retract_velocity_by_norm()`; `ellipsoid_positive::retract_velocity_by_density()`, `retract_velocity_by_e()`; overrides in `UnitMassSphere` and `PosUnitMassSphere` |
| `staging/pos_sphere_study.cc` | velocities of `FlooredPosSphere` (clamped nodes do not move) and `ExpPosSphere`; variant `floor-cont-cap`; summary column `ls_exact` |
| `staging/test/positive.cc` | `check_exact_line_search`: velocities against central differences (norm, density, e), the exact step on the density curve (interior zero of φ′, at least as good as a grid of 2000 points), the closed-form ray step (stationary, at least as good as a grid) |
| `staging/pos_sphere_cpu.sh`, `pos_sphere_econn_cpu.sh`, `pos_sphere_exact_cpu.sh`, `plot_pos_sphere_cpu.py` | the studies and figures of this report; `pos_sphere_analysis.sh` skips its sweeps with `MATRIX_ONLY=1` |

`test_positive` passes in Release and Debug.

## 8. Reproduction

```sh
B=build_cpu   # frozen binary in $B/bin, GNU xargs linked in $B/gnubin (pos_sphere_lib.sh uses xargs -d)
MATRIX_ONLY=1 PATH=$B/gnubin:$PATH staging/pos_sphere_cpu.sh staging/pos_sphere_analysis.sh $B/bin $B/cpu_iv 3
PATH=$B/gnubin:$PATH staging/pos_sphere_cpu.sh staging/pos_sphere_econn_cpu.sh $B/bin $B/cpu_ii 3
PATH=$B/gnubin:$PATH staging/pos_sphere_cpu.sh staging/pos_sphere_exact_cpu.sh $B/bin $B/cpu_iii 3
staging/plot_pos_sphere_cpu.py analysis $B/cpu_iv $B/fig    # likewise econn ($B/cpu_ii), exact ($B/cpu_iii)
$B/par_test.sh 4 5 8 11 && python3 $B/par_compare.py $B     # parallel-jobs test
```

Wall time, sequential, 3 repeats: IV 799 s, II 818 s, III 1232 s.

## 9. Update: final exact line search, e-retraction with the exact step, norm-cap, profiles

**Exact line search, final form** (`exact_line_search()`, `tex/exact_line_search.tex` Section 7). Where a closed form
exists (the normalization ray: unit sphere, and norm-cap below), it is used. Otherwise the step is a root of φ′, found
by a derivative-only root finder: the first sign change of φ′ is bracketed by doubling (the extrapolation phase of
Nocedal and Wright, Algorithm 3.5, pp. 60–61), and the root located by the Illinois method (Dowell and Jarratt, BIT 11,
1971); accepted when |φ′| ≤ max(1e-10·|φ′(0)|, δ_g) with the rounding level δ_g of φ′, or at the cap; no Wolfe
conditions. A final descent check (one function value) halves the step if φ(t) > φ(0) + δ. A bracketing-and-zoom
variant (N&W Algorithms 3.5/3.6 with c₁ = 0 and cubic interpolation) was also tried: without the noise floor δ_g it
needed 26–40 evaluations per step near convergence, with it 4.2/8.6/8.5 (EA/mass/Frobenius, Q1, radius 10) against
3.6/4.3/4.2 for the derivative-only search, with the same iterations on the density path. The exact panels of II and
III were rerun with the final binary (`build_cpu/bin_v4`).

**norm-cap** (new variant, `NormPosSphere`): the normalization retraction on the positive sphere with the positivity
cap t ≤ θ·min_{η_i<0} x_i/|η_i| (= 2θ t_max), i.e. the fraction-to-the-boundary rule for x + tη. Its exact step is the
closed-form quartic of the ray, capped (`ray_step(x, v, t_max)`). Study: `staging/pos_sphere_norm_cpu.sh`.

**Retractions and exact steps:**

| retraction | closed form of the exact step | defined (and positive) for | convexity of φ |
|---|---|---|---|
| normalization (unit sphere) | yes: positive root of a quartic | all t ≥ 0; not positive in general | no |
| exponential map (great circle) | yes: arctan(ν t*_ray)/ν (same points as the ray) | all t | no |
| orthographic | yes: t*_ray/√(1+ν² t*²_ray) | tν < 1 | no |
| norm-cap (normalization on M⁺) | yes: the quartic, capped | positive for t < 2 t_max; cap 2θ t_max | no |
| density (m) | no (algebraic, n branch points) | t < t_max = min x_i/(2\|η_i\|) | strictly convex, barrier at t_max (Z-matrix, connected) |
| floored density (floor, cont.) | no | all t ≥ 0 (clamped at ε) | as density up to the first clamp; beyond, may decrease towards t → ∞ |
| e (exponential) | no (exponential sums) | all t, positive (floor DBL_MIN) | no (φ″ has a sign-indefinite term; min φ″ = −7.6 observed) |

**Results** (converged of 16; CPU relative to the unit sphere with the same rule; details in the tex):

- *Unit sphere, closed form:* mass and Frobenius converge in 15 of 16 (Armijo: 1).
- *norm-cap:* exact: mass 10, Frobenius 10; Armijo: 5, 8; EA: only radius 4 (the cap collapses with inexact solves,
  as for A). With the mass metric and the exact step it is the fastest positive version on Q1, P1 (0.37× the unit
  sphere).
- *Density path, exact vs Armijo:* mass/Frobenius gain the radius-4 cases (floor+A, cont.+A: 12 vs 10 and 9) but take
  1.7–2.0× (A, floor+A) and 3.2–3.4× (cont.+A) the CPU time of Armijo where both converge. EA: same cases, +2 to +14 %.
- *e-path, exact vs Armijo:* EA e-cap-up 11 vs 8, mass/Frobenius e-cap, e-cap-up 10 vs 8; 1.3–2.2× the CPU time.
- *Performance profiles* (`profiles_case{1,2}_{ea,mass,frobenius,best}.pdf`, `tab_profiles.tex`, 73 versions; case 1: Q1, P1 × radius, case
  2: all 16): the best positive versions are EA with floor (τ = 1: ρ(1) = 0.63 / 0.50, geometric mean ratio 1.03),
  then floor with Armijo, cont., cont.+A and clip-e-cap-up (8/8, 13/16, geometric means 1.12–1.46); for mass and
  Frobenius the most robust are floor+A and cont.+A with the exact step (12/16), the fastest robust one on Z-matrices
  norm-cap with the exact step.

**Load during the runs.** `WAIT_LOAD=25` (see `pos_sphere_cpu.sh`); the repeats of II-exact, III-exact and norm-cap
started at load averages 2.6–4.8.

**Non-converged runs in the tables.** The comparison tables give, for runs that do not reach 1e-9, the reason (L: the
1000-iteration limit; Z: the line search returned step 0), the CPU time until the stop and the final residual. A limit
of 10 000 iterations makes only 14 of the 146 L runs of the density study converge, after 1284–8990 iterations (EA
floor+A on Q1, P1, P2; mass A and floor+A on P2, radius 4); the others stagnate at residuals 1e-8–1e-7
(`build_cpu/iter10k_test`, not timed). The limit stays at 1000.
