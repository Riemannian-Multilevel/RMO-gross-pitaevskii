# The e-connection on the positive sphere: retractions and step sizes

Date: 2026-10-12. Code: `proto/pos-sphere` at `21b3ee3` plus the working-tree changes listed in section 1. Runs:
`staging/pos_sphere_econn.sh` (148 runs, Mac, 9 parallel jobs, 1.5 min), tables from
`staging/pos_sphere_econn_tables.py`.

## Summary

- **The e-retraction works as a retraction.** `ellipsoid_positive::retract_by_e` is x_i ← c·x_i·exp(h v_i/x_i). It is
  the e-geodesic of the densities: p(h) ∝ p·exp(h u), with u_i = 2 v_i/x_i. It is defined and positive for every step,
  so it needs no fraction-to-the-boundary rule. Its inverse, `retract_inv_by_e`, has a closed form. Both pass the new
  checks in `test_positive`.
- **With the mass metric it is entropic mirror descent.** The update is p' ∝ p·exp(−4h ∇F(p)), with F the density
  energy of BlackboardS.tex. With the Frobenius metric it is the same update with the step scaled per node, 4h·d_i.
- **The paper's global m-convexity does not carry over to the e-geodesic.** From a peaked point to the constant one,
  f along the e-geodesic has φ'' down to −7.6. The supporting inequality f(x̂) − f(x) − Df(x)[L^e_x(x̂)] ≥ 0 fails
  there (−1.02).
- **Small domain (radius 4): every metric and step rule that converges on the unit sphere also converges with the
  e-retraction, except P2 with the mass metric.** With the energy-adaptive (EA) metric and step 1 or Armijo, it is the fastest retraction: 26–30
  iterations against 30–33 for the norm retraction and 31–34 for the density retraction.
- **Large domains (radius 6 and 10): the uncapped e-retraction is not safe with any step rule.** Fixed steps (EA, step
  1) diverge. Armijo and BB stall: the line search shrinks the step to about 1/max_i |v_i|/x_i, and this rate grows to
  1e10…1e304 once the tails of x collapse.
- **Convergence needs a relative step limit, applied through the existing fraction-to-the-boundary mechanism
  (`max_step`).** Recommended limit, by metric (variant names as in the tables):
  - EA: clip the direction, so that η_i ≥ −x_i holds as in exact arithmetic, and cap growth: h ≤ θ / max_i (v_i)₊/x_i
    (`clip-e-cap-up`). It converges with all three step rules on every Z-matrix case (Q1, P1) and on P2. On Q2 it
    converges at radius 4, reaches the KKT floor at radius 6 and stalls at 4e-5 at radius 10.
  - Mass and Frobenius: the growth cap (`e-cap-up`) with BB steps. It converges on every Z-matrix case (Q1, P1), and at
    radius 10 it is 1.0–1.9× faster than the density retraction.
  - On the non-Z discretizations (Q2, P2), mass and Frobenius converge only at radius 4 with any e variant; at radius 6
    and 10 they stall (section 5).
- **The unit-sphere retractions behave the same as each other.** Norm, orthographic and exponential map converge in
  the same cases with comparable iteration counts. All three end with negative components on large domains. Under the
  mass metric, `retract_by_exp` is the Levi-Civita (α = 0) geodesic of the Fisher–Rao geometry.
- **The step rule matters more than the retraction for mass and Frobenius.** Monotone Armijo starting at 1 converges
  on the unit sphere in 1 of 12 cases (mass: 3 of 12 when residuals below 1e-8 count); BB steps converge in 12 of 12. The step limit of the density and e-cap
  retractions doubles as a good starting step: with monotone Armijo they converge in 5–6 of 12, against 1 of 12 for
  the unit sphere.

## 1. What was added

| file | change |
|---|---|
| `include/rmo/gpe/manifold.h` | `ellipsoid_positive::retract_by_e`, `retract_inv_by_e` |
| `staging/test/positive.cc` | `check_e_retraction`: R(0) = x; positive and on the sphere up to 1e4× the density step limit; DR(0) = id; inverse in both directions; geometric curve R_x(t L_x(x̂)) ∝ x^(1−t) x̂^t. Also an INFO line for the energy along the e-geodesic (not a check). |
| `staging/pos_sphere_study.cc` | variants `e`, `e-cap`, `e-cap-up`, `clip-e`, `clip-e-cap`, `clip-e-cap-up` (positive sphere) and `ortho`, `exp` (unit sphere); column `rel_rate` = max_i \|η_i\|/x_i in `iter_*.csv` |
| `staging/pos_sphere_econn.sh` | the study; `staging/pos_sphere_econn_tables.py` makes the tables below |

**Evaluation.** The e-retraction is computed in logarithms, shifted by their maximum. Components that underflow are
set to the smallest normal number, `DBL_MIN` ≈ 2.2e-308, so the iterate stays strictly positive.

**The `ortho` and `exp` wrappers renormalize after the step,** as `PosUnitMassSphere` does, against the drift of
rounding errors. `ortho` uses the step limit 1/‖v‖_M of `retract_by_ortho` with θ = 0.99. `exp` has no step limit.

**Running the script on the Mac needs GNU xargs.** `pos_sphere_lib.sh` calls `xargs -d`, which the BSD `xargs` on
macOS does not have. Put a `xargs → /opt/local/bin/gxargs` link first on the PATH, for example
`PATH=<dir with link>:$PATH staging/pos_sphere_econn.sh …`.

## 2. Geometry

The density map p_i = d_i x_i² sends the positive sphere onto the open simplex. The mass metric is a quarter of
Fisher–Rao (BlackboardS.tex). The three α-connections of Fisher–Rao give three retractions of the positive sphere:

| α | connection | curve in p | in x | here |
|---|---|---|---|---|
| −1 | m (mixture) | p + h J_x v | sqrt(x² + 2h x v) | density retraction, `PosUnitMassSphere`; has a step limit |
| 0 | Levi-Civita | great circle | cos(h‖v‖)x + sin(h‖v‖)v/‖v‖ | `retract_by_exp` (mass metric); can leave the orthant |
| +1 | e (exponential) | p·exp(h u)/Z | c·x·exp(h v/x) | `retract_by_e`; no step limit |

`retract_by_ortho` is not an α-geodesic. Like the e-retraction, though, it moves along v and then rescales the base
point. In log coordinates the e-retraction is exactly this "step, then correct along a fixed direction": the
correction direction is the constant vector, where for ortho it is x.

**The e-step is mirror descent.** The mass gradient is grad_M f = D⁻¹A(x)x − λx (`kernels::grad_mass`), so
v_i/x_i = λ − (Ax)_i/(d_i x_i). Since ∂F/∂p_i = (Ax)_i/(2 d_i x_i), the e-step is
p' ∝ p·exp(−4h ∇F(p)). This is exponentiated-gradient (entropic mirror) descent on the simplex. Its natural step
condition is relative smoothness of F with respect to the entropy, not a boundary distance.

The density Hessian (BlackboardS.tex) has the terms (−B_ij)·sqrt(p_i p_j)·(ξ_i/p_i − ξ_j/p_j)². Their ratio to the
entropy Hessian Σ ξ_i²/p_i grows like sqrt(p_j/p_i) between neighbours. So the admissible step falls with the
neighbour density ratio, which is large exactly in the decaying tails. That explains the large-domain behaviour in
section 4.

**No e-convexity.** `test_positive` reports the second derivative of φ(t) = f(R^e_x(t L^e_x(x̂))):

| pair | f(x̂) − f(x) − Df(x)[L^e_x(x̂)] | min φ'' |
|---|---|---|
| smooth test points | +7.9e-2 | +1.34 |
| peak (Gaussian in the DoF index) to constant | **−1.02** | **−7.60** |

The global Ret^m-convexity and the supporting inequality (GP_exact_support) are specific to the m-geodesic. For the
e-retraction, the convergence theory would have to come from mirror descent (relative smoothness, KL), not from
retraction convexity.

## 3. Setup

- **Problem:** lumped mass, Neumann boundary, β = 100, start constant (thresholded, normalized); tolerance 1e-9 on the
  residual, 1000 iterations.
- **Discretizations:** Q1, P1 (level 6; Z-matrix, positive ground state) and Q2, P2 (level 5; non-Z). Radius 4, 6
  and 10.
- **Step rules:**
  - EA: fixed step 1, Armijo starting at 1, BB alternating with the max reference (nonmonotone).
  - Mass and Frobenius: Armijo starting at 1, and BB alternating with the monotone or the max reference.
  - Line search: β = 0.5, at most 40 trials, minimum step 1e-12.
- **Fraction θ = 0.99** for every step limit (density, e-cap, e-cap-up, ortho).
- **Sweeps:**
  - fixed steps: Q1 at radius 4 and 10 for mass and Frobenius; Q1 and Q2 at radius 10 for EA;
  - 5000-iteration runs of e, e-cap and e-cap-up: Q1 and P1, radius 10, mass and Frobenius with BB.

**Retractions:**

- unit sphere: `norm` (`UnitMassSphere`), `ortho`, `exp`;
- positive sphere:
  - `m`: the density retraction;
  - `e`: e-retraction, no limit;
  - `e-cap`: h ≤ θ / max_i |v_i|/x_i, so every component changes by at most a factor exp(θ);
  - `e-cap-up`: h ≤ θ / max_i (v_i)₊/x_i, so a component grows by at most exp(θ) and shrinks freely;
  - `clip-*`: EA only, the direction with w = A(x)⁻¹Dx clipped at 0, as in the `clip` variant.

## 4. Results

**Convergence per metric and step rule:** number of runs out of 12 (4 discretizations × 3 radii) that reach residual
1e-9. "+k" counts runs that stop at the ground-state energy (|E − E*| < 1e-10) with residual below 1e-8; these are at
the KKT floor of the non-Z discretizations, where the unit-sphere ground state has negative entries of about −2e-9.

| metric, step rule | norm | ortho | exp | m | e | e-cap | e-cap-up | clip-e | clip-e-cap | clip-e-cap-up |
|---|---|---|---|---|---|---|---|---|---|---|
| energy_adaptive_step1 | 12/12 | 12/12 | 12/12 | 4/12 | 4/12 | 7/12 | 7+3/12 | 4/12 | 10+1/12 | 10+1/12 |
| energy_adaptive_armijo | 12/12 | 12/12 | 12/12 | 4/12 | 7+1/12 | 7/12 | 7+3/12 | 9+1/12 | 10+1/12 | 10+1/12 |
| energy_adaptive_bb_max | 12/12 | 12/12 | 12/12 | 4/12 | 7+1/12 | 7/12 | 10+1/12 | 8+1/12 | 10+1/12 | 10+1/12 |
| mass_armijo | 1+2/12 | 1/12 | 1/12 | 5/12 | 1/12 | 6/12 | 6/12 |  |  |  |
| mass_bb_monotone | 12/12 | 12/12 | 12/12 | 7/12 | 5/12 | 9/12 | 8/12 |  |  |  |
| mass_bb_max | 12/12 | 12/12 | 12/12 | 7/12 | 4/12 | 9/12 | 8/12 |  |  |  |
| frobenius_armijo | 1/12 | 1/12 | 1/12 | 6/12 | 1/12 | 6/12 | 6/12 |  |  |  |
| frobenius_bb_monotone | 12/12 | 12/12 | 12/12 | 9/12 | 6/12 | 9/12 | 9/12 |  |  |  |
| frobenius_bb_max | 12/12 | 12/12 | 12/12 | 9/12 | 6/12 | 9/12 | 8/12 |  |  |  |

The unit-sphere columns count runs whose iterates turn negative on large domains ("neg" in the appendix). They are
references for the rate, not solutions of the positive problem.

### 4.1 Energy-adaptive metric

**Radius 4.** All retractions converge with all step rules. With step 1 or Armijo, the e-retraction is the fastest
(Q1: 29 iterations; norm 33, ortho 32, exp 32, m 34). With BB, all are within 18–24 iterations.

**Radius 6 and 10: uncapped e.** Step 1 fails in 8 of 8 cases (the iteration ends at a residual of 0.8 or more). Armijo and BB converge at radius 6 on Q1, P1 and
P2, and reach the KKT floor on Q2. At radius 10 they stall in all four discretizations.

The trace of Q1, radius 10, Armijo shows the failure mode:

| k | residual | min x | step | rel_rate = max\|η\|/x |
|---|---|---|---|---|
| 0 | 1.5e+01 | 5.0e-02 | 0.5 | 1.6e+01 |
| 1 | 2.5e+01 | 2.3e-04 | 1 | 2.8e+01 |
| 2 | 2.7e+00 | 7.2e-15 | 3.7e-09 | 1.1e+10 |
| 3 | 2.7e+00 | 2.0e-30 | 1.2e-07 | 1.1e+27 |
| 4 | 2.7e+00 | 2.2e-308 | 4.8e-07 | 9.5e+304 |
| 8 | 2.7e+00 | 2.2e-308 | 0 (stalled) | 5.4e+304 |

What happens, step by step:

- **The first full steps collapse the tails.** The bulk grows by exp(h·w/(cx)) and the normalization pushes the tails
  down by many orders of magnitude.
- **The EA direction then grows without bound.** v_i/x_i = w_i/(c x_i) − 1 becomes enormous, because w = A⁻¹Dx decays
  much more slowly than x.
- **Fixed steps overshoot; the line search stalls.** With a fixed step, a tail node grows by exp(1e80) and the energy
  jumps (Q1, radius 6, step 1: E = 218.9). Armijo shrinks the step to about 1/rel_rate, until the trial step falls
  below the 1e-12 minimum and the line search stalls.
- **The unit sphere does not have this problem.** With step 1, the norm retraction is inverse iteration, x' ∝ w, which
  has no exponential overshoot.

**Two safeguards are needed, and they cover each other's failure modes:**

1. **Clip the direction.** In exact arithmetic v_i ≥ −x_i, but inexact solves of A(x)w = Dx give w_i < 0 in the tails,
   and then v_i/x_i ≪ −1. Without the clip, the growth cap alone is slow: Q1, radius 6, `e-cap-up`, step 1 takes 732
   iterations, against 36 with the clip.
2. **Cap the growth (or the rate).** The clip alone does not stop the growth overshoot: `clip-e` with step 1 diverges
   at radius 6 and 10.

**`clip-e-cap-up` is the safe EA combination.** It converges with every step rule on Q1 and P1 at all radii (radius
10: 24–38 iterations, against 21–36 for the unit sphere) and on P2 (radius 10: 23–31). With fixed steps it converges
for step 0.25–1.5 on Q1 at radius 10.

**On Q2 the EA runs do not fully converge.** At radius 6 they reach the KKT floor (residual 6.8e-9, |E − E*| = 0,
free residual 1e-8, ν_min = −4.8e-9, 116 active nodes). At radius 10 they stall at residual 4e-5 with
|E − E*| = 6e-10 and ν_min = −4.9e-7. This is slow convergence of the boundary nodes, not divergence.

**The density retraction (m) fails in 8 of 12 EA cases** (the step becomes undefined after about 160 iterations).
This is the known large-domain failure that the earlier `clip` and `floor` variants address.

### 4.2 Mass and Frobenius metrics

**BB is the step rule for these metrics on every retraction.** Monotone Armijo starting at 1 does not converge in
1000 iterations on the unit sphere (11 of 12 runs per metric stop at a residual of 6e-9 to 9e-6), and BB converges in
12 of 12.
For the positive sphere the same ranking holds.

**Radius 4, all discretizations.** With BB, e converges, except P2 with the mass metric. For Frobenius it needs at
most 1.4× the iterations of the unit sphere; for mass, 1.2–1.7× on Q1, P1 and Q2. It needs fewer than the density
retraction except Q2 with mass and the max reference (107 against 90).

**P2 with the mass metric is the exception at radius 4.**

- The density retraction fails after 319 iterations.
- e stalls at 9e-2 (BB max) and 4e-4 (BB monotone).
- e-cap converges in 610–679 iterations; e-cap-up in 267–287.

**Radius 6 and 10, Z-matrix discretizations (Q1, P1).**

- **Uncapped e is slow and unreliable.** Q1, radius 6: 84–172 iterations; P1, radius 6: 385–863 or a stall at
  2e-7; radius 10: stalls.
- **More iterations do not help.** With a 5000-iteration limit, all 8 BB runs at radius 10 stop early on a stalled
  line search, after 7 to 2471 iterations, at a residual of 3e-7 to 7. Mirror descent near the boundary is slow, and
  the tails collapse as in 4.1.
- **e-cap converges in every case:** 58–397 iterations with BB.
- **e-cap-up converges in 8 of 8 BB runs:** 55–220 iterations, against 78–426 for the density retraction. At radius 10:
  - Q1: 136–138 against 141–145, about equal;
  - P1: 150–220 against 264–426, 1.8× faster with the mass metric and 1.9× with Frobenius.

**Radius 6 and 10, non-Z discretizations (Q2, P2).**

- **The density retraction fails** in 21 of 24 runs (undefined step). Only P2 at radius 6 with Frobenius converges,
  with all three step rules, in 730–731 iterations.
- **e-cap is slow on P2 at radius 6** (773–930 iterations) **and stalls in every other case.** It stalls far from the
  solution: Q2, radius 10, Frobenius at |E − E*| = 1.2.
  - Active nodes (x_i → 0 with a positive multiplier) have v_i/x_i → −∞.
  - So the two-sided cap → 0, and every step is capped.
- **e-cap-up stalls much closer:**
  - Q2, radius 6, mass with BB: residual 1.4e-6, |E − E*| < 1e-15;
  - Q2, radius 10, Frobenius with BB: 5e-3, |E − E*| = 2.5e-5;
  - P2, radius 10: 2e-2 to 3e-1.
  - P2, radius 6, Frobenius converges with Armijo (735 iterations) and monotone BB (472), but not with the max
    reference (stall at 9e-8). With mass, Armijo converges (597) and BB stalls at 3e-3.
- **Remaining failure: collapsed nodes cannot recover.** A node that falls to about 1e-300 during the transient, but
  that the solution needs, cannot grow back: its rate v_i/x_i ≈ 1e300 limits every step to about 1e-300.

### 4.3 Fixed steps (Q1)

Fixed steps do not ensure convergence with any retraction. Their window is narrow:

- **Mass:** steps 4e-3 and 1.6e-2 converge at radius 4; 4e-3 converges at radius 10 (934 iterations).
- **Frobenius:** steps 6.4e-2 and 0.25 converge at radius 4; 1.6e-2 converges at radius 10 (594 iterations).
- **In this window all retractions take the same iterations,** within 2.

Beyond the window:

- **Unit sphere: diverges** (residual 2 to 1e2, negative iterates).
- **Density retraction:**
  - radius 10: converges in 181–327 iterations;
  - radius 4: stalls (Frobenius, step 1: residual 0.2; mass, steps 6.4e-2 to 1: 2e-5 to 5e-5).
  - At radius 10 its cap acts as a step-size limit.
- **e-cap and e-cap-up:** converge at radius 10 (195–412 iterations). At radius 4 they stall: at 3e-5 to 1e-3 with
  mass, and at 0.3–0.7 with Frobenius at step 1.
- **Uncapped e:**
  - mass, radius 4: fails with an infinite exponent after 2–7 steps;
  - Frobenius, radius 4, step 1: stalls at 1;
  - radius 10: stalls at 0.7–6.

**With EA at radius 10:**

- the unit sphere converges for steps 0.25–1 and diverges at 1.5;
- `clip-e-cap(-up)` converges on Q1 for steps 0.25–1.5 (38–180 iterations);
- on Q2 it reaches 4e-5.

## 5. Verdict per retraction

**Step rules that converge in every case**, among the tested rules. "Z" means Q1 and P1.

| retraction | EA | mass, Frobenius |
|---|---|---|
| norm, ortho, exp (unit) | step 1, Armijo, BB (12/12) | BB (12/12) |
| m (density) | radius 4 only (any rule) | BB on Z (6/6); non-Z: radius 4 (except P2 with mass) and P2, radius 6, Frobenius |
| e | radius 4 only (any rule); Armijo/BB also radius 6 | radius 4 with BB, except P2 with mass |
| e-cap | any rule up to radius 6 (Q1, P1, P2; Q2 radius 4) | BB on Z (6/6) |
| e-cap-up | BB on Z and P2 (Q2: KKT floor at radius 6); Armijo and step 1 stall at 2e-9 to 2e-7 at radius 10 | BB on Z (6/6), faster than m |
| clip-e-cap, clip-e-cap-up | every rule on Z and P2; Q2 to the KKT floor at radius 6, 4e-5 at radius 10 | – |

**Recommendation.** Two choices make the e-retraction usable:

1. **A growth cap in `max_step`** for the e-retraction: h ≤ θ / max_i (v_i)₊/x_i.
2. **For EA, the clip of w at 0.** A clip in the oracle is not part of the retraction, so it would need its own option.

With these, every Z-matrix case converges, and on large domains the e-retraction matches or beats the density
retraction (P1, radius 10: 1.8–1.9× fewer iterations with BB). On the non-Z discretizations neither positive-sphere
retraction is robust.

## 6. Open items

- [ ] **Non-Z discretizations:** a lower bound on the e-retraction, like the `floor` variants, so that collapsed nodes
      can recover. Exclude nodes below the floor from the cap, as `FlooredPosSphere` does.
- [ ] **A KL trust region** KL(p'‖p) ≤ δ in place of the componentwise growth cap. This is the natural step bound of
      mirror descent; it needs a 1D root find per step.
- [ ] **Residual test for constrained solutions:** report convergence on the KKT measures (free residual and ν_min,
      already in the CSV), not on the unconstrained residual. The Q2 runs at radius 6 are converged in that sense.
- [ ] **Check whether to promote the e-retraction to the library.** Only the free functions and the tests are in
      `manifold.h`; the `ManifoldBase` wrappers live in the study. A library class would also need the differentials
      for the coarse models.
- [ ] **Mac portability of `pos_sphere_lib.sh`** (`xargs -d`).

## Appendix: all runs

Cells: iterations to residual 1e-9; F k: step undefined after k iterations (density retraction) or infinite exponent
(e); S (r): stopped after 1000 iterations or on a stalled line search, at residual r; "neg": nonpositive entries; dE:
|E − E*| > 1e-8 at convergence (does not occur).

### Main matrix

| run | norm | ortho | exp | m | e | e-cap | e-cap-up | clip-e | clip-e-cap | clip-e-cap-up |
|---|---|---|---|---|---|---|---|---|---|---|
| p1_r4_energy_adaptive_armijo | 30 | 29 | 30 | 32 | 27 | 32 | 32 | 27 | 32 | 32 |
| p1_r4_energy_adaptive_bb_max | 19 | 18 | 20 | 19 | 21 | 21 | 20 | 21 | 21 | 20 |
| p1_r4_energy_adaptive_step1 | 30 | 29 | 30 | 32 | 27 | 32 | 32 | 27 | 32 | 32 |
| p1_r4_frobenius_armijo | 714 | 587 | 570 | 682 | 557 | 567 | 564 |  |  |  |
| p1_r4_frobenius_bb_max | 114 | 111 | 121 | 216 | 120 | 144 | 145 |  |  |  |
| p1_r4_frobenius_bb_monotone | 120 | 107 | 108 | 206 | 121 | 130 | 117 |  |  |  |
| p1_r4_mass_armijo | S (1e-08) | S (2e-08) | S (2e-08) | S (1e-08) | S (2e-08) | S (2e-08) | S (2e-08) |  |  |  |
| p1_r4_mass_bb_max | 107 | 126 | 115 | 236 | 190 | 394 | 182 |  |  |  |
| p1_r4_mass_bb_monotone | 118 | 130 | 126 | 206 | 183 | 378 | 152 |  |  |  |
| p1_r6_energy_adaptive_armijo | 30 | 30 | 27 | F162 | 27 | 31 | 31 | 27 | 31 | 31 |
| p1_r6_energy_adaptive_bb_max | 19 | 19 neg | 18 neg | F162 | 23 | 29 | 22 | 23 | 29 | 22 |
| p1_r6_energy_adaptive_step1 | 30 | 30 | 27 | F162 | S (2e+00) | 31 | 31 | S (2e+00) | 31 | 31 |
| p1_r6_frobenius_armijo | S (7e-08) neg | S (4e-08) neg | S (4e-08) neg | 311 | S (1e-08) | 333 | 260 |  |  |  |
| p1_r6_frobenius_bb_max | 79 neg | 73 | 85 neg | 316 | 385 | 284 | 121 |  |  |  |
| p1_r6_frobenius_bb_monotone | 84 | 83 neg | 98 neg | 316 | 546 | 298 | 178 |  |  |  |
| p1_r6_mass_armijo | S (6e-09) neg | S (3e-08) neg | S (3e-08) neg | 177 | S (3e-08) | 384 | 281 |  |  |  |
| p1_r6_mass_bb_max | 86 neg | 97 neg | 80 neg | 162 | S (2e-07) | 397 | 125 |  |  |  |
| p1_r6_mass_bb_monotone | 87 neg | 83 | 89 neg | 162 | 863 | 397 | 125 |  |  |  |
| p1_r10_energy_adaptive_armijo | 32 neg | 30 neg | 32 neg | F161 | S (5e-02) | S (7e-08) | S (9e-09) | 52 | 35 | 33 |
| p1_r10_energy_adaptive_bb_max | 21 neg | 19 neg | 22 neg | F161 | S (4e+00) | S (5e-07) | 27 | 871 | 41 | 24 |
| p1_r10_energy_adaptive_step1 | 32 neg | 30 neg | 32 neg | F161 | S (2e+00) | S (7e-08) | S (9e-09) | S (2e+00) | 35 | 33 |
| p1_r10_frobenius_armijo | S (5e-08) neg | S (7e-08) neg | S (8e-08) neg | 426 | S (8e-03) | 335 | 248 |  |  |  |
| p1_r10_frobenius_bb_max | 90 neg | 92 neg | 89 neg | 426 | S (7e-02) | 335 | 220 |  |  |  |
| p1_r10_frobenius_bb_monotone | 79 neg | 96 neg | 92 neg | 426 | S (6e-03) | 335 | 220 |  |  |  |
| p1_r10_mass_armijo | S (4e-08) neg | S (8e-08) neg | S (4e-08) neg | 448 | S (8e-02) | 352 | 252 |  |  |  |
| p1_r10_mass_bb_max | 95 neg | 91 neg | 103 neg | 264 | S (8e-01) | 352 | 150 |  |  |  |
| p1_r10_mass_bb_monotone | 112 neg | 86 neg | 90 neg | 264 | S (6e-03) | 352 | 150 |  |  |  |
| p2_r4_energy_adaptive_armijo | 30 | 29 | 29 | 31 | 26 | 31 | 31 | 26 | 31 | 31 |
| p2_r4_energy_adaptive_bb_max | 19 | 18 | 19 | 19 | 21 | 20 | 20 | 21 | 20 | 20 |
| p2_r4_energy_adaptive_step1 | 30 | 29 | 29 | 31 | 26 | 31 | 31 | 26 | 31 | 31 |
| p2_r4_frobenius_armijo | S (9e-06) | S (4e-06) | S (9e-06) | S (2e-06) | S (4e-06) | S (6e-06) | S (6e-06) |  |  |  |
| p2_r4_frobenius_bb_max | 210 | 160 | 257 | 533 | 192 | 209 | 195 |  |  |  |
| p2_r4_frobenius_bb_monotone | 195 | 177 | 214 | 506 | 224 | 193 | 197 |  |  |  |
| p2_r4_mass_armijo | S (1e-06) | S (3e-07) | S (6e-07) | S (3e-08) | S (3e-07) | S (6e-08) | S (3e-08) |  |  |  |
| p2_r4_mass_bb_max | 189 | 181 | 194 | F319 | S (9e-02) | 679 | 267 |  |  |  |
| p2_r4_mass_bb_monotone | 183 | 170 | 208 | F319 | S (4e-04) | 610 | 287 |  |  |  |
| p2_r6_energy_adaptive_armijo | 29 | 29 | 26 | F165 | 27 | 31 | 31 | 27 | 31 | 31 |
| p2_r6_energy_adaptive_bb_max | 18 | 21 neg | 17 | F165 | 22 | 29 | 22 | 22 | 29 | 22 |
| p2_r6_energy_adaptive_step1 | 29 | 29 | 26 | F165 | S (2e+00) | 31 | 31 | S (2e+00) | 31 | 31 |
| p2_r6_frobenius_armijo | S (8e-08) neg | S (5e-08) neg | S (6e-08) neg | 730 | S (1e-02) | 853 | 735 |  |  |  |
| p2_r6_frobenius_bb_max | 135 neg | 144 neg | 132 neg | 731 | S (1e-01) | 801 | S (9e-08) |  |  |  |
| p2_r6_frobenius_bb_monotone | 150 neg | 146 neg | 134 neg | 731 | S (5e-02) | 773 | 472 |  |  |  |
| p2_r6_mass_armijo | S (3e-08) neg | S (4e-08) neg | S (3e-08) neg | F203 | S (3e-02) | 930 | 597 |  |  |  |
| p2_r6_mass_bb_max | 129 neg | 134 neg | 135 neg | F203 | S (9e-01) | 920 | S (3e-03) |  |  |  |
| p2_r6_mass_bb_monotone | 128 neg | 137 neg | 155 neg | F203 | S (2e-02) | 920 | S (3e-03) |  |  |  |
| p2_r10_energy_adaptive_armijo | 31 neg | 29 neg | 32 neg | F164 | S (8e-03) | S (3e-08) | S (2e-09) | S (7e-08) | 32 | 31 |
| p2_r10_energy_adaptive_bb_max | 21 neg | 19 neg | 20 neg | F164 | S (4e+00) | S (7e-08) | 23 | S (1e-07) | 37 | 23 |
| p2_r10_energy_adaptive_step1 | 31 neg | 29 neg | 32 neg | F164 | S (3e+00) | S (3e-08) | S (2e-09) | S (3e+00) | 32 | 31 |
| p2_r10_frobenius_armijo | S (9e-08) neg | S (6e-08) neg | S (5e-08) neg | F166 | S (5e-01) | S (6e-01) | S (3e-01) |  |  |  |
| p2_r10_frobenius_bb_max | 115 neg | 111 neg | 117 neg | F166 | S (6e+00) | S (6e-01) | S (2e-01) |  |  |  |
| p2_r10_frobenius_bb_monotone | 120 neg | 111 neg | 113 neg | F166 | S (3e-01) | S (6e-01) | S (2e-01) |  |  |  |
| p2_r10_mass_armijo | S (4e-08) neg | S (4e-08) neg | S (1e-07) neg | F174 | S (6e-02) | S (3e-02) | S (2e-02) |  |  |  |
| p2_r10_mass_bb_max | 96 neg | 107 neg | 116 neg | F174 | S (4e+00) | S (3e-02) | S (2e-02) |  |  |  |
| p2_r10_mass_bb_monotone | 106 neg | 119 neg | 110 neg | F174 | S (1e-01) | S (3e-02) | S (2e-02) |  |  |  |
| q1_r4_energy_adaptive_armijo | 33 | 32 | 32 | 34 | 29 | 34 | 34 | 29 | 34 | 34 |
| q1_r4_energy_adaptive_bb_max | 21 | 20 | 21 | 20 | 21 | 21 | 22 | 21 | 21 | 22 |
| q1_r4_energy_adaptive_step1 | 33 | 32 | 32 | 34 | 29 | 34 | 34 | 29 | 34 | 34 |
| q1_r4_frobenius_armijo | S (5e-08) | S (6e-08) | S (6e-08) | S (5e-08) | S (4e-08) | S (6e-08) | S (1e-08) |  |  |  |
| q1_r4_frobenius_bb_max | 43 | 47 | 50 | 68 | 61 | 49 | 48 |  |  |  |
| q1_r4_frobenius_bb_monotone | 44 | 45 | 46 | 68 | 45 | 52 | 50 |  |  |  |
| q1_r4_mass_armijo | S (9e-09) | S (5e-08) | S (4e-08) | S (2e-08) | S (1e-08) | S (1e-08) | S (2e-08) |  |  |  |
| q1_r4_mass_bb_max | 44 | 43 | 42 | 62 | 53 | 59 | 47 |  |  |  |
| q1_r4_mass_bb_monotone | 44 | 43 | 51 | 61 | 53 | 61 | 50 |  |  |  |
| q1_r6_energy_adaptive_armijo | 33 | 33 | 30 | F165 | 31 | 46 | 732 | 31 | 36 | 36 |
| q1_r6_energy_adaptive_bb_max | 22 | 22 | 21 | F165 | 24 | 28 | 28 | 24 | 32 | 23 |
| q1_r6_energy_adaptive_step1 | 33 | 33 | 30 | F165 | S (8e-01) | 46 | 732 | S (9e-01) | 36 | 36 |
| q1_r6_frobenius_armijo | S (6e-08) neg | S (8e-08) neg | S (8e-08) neg | 99 | S (2e-08) | 105 | 88 |  |  |  |
| q1_r6_frobenius_bb_max | 41 | 47 | 42 | 93 | 125 | 105 | 55 |  |  |  |
| q1_r6_frobenius_bb_monotone | 51 | 42 | 47 | 93 | 84 | 105 | 55 |  |  |  |
| q1_r6_mass_armijo | S (5e-08) neg | S (2e-08) neg | S (5e-08) neg | 93 | S (6e-08) | 80 | 118 |  |  |  |
| q1_r6_mass_bb_max | 53 | 54 | 49 | 78 | 172 | 58 | 58 |  |  |  |
| q1_r6_mass_bb_monotone | 48 neg | 47 neg | 51 neg | 78 | 131 | 64 | 58 |  |  |  |
| q1_r10_energy_adaptive_armijo | 36 neg | 34 neg | 37 neg | F162 | S (3e+00) | S (3e-06) | S (2e-07) | 242 | 38 | 38 |
| q1_r10_energy_adaptive_bb_max | 24 neg | 23 neg | 24 neg | F162 | S (1e+00) | S (5e-05) | 33 | S (1e+00) | 45 | 27 |
| q1_r10_energy_adaptive_step1 | 36 neg | 34 neg | 37 neg | F162 | S (9e-01) | S (3e-06) | S (2e-07) | S (9e-01) | 38 | 38 |
| q1_r10_frobenius_armijo | S (4e-07) neg | S (4e-07) neg | S (3e-07) neg | 167 | S (1e-05) | 285 | 170 |  |  |  |
| q1_r10_frobenius_bb_max | 86 neg | 85 neg | 91 neg | 145 | S (7e+00) | 285 | 138 |  |  |  |
| q1_r10_frobenius_bb_monotone | 81 neg | 81 neg | 77 neg | 145 | S (2e-04) | 285 | 138 |  |  |  |
| q1_r10_mass_armijo | S (9e-08) neg | S (1e-07) neg | S (1e-07) neg | 187 | S (1e-01) | 257 | 343 |  |  |  |
| q1_r10_mass_bb_max | 83 neg | 80 neg | 80 neg | 141 | S (5e-03) | 220 | 136 |  |  |  |
| q1_r10_mass_bb_monotone | 88 neg | 77 neg | 94 neg | 141 | S (9e-04) | 220 | 136 |  |  |  |
| q2_r4_energy_adaptive_armijo | 33 | 32 | 33 | 34 | 30 | 35 | 35 | 30 | 35 | 35 |
| q2_r4_energy_adaptive_bb_max | 21 | 20 | 21 | 21 | 22 | 23 | 24 | 22 | 23 | 24 |
| q2_r4_energy_adaptive_step1 | 33 | 32 | 33 | 34 | 30 | 35 | 35 | 30 | 35 | 35 |
| q2_r4_frobenius_armijo | S (1e-07) | S (1e-07) | S (1e-07) | S (1e-08) | S (1e-08) | S (5e-08) | S (9e-08) |  |  |  |
| q2_r4_frobenius_bb_max | 71 | 64 | 62 | 92 | 69 | 79 | 78 |  |  |  |
| q2_r4_frobenius_bb_monotone | 71 | 74 | 69 | 88 | 69 | 72 | 77 |  |  |  |
| q2_r4_mass_armijo | 232 | 174 | 242 | 120 | 139 | 149 | 164 |  |  |  |
| q2_r4_mass_bb_max | 66 | 59 | 65 | 90 | 107 | 87 | 70 |  |  |  |
| q2_r4_mass_bb_monotone | 52 | 57 | 60 | 84 | 78 | 71 | 71 |  |  |  |
| q2_r6_energy_adaptive_armijo | 33 neg | 33 neg | 29 neg | F163 | S (7e-09) | S (1e-06) | S (7e-09) | S (7e-09) | S (7e-09) | S (7e-09) |
| q2_r6_energy_adaptive_bb_max | 22 neg | 22 neg | 20 neg | F163 | S (7e-09) | S (8e-07) | S (7e-09) | S (7e-09) | S (7e-09) | S (7e-09) |
| q2_r6_energy_adaptive_step1 | 33 neg | 33 neg | 29 neg | F163 | S (1e+00) | S (1e-06) | S (7e-09) | S (2e+00) | S (7e-09) | S (7e-09) |
| q2_r6_frobenius_armijo | S (1e-07) neg | S (5e-08) neg | S (2e-07) neg | F167 | S (6e-05) | S (8e-02) | S (2e-02) |  |  |  |
| q2_r6_frobenius_bb_max | 64 neg | 69 neg | 71 neg | F167 | S (1e-02) | S (8e-02) | S (2e-05) |  |  |  |
| q2_r6_frobenius_bb_monotone | 62 neg | 66 neg | 74 neg | F167 | S (6e-05) | S (8e-02) | S (2e-05) |  |  |  |
| q2_r6_mass_armijo | S (2e-08) neg | S (2e-08) neg | S (2e-08) neg | F171 | S (9e-02) | S (4e-02) | S (3e-05) |  |  |  |
| q2_r6_mass_bb_max | 63 neg | 60 neg | 61 neg | F171 | S (5e-01) | S (4e-02) | S (1e-06) |  |  |  |
| q2_r6_mass_bb_monotone | 56 neg | 55 neg | 60 neg | F171 | S (3e-02) | S (4e-02) | S (1e-06) |  |  |  |
| q2_r10_energy_adaptive_armijo | 37 neg | 35 neg | 37 neg | F179 | S (3e+00) | S (4e-04) | S (5e-05) | S (4e-05) | S (4e-05) | S (4e-05) |
| q2_r10_energy_adaptive_bb_max | 23 neg | 23 neg | 23 neg | F179 | S (2e+01) | S (5e-04) | S (4e-05) | S (2e+00) | S (4e-05) | S (4e-05) |
| q2_r10_energy_adaptive_step1 | 37 neg | 35 neg | 37 neg | F179 | S (2e+00) | S (4e-04) | S (4e-05) | S (2e+00) | S (4e-05) | S (4e-05) |
| q2_r10_frobenius_armijo | S (9e-07) neg | S (4e-07) neg | S (4e-07) neg | F164 | S (2e-01) | S (2e+00) | S (1e-01) |  |  |  |
| q2_r10_frobenius_bb_max | 117 neg | 126 neg | 109 neg | F164 | S (2e+01) | S (2e+00) | S (5e-03) |  |  |  |
| q2_r10_frobenius_bb_monotone | 112 neg | 109 neg | 110 neg | F164 | S (4e-01) | S (2e+00) | S (5e-03) |  |  |  |
| q2_r10_mass_armijo | S (4e-08) neg | S (3e-08) neg | S (4e-08) neg | F161 | S (3e-01) | S (1e+00) | S (2e-01) |  |  |  |
| q2_r10_mass_bb_max | 93 neg | 85 neg | 90 neg | F161 | S (4e-01) | S (1e+00) | S (3e-01) |  |  |  |
| q2_r10_mass_bb_monotone | 87 neg | 97 neg | 91 neg | F161 | S (4e-01) | S (1e+00) | S (3e-01) |  |  |  |

### Fixed steps

| run | norm | ortho | exp | m | e | e-cap | e-cap-up | clip-e | clip-e-cap | clip-e-cap-up |
|---|---|---|---|---|---|---|---|---|---|---|
| q1_r10_energy_adaptive_t0.25 | 175 neg | 175 neg | 175 neg | F184 | S (8e-01) | S (5e-04) | S (1e-04) | S (1e+00) | 180 | 180 |
| q1_r10_energy_adaptive_t0.5 | 82 neg | 77 neg | 82 neg | F165 | S (9e-01) | S (1e-04) | S (3e-05) | S (9e-01) | 112 | 112 |
| q1_r10_energy_adaptive_t1 | 36 neg | 34 neg | 37 neg | F162 | S (9e-01) | S (3e-06) | S (2e-07) | S (9e-01) | 38 | 38 |
| q1_r10_energy_adaptive_t1.5 | S (1e+00) neg | S (2e+00) neg | S (2e+00) neg | F162 | S (9e-01) | S (2e-06) | S (2e-02) | S (9e-01) | 39 | 70 |
| q2_r10_energy_adaptive_t0.25 | 179 neg | 178 neg | 178 neg | F183 | S (2e+00) | S (1e-03) | S (1e-04) | S (2e+00) | S (4e-05) | S (4e-05) |
| q2_r10_energy_adaptive_t0.5 | 84 neg | 81 neg | 84 neg | F180 | S (2e+00) | S (1e-03) | S (5e-05) | S (2e+00) | S (4e-05) | S (4e-05) |
| q2_r10_energy_adaptive_t1 | 37 neg | 35 neg | 37 neg | F179 | S (2e+00) | S (4e-04) | S (4e-05) | S (2e+00) | S (4e-05) | S (4e-05) |
| q2_r10_energy_adaptive_t1.5 | S (3e+00) neg | S (3e+00) neg | S (3e+00) neg | F179 | S (3e+00) | S (4e-04) | S (2e-02) | S (3e+00) | S (4e-05) | S (4e-05) |
| r10_frobenius_t1e-3 | S (1e-01) | S (1e-01) | S (1e-01) | S (1e-01) | S (1e-01) | S (1e-01) | S (1e-01) |  |  |  |
| r10_frobenius_t4e-3 | S (2e-04) | S (2e-04) | S (2e-04) | S (2e-04) | S (2e-04) | S (2e-04) | S (2e-04) |  |  |  |
| r10_frobenius_t1.6e-2 | 595 | 594 | 594 | 597 | 594 | 594 | 594 |  |  |  |
| r10_frobenius_t6.4e-2 | S (3e+01) neg | S (4e+01) neg | S (5e+01) neg | 303 | S (6e+00) | 301 | 236 |  |  |  |
| r10_frobenius_t2.5e-1 | S (4e+01) neg | S (6e+01) neg | S (2e+01) neg | 181 | S (6e+00) | 285 | 195 |  |  |  |
| r10_frobenius_t1 | S (5e+01) neg | S (6e+01) neg | S (2e+00) neg | 186 | S (8e-01) | 285 | 248 |  |  |  |
| r10_mass_t1e-3 | S (4e-03) | S (4e-03) | S (4e-03) | S (4e-03) | S (4e-03) | S (4e-03) | S (4e-03) |  |  |  |
| r10_mass_t4e-3 | 934 | 934 | 934 | 936 | 934 | 934 | 934 |  |  |  |
| r10_mass_t1.6e-2 | S (3e+01) neg | S (6e+01) neg | S (6e+01) neg | 327 | S (7e-01) | 309 | 289 |  |  |  |
| r10_mass_t6.4e-2 | S (1e+02) neg | S (5e+01) neg | S (4e+01) neg | 214 | S (8e-01) | 318 | 220 |  |  |  |
| r10_mass_t2.5e-1 | S (5e+01) neg | S (5e+01) neg | S (2e+01) neg | 185 | S (7e-01) | 257 | 218 |  |  |  |
| r10_mass_t1 | S (6e+01) neg | S (5e+01) neg | S (5e+01) neg | 284 | S (7e-01) | 287 | 412 |  |  |  |
| r4_frobenius_t1e-3 | S (2e-01) | S (2e-01) | S (2e-01) | S (2e-01) | S (2e-01) | S (2e-01) | S (2e-01) |  |  |  |
| r4_frobenius_t4e-3 | S (6e-02) | S (6e-02) | S (6e-02) | S (6e-02) | S (6e-02) | S (6e-02) | S (6e-02) |  |  |  |
| r4_frobenius_t1.6e-2 | S (7e-04) | S (7e-04) | S (7e-04) | S (7e-04) | S (8e-04) | S (8e-04) | S (8e-04) |  |  |  |
| r4_frobenius_t6.4e-2 | 883 | 883 | 883 | 883 | 884 | 884 | 884 |  |  |  |
| r4_frobenius_t2.5e-1 | 219 | 219 | 219 | 219 | 220 | 220 | 220 |  |  |  |
| r4_frobenius_t1 | S (3e+01) neg | S (2e+01) neg | S (3e+00) neg | S (2e-01) | S (1e+00) | S (3e-01) | S (7e-01) |  |  |  |
| r4_mass_t1e-3 | S (6e-04) | S (6e-04) | S (6e-04) | S (6e-04) | S (6e-04) | S (6e-04) | S (6e-04) |  |  |  |
| r4_mass_t4e-3 | 835 | 835 | 835 | 835 | 836 | 836 | 836 |  |  |  |
| r4_mass_t1.6e-2 | 202 | 202 | 202 | 201 | 202 | 202 | 202 |  |  |  |
| r4_mass_t6.4e-2 | S (7e+01) neg | S (6e+01) neg | S (6e+00) neg | S (5e-05) | F7 | S (2e-04) | S (3e-05) |  |  |  |
| r4_mass_t2.5e-1 | S (7e+01) neg | S (8e+01) neg | S (6e+00) neg | S (2e-05) | F3 | S (2e-04) | S (1e-03) |  |  |  |
| r4_mass_t1 | S (8e+01) neg | S (8e+01) neg | S (3e+01) neg | S (2e-05) | F2 | S (2e-04) | S (9e-05) |  |  |  |

### 5000 iterations, radius 10

| run | e | e-cap | e-cap-up |
|---|---|---|---|
| p1_r10_frobenius_bb_max | S (3e-03) | 335 | 220 |
| p1_r10_frobenius_bb_monotone | S (4e-03) | 335 | 220 |
| p1_r10_mass_bb_max | S (8e-01) | 352 | 150 |
| p1_r10_mass_bb_monotone | S (6e-03) | 352 | 150 |
| q1_r10_frobenius_bb_max | S (7e+00) | 285 | 138 |
| q1_r10_frobenius_bb_monotone | S (3e-07) | 285 | 138 |
| q1_r10_mass_bb_max | S (5e-03) | 220 | 136 |
| q1_r10_mass_bb_monotone | S (5e-05) | 220 | 136 |
