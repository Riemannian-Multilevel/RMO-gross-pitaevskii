# `rmo::cc`: continuous cuts on the product Bernoulli manifold

This document covers the `rmo::cc` module — the binary continuous-cuts image
segmentation problem (Sec. 6.3 of arXiv:2607.09517) implemented inside the existing
`rmo::ropt` multilevel-optimization framework, mirroring the split already used for
the Gross–Pitaevskii problem (`rmo::gpe`): manifold, transfer operators, oracles, and
the multilevel scheme live in the problem-independent `rmo::ropt` core; `rmo::cc`
supplies the parts specific to continuous cuts.

It replaces three earlier documents (`doc/plan_continuous_cuts.tex`, the original
design note; `REVIEW-continuous-cuts.md`, an audit against the paper and its
reference implementation; and a planning brief, `fable.md`) with one, reflecting the
current, corrected state of the module rather than the history of getting there.

**Status in one paragraph.** Every module is implemented and checked against the
paper's own reference implementation (`RMO-continuous-cuts`, package
`bernoulli_multilevel`) to high precision. An initial version had a "no speedup"
result on its own 41×41 test problem; that was traced to two parameter bugs and
fixed (§3). The module now also runs the paper's actual 960×1280 → 240×320
cow-image problem (§4.2), not just a synthetic disk. CPU-time benchmarking (as
opposed to counting outer iterations, which is what the paper's own figures plot)
shows a genuine but narrow win: a coarse correction is expensive enough that it
only pays for itself early in a run, on the paper's own problem, with one specific
multilevel design (§4); deeper hierarchies and the smaller synthetic problem show
no CPU-time benefit at all.

## Contents

1. [Design](#1-design)
2. [Verification](#2-verification)
3. [Bugs found and fixed](#3-bugs-found-and-fixed)
4. [Performance: CPU time, not just iterations](#4-performance-cpu-time-not-just-iterations)
5. [Known limitations and future work](#5-known-limitations-and-future-work)
6. [Reproduction](#6-reproduction)

---

## 1. Design

### Why this module is simpler than `rmo::gpe`

- **No linear solves.** The Fisher–Rao metric is diagonal,
  $G(\phi) = \operatorname{diag}(1/(\phi(1-\phi)))$; every Table 9 formula is a
  closed-form elementwise expression. The only matrix-shaped objects are the sparse
  gradient operator $D$ and the grid-transfer operators.
- **Plain array indexing, no finite elements.** The grid-transfer operators —
  injection $J_h^H$, bilinear interpolation $B_H^h$, full-weighting
  $F_h^H = \tfrac14(B_H^h)^\top$ — are index/parity arithmetic on the pixel array
  (matching the reference's own `primitives.py`, e.g. `phi[::2,::2]` for
  injection), not a finite-element mesh transfer.

### Components

| Component | File | Notes |
|---|---|---|
| Pixel grid + forward difference $D$ | `cc/grid.h` | Plain row-major indexing; no deal.II mesh dependency. |
| Fisher–Rao metric | `cc/metric.h` | Inner product, norm, diagonal weight $\phi(1-\phi)$. |
| Manifold geometry (Table 9) | `cc/manifold.h` | Retraction/lifting, see below. |
| Objective (value/gradient) | `cc/cc.h` | Ambient (Euclidean) gradient; converted to Riemannian by the oracle. |
| Fine-level oracle | `cc/oracle.h` | `FisherRaoOracle`; no linear solve. |
| Grid transfer | `cc/interpolate.h` | Injection, bilinear, full-weighting; generalized to even sizes and composed transfers (§3, §4.2). |
| Point map + vector transport (Table 10) | `cc/transport.h` | Options 1 and 4 (only two implemented; see §5). |
| Coarse-model oracle | `cc/oracle_coarse.h` | `ContinuousCutsCoarseOracle`. |
| Coarse-correction trigger (eq. 16) | `cc/condition.h` | `ScaledCoarseCondition`, pluggable, with a minimum-fine-gradient-norm floor (§4.3). |
| Drivers | `src/main_cc*.cc` | Single-level, two-level (synthetic disk and cow image), 3-/4-level, CPU-time benchmarks. |
| Tests | `test/cc/*.cc`, `test/cc/*.py` | Golden values and reference-trajectory comparisons (§2). |

### Manifold: retraction and lifting

Both reduce to a logit-space identity:
$$
  R_\phi(v) = \sigma\big(\operatorname{logit}(\phi) + G(\phi) v\big), \qquad
  \mathcal{L}_\phi(z) = G(\phi)^{-1}\big(\operatorname{logit}(z) - \operatorname{logit}(\phi)\big),
$$
with $\sigma$ the logistic sigmoid and $G(\phi)v = v/(\phi(1-\phi))$. Implementing
them through stable `sigmoid`/`logit` (rather than the raw exponential form)
avoids overflow near $\phi \in \{0,1\}$ and matches the identity the paper itself
uses to simplify the coarse-model correction term.

### Objective and gradient

$$
  E_\varepsilon^{CC}(\phi) = \rho^\top\phi + \alpha\sum_i\sqrt{(D_1\phi)_i^2+(D_2\phi)_i^2+\varepsilon^2}.
$$
The ambient gradient is $\rho + \alpha D^\top(I_2\otimes\operatorname{diag}(\omega))D\phi$
with $\omega_i = 1/\sqrt{(D_1\phi)_i^2+(D_2\phi)_i^2+\varepsilon^2}$; the Riemannian
gradient is $G(\phi)^{-1}$ applied to it, i.e. an elementwise multiply by
$\phi(1-\phi)$ (Table 9) — no linear solve, unlike GP's mass/energy metrics.

**$\varepsilon$ convention.** Eq. (42) squares $\varepsilon$ under the root; the
reference's `objective.py` adds its `eps` argument unsquared, and the paper states
its own figures were produced by the reference. Convention used throughout this
module: $\varepsilon_{\text{paper}} = \sqrt{\varepsilon_{\text{ref}}}$ — e.g.
$\varepsilon = 10^{-2}$ fine ($\sqrt{10^{-4}}$), $\approx 3.16\times10^{-2}$ coarse
($\sqrt{10^{-3}}$). The reference's own numbers passed unsquared would make the
smoothing 100× sharper than anything the reference (and hence the paper) ran.

### Coarse model

$q_k(z) = E_H(z) - w_k^\top(\operatorname{logit}(z) - \operatorname{logit}(\psi_k))$,
with Riemannian gradient
$$
  \operatorname{grad}q_k(z) = \operatorname{diag}(z(1-z))\,g - w_k, \qquad g = \operatorname{grad}_{\mathrm{amb}}E_H(z).
$$
An earlier version computed $\operatorname{diag}(z(1-z))(g-w_k)$ (also rescaling
$w_k$), which is wrong: the correct formula follows from the metric adjoint of
$\mathrm D\mathcal L_{\psi_k}(z)$ being the *identity* under the Fisher–Rao metric
(checked both analytically and numerically to $10^{-10}$) — not the Euclidean
self-adjoint object `retract_inv_diff_adjoint` computes, which answers a different
question. The bug was caught by testing Proposition 3.2's first-order coherence
identity, not by re-reading the derivation.

### Grid transfer (Table 10)

Options 1 and 4 use identical $P,R$ and differ only in which restriction builds
$\psi=r(\phi)$ (injection vs. full-weighting), so one class,
`GeometricBilinearTransport`, covers both:
$$
  P^\phi_\psi(v) = G(\phi)^{-1}\big(B_H^h\,G(\psi)v\big), \qquad
  R^\psi_\phi(v) = (B_H^h)^\top v = 4F_h^H v \text{ (unweighted)}.
$$
Options 2, 3, 5 are not implemented (2/5 underperform in the paper's own results;
3 needs a Moore–Penrose pseudoinverse for marginal benefit).

`BernoulliGridTransfer` originally required `fine = 2·coarse − 1` (one exact
mesh-refinement step). Generalized to `coarse = ceil(fine/2)` for either parity of
`fine`, matching the reference's own `r_injection` (`phi[::2,::2]`) exactly; the
even-size boundary case (a stencil neighbor one past the coarse grid's last
row/column) drops that term rather than renormalizing, matching the reference's
zero-padded `conv2d`. `ComposedGridTransfer`/`ComposedVectorTransport` chain
several factor-2 hops into one coarse-model transition (matching the reference's
`operators.make_composed_ops`), needed for the paper's own 960×1280 → 240×320
problem (two pooling steps) without exposing an intermediate level to
`FullApproximationScheme`. Both verified against the reference numerically,
including per-hop metric weighting for the composed vector transport (§2).

### Coarse-correction trigger (eq. 16)

$$
  \|R\operatorname{grad}f_h\|_\psi \geq \max(\eta\|\operatorname{grad}f_h\|_\phi, \mu).
$$
`ScaledCoarseCondition(grid_scale, min_fine_norm = 0)` implements this with two
adjustments, both explicit and labelled rather than silently folded in:

- **`grid_scale`**, multiplying the coarse norm before either comparison, is the
  reference's own per-approach compensation for how much a given restriction
  shrinks or grows the gradient norm (`operators.get_grid_scale`) — absent from
  the paper itself. For Option 1, `grid_scale = 2^n_pools` (2 for one factor-2
  step, 4 for the composed two-step cow-image transition).
- **`min_fine_norm`** (default 0, i.e. disabled) is a floor on the *fine* gradient
  norm, below which no correction is considered regardless of the coarse side —
  GPE's `DefaultCoarseCondition` has always had this (`norm_level > options.eps`,
  wired to GPE's `--eps` CLI flag); eq. (16) itself has no such term. It matters
  when the coarse norm stays orders of magnitude above the fine norm for an
  entire run rather than the two converging together (§4.3).

---

## 2. Verification

Every non-trivial formula is checked two ways: an internal consistency check
(e.g. a gradient against a central difference), and against the paper's own
reference implementation (`RMO-continuous-cuts`, not part of this repository).

**Golden-value tests** (`test/cc/derive_golden_values.py` calls the reference's
own functions directly on fixed inputs; the literal constants in
`test_manifold.cc`, `test_oracle.cc`, `test_interpolate.cc` are copies of its
output, kept in sync by hand):

- `cc/metric.h`, `cc/manifold.h` — retraction/lifting inverse pair, differentials — match `manifold.py` to $10^{-9}$.
- `cc/cc.h` — ambient gradient matches torch autograd to $10^{-6}$.
- `cc/interpolate.h`, `cc/transport.h` — injection, bilinear, `Tfine`, Option 1/4 equivalence — match `primitives.py`/`operators.py` coefficient for coefficient, including the even-size and composed-transfer generalizations (verified against `make_composed_ops` on an 8×8 → 4×4 → 2×2 grid).
- `cc/oracle_coarse.h` — Proposition 3.2 first-order coherence, tested via `test_oracle_coarse.cc`'s full `CoarseOracleBase` machinery under both point restrictions.

**Trajectory comparisons** (the algorithm end-to-end, not just primitives —
`test/cc/compare_reference.py` and `test/cc/prepare_cow_data.py`):

- Single-level RGD on the 41×41 synthetic disk matches the Python reference to 6
  digits at iteration 1 and $5\times10^{-7}$ relative at iteration 300.
- Single-level RGD on the paper's actual 960×1280 cow-image `rho` matches
  `solve_single_level` on the identical array to 6 significant digits at every
  checked iteration (1, 2, 3, 5, 10, 20) — the first check of this port's
  fine-level code at the paper's own resolution.
- `prepare_cow_data.py`'s preprocessing (grayscale, 2× bilinear enlarge, Gaussian
  blur, seed-rectangle means, composed average-pooling) is delegated to skimage —
  the same code path the paper's own figures were produced with — rather than a
  fresh C++ reimplementation risking silent divergence; checked against the
  reference's actual `avg_pool2d` call (max abs. diff $1.2\times10^{-7}$).

**Gaps**: `derive_golden_values.py`'s docstring claimed it works without
scikit-image; it doesn't (`bernoulli_multilevel/__init__.py` imports a module that
needs it) — fixed by installing a stub, the same workaround `compare_reference.py`
uses. Tests draw unseeded random points from the interior of $(0,1)$, so the
boundary-clamp regime (`MIN_WEIGHT`) is never exercised by them.

---

## 3. Bugs found and fixed

| # | Bug | Where | Fix |
|---|---|---|---|
| 1 | Coarse Armijo damped to `alpha=0.01` instead of the reference's undamped `1.0`, leaving the coarse model unminimised | `main_cc_coarse.cc` | `ls.alpha = 1.0`, relying on the existing `MIN_WEIGHT` boundary clamp instead of damping |
| 2 | `coarse_every = 1` (no lockout on consecutive coarse corrections), violating the paper's §6.4 rule | `main_cc_coarse.cc` | `coarse_every = 2` |
| 3 | Eq. (16)'s $\mu$ gate applied to the *fine* norm instead of the *coarse* one | `cc/condition.h` | `ScaledCoarseCondition` gates on the scaled coarse norm, matching eq. (16) literally |
| 4 | $\varepsilon$ passed unsquared into a formula that squares it (100× sharper smoothing than the reference ever ran) | `main_cc.cc`, `main_cc_coarse.cc` | $\varepsilon_{\text{paper}} = \sqrt{\varepsilon_{\text{ref}}}$ convention adopted throughout |
| 5 | Armijo line search accepted a step that **failed** sufficient decrease once `alpha < ls.min`, returning `ls.min` as if it had succeeded | `ropt/descent.h` | Returns 0 (matching the existing max-iterations contract) instead of silently accepting the failed step |
| 6 | `LinearCombination::add_component` asserted a non-zero weight, aborting `test_arpack` (`beta=0`) in every Debug build | `lac.h` | Skip zero-weight components instead of asserting |
| 7 | Unconditional `std::cerr << "level: "` debug print, 300+ lines of noise per run | `ropt/fas.h` | Removed |
| 8 | Legacy `add_test(descent test_descent COMMAND test_descent)` signature | `CMakeLists.txt` | `add_test(NAME descent COMMAND test_descent)` |
| 9 | Unseeded RNG (`std::random_device`); test failures not reproducible | `util/random.h` | Seed from `RMO_TEST_SEED` if set, else `std::random_device`; print the seed used either way |
| 10 | `BernoulliGridTransfer` required `fine = 2·coarse − 1`; the paper's own 960×1280 → 240×320 problem needs even sizes and two composed pooling steps | `cc/interpolate.h`, `cc/transport.h` | Generalized to `coarse = ceil(fine/2)`; added `ComposedGridTransfer`/`ComposedVectorTransport` |
| 11 | No minimum-fine-gradient-norm floor, so the coarse trigger never shuts off once the coarse norm permanently dwarfs the fine one (145 of 150 eligible iterations fire on the cow image) | `cc/condition.h` | `min_fine_norm` parameter, opt-in, default 0 |
| 12 | No differential test compared the *algorithm* (trigger → coarse solve → prolongation → line search) against the reference on the same problem — only primitives were golden-tested | `test/cc/` | `compare_reference.py`, `prepare_cow_data.py` added |

Bug 1+2 together were the entire cause of an earlier "no speedup" conclusion on
the 41×41 synthetic disk, previously (incorrectly) attributed to the vector
transport's structural properties; see §4.1 for the corrected measurement.

---

## 4. Performance: CPU time, not just iterations

The paper's own figures plot energy against outer iteration count. A coarse
correction also costs real CPU time — a smaller solve on the coarse grid plus two
grid transfers — so an iteration-count win does not automatically mean a
CPU-time win. All numbers below are single-threaded
(`DEAL_II_NUM_THREADS=1`; deal.II's own thread pool otherwise parallelizes large
vector operations independently of `OMP_NUM_THREADS`, inflating `cpu_time()`
non-deterministically) and use two convergence targets: **90% converged**
(within 10% of the single-level run's own total energy drop — a scale-independent
early milestone) and **fully converged** (matches single-level's own final
energy).

### 4.1 Synthetic disk (41–321 px), single- vs. two-level

One coarse level, one exact mesh-doubling (`fine = 2·coarse − 1`), `grid_scale=2`.

| $n_{\text{fine}}$ | $n_{\text{dofs}}$ | target | SL it | SL cpu (s) | ML it | ML cpu (s) | ML coarse corr. | speedup (SL/ML) |
|---|---|---|---|---|---|---|---|---|
| 41  | 1,681   | 90%  | 9   | 1.29e-3 | 1   | 1.70e-3 | 4 | 0.76 |
| 41  | 1,681   | full | 161 | 2.19e-2 | 161 | 2.73e-2 | 4 | 0.80 |
| 81  | 6,561   | 90%  | 9   | 4.80e-3 | 1   | 6.18e-3 | 2 | 0.78 |
| 81  | 6,561   | full | 161 | 8.13e-2 | 158 | 1.07e-1 | 2 | 0.76 |
| 161 | 25,921  | 90%  | 9   | 1.98e-2 | 1   | 2.38e-2 | 2 | 0.83 |
| 161 | 25,921  | full | 161 | 3.35e-1 | 157 | 4.38e-1 | 2 | 0.76 |
| 321 | 103,041 | 90%  | 9   | 8.51e-2 | 1   | 9.11e-2 | 5 | 0.93 |
| 321 | 103,041 | full | 161 | 1.42e0  | 161 | 2.26e0  | 5 | 0.63 |

(Coarse solver `max_iter=5`; halving it from the paper/reference's 10 changed
these ratios by only a few points either way — the dominant cost is fixed
per-correction overhead, not the coarse iteration budget.)

**Reading**: single-level is faster at *every* size and *both* targets. The
one-iteration jump the bug fixes restored (§3, bugs 1–2) is real, but the coarse
correction that produces it costs about as much as the 9–16 fine iterations it
replaces — the fine-level objective here is cheap per iteration (a forward
difference and an elementwise divide, no linear solve), unlike the
linear-solver-dominated problems multigrid is usually built for, so there is
little expensive work for a correction to substitute for. The ratio does not
improve with problem size (both levels' cost scales the same way with
resolution), so extrapolating to the paper's much larger image gives no reason
to expect a win — but it does show one, once actually measured (§4.2).

### 4.2 The paper's own cow image (960×1280 → 240×320)

Single coarse level, composed two-step transfer (`n_pools=2`, `grid_scale=4`),
300 fine iterations.

| driver | corrections | final $E$ | cpu(300) | 90%-converged speedup | fully-converged speedup |
|---|---|---|---|---|---|
| single-level (baseline) | — | −63,143 | 47.5 s | — | — |
| two-level, ungated | 145 | −63,119 | 145.6 s | **3.16** | 0.317 |
| two-level, `min_fine_norm=10` | 59 | **−63,147** | 85.9 s | **3.20** | 0.615 |

**Reading**: this is the first (and only) target anywhere in this document where
the two-level driver beats single-level in CPU time — 3.2× faster to 90%
convergence, matching the paper's own headline claim of an early, large jump.
Ungated, it is a clean *loss* by 300 iterations (3× slower, worse final energy):
the scaled coarse gradient norm stays 5,000–9,000× the fine norm for the entire
run (`mu=0.5` and `grid_scale=4`, tuned by observation on the 1,681-pixel disk,
gate nothing on this 1,228,800-pixel problem), so the trigger fires on 145 of 150
eligible iterations, most of them past the point of doing further good. Gating on
a minimum fine-gradient norm (bug 11) more than halves the corrections, cuts CPU
time 41%, and — because the removed corrections were actively net-negative, not
merely wasted — leaves the *final* energy better than single-level's. It does not
fully close the gap: 59 corrections still fire before the floor is crossed.

### 4.3 3- and 4-level hierarchies, same cow image

Replicates the reference's own "best configurations"
(`examples/levels_2_3_4.py`): every adjacent level pair here is a single,
un-composed factor-2 step, so these use plain `BernoulliGridTransfer` and
`FullApproximationScheme`'s ordinary N-level recursion — no composed transfer.
All multilevel rows use `min_fine_norm=10`; single-level is shared across rows.

| hierarchy | levels (px) | corrections | final $E$ | cpu(300) | 90%-converged (it, speedup) | fully-converged (it, speedup) |
|---|---|---|---|---|---|---|
| single-level | 960×1280 | — | −63,143 | 47.5 s | it=19 | it=300 |
| 2-level | 240×320 / 960×1280 (composed) | 59 | −63,147 | 85.9 s | it=1, **3.20** | it=262, 0.615 |
| 3-level | 240×320 / 480×640 / 960×1280 | 31 | **−63,150** | 142.6 s | it=1, 0.618 | it=253, 0.363 |
| 4-level | 120×160 / 240×320 / 480×640 / 960×1280 | 35 | −63,148 | 118.5 s | it=1, 0.922 | it=216, 0.490 |

**Reading**: going deeper does not help CPU time, at either target — the
composed 2-level driver's early win vanishes once a genuine intermediate level is
inserted (0.62× and 0.92×, both slower than single-level). This is structural:
`FullApproximationScheme`'s recursion actually *solves* at every intermediate
level (checking eq. (16) again and potentially firing its own nested correction)
rather than treating the whole coarse chain as one geometric jump to a single
bottom-level solve — exactly the per-level cost the composed transfer exists to
skip. All three multilevel configurations do reach a better final energy than
single-level, so a deeper hierarchy is not wasted work, but none recovers
single-level's CPU-time lead by 300 iterations.

**Net conclusion**: on every problem tested, the *only* configuration that is
ever faster than single-level in wall-clock terms is the composed two-level
design, and only up to an early convergence target — never for the full run,
and never for a genuine 3+ level hierarchy. Whether a different coarse solve
budget, gate, or a genuinely more expensive fine-level problem changes this is
open (§5).

---

## 5. Known limitations and future work

- **Options 2, 3, 5** (Table 10) are not implemented — see §1 for why 1 and 4
  were chosen first; adding them is mechanical but not done.
- **Larger/harder problems**: the CPU-time story in §4 might look different on a
  problem where the fine-level objective itself is expensive (e.g. involves a
  linear solve) — untested here, since `rmo::cc`'s objective never needs one.
- **Trigger tuning**: `mu` and `grid_scale`, calibrated by observation on the
  41×41 disk, do not transfer to the cow image's scale (§4.2); a formulation that
  scales with problem size (or a different gate entirely — three variants are
  compared in §1) is unexplored.
- **`FISHER_RAO`** is in the shared `MetricKind` enum but not CLI-parseable
  (`option.h`'s list feeds GP's `--metric` flag, which has no handler for it);
  add it once a CC driver takes CLI options of its own.
- Housekeeping noted in earlier drafts of this document (duplicate fork branches,
  two pre-existing test failures unrelated to this module — `arpack` in Debug
  before bug 6's fix, `operator`'s `distribute_mg_dofs()` call) — resolved or
  superseded; see git history if needed.

---

## 6. Reproduction

```sh
# build and test
git submodule update --init fmt
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
(cd build && OMP_NUM_THREADS=1 ctest)   # 10/10

# reference-trajectory checks (needs a RMO-continuous-cuts checkout; torch + numpy only,
# scikit-image is stubbed)
python3 test/cc/compare_reference.py /path/to/RMO-continuous-cuts --every 50

# synthetic-disk CPU-time comparison (§4.1)
DEAL_II_NUM_THREADS=1 ./build/main_cc_bench

# cow-image problem (§4.2, §4.3) -- generates ~13 MB of raw rho data, gitignored
python3 test/cc/prepare_cow_data.py /path/to/RMO-continuous-cuts
DEAL_II_NUM_THREADS=1 ./build/main_cc_cow          # single- vs two-level, ~2 min
DEAL_II_NUM_THREADS=1 ./build/main_cc_cow_levels   # 3-/4-level, ~5 min
```
