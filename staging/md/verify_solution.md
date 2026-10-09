# Design: verifying consistent and mass-lumped solutions (`proto/verify-solution`)

Branch `proto/verify-solution`, 5 commits on top of `efc7292` (state of 2026-10-07):

| Commit | Content |
|---|---|
| `c731b4b` | `NumberGenerator`, `--initial (constant\|random)`, `--seed` |
| `6847a38` | `--initial cosine`: cosine bump of radius `--initial-arg` |
| `5aac87d` | `staging/main_export.cc`: coordinates, lumped weights, constraints, matrices |
| `c541471` | `staging/verify_solutions.py`: checks of consistent and lumped ground states |
| `7ff57e5` | `verify_solutions.py`: rename `--initial bump` to `cosine` |

## 1. Motivation

- Solutions of the consistent and the mass-lumped discretization (`--mass-lumping`) should be
  checked against each other, against known solutions, and for internal consistency, for the∑
  potentials `zero`, `square`, `optical_lattice` and nonlinearities β ∈ {0, 100, 10000}.
- With the default boundary (Neumann) and V = 0, the old fixed starting value x₀ = 1 is the
  exact ground state: S·1 = 0 and M_φφ(x)x = c²Mx for x = c·1, so the residual at x₀ is about
  1e-16 and the solver stops before the first step. A verification needs other starting values.
- Exact checks (normalization xᵀMx = 1, energy, residual) need the matrices and lumped weights,
  which the solver programs don't write. They are exported by a separate program, so the solvers
  stay free of export code.

## 2. Starting value: `--initial`, `--initial-arg`, `--seed`

- `InitialValue` (`option_types.h`), `GPE_Options::initial`, `initial_arg` (`std::optional`) and
  `seed`; all three are in `BOOST_DESCRIBE_STRUCT`.
  - `constant`: x = value, default 1 (the previous behaviour).
  - `cosine`: `CosineBump(r)`, u₀(x) = Π_d cos(πx_d / 2r) for |x_d| < r and 0 otherwise; r
    defaults to the domain radius. Interpolated with `VectorTools::interpolate`.
  - `random`: uniform in [0.5, 1.5].
- `--initial-arg` is parsed strictly (`std::stod` must consume the whole string) and rejected for
  `random`, whose only parameter is `--seed`.
- `ModelBuilder::initial_value()` builds the vector (without constraints and normalization); the
  experiments in `main_coarse.cc` and `main.cc` call it, then `distribute` and normalize as before.
- `NumberGenerator` (`util/random.h`) replaces the free functions `normrnd`/`unifrnd`/`randi` and
  the per-thread engine: a singleton around one `std::mt19937`, seeded with `--seed` (default 42)
  at program start, so runs are reproducible. Not thread-safe; draw numbers from one thread only.
  All previous callers (manifold, metric, tests, benchmarks) use it.

## 3. `staging/main_export.cc`

- Same mesh options as the solvers (`gpe_cli_options()`, `--levels`/`--multilevel`), so the DoF
  numbering matches their `--output-bin` files. Per level L, base name `<name>_lvl<L>`:
  - `_coords.bin`: support points (`write_support_points`, format of `--output-bin`).
  - `_weights.bin`: lumped weights mᵢ = ∫φᵢ, assembled **without** constraints, so Dirichlet
    DoFs get their true ∫φᵢ instead of the artificial diagonal of constrained rows.
  - `_constrained.bin`: 1 for constrained (Dirichlet) DoFs, 0 otherwise.
  - With `--matrices`, in Matrix Market format (read with `scipy.io.mmread`): `_M.mtx`
    (consistent mass), `_S.mtx` (stiffness), `_A0.mtx` (S + M_V for `--potential`, with the
    lumped potential term under `--mass-lumping`).
- Uses `GrossPitaevskiiPackage::get_quadrature()`, so matrices are assembled with exactly the
  solvers' quadrature.
- Checked: weights sum to |Ω|; away from the boundary they equal the row sums of M and the rows
  of S sum to zero (≤ 9e-16); the off-diagonal part of the lumped A₀ equals S.

## 4. `staging/verify_solutions.py`

- For every potential × mass variant × level: one `main_export --matrices`; then for every β one
  `main_coarse --metric none` (single-level, energy-adaptive) with `--output-bin`, in its own
  directory, run in parallel (`--jobs`).
- Checks of the final iterate (required unless noted):

| Check | Method |
|---|---|
| converged | final residual ≤ `--tol-residual` |
| normalized | xᵀMx = 1, with the lumped or the consistent M |
| energy | ½xᵀA₀x + β/4 Σ mᵢxᵢ⁴ recomputed (lumped; consistent only for β = 0) |
| residual | M-norm of Ax − λMx recomputed, as in the program (same cases) |
| positive | min xᵢ ≥ −tol on free DoFs (required for lumped, reported for consistent) |
| symmetric | invariance under x_d → −x_d and swaps, nodes matched with `cKDTree` |
| reference | node error and energy error against a known solution (below) |
| ritz | E_consistent ≥ E_exact for `zero`, Dirichlet, β = 0 (exact quadrature) |

- References: `zero` (Dirichlet, β = 0): Π_d cos(πx_d/2R)/R^{dim/2}, E = dim/2·(π/2R)²;
  `square` (β = 0): π^{-dim/4} e^{−|x|²/2}, E = dim/2; `optical_lattice` (β = 0): separable,
  1D eigenfunction from `scipy.sparse.linalg.eigsh`; `zero` (Neumann, any β): constant,
  E = β/(4|Ω|).
- Across levels: observed rates of the energy self-differences, of the reference error, and of
  the consistent–lumped energy gap (expected 2). Output: `results.csv`, `rates.csv`,
  `report.md`, and SVG plots (solutions and differences per level, cut along x₂ = 0, energy
  convergence). Exit code 1 if a required check fails.
- Defaults: levels 5–7, Dirichlet, radius 8, `--initial random`, tight `--tol-inner-res`
  (inexact inner solves produce negative entries of order 1e-5 even for the lumped variant).

## 5. Results

Measured with the version of the branch before the rework of the starting value (random start
with a fixed seed 0 instead of `--seed 42`); converged energies are unaffected, iteration counts
may differ.

- 48 of 54 default cases pass. Normalization error ≤ 4e-16; recomputed energy and residual,
  symmetry, lumped positivity and the Rayleigh–Ritz bound hold.
- `square` and `zero` (β ≤ 100): energies, reference error and consistent–lumped gap converge
  at rate ≈ 2.0.
- `optical_lattice` (β = 0) is asymptotic only for h ≤ 0.125 (level ≥ 8 at radius 8): on levels
  7–9 both variants converge at rate 2 to E = 15.0806. On coarse meshes the nodal rule of the
  lumped potential term aliases the lattice (period 2, amplitude 100): at h = 1, E_lumped = 1.33.
- The 6 failures are `zero`, β = 10000: no convergence within 2000 iterations (residuals 4e-7 to
  2e-4), a convergence problem of the energy-adaptive method in this regime.
- Consistent solutions have negative entries on coarse meshes (down to −0.18 for the lattice),
  as expected without the M-matrix property.

## 6. Open issues

- `cosine` with r = R is the exact discrete solution for `zero` (Dirichlet, β = 0) on the uniform
  Q1 mesh: the verification converges in 0 iterations. Keep `random` as default.
- Consistent energy and residual for β > 0 can't be recomputed from matrices (the quartic term is
  assembled with quadrature); exporting the nonlinear term would close this gap.
- Normalization and symmetry use the exported data; meshes that are not symmetric skip the
  symmetry check (reported as `nan`).
- Neumann, V ≡ 0, β = 0 makes the energy-adaptive matrix singular; with a non-constant start CG
  diverges. A guard (error message or a shift A + εM) is not implemented.
- With the pybind11 interface (`doc/pybind11_interface.md`), the verifier could call the solvers
  directly instead of through subprocesses and files.
