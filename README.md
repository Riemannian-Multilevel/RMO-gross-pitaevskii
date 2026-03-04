# GPE Discretization Architecture

This project implements a high-performance solver for the Gross-Pitaevskii Equation (GPE). The codebase follows a tiered architecture to maintain a strict separation of concerns 
between geometry, algebra, and optimization.

---

## Design Philosophy

### 1. Tiered Responsibility
The system is divided into three primary layers:

| Layer | Component | Responsibility                                                                                                        |
| :--- | :--- |:----------------------------------------------------------------------------------------------------------------------|
| **Discretization** | `GrossPitaevskiiPackage` | Handles mesh generation, refinement, FE space distribution, and geometric mapping (Simplex vs. Quad).                 |
| **Algebraic** | `GrossPitaevskiiProblem` | A container for matrices (A0, M, Mpp). Manages the iterative assembly of physical operators.                          |
| **Optimization** | `EnergyOracle` | Bridges the physics to the Riemannian Gradient Descent algorithm.                                         |

### 2. Efficiency in Multi-Physics
The triangulation and FE space often remain constant while physical parameters (like the potential V or the coupling constant beta) change.
Keeping this in mind, you can instantiate one `GrossPitaevskiiPackage` and generate multiple `GrossPitaevskiiProblem` instances from it.

This avoids redundant mesh generation, refinement, and sparsity pattern computation.

---

## Implementation Details

### Lazy Assembly with mutable

All classes are const-correct, with the exception of `GrossPitaevskiiProblem`, which
includes `Mpp` as a `mutable` variable. This is because `Mpp` needs to be recomputed whenever
the solution density `|phi|^2` changes. Function and gradient evaluation
require `Mpp` to be kept updated at all times.

_Mechanism_: the Oracle can trigger `assemble_nonlinear_term(x)` inside its `update()` method even when holding a `const` reference to the problem. 
This keeps the API clean while ensuring the physics are updated before every gradient calculation.

---

## Simulation Workflow

The `EnergySimulator` serves as the orchestrator for the entire lifecycle:

* _Initialize package_: Set up the mesh and finite element space.
* _Generate problem_: Assemble the stationary matrices (`A0` and `M`).
* _Setup oracle_: Link the matrices and the coupling constant `beta` (`A = A0 + beta*Mpp`)
* _Descent method_: Execute `gradient_descent()` until the residual and eigenvalue (`lambda`) converge.

---

## Operator and Preconditioner Abstractions

We have the following classes for linear algebra operations:

* `LinearCombination`: Efficiently handles `A = A0 + beta * Mpp` without full matrix-matrix addition.
* `InverseMatrix`: A wrapper for iterative solvers (CG, GMRES, MINRES) used in gradient computation.

To maximize performance and minimize memory usage, the pipeline strictly separates the concepts 
of _operator_ and _preconditioner_.

**`OperatorType` (Matrix-Free):** The Krylov solvers (CG, GMRES) only require the ability to compute matrix-vector products. 
We pass a `LinearCombination` object as the `OperatorType`. This avoids the expensive allocation and assembly of a full 
$A = A_0 + \beta M_{pp}$ matrix, evaluating the sum on-the-fly via `vmult()`. 
This can be extended to a matrix-free implementation of the terms $A_0$ and $M_{pp}$.

**`MatrixType` (Explicitly Assembled):** Preconditioners like `SparseILU`, `PreconditionJacobi`, and `PreconditionSSOR`
require explicit access to matrix entries (e.g., extracting the diagonal or performing triangular sweeps). For these, 
we explicitly assemble and pass a `SparseMatrix<double>` as argument.

---

## TODOs

Evaluating the GP functional does not require any matrix inversion, or preconditioning.
For Armijo line search, this needs to be done many times per iteration.
Therefore, use a matrix-free approach for `EnergyOracle::value()` to save computation time.

Currently, `EnergyOracle::update()`, `EnergyOracle::value()` and `EnergyOracle::gradient()`
all have a vector argument `Vector<double> x`. For `value()` and `gradient()` to be correct,
`update()` must be called for the same argument. This can easily lead to inconsistencies.
An alternative is for `update()` to set two boolean flags - `needs_assembly` and `needs_gradient`.
`gradient()` can check these flags, assemble $A_x$ as-needed (for a **stored** vector),
and be called without argument.

For `EnergyOracle::gradient()`, which uses matrix inversion (`InverseMatrixType`)
some kind of preconditioning is useful. For an explicitly stored matrix, many
different choices are possible (Jacobi, SSOR, ILU, AMG, ...). For matrix-free
evaluation, we have *geometric multigrid* at our disposal. While working approaches
can be as effective as AMG (*algebraic multigrid*), an implementation requires
correct handling of boundary conditions. This is significantly more involved than
the black-box algebraic preconditioners mentioned above.

If we want to compare an "algebraic" coarse model, where the coarse mass $M_H$ and stiffness $S_H$
matrices arise from restriction $I_h^H$ and interpolation $I_H^h$ of "canonical" matrices $M_h$, $S_h$, we need
explicit access to restriction and interpolation. `deal.II` only provides said access
for an explicit geometric multigrid hierarchy, using the same mesh handler (`DoFHandler`.)
When using multiple meshes, arising through global refinement with independent `DoFHandler` objects,
only black-box interpolation and restriction is available. While this may still be sufficient
for matrix-vector products (i.e. $M_H v := I_h^H M_h I_H^h v$), without explicit matrix form
the behavior of the interpolators should be verified on test examples.

---

## Quick Start

To run a simulation, initialize the `EnergySimulator` with your discretization options and call `run` with your desired physical parameters.

```cpp
// Configure discretization (Mesh, FE Degree, etc.)
GPE_Options options;
options.beta = 100.0;
options.degree = 1;

// Configure Solver (Tolerances, Max Iterations)
GdOptions gd_options;
gd_options.tol_residual = 1e-4;
gd_options.step_size = 1.0;

// Initialize Simulator
unsigned int n_refinements = 8;
EnergySimulator<2> simulator(options, n_refinements);

// Execute solve with a Square potential
auto solution = simulator.run(Square<2>(), gd_options, options.beta);