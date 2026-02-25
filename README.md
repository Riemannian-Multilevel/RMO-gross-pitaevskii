# GPE Discretization Architecture

This project implements a high-performance solver for the Gross-Pitaevskii Equation (GPE). The codebase follows a tiered architecture to maintain a strict Separation of Concerns between geometry, algebra, and optimization.

---

## Design Philosophy

### 1. Tiered Responsibility
The system is divided into three primary layers:

| Layer | Component | Responsibility |
| :--- | :--- | :--- |
| **Discretization** | `GrossPitaevskiiPackage` | The "Where" and "How": Handles mesh generation, refinement, FE space distribution, and geometric mapping (Simplex vs. Quad). |
| **Algebraic** | `GrossPitaevskiiProblem` | The "What" and "Physics": A container for matrices (A0, M, Mpp). Manages the iterative assembly of physical operators. |
| **Optimization** | `EnergyOracle` | The "Math": Bridges the physics to the Riemannian Gradient Descent algorithm. |

### 2. Efficiency in Multi-Physics
In complex simulations, the triangulation and FE space often remain constant while physical parameters (like the potential V or the coupling constant beta) change.
* Advantage: You can instantiate one `GrossPitaevskiiPackage` and generate multiple `GrossPitaevskiiProblem` instances from it.
* Resource Savings: This avoids redundant mesh generation, refinement, and sparsity pattern computation.

---

## Implementation Details

### Lazy Assembly with mutable
The non-linear term `Mpp` is marked as mutable within the `GrossPitaevskiiProblem` class. This allows the `EnergyOracle` to remain logically constant.

* Purpose: In GPE, `Mpp` must be recomputed whenever the solution density `|phi|^2` changes.
* Mechanism: the Oracle can trigger `assemble_nonlinear_term(x)` inside its `initialize()` method even when holding a `const` reference to the problem. 
This keeps the API clean while ensuring the physics are updated before every gradient calculation.

---

## Simulation Workflow

The `EnergySimulator` serves as the orchestrator for the entire lifecycle:

1. Initialize Package: Set up the mesh and FE space.
2. Generate Problem: Assemble the stationary matrices (`A0` and `M`).
3. Setup Oracle: Link the matrices and the coupling constant `beta`.
4. Descent: Execute the `gradient_descent` loop until the residual and eigenvalue (`lambda`) converge.

---

## Related Components

* `LinearCombination`: Efficiently handles `A = A0 + beta * Mpp` without full matrix-matrix addition.
* `InverseMatrix`: A wrapper for iterative solvers (CG, GMRES, MINRES) used in gradient computation.