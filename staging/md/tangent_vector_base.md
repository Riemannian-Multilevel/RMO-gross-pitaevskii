# Plan: `VectorBase` and `TangentVector` for the `ropt/` layer

Goal: a switchable vector interface (`VectorBase`, with `dealii::Vector` as one implementation) and
a distinct tangent vector type, so that point and tangent arguments of the Riemannian interfaces
cannot be swapped.

## What the code shows

- **The boundary is clean.** Every `ropt/` interface (`OracleBase`, `ManifoldBase`, `MetricBase`,
  `ResidualBase`, the transfer and transport bases, `IterationBase`) takes
  `dealii::Vector<double>`. The `gpe/` code, meaning `kernels.h`, the `ellipsoid::` and `metric::`
  functions, `lac.h` and the Krylov solvers, is deal.II-specific by nature and doesn't need to be
  abstracted.
- **Argument order is inconsistent, which is the bug to guard against.** The tangent vector comes
  first in `retract(z, x)` and `retract_inv(v, x)`, but the point comes first in `retract_diff(x, v,
  w)`.  `vector_prolongation` takes `(x_fine, y_coarse, …)` while `vector_restriction` takes
  `(y_coarse, x_fine, …)`.
- **One vector changes type in place.** At `fas.h:274`, `zk` is a coarse point, and `retract_inv(zk,
  state.y)` overwrites it with a tangent vector at `y`. Distinct types make that impossible, so this
  call becomes out-of-place.
- **Few calls go through the interface.** `retract_inv` is the only retraction called through
  `ManifoldBase` (`fas.h:274`). `retract_diff` and `retract_inv_diff` are only reached through
  `ellipsoid::` directly.  `MetricBase::apply` is used once, at `gpe/residual.h:151`.
- **deal.II ≥ 9.6 is required.** deal.II itself appears to have deprecated its own virtual vector
  base (`LinearAlgebra::VectorSpaceVector`) in favour of templates (not verified). This doesn't
  argue against a virtual `VectorBase`: the design below keeps virtual calls to whole-vector
  operations only, never per element.

## Design

### 1. `VectorBase` (new `include/rmo/vector.h`)

A virtual interface with only the whole-vector operations the `ropt/` layer needs. There is no
virtual `operator[]`.

```cpp
class VectorBase {
public:
    virtual ~VectorBase() = default;
    [[nodiscard]] virtual std::unique_ptr<VectorBase> clone() const = 0;       // deep copy
    [[nodiscard]] virtual std::unique_ptr<VectorBase> clone_zero() const = 0;  // same layout, zeroed
    [[nodiscard]] virtual std::size_t size() const = 0;

    virtual void set_zero() = 0;
    virtual void equ(double a, const VectorBase& u) = 0;             // *this = a u
    virtual void add(double a, const VectorBase& u) = 0;             // *this += a u
    virtual void sadd(double s, double a, const VectorBase& u) = 0;  // *this = s *this + a u
    virtual void scale(double s) = 0;
    [[nodiscard]] virtual double dot(const VectorBase& u) const = 0;  // Euclidean
    [[nodiscard]] virtual double l2_norm() const = 0;
    [[nodiscard]] virtual double min() const = 0;                     // CycleInfo::min_value
};
```

- **Backend adapter.** `template <typename V> class DealiiVector final : public VectorBase` owns a
  `V` and exposes `V& data()`. One adapter covers `dealii::Vector<double>` today and
  `LinearAlgebra::distributed::Vector` or the Trilinos/PETSc vectors later, since they share the
  same API.
- **Unwrap helper.** `as_dealii<V = Vector<double>>(VectorBase&)` uses `static_cast` in release
  builds and `dynamic_cast` plus `Assert` in debug builds. Mixing backends fails in debug builds
  rather than silently.
- **Cost.** One virtual call per whole-vector operation is negligible at the problem sizes used
  here. The `gpe/` kernels already allocate their temporaries, so allocation doesn't change.

### 2. `TangentVector`: built by composition, not inheritance

A concrete, `final`, value-semantic wrapper rather than a `TangentVectorBase : VectorBase`. With
inheritance, a tangent vector converts implicitly to `const VectorBase&`, so passing a tangent where
a point is expected still compiles. The type check only works if no implicit conversion exists in
either direction.

```cpp
class TangentVector final {
public:
    explicit TangentVector(std::unique_ptr<VectorBase> v);
    static TangentVector zero_at(const VectorBase& x);   // x.clone_zero()
    TangentVector(const TangentVector&);                 // clone
    TangentVector& operator=(const TangentVector&);
    TangentVector(TangentVector&&) = default;

    // Closed under linear combination within one tangent space
    void equ(double a, const TangentVector& u);
    void add(double a, const TangentVector& u);
    void sadd(double s, double a, const TangentVector& u);
    void scale(double s);
    std::size_t size() const;

    // Explicit unwrapping, for implementations only
    VectorBase&       vector();
    const VectorBase& vector() const;
private:
    std::unique_ptr<VectorBase> m_v;
};
```

- **No `dot()` on tangent vectors, on purpose.** Inner products of tangent vectors must go through
  `MetricBase::inner`, so a tangent vector can't accidentally get a Euclidean inner product.
- **Points stay `VectorBase`.** The manifolds are embedded, and inside `ropt/` every non-tangent
  vector is a point.  Wrapping points as well only adds checking against ambient vectors (like `Mx`
  or the residual vector), which never cross `ropt/` interfaces. Revisit this if they ever do.
- **The base point is not stored.** Pointer identity breaks on copies (`zk` against `state.y`, for
  example).  Instead, add an optional `virtual void ManifoldBase::assert_tangent(const VectorBase&
  x, const TangentVector& v) const`. Its default does nothing; `UnitMassSphere` checks |xᵀMv| ≤
  tol·‖v‖. Call it under `Assert` at the boundaries of the solver and the transports.

### 3. Interface changes

| Interface | New signature |
|---|---|
| `ManifoldBase` | `retract(const TangentVector& z, VectorBase& x, double h)`<br>`retract(const TangentVector& z, const VectorBase& x, VectorBase& out, double h)`<br>**`retract_inv(const VectorBase& z, const VectorBase& x, TangentVector& out)`**: out-of-place<br>`retract_diff(const VectorBase& x, const TangentVector& v, const TangentVector& w, TangentVector& out)`<br>`retract_inv_diff(x, zeta, const TangentVector& u, TangentVector& out)` and its `_adjoint`<br>new: `create_vector()`, `assert_tangent()` |
| `MetricBase` | `inner(const TangentVector&, const TangentVector&)`, `norm(const TangentVector&)`, `apply(VectorBase& dst, const TangentVector& src)` |
| `OracleBase` | `update(const VectorBase&)`, `value(const VectorBase&)`, `directional_derivative(const VectorBase& x, const TangentVector& z)`, `gradient(const VectorBase& x, TangentVector& out[, double])` |
| `IterationBase` | `shared_ptr<const VectorBase>`, `directional_derivative(const TangentVector&)`, `gradient(TangentVector&)` |
| `ResidualBase` | `residual(const VectorBase&)` |
| `ManifoldTransferBase` | `restriction` and `prolongation` on `VectorBase`; `diff_*` take and return `TangentVector` |
| `VectorTransportBase` | both functions take `(x_fine, y_coarse, const TangentVector& v, TangentVector& dst)`, so **the point arguments come in the same order in both**. Types can't catch a fine/coarse swap, but the existing `AssertDimension` checks do. |
| `LinearTransferBase` | `const VectorBase&` → `VectorBase&`; it acts on ambient vectors and is applied to both points and unwrapped tangent vectors |
| `CoarseState` | `x` and `y` become `unique_ptr<VectorBase>`; `y_grad`, `x_grad`, `x_grad_restr` and `w` become `TangentVector`. It is built from prototype vectors rather than `(n_fine, n_coarse)`. |
| `SolverBase::cycle`, `x_hist` | `VectorBase&`; the history becomes `std::vector<std::unique_ptr<VectorBase>>` |

`ManifoldBase::create_vector()` is the allocator. The manifold knows its embedding space, and for
distributed vectors it will also know the partitioner. That replaces `Vector<double>(n)` in `ropt/`,
which can't know the backend.

### 4. `gpe/` side: unwrap once, at the override

The virtual overrides convert the arguments and call the existing kernels unchanged:

```cpp
void retract(const TangentVector& z, VectorBase& x, double h) const override {
    ellipsoid::retract_by_norm(M, as_dealii(z), as_dealii(x), h);
}
```

The `ellipsoid::` and `metric::` namespaces, `kernels.h`, `lac.h`, `LinearCombination`, the
`*Inverse` classes, `fe/`, and output and serialization all stay on
`dealii::Vector<double>`. `OperatorMetric<Operator>` unwraps in `inner` and `apply`.

## Phases

Each phase should leave everything compiling and `ctest` passing.

1. **Add the types.** Add `vector.h` with `VectorBase`, `DealiiVector<V>`, `as_dealii` and
   `TangentVector`. Add a unit test for clone semantics, the mixed-backend assertion, and a check
   that `TangentVector` and `VectorBase` don't convert into each other
   (`static_assert(!std::is_convertible_v<…>)`). Nothing else changes.
2. **Leaf interfaces.** Port `ResidualBase`, `MetricBase` and `LinearTransferBase`, plus their
   implementations in `gpe/residual.h`, `ropt/metric.h` and `fe/interpolate.h`.
3. **`ManifoldBase` and `UnitMassSphere`.** Add `create_vector()` and `assert_tangent()`, and make
   `retract_inv` out-of-place.
4. **`OracleBase` and `IterationBase`, with all oracles.** This covers `gpe/oracle.h`,
   `oracle_coarse.h` and `iteration.h`, plus `CoarseState` and `CoarseOracleBase`. It is the largest
   mechanical step: about 15 oracle classes, each with roughly 5 overrides.
5. **Transfers and transports.** Port the `ManifoldTransfer` and `*Transport` classes and unify the
   argument order.
6. **Solvers.** Port `armijo_line_search`, `cycle_smooth`, `cycle_eval`, `GradientDescent` and
   `FullApproximationScheme`.
   - `armijo_line_search` can drop its `step` copy by using `retract(eta, x, x_trial, alpha)`.
   - The FAS block at `fas.h:263-279` becomes:
     ```cpp
     auto zk = state.y->clone();
     cycle(…, *zk, …);
     TangentVector eta = TangentVector::zero_at(*state.y);
     manifold.retract_inv(*zk, *state.y, eta);
     vector_transport.vector_prolongation(*state.x, *state.y, eta, dk);
     ```
7. **Drivers and tests.**
   - `src/main*.cc` and `staging/test/{arpack,gradient}.cc` construct a
     `DealiiVector<Vector<double>>` and use `.data()` for `output_vtk`, `output_bin` and the ARPACK
     comparisons.
   - `gradient_problems.h` and `finite_difference.h` will need a tangent overload.
   - Check that the convergence tables are bit-identical to the reference runs in `data/org` for at
     least one SL case and one ML case (for example `sl_b100_l10` and
     `ml_mass_proj_b100_l10_depth2`).

Phases 2–6 can't be split across commits cleanly without temporary dual overloads, because each
interface is used by the next one. Practically, it's one branch with a commit per phase, using
deprecated `Vector<double>` forwarding overloads only where they keep `main_*.cc` compiling in
between.

## Open points

- **`assert_tangent` tolerance.** For the energy-adaptive and coarse gradients, tangency holds only
  up to round-off times the condition number. Start with a loose tolerance (1e-8 relative) in debug
  builds only, and tighten it if it never fires.
- **`min()` on `VectorBase`.** It's GPE-flavoured: it's there for positivity of the ground
  state. The alternative is to move `min_value` into a problem hook on `ResidualBase` or the
  observer and keep `VectorBase` purely about vector spaces. Leaning towards the hook.
- **Templating `gpe/` on the vector type is out of scope.** With `DealiiVector<V>` in place,
  templating `gpe/` on `V` (for `distributed::Vector` or MPI) would be a later, separate step.
