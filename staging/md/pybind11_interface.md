# Design: a pybind11 interface for `main` / `main_coarse`

Design document only; no code has been changed. It is based on the state of `main` on
2026-10-07.

## 1. Starting point

- **What both programs do:** parse options into structs, choose dimension and system type at
  runtime (`with_dimension`, `with_system`), build a discretization, and run gradient descent
  (`main`, `SingleLevelExperiment`) or FAS (`MultiLevelExperiment`). The output is a convergence
  table on `std::cout`, plus `.bin`/`.vtk` files.
- **Three properties of the C++ code shape the interface:**
  1. **Templates everywhere:** dim ∈ {1,2,3} × System ∈ {consistent, lumped} gives 6
     instantiations. `FullApproximationScheme::cycle` is additionally templated on the oracle
     types, so the metric (mass/Frobenius) is fixed at compile time. Python can't instantiate
     templates, so the choice has to be type-erased in C++.
  2. **References instead of ownership:** `GradientDescent` holds `OracleBase&`, `ResidualBase&`
     and `ManifoldBase&` (it even deletes the rvalue constructors). The functional holds
     `System&`, oracles hold `Functional&`, and `UnitMassSphere` and the metrics hold
     `const Operator&`. If these objects were bound individually, Python would have to manage the
     lifetimes of a whole object graph with `keep_alive`, which is error-prone.
  3. **deal.II types** (`Vector`, `SparseMatrix`, `DoFHandler`, `AffineConstraints`,
     `MGLevelObject`) shouldn't appear in Python. Data crosses the boundary as NumPy/SciPy arrays.

**Conclusion:** bind a few owning facade classes, not the building blocks.

## 2. Architecture

```
Python  rmo (pybind11 module)
          ├── options: GPEOptions, DescentOptions, SolverOptions, FASOptions,
          │            CoarseModelOptions, OutputOptions
          ├── enums:   Potential, BoundaryCondition, MeshKind, Ordering, SolverMethod,
          │            Precondition, MetricKind, Transport, Interpolate, InitialValue
          ├── Discretization  (mesh, DoFs, matrices; inspection)
          ├── SingleLevel     (gradient descent)   ┐ type-erased facades,
          ├── MultiLevel      (FAS)                ┘ own the whole object graph
          └── Result, CycleInfo, observer callback
C++     rmo/gpe/experiment.h  (new; experiment classes moved out of main_coarse.cc)
          └── ExperimentBase (virtual)  <-  Experiment<System>  for 6 instantiations
```

- **Prerequisite refactoring:** move `SingleLevelExperiment`, `MultiLevelExperiment`,
  `build_transfers`, `build_cond_metric` and `output_bin` out of `main_coarse.cc` into a header
  (e.g. `rmo/gpe/experiment.h`). Then the programs and the bindings use the same code.
  `main.cc`'s `solve()` is the single-level case of `SingleLevelExperiment` and can be replaced
  by it.
- **Type erasure:** a virtual base class with what Python needs, implemented by
  `Experiment<System>`. A factory chooses the instantiation from `options.dimension` and
  `options.mass_lumping`, using the existing `with_dimension`/`with_system`:

  ```cpp
  class ExperimentBase {
  public:
      virtual ~ExperimentBase() = default;
      virtual unsigned n_dofs() const = 0;
      virtual Vector<double> initial_value(InitialValue) const = 0;
      virtual Result run(const Vector<double>& x0, MetricKind metric) = 0;
      // coordinates, weights, matrices
      virtual DiscretizationData discretization(unsigned level) const = 0;
  };
  std::unique_ptr<ExperimentBase>
  make_single_level(const PotentialSpec&, unsigned level, const GPE_Options&,
                    const SolverOptions&, const DescentOptions&);
  std::unique_ptr<ExperimentBase>
  make_multilevel(const PotentialSpec&, std::vector<unsigned> levels, ...);
  ```

  That's 6 instantiations for single-level and 6 × 2 (mass, Frobenius) for FAS. Compiling them
  all in one module is slow, so split it over several translation units with explicit
  instantiations.

## 3. What to export, and what not

### Exported

- **Option structs:** `GPEOptions`, `DescentOptions` (with nested `ls`), `SolverOptions`,
  `FASOptions`, `CoarseModelOptions`, `OutputOptions` (from `option_types.h`).
  - Read/write fields, keyword constructors with the CLI defaults.
  - **Generated from the existing `BOOST_DESCRIBE_STRUCT` declarations** (§4), so they stay in
    sync with the C++ structs.
  - `MG_Options` isn't needed: the levels are an argument.
- **All enums** (from `option_types.h`, plus `InitialValue`): `py::enum_`, also generated from
  `BOOST_DESCRIBE_ENUM`.
- **`SingleLevel(potential, level, gpe, solver, descent)`**, with `.n_dofs`,
  `.initial_value(kind)` and `.run(x0=None, metric=ENERGY_ADAPTIVE)`.
  - From `SingleLevelExperiment` (`main_coarse.cc`) and `main.cc`'s `solve()`.
  - Replaces `main` and `main_coarse --metric none`.
- **`MultiLevel(potential, levels, gpe, fas, coarse_model, solver, descent)`**, with
  `.run(x0, metric)` and `.cycle_log`.
  - From `MultiLevelExperiment` and `FullApproximationScheme`.
  - Replaces `main_coarse --multilevel`.
- **`Result`:** `x` (ndarray), `history` (list of ndarrays, or `iterates(every=k)`), `table`
  (list of `CycleInfo`, or `to_dicts()` for pandas), `iterations`, `converged`.
  - From `GradientDescent::history`, `FAS::history` and `CycleInfo`.
  - Replaces `--output-bin` and the table on stdout.
- **`CycleInfo`** (`solver.h`): fields from `BOOST_DESCRIBE_STRUCT`, `extra` as a dict.
- **Observer callback** `run(..., callback=lambda info: ...)`: a trampoline class
  (`PYBIND11_OVERRIDE`) for `IterationObserver<CycleInfo>`, for live progress or early analysis.
- **`Discretization(gpe, level)`**, with `.coords`, `.weights`, `.constrained`, `.n_cells` and
  `.matrices()` returning `M`, `S`, `A0` as SciPy CSR matrices (built from the CSR arrays, one
  copy).
  - From `GrossPitaevskiiPackage`, `fe::assemble_*` and `make_sparsity_pattern`.
  - Replaces `main_export`.
- **`write_vtk(result, name, every)`, `write_bin(result, name)`:** from `util.h` `output_vtk` and
  `main_coarse.cc` `output_bin`; replace `--output-vtk`, `--output-every` and `--output-bin`.
- **Optional: `Functional(...)`** with `value`, `gradient`, `residual`, `update`, and
  `mass_metric`/`energy_metric` norms.
  - From `GrossPitaevskiiFunctional`, `GrossPitaevskiiResidual` and `gpe/metric.h`.
  - For experiments in Python; keep it **inside the facade** (e.g. `SingleLevel.functional`) so
    the references stay valid.

### Not exported, with reasons

- **`rmo/fe`:** `assemble_system` and all `assemble_*` kernels, `make_nodal_quadrature`,
  `FeSpace`/`FeSpaceMG`, `renumber_dofs*`, `LinearTransfer*`.
  - Templated on `dim` and matrix types, and they work on deal.II objects. Only their results
    (matrices, weights) are useful in Python, via `Discretization`.
- **`rmo/gpe`:** `kernels.h` (`grad_mass`, `coarse_*`), `GrossPitaevskiiSystemBase`, the coarse
  oracles, `MassTransfer`, all transport classes, `ManifoldTransfer`, `UnitMassSphere`.
  - Internal building blocks with reference members. They're selected through enums
    (`CoarseModelOptions.transport_t` …), as the CLI already does.
- **`rmo/ropt`:** `OracleBase`, `CoarseOracleBase`, `GradientDescent` and
  `FullApproximationScheme` themselves, `ManifoldBase`, `VectorTransportBase`, `LevelMetric`,
  `MGLevelObject`.
  - Bound individually, they'd force Python to manage lifetimes, and FAS can't be instantiated
    from Python. The facades own them.
- **`rmo/lac.h`:** `LinearCombination`, `PreconditionInverse`, `DiagonalInverse`,
  `InverseMatrix`, `SpdNorm`.
  - Solver internals. Their observable effects go through `SolverOptions` and
    `GradInfo`/`CycleInfo`.
- **`rmo/util`:** `with_dimension`, `with_system`, `string_to_enum`, `dump_options`.
  - Replaced by the factory and by `py::enum_`; `dump_options` becomes `repr()`.

**Rule of thumb:** export what currently travels across the command line (options, the choice of
program) or ends up in files (iterates, table, matrices). Anything that only wires up the object
graph stays in C++.

## 4. Details

- **Bindings generated from Boost.Describe.** `option_types.h` and `option.h` already describe
  every option struct and enum. A small helper generates the bindings:

  ```cpp
  template <class T> void bind_struct(py::module_& m, const char* name) {
      using namespace boost::describe;
      py::class_<T> c(m, name);
      c.def(py::init<>());
      boost::mp11::mp_for_each<describe_members<T, mod_public>>(
          [&](auto D) { c.def_readwrite(D.name, D.pointer); });
  }
  template <class E> void bind_enum(py::module_& m, const char* name) {
      py::enum_<E> e(m, name);
      boost::mp11::mp_for_each<boost::describe::describe_enumerators<E>>(
          [&](auto D) { e.value(D.name, D.value); });
  }
  ```

  - Fields still missing from the descriptions (`mass_lumping`, `potential`, `smooth_t`, …)
    need adding there; that also makes `dump_options` complete.
  - The defaults come from `*_cli_options()` today. They should move into default member
    initializers in the structs, so the CLI and Python share one source.
- **Potential:** a value of `Potential` plus parameters (`potential_expr`, and later the planned
  `lattice_amplitude`/`potential_value`), packed into `PotentialSpec`.
  - A Python callable V(x) is possible but expensive: assembly evaluates V at every quadrature
    point, one call each, with the GIL held. That's fine for one-off assembly on small meshes.
  - For larger meshes, a vectorized variant (evaluate V once on all quadrature points as an
    `(n, dim)` array) would need an extension to the assembly; that's a phase 2 item.
- **Vectors:**
  - Input `x0`: a NumPy array (converted with `py::array_t<double, c_style | forcecast>`) is
    copied into a `Vector<double>`, with the length checked against `n_dofs`.
  - Output: `x` and the iterates as new arrays (copied). Zero-copy via the buffer protocol is
    possible, but would make the arrays depend on the `Result` object's lifetime, which isn't
    worth it.
  - Matrices: `scipy.sparse.csr_matrix((data, indices, indptr))` built from `SparseMatrix`
    (one copy).
- **Errors:**
  - At module import, call `dealii::deal_II_exceptions::disable_abort_on_exception()`.
    Otherwise a failing `Assert` in a debug build aborts the whole Python process, instead of
    raising a `RuntimeError` (deal.II's exceptions derive from `std::exception`, which pybind11
    translates).
  - `AssertThrow` (CG non-convergence, invalid options) already throws.
- **GIL and output:**
  - `run()` releases the GIL (`py::gil_scoped_release`) and only reacquires it when calling a
    Python callback.
  - The convergence table currently goes to an `std::ostream&`. In the facade, pass an
    `ostringstream` (`Result.table_text`) or wrap it in `py::scoped_ostream_redirect`. The
    structured data comes from the `CycleInfo` list.
- **Build:**
  - pybind11 isn't installed. MacPorts Python is 3.14 (`/opt/local/bin/python3`), so you'd need
    `py314-pybind11` or `pip install pybind11`.
  - deal.II is a shared library (`libdeal_II.9.7.1.dylib`), so it can be linked into a module.
  - CMake: `find_package(pybind11 CONFIG)`, `pybind11_add_module(rmo bindings/*.cc)`,
    `deal_ii_setup_target(rmo)`, with the rpath set to deal.II's libraries.
  - The `-openmp-simd` flag from `CMakeLists.txt:110` (it is parsed as `-o penmp-simd`)
    shouldn't be carried over.

## 5. Mapping CLI → Python (examples)

- `main --levels 5 --boundary dirichlet --mass-lumping`:

  ```python
  gpe = rmo.GPEOptions(bc=rmo.BoundaryCondition.DIRICHLET, mass_lumping=True)
  rmo.SingleLevel(rmo.Potential.SQUARE, 5, gpe, slv, gd).run()
  ```

- `--levels 4,5,6` (the loop in `main`):
  `[rmo.SingleLevel(pot, L, gpe, slv, gd).run() for L in (4, 5, 6)]`
- `main_coarse --multilevel 7,6,5 --metric mass --transport adjoint_restriction`:

  ```python
  cm = rmo.CoarseModelOptions(metric_t=rmo.MetricKind.MASS,
                              transport_t=rmo.Transport.ADJOINT_RESTRICTION)
  rmo.MultiLevel(pot, [7, 6, 5], gpe, fas, cm, slv, gd).run(metric=rmo.MetricKind.MASS)
  ```

- `--initial random`: `exp.run(x0=exp.initial_value(rmo.InitialValue.RANDOM))`, or a custom
  `x0` array.
- `--output-bin=name`: `result.history` / `result.x` (in memory), or
  `rmo.write_bin(result, "name")`.
- `--output-vtk=name --output-every 5`: `rmo.write_vtk(result, "name", every=5)`.
- `main_export --matrices`: `rmo.Discretization(gpe, level).matrices()`.
- The table on stdout: `pd.DataFrame(result.table_dicts())`.

## 6. Suggested order

1. Move the experiment classes into `rmo/gpe/experiment.h`, add `ExperimentBase` and the
   factory. That's useful even without Python: `main.cc` would shrink to the single-level case.
2. Complete the option defaults in the structs and the `BOOST_DESCRIBE` declarations.
3. Module with options, enums, `SingleLevel`, `Result`/`CycleInfo`. Check: identical energies
   and tables to `main_coarse --metric none`.
4. `MultiLevel`, and the observer callback.
5. `Discretization` (replaces `main_export`), plus `verify_solutions.py` rewritten on top of the
   module, without subprocesses.
6. Optional: the functional/metric interface, and vectorized Python potentials.
