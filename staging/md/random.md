# Plan: random number generation

Goal: replace the `NumberGenerator` singleton of `include/rmo/util/random.h` with explicitly passed `Random`
instances, so that the numbers drawn at one site no longer depend on draws elsewhere in the program.

## How `util/random.h` works today

`NumberGenerator` is a singleton holding one `std::mt19937`. It starts seeded from `std::random_device`, and each
driver reseeds it with `--seed` (default 42). The distribution parameters are passed on every call
(`normrnd(mean, stddev, …)`, `unifrnd(a, b, …)`), and a new `std::*_distribution` is built each time.

There are five draw sites (`randi`, `rand_sparse_vec_idx` and the scalar `normrnd`/`unifrnd` overloads are never
called):

| Site | Draws |
|---|---|
| `gpe/manifold.h:32` `ellipsoid::random_point` | `normrnd` over a vector |
| `gpe/metric.h:90, 124, 172` `random_tangent_vector` (energy, mass, Frobenius) | `normrnd` over a vector |
| `gpe/model.h:100` `initial_value()` with `--initial random` | `unifrnd(0.5, 1.5)` |

Six places call `seed()`: `src/main.cc:102`, `main_matrix.cc:188`, `main_bench.cc:71`, `main_coarse.cc:464`,
`staging/test/gradient.cc:72` and `arpack.cc:81`.

## Distribution parameters stay per call

The TODO in `random.h` asks whether mean and standard deviation should become constructor arguments. Keep them per
call, and delete the TODO:

- **Parameters belong to the draw, not the generator.** The initial value uses U(0.5, 1.5) and the test
  perturbations use N(0, 1). If the singleton held a single distribution, those call sites would overwrite each
  other's parameters.
- **Constructing a distribution is cheap.** Its state is essentially its parameters. The only cost is that
  `std::normal_distribution` caches a second value, which a fresh object throws away. That only affects the unused
  scalar `normrnd`.
- **Callers that want fixed parameters can own the distribution:**
  `std::normal_distribution d(0, 1); d(rng.engine());`.

The real problem is the singleton, not the parameters.

## A distribution options class, by analogy with `dealii::SolverControl`

`SolverControl` separates the configurable knobs (`max_steps`, `tolerance`, logging flags) from the solver that uses
them. One generic solver runs with any control object, and the control can be filled from a `ParameterHandler`
(`declare_parameters`/`parse_parameters`). It also records the state of the last solve (`last_step()`,
`last_value()`).

The analogue here: `Random` plays the role of the solver (the engine), and a distribution object carries the knobs of
one draw. Like a `SolverControl`, the distribution object does not own the engine, and the seed stays with `Random`:
the seed belongs to the consumer, the distribution to the draw.

What is configurable today is hard-coded at the draw sites: U(0.5, 1.5) in `initial_value()` (`model.h:100`), and
N(0, 1) as default arguments of `random_point` and `random_tangent_vector`. `--initial-arg` is explicitly rejected for
`--initial random` (`option.h:308`). A distribution object is what makes these configurable without new parameters
on every kernel.

### Possibilities

**A. Option struct in the style of `SolverOptions`.**

```cpp
enum class Distribution { NORMAL, UNIFORM };

struct DistributionOptions
{
    Distribution kind;  // which distribution
    double a;           // NORMAL: mean,   UNIFORM: lower bound
    double b;           // NORMAL: stddev, UNIFORM: upper bound
};
BOOST_DESCRIBE_STRUCT(DistributionOptions, (), (kind, a, b));
```

Matches the existing `*Options` structs, `BOOST_DESCRIBE` and the `po::` parsing in `option.h`. The drawback is that
`a` and `b` mean different things per `kind`, and nothing stops a negative standard deviation or `a > b` before the
draw.

**B. A variant of typed parameter structs.**

```cpp
struct Normal  { double mean = 0.0; double stddev = 1.0; };
struct Uniform { double a = 0.0;    double b = 1.0; };
using Distribution = std::variant<Normal, Uniform>;

class Random
{
public:
    template <typename RangeType>
    void fill(const Distribution& dist, RangeType& vec);   // std::visit to the matching std:: distribution
    // normrnd() and unifrnd() remain as shorthands for fill(Normal{…}, vec) and fill(Uniform{…}, vec)
};
```

Each field has one meaning, the constructors can check `stddev > 0` and `a < b`, and adding a distribution (log-normal,
for instance) is a new alternative that `std::visit` forces every consumer to handle. Kernels take
`const Distribution&` instead of `(mean, stddev)`, so `random_point` is no longer tied to a normal distribution. The
standard library already has `std::normal_distribution<>::param_type` and friends, but own structs give named fields
and keep the interface unchanged if the Boost distributions (phase 6) replace the `std::` ones.

**C. A stateful control, like the state half of `SolverControl`.** It would record what was drawn, for example the
number of values or the min and max of the last fill. There is nothing to converge, so `last_step()`/`last_value()`
have no real counterpart. The only useful piece of state is the number of values drawn, for debugging
reproducibility, and that fits in `Random` itself if it's ever needed. Not recommended.

**D. The distribution stored inside `Random`.** This is the TODO in `random.h`. Rejected above: one instance could no
longer serve draws with different parameters, and with one instance per consumer it would gain nothing over passing
the distribution.

**E. A `RandomBase` hierarchy: distribution classes with fixed parameters.** Kernels such as
`random_tangent_vector` take a single object that knows its distribution, and the caller picks the subclass:
`NormalRandom`, `UniformRandom`, and so on. This follows the `*Base` style of `MetricBase` and `ManifoldBase`.

The base must *reference* a shared engine, not own one. If `RandomBase` owned the engine, a consumer that needs two
distributions would need two engines:

```cpp
NormalRandom  normal(seed, 0.0, 1.0);
UniformRandom uniform(seed, 0.5, 1.5);   // same seed → same mt19937 output
```

Both objects would compute their values from the same stream of engine output, so the normal and uniform draws would
be strongly correlated. Every consumer would then have to derive a separate seed per distribution with `seed_seq`,
and the "one seed per consumer" rule would no longer be enough. Copying is the other trap: copying an object that owns
its engine copies the engine state, so the copy silently repeats the original's sequence. This is the reason the
standard library keeps engines and distributions separate.

With a reference to a shared engine:

```cpp
class RandomBase
{
public:
    explicit RandomBase(Random& rng) : m_rng(rng) {}
    RandomBase(Random&&) = delete;               // as OperatorMetric: reject temporaries
    virtual ~RandomBase() = default;

    virtual double operator()() = 0;
    virtual void fill(std::span<double> v) = 0;  // dealii::Vector<double> is contiguous

protected:
    Random& m_rng;
};

class NormalRandom : public RandomBase
{
public:
    NormalRandom(Random& rng, double mean, double stddev);  // validates stddev > 0
    double operator()() override { return m_dist(m_rng.engine()); }
    void fill(std::span<double> v) override;
private:
    std::normal_distribution<double> m_dist;     // keeps its cached second value across calls
};
```

- **Kernels take a single argument:** `random_tangent_vector(x, M, v, RandomBase& dist)`.
- **One engine per consumer:** several distributions draw from it in turn, so they aren't correlated, and the seeding
  rule stays one seed per consumer.
- **Parameters are fixed and checked once,** in the constructor.
- **`fill` takes `std::span<double>`.** It can't be a virtual template over `RangeType`. `dealii::Vector<double>`
  iterators are raw pointers, so `std::span<double>(v.begin(), v.end())` works.
- **The lifetime rule is the same as for `OperatorMetric`:** the engine must outlive the distribution, and
  temporaries are rejected.

Compared with B:

| | E: `RandomBase` hierarchy | B: `std::variant<Normal, Uniform>` |
|---|---|---|
| Kernel arguments | one (`RandomBase&`) | two (`Random&`, `const Distribution&`) |
| Adding a distribution | new subclass, no existing code changes | new variant alternative, every `std::visit` must handle it |
| Command-line config | needs a factory from parsed options to a `unique_ptr<RandomBase>` | parses directly into a value stored in `GPE_Options` |
| Value semantics | no copies, reference to the engine | copyable, no lifetime constraints |
| Cost | one virtual call per `fill`, negligible | none |

### Recommendation

The choice between B and E depends on where distributions are chosen. If they're mostly chosen in code and should be
extensible, E fits better. If they're mostly chosen from the command line and stored in options, B is simpler. The two
combine well: B as the value stored in the options, plus a small factory that builds the `RandomBase` the kernels
take.

With B, filled from the command line like A:

- `GPE_Options` gets a `Distribution initial_dist` (default `Uniform{0.5, 1.5}`, today's behaviour), used by
  `initial_value()` for `--initial random`.
- One option parses into it, for example `--initial-dist uniform:0.5,1.5` or `--initial-dist normal:0,1`, which
  validates the parameters at parse time. Alternatively, `--initial-arg` accepts the two numbers for
  `--initial random`, which avoids a new option but keeps the distribution fixed.
- `test_gradient` keeps `Normal{}` for its perturbations; it doesn't need a command-line option.
- Skip the `SolverControl` state (C) and the `ParameterHandler` integration: the repo parses options with
  `boost::program_options`, not `ParameterHandler`.

## Problems with the singleton

1. **The numbers depend on draw order.** `test_gradient` runs seven problems in sequence (`gradient.cc:317-325`),
   each with `--trials` draws. So trial 3 of `mass` changes when `--trials` changes, or when the `energy` problem
   draws one more vector. A failing trial can't be reproduced on its own.
2. **Hidden global state.** The kernels say "Seed is set globally". You can't see from a signature that a function
   consumes randomness, and tests can't give it an independent generator.
3. **Seeding is non-deterministic by default.** Any program that forgets `seed()` silently gets `random_device`.
   Every driver seeds today, but nothing enforces it.
4. **Same seed, different numbers on another standard library.** `mt19937` is specified exactly, but the
   `std::*_distribution` algorithms are implementation-defined. libstdc++ and libc++ (macOS) give different numbers
   for the same seed. CI only runs on `ubuntu-latest`, so this is latent.
5. **Not thread-safe.** The class comment already says so. That's harmless today, but it rules out drawing from
   deal.II's threaded loops.

## Design: an explicit `Random`, one instance per consumer

```cpp
// include/rmo/util/random.h
class Random
{
public:
    using engine_type = std::mt19937;

    //! Same sequence as NumberGenerator after seed(seed)
    explicit Random(unsigned seed) : m_engine(seed) {}

    //! For seeds derived from several values, see "Seeding several instances"
    explicit Random(std::seed_seq& seq) : m_engine(seq) {}

    double normrnd(double mean, double stddev);
    template <typename RangeType> void normrnd(double mean, double stddev, RangeType& vec);
    double unifrnd(double a, double b);
    template <typename RangeType> void unifrnd(double a, double b, RangeType& vec);

    engine_type& engine() { return m_engine; }

private:
    engine_type m_engine;
};
```

- **No default constructor.** Every instance is seeded explicitly. Entropy seeding becomes an explicit opt-in that
  prints its seed (for example `--seed random`).
- **Separate instances fix the order dependence.** No substream mechanism is needed: when each consumer owns its
  `Random`, its numbers depend only on its seed and its own earlier draws.
- **The `Random&` is passed explicitly** to the functions that draw:
  - `ellipsoid::random_point(x, M, rng, mean, stddev)`
  - `metric::{energy,mass,frobenius}::random_tangent_vector(…, rng, mean, stddev)`
  - the `random_point` and `random_tangent_vector` wrappers in `staging/test/gradient_problems.h`
- **Instances per consumer:**
  - `test_gradient`: one `Random rng(seed)` per problem in `check_problem`. Trial k of `mass` no longer depends on
    the `energy` problem, and doesn't depend on `--trials` either, because a longer run only adds trials at the end.
    Trial k still depends on trials 0..k-1 of the same problem, which are cheap to rerun.
  - `ModelBuilder::initial_value()`: a local `Random rng(options.seed)` built from `GpeOptions::seed`, so its
    signature stays the same.

### Seeding several instances

Instances with the same seed produce the same sequence. That only matters if two of them must be independent and are
used together. For example, if several levels each drew a random initial value from `Random(seed)`, every coarse
vector would be the first `n_l` values of the fine one. Today only the finest level draws (`main_coarse.cc:276`), so
the same seed everywhere is fine.

When distinct seeds are needed, derive them at the call site:

```cpp
std::seed_seq seq{seed, level};
Random rng(seq);
```

Prefer this over `Random(seed + level)`. With `seed + k`, a run with `--seed 42` and one with `--seed 43` reuse each
other's sequences shifted by one: run 42's level 1 equals run 43's level 0. Mixing the values through `seed_seq`
avoids that.

## Comparison

| | `NumberGenerator` (now) | `Random` (proposed) |
|---|---|---|
| State | global singleton | one object per consumer, passed by the caller |
| Seeding | `random_device`, then reseeded by `main()` | required constructor argument |
| Reproducing one problem or initial value | depends on every earlier draw | depends only on its seed |
| Distribution parameters | per call | per call (unchanged) |
| Threads | unsafe | one `Random` per thread or task |

## Phases

1. **Add `Random`, with `Random(seed)` producing the same sequence as `NumberGenerator::seed(seed)`.** Turn
   `NumberGenerator` into a thin wrapper around a single `Random`. Nothing observable changes.
2. **Pass a `Random&` through the kernels.** Change `ellipsoid::random_point`, the three `random_tangent_vector`
   functions and the `gradient_problems.h` wrappers. Callers pass the singleton's instance in the same order as now,
   so output stays bit-identical. Check this with `ctest`, plus a diff of one `--initial random` run before and
   after.
3. **One instance per consumer.** Use one per problem in `test_gradient` and a local one in `initial_value()`. **This
   deliberately changes the drawn numbers** compared with the shared singleton. If any results in `data/org` used
   `--initial random`, tag the commit before this phase, or regenerate them.
4. **Remove `NumberGenerator`.** Also remove the six `seed()` calls, and either delete the unused `randi`,
   `rand_sparse_vec_idx` and scalar overloads, or rewrite `rand_sparse_vec_idx` with `std::ranges::sample`, which
   avoids shuffling all `n` indices.
5. **Distribution objects.** For B: add `Normal`, `Uniform`, `Distribution` and `Random::fill`, and change the
   kernels to take `const Distribution&`. For E: add `RandomBase`, `NormalRandom` and `UniformRandom`, and change the
   kernels to take `RandomBase&`. In both cases, add `GPE_Options::initial_dist` with its command-line option (with
   E, through a factory). With the defaults
   (`Uniform{0.5, 1.5}` and `Normal{}`), the output stays identical to phase 3. This phase is independent of phases 1–4
   and can also come first.
6. **Optional, if results must match across platforms:** replace the `std::` distributions with
   `boost::random::normal_distribution` and `uniform_real_distribution`. Boost is already a dependency, and its
   distributions should give the same numbers on every platform for a given Boost version (not verified). Also print
   the seed in the run's output header.

## Open points

- **Random initial values on several levels.** If a coarser level ever draws a random starting value, give each
  level its own seed via `std::seed_seq{seed, level}`, as described above.
- **Engine.** Keeping `mt19937` makes phases 1 and 2 bit-identical. Switching to `mt19937_64` or a counter-based
  generator (Philox) is only worth it if many parallel generators are needed.
