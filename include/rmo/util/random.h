#ifndef RMO_UTIL_RANDOM_H
#define RMO_UTIL_RANDOM_H

#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include <numeric>

/**
 * @file
 * @brief Random numbers for test data and starting values: distributions (DistributionBase,
 * Distribution) that draw from an Engine owned by the caller.
 *
 * Each consumer seeds its own Engine, e.g. from GPE_Options::seed (option `--seed`). Its numbers
 * then depend only on its seed and its own draws.
 *
 * Distributions on the same engine use successive outputs of it, so their numbers are independent;
 * the order of their draws still determines which numbers each one gets. Distributions on separate
 * engines with the same seed would instead use the same outputs, and be correlated.
 *
 * \code{.cpp}
 * Engine engine(options.seed);
 * NormalDistribution normal(engine);              // N(0, 1)
 * UniformDistribution uniform(engine, 0.5, 1.5);  // U[0.5, 1.5)
 *
 * Vector<double> x(n);
 * uniform.fill(x);                                // dealii::Vector<double> converts to std::span
 * ellipsoid::random_point(x, M, normal);          // kernels take a DistributionBase<double>&
 * \endcode
 *
 * Not thread-safe: use an engine and its distributions from one thread at a time.
 */
namespace rmo {

//! Default of the command-line option --seed
inline constexpr unsigned default_seed = 42;


/**
 * @brief Engine of all distributions.
 *
 * The engine of the former NumberGenerator: a distribution on `Engine(seed)` draws the same numbers
 * as NumberGenerator after `seed(seed)`.
 */
using Engine = std::mt19937;


/**
 * @brief Distribution with fixed parameters, drawing values of type @p ElementType from an Engine.
 *
 * The engine is referenced, not owned, and must outlive the distribution. Functions that draw
 * random values take a `DistributionBase<double>&`, so the caller chooses the distribution.
 *
 * @tparam ElementType Type of the drawn values, e.g. `double` or `int`.
 */
template <typename ElementType>
class DistributionBase
{
public:
    //! @p engine must outlive this object; temporaries do not bind to the non-const reference.
    explicit DistributionBase(Engine& engine)
        : m_engine(engine) {}
    virtual ~DistributionBase() = default;

    //! Not copyable: a copy through a base reference would slice.
    DistributionBase(const DistributionBase&) = delete;
    DistributionBase& operator=(const DistributionBase&) = delete;

    //! The engine drawn from, e.g. for rand_sparse_vec_idx().
    Engine& engine() { return m_engine; }

    //! One value of the distribution.
    virtual ElementType operator()() = 0;

    //! Fills @p v with values of the distribution.
    virtual void fill(std::span<ElementType> v) = 0;

protected:
    Engine& m_engine;  //!< Engine drawn from, not owned
};


namespace detail
{

/**
 * @brief Checks the parameters of @p StdDistribution before it is constructed.
 *
 * The standard states these checks as preconditions of the constructors, and violating a
 * precondition is undefined behavior. Implementations check some of them at most: libstdc++ checks
 * a <= b for std::uniform_real_distribution, but not that b - a is finite. Here they all throw
 * std::invalid_argument.
 *
 * `check()` takes the constructor arguments of @p StdDistribution, with the same defaults. A
 * distribution without a specialization does not compile in Distribution.
 *
 * References, by section label of the C++ standard (e.g. https://eel.is/c++draft/res.on.required):
 * - [res.on.required]: violating a "Preconditions:" element is undefined behavior
 * - [rand.dist.norm.normal]: 0 < stddev
 * - [rand.dist.uni.real]: a <= b and b - a <= numeric_limits<RealType>::max()
 * - [rand.dist.uni.int]: a <= b
 */
template <typename StdDistribution>
struct ParameterDomain;

//! [rand.dist.norm.normal]: requires stddev > 0.
template <typename Real>
struct ParameterDomain<std::normal_distribution<Real>>
{
    //! @throws std::invalid_argument unless stddev > 0
    static void check(Real /*mean*/ = 0, Real stddev = 1)
    {
        if (!(stddev > 0)) {
            throw std::invalid_argument("normal distribution: stddev must be positive");
        }
    }
};

//! [rand.dist.uni.real]: requires a <= b and a finite b - a.
template <typename Real>
struct ParameterDomain<std::uniform_real_distribution<Real>>
{
    //! @throws std::invalid_argument unless a <= b and b - a is finite
    static void check(Real a = 0, Real b = 1)
    {
        if (!(a <= b && std::isfinite(b - a))) {
            throw std::invalid_argument("uniform distribution: requires a <= b and finite b - a");
        }
    }
};

//! [rand.dist.uni.int]: requires a <= b.
template <typename Int>
struct ParameterDomain<std::uniform_int_distribution<Int>>
{
    //! @throws std::invalid_argument unless a <= b
    static void check(Int a = 0, Int b = std::numeric_limits<Int>::max())
    {
        if (!(a <= b)) {
            throw std::invalid_argument("uniform integer distribution: requires a <= b");
        }
    }
};

//! @p StdDistribution constructed from @p params, after ParameterDomain::check().
template <typename StdDistribution, typename... Params>
StdDistribution make_checked(Params... params)
{
    ParameterDomain<StdDistribution>::check(params...);

    return StdDistribution(params...);
}

} // namespace detail


/**
 * @brief The standard distribution @p StdDistribution, with parameters fixed at construction.
 *
 * The standard distribution is kept between calls, since std::normal_distribution generates values
 * in pairs and returns the second one on the next call.
 *
 * @tparam StdDistribution A distribution of `<random>` with a detail::ParameterDomain
 * specialization, e.g. `std::normal_distribution<double>`.
 */
template <typename StdDistribution>
class Distribution final : public DistributionBase<typename StdDistribution::result_type>
{
    using Element = typename StdDistribution::result_type;

public:
    /**
     * @param engine Engine to draw from; must outlive this object.
     * @param params Constructor arguments of @p StdDistribution, e.g. (mean, stddev) or (a, b);
     *               omitted trailing arguments take their defaults.
     * @throws std::invalid_argument for parameters outside the domain, see detail::ParameterDomain.
     */
    template <typename... Params>
    explicit Distribution(Engine& engine, Params... params)
        : DistributionBase<Element>(engine)
        , m_dist(detail::make_checked<StdDistribution>(params...))
    {}

    Element operator()() override
    {
        return m_dist(this->m_engine);
    }

    void fill(std::span<Element> v) override
    {
        for (auto& elem : v) {
            elem = m_dist(this->m_engine);
        }
    }

private:
    StdDistribution m_dist;
};

//! Normal distribution \f$ N(\mu, \sigma^2) \f$, parameters (mean, stddev); default N(0, 1).
using NormalDistribution     = Distribution<std::normal_distribution<double>>;
//! Uniform distribution on \f$ [a, b) \f$, parameters (a, b); default [0, 1).
using UniformDistribution    = Distribution<std::uniform_real_distribution<double>>;
//! Uniform distribution on the integers \f$ a, \dots, b \f$; default 0, ..., INT_MAX.
using UniformIntDistribution = Distribution<std::uniform_int_distribution<int>>;


/**
 * @brief @p s distinct indices of \f$ 0, \dots, n-1 \f$, in random order.
 *
 * @param s Number of indices.
 * @param n Number of candidate indices.
 * @param engine Engine to draw from, e.g. DistributionBase::engine().
 * @throws std::invalid_argument unless \f$ 0 \le s \le n \f$.
 */
inline std::vector<int> rand_sparse_vec_idx(int s, int n, Engine& engine)
{
    if (bool valid = (s <= n && s >= 0); !valid) {
        throw std::invalid_argument("invalid sparse range");
    }

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::ranges::shuffle(idx, engine);
    return {idx.begin(), idx.begin() + s};
}

} // namespace rmo

#endif //RMO_UTIL_RANDOM_H
