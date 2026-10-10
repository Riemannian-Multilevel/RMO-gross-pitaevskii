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
 * @brief Pseudo-random numbers for test data and starting values (DistributionBase, Distribution).
 */
namespace rmo {

//! Default of the command-line option --seed
inline constexpr unsigned default_seed = 42;
 

//! Engine of all distributions; the same as the former NumberGenerator, so equal seeds give the same numbers
using Engine = std::mt19937;


//! Distribution drawing from an engine it references (not owns): the engine must outlive the distribution
template <typename ElementType>
class DistributionBase
{
public:
    explicit DistributionBase(Engine& engine)
        : m_engine(engine) {}
    virtual ~DistributionBase() = default;

    // Passed by reference: a copy through a base reference would slice
    DistributionBase(const DistributionBase&) = delete;
    DistributionBase& operator=(const DistributionBase&) = delete;

    Engine& engine() { return m_engine; }

    virtual ElementType operator()() = 0;
    virtual void fill(std::span<ElementType> v) = 0;

protected:
    Engine& m_engine;
};


namespace detail
{

// This adds additional precondition checks, over those that may be provided by an implementation.
// For example, libstdc++ checks a <= b for the real uniform distribution, but not if (b - a) is finite.

template <typename StdDistribution>
struct ParameterDomain;

template <typename Real>
struct ParameterDomain<std::normal_distribution<Real>>
{
    static void check(Real /*mean*/ = 0, Real stddev = 1)
    {
        if (!(stddev > 0)) {
            throw std::invalid_argument("normal distribution: stddev must be positive");
        }
    }
};

template <typename Real>
struct ParameterDomain<std::uniform_real_distribution<Real>>
{
    static void check(Real a = 0, Real b = 1)
    {
        if (!(a <= b && std::isfinite(b - a))) {
            throw std::invalid_argument("uniform distribution: requires a <= b and finite b - a");
        }
    }
};

template <typename Int>
struct ParameterDomain<std::uniform_int_distribution<Int>>
{
    static void check(Int a = 0, Int b = std::numeric_limits<Int>::max())
    {
        if (!(a <= b)) {
            throw std::invalid_argument("uniform integer distribution: requires a <= b");
        }
    }
};

template <typename StdDistribution, typename... Params>
StdDistribution make_checked(Params... params)
{
    ParameterDomain<StdDistribution>::check(params...);
    
    return StdDistribution(params...);
}

} // namespace detail


//! Distribution @p StdDistribution with parameters fixed at construction, e.g. (mean, stddev) or (a, b)
template <typename StdDistribution>
class Distribution final : public DistributionBase<typename StdDistribution::result_type>
{
    using Element = typename StdDistribution::result_type;

public:
    //! @throws std::invalid_argument for parameters outside the domain of @p StdDistribution
    template <typename... Params>
    explicit Distribution(Engine& engine, Params... params)
        : DistributionBase<Element>(engine), m_dist(detail::make_checked<StdDistribution>(params...))
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
    // Kept across calls: std::normal_distribution caches the second value of each generated pair
    StdDistribution m_dist;
};

using NormalDistribution     = Distribution<std::normal_distribution<double>>;
using UniformDistribution    = Distribution<std::uniform_real_distribution<double>>;
using UniformIntDistribution = Distribution<std::uniform_int_distribution<int>>;


//! @p s distinct indices of 0, ..., n-1, in random order
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
