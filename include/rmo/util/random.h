#ifndef RMO_UTIL_RANDOM_H
#define RMO_UTIL_RANDOM_H
#include <random>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace rmo {

/**
 * @brief Seeds @ref random_engine from the environment variable @c RMO_TEST_SEED
 * when set (base 10), otherwise from @c std::random_device. Either way the seed
 * actually used is printed to stderr once, on first use, so a test failure that
 * depended on the draw can be reproduced by rerunning with
 * @c RMO_TEST_SEED=<printed value> set.
 */
inline unsigned int seed_from_env()
{
    if (const char* env = std::getenv("RMO_TEST_SEED")) {
        return static_cast<unsigned int>(std::strtoul(env, nullptr, 10));
    }
    return std::random_device{}();
}

inline std::mt19937& random_engine()
{
    static thread_local std::mt19937 twister = [] {
        const unsigned int seed = seed_from_env();
        std::cerr << "rmo::random_engine: seed = " << seed
                  << " (set RMO_TEST_SEED=" << seed << " to reproduce)" << std::endl;
        return std::mt19937(seed);
    }();
    return twister;
}

inline double
normrnd(double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);

    return dist(random_engine());
}

template <typename RangeType>
void
normrnd(double mean, double stddev, RangeType& vec)
{
    std::normal_distribution<double> dist(mean, stddev);

    for (auto& x : vec) {
        x = dist(random_engine());
    }
}

inline double
unifrnd(double a, double b) {
    std::uniform_real_distribution<double> dist(a, b);

    return dist(random_engine());
}

template <typename RangeType>
void
unifrnd(const double a, const double b, RangeType& vec)
{
    std::uniform_real_distribution<double> dist(a, b);

    for (auto& x : vec) {
        x = dist(random_engine());
    }
}

inline int
randi(const int a, const int b) {
    std::uniform_int_distribution<int> dist(a, b);

    return dist(random_engine());
}

template <typename RangeType>
void
randi(const int a, const int b, RangeType& vec)
{
    std::uniform_int_distribution<int> dist(a, b);

    for (auto& x : vec) {
        x = dist(random_engine());
    }
}

// TODO: use a set-based approach if `s` is much smaller than `n`
inline std::vector<int>
RandSparseVecIdx(int s, int n) {
    if (bool valid = (s <= n && s > 1); !valid) {
        throw std::invalid_argument("invalid sparse range");
    }

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::ranges::shuffle(idx, random_engine());
    return {idx.begin(), idx.begin() + s};
}

} // namespace rmo

#endif //RMO_UTIL_RANDOM_H
