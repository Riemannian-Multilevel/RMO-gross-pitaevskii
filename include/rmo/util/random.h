#ifndef RMO_UTIL_RANDOM_H
#define RMO_UTIL_RANDOM_H

#include <random>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <utility>

/**
 * @file
 * @brief Pseudo-random numbers for test data and starting values (NumberGenerator).
 */
namespace rmo {

//! Default of the command-line option --seed
inline constexpr unsigned default_seed = 42;

/**
 * @brief Generator of all pseudo-random numbers of the program (a singleton), seeded from
 * `std::random_device` until seed() is called. Not thread-safe: draw numbers from one thread only.
 *
 * \code{.cpp}
 * // Seed with a fixed value, for reproducible runs
 * NumberGenerator::get().seed(42);
 * // Or with a seed sequence
 * std::seed_seq ss{1, 2, 3, 4, 5};
 * NumberGenerator::get().seed(ss);
 * \endcode
 */
class NumberGenerator
{
private:
    std::mt19937 twister;

    NumberGenerator()
        : twister(std::random_device{}())
    {}

public:
    NumberGenerator(const NumberGenerator&) = delete;
    NumberGenerator(NumberGenerator&&) = delete;
    NumberGenerator& operator=(const NumberGenerator&) = delete;
    NumberGenerator& operator=(NumberGenerator&&) = delete;

    static NumberGenerator& get()
    {
        static NumberGenerator gen;
        return gen;
    }

    // Seed the generator with a custom seed or seed sequence
    template <typename SeedArg>
    void seed(SeedArg&& s)
    {
        twister.seed(std::forward<SeedArg>(s));
    }

    // Reseed the generator from std::random_device
    void seed()
    {
        twister.seed(std::random_device{}());
    }

    double normrnd(double mean, double stddev)
    {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(twister);
    }

    template <typename RangeType>
    void normrnd(double mean, double stddev, RangeType& vec)
    {
        std::normal_distribution<double> dist(mean, stddev);
        for (auto& x : vec) {
            x = dist(twister);
        }
    }

    double unifrnd(double a, double b) {
        std::uniform_real_distribution<double> dist(a, b);
        return dist(twister);
    }

    template <typename RangeType>
    void unifrnd(const double a, const double b, RangeType& vec)
    {
        std::uniform_real_distribution<double> dist(a, b);

        for (auto& x : vec) {
            x = dist(twister);
        }
    }

    int randi(const int a, const int b) {
        std::uniform_int_distribution<int> dist(a, b);
        return dist(twister);
    }

    template <typename RangeType>
    void randi(const int a, const int b, RangeType& vec)
    {
        std::uniform_int_distribution<int> dist(a, b);
        for (auto& x : vec) {
            x = dist(twister);
        }
    }

    std::vector<int> rand_sparse_vec_idx(int s, int n) {
        if (bool valid = (s <= n && s >= 0); !valid) {
            throw std::invalid_argument("invalid sparse range");
        }

        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);

        std::ranges::shuffle(idx, twister);
        return {idx.begin(), idx.begin() + s};
    }
};

} // namespace rmo

#endif //RMO_UTIL_RANDOM_H
