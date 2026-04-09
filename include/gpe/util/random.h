#ifndef RANDOM_H
#define RANDOM_H
#include <random>
#include <algorithm>
#include <stdexcept>

namespace gpe {

/// XXX: allow user-specified seed
inline std::mt19937& random_engine()
{
    static thread_local std::mt19937 twister{std::random_device{}()};
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

} // namespace gpe

#endif // RANDOM_H
