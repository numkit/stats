// toolboxes/stats/src/distributions/hypergeom.cpp
//
// Hypergeometric distribution: drawing N items without replacement from a
// population of M with K successes. f(k; M, K, N) = C(K, k)·C(M-K, N-k)/C(M, N).

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/hypergeom.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <keel/random.h>

#include "hypergeom_detail.hpp"

namespace numkit::stats {

namespace {
inline bool hyge_argcheck(keel::f64 M, keel::f64 K, keel::f64 N) noexcept
{
    return params_valid(M, K, N);
}
} // namespace

Value hygepdf(const Value &k, keel::f64 M, keel::f64 K, keel::f64 N, keel::Allocator *mr)
{
    if (!hyge_argcheck(M, K, N))
        return elementwise(k, [](keel::f64){ return std::numeric_limits<keel::f64>::quiet_NaN(); }, mr);
    return elementwise(k, [=](keel::f64 ki){ return hyge_pmf(ki, M, K, N); }, mr);
}

Value hygecdf(const Value &k, keel::f64 M, keel::f64 K, keel::f64 N, keel::Allocator *mr)
{
    if (!params_valid(M, K, N))
        return elementwise(k, [](keel::f64){ return std::numeric_limits<keel::f64>::quiet_NaN(); }, mr);
    return elementwise(k, [=](keel::f64 ki){ return hyge_cdf_scalar(ki, M, K, N); }, mr);
}

Value hygeinv(const Value &q, keel::f64 M, keel::f64 K, keel::f64 N, keel::Allocator *mr)
{
    if (!params_valid(M, K, N))
        return elementwise(q, [](keel::f64){ return std::numeric_limits<keel::f64>::quiet_NaN(); }, mr);
    return elementwise(q, [=](keel::f64 qi){ return hyge_inv_scalar(qi, M, K, N); }, mr);
}

Value hygernd(::numkit::ops::RngContext &rng, keel::f64 M, keel::f64 K, keel::f64 N, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!params_valid(M, K, N) || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize cnt = rows * cols;
    // Inverse-cdf walk per draw. M, K, N fixed so the walk is fast.
    for (keel::usize i = 0; i < cnt; ++i) od[i] = hyge_inv_scalar(keel::random::uniform_f64(gen), M, K, N);
    return out;
}

std::tuple<keel::f64, keel::f64> hygestat(keel::f64 M, keel::f64 K, keel::f64 N)
{
    if (!params_valid(M, K, N) || M < 1.0) {
        const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    const keel::f64 mean = N * K / M;
    if (M < 2.0) return std::make_tuple(mean, std::numeric_limits<keel::f64>::quiet_NaN());
    const keel::f64 var = N * K * (M - K) * (M - N) / (M * M * (M - 1.0));
    return std::make_tuple(mean, var);
}

} // namespace numkit::stats
