// toolboxes/stats/src/distributions/hypergeom.cpp
//
// Hypergeometric distribution: drawing N items without replacement from a
// population of M with K successes. f(k; M, K, N) = C(K, k)·C(M-K, N-k)/C(M, N).

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/hypergeom.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>

#include "hypergeom_detail.hpp"

namespace numkit::stats {

Value hygepdf(const Value &k, keel::f64 M, keel::f64 K, keel::f64 N, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki) { return keel::stats::hyge_pdf(ki, M, K, N); }, mr);
}

Value hygecdf(const Value &k, keel::f64 M, keel::f64 K, keel::f64 N, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki) { return keel::stats::hyge_cdf(ki, M, K, N); }, mr);
}

Value hygeinv(const Value &q, keel::f64 M, keel::f64 K, keel::f64 N, keel::Allocator *mr)
{
    return elementwise(q, [=](keel::f64 qi) { return keel::stats::hyge_inv(qi, M, K, N); }, mr);
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

keel::Tuple<keel::f64, keel::f64> hygestat(keel::f64 M, keel::f64 K, keel::f64 N)
{
    const auto m = keel::stats::hyge_stat(M, K, N);
    return keel::make_tuple(m.mean, m.var);
}

} // namespace numkit::stats
