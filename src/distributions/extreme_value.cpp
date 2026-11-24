// toolboxes/stats/src/distributions/extreme_value.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/extreme_value.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include "extreme_value_detail.hpp"

namespace numkit::stats {

Value evpdf(const Value &x, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::ev_pdf(xi, mu, sigma); }, mr);
}

Value evcdf(const Value &x, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::ev_cdf(xi, mu, sigma); }, mr);
}

Value evinv(const Value &p, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::ev_inv(pi, mu, sigma); }, mr);
}

Value evrnd(::numkit::ops::RngContext &rng, keel::f64 mu, keel::f64 sigma, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (sigma <= 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        // genRes53 — MATLAB-canonical 53-bit uniform in [0, 1). Direct
        // call (bypasses std::uniform_real_distribution whose uint32->
        // double mapping is implementation-defined and breaks parity).
        const keel::f64 u = gen.genRes53();
        // Avoid log(0) for u==0; genRes53 can return exactly 0 (low
        // probability) — guard.
        const keel::f64 safe = (u > 0.0) ? u : keel::Limits<keel::f64>::kMin;
        // MATLAB convention: evrnd is Gumbel-MIN (Type-I extreme value
        // for minima), so x = mu + sigma * log(-log(u)) -- direct u, NOT
        // 1-u. (We previously used Gumbel-MAX log(-log1p(-u)).)
        od[i] = mu + sigma * keel::log(-keel::log(safe));
    }
    return out;
}

keel::Tuple<keel::f64, keel::f64> evstat(keel::f64 mu, keel::f64 sigma)
{
    const auto m = keel::stats::ev_stat(mu, sigma);
    return keel::make_tuple(m.mean, m.var);
}

} // namespace numkit::stats
