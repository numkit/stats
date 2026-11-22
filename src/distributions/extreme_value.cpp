// toolboxes/stats/src/distributions/extreme_value.cpp

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/extreme_value.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <mutex>

#include "extreme_value_detail.hpp"

namespace numkit::stats {

namespace {
inline bool ev_argcheck(keel::f64 sigma) noexcept
{
    return sigma > 0.0;
}
} // namespace

Value evpdf(const Value &x, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    if (!ev_argcheck(sigma))
        return elementwise(x, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    const keel::f64 inv_s = 1.0 / sigma;
    return elementwise(x, [=](keel::f64 xi) {
        const keel::f64 t = (xi - mu) * inv_s;
        return inv_s * keel::exp(t - keel::exp(t));
    }, mr);
}

Value evcdf(const Value &x, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    if (sigma <= 0.0)
        return elementwise(x, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    return elementwise(x, [=](keel::f64 xi) {
        const keel::f64 t = (xi - mu) / sigma;
        return -keel::expm1(-keel::exp(t));
    }, mr);
}

Value evinv(const Value &p, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    if (sigma <= 0.0)
        return elementwise(p, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    return elementwise(p, [=](keel::f64 pi) {
        if (!(pi >= 0.0 && pi <= 1.0)) return keel::Limits<keel::f64>::quiet_nan();
        if (pi == 0.0) return -keel::Limits<keel::f64>::infinity();
        if (pi >= 1.0) return  keel::Limits<keel::f64>::infinity();
        // x = mu + sigma · log(-log1p(-p))
        return mu + sigma * keel::log(-keel::log1p(-pi));
    }, mr);
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

std::tuple<keel::f64, keel::f64> evstat(keel::f64 mu, keel::f64 sigma)
{
    if (sigma <= 0.0) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(mu - sigma * kEulerGamma,
                           sigma * sigma * kPi2Over6);
}

} // namespace numkit::stats
