// toolboxes/stats/src/distributions/rician.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/rician.hpp>

#include <numkit/builtin/specfun.hpp>   // besseli, marcumq
#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>
#include <keel/math.h>


#include "rician_detail.hpp"

namespace numkit::stats {

namespace {
inline bool rice_argcheck(keel::f64 s, keel::f64 sigma) noexcept
{
    return sigma > 0.0 && s >= 0.0;
}
} // namespace

Value ricepdf(const Value &x, keel::f64 s, keel::f64 sigma, keel::Allocator *mr)
{
    if (!rice_argcheck(s, sigma))
        return elementwise(x, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    const keel::f64 s2 = s * s;
    const keel::f64 sg2 = sigma * sigma;
    return elementwise(x, [&](keel::f64 xi) {
        if (xi < 0.0) return 0.0;
        if (xi == 0.0) {
            // Limit: at x=0 the PDF is 0 unless s=0 (Rayleigh case).
            return (s == 0.0) ? 0.0 : 0.0;
        }
        const keel::f64 arg = xi * s / sg2;
        const keel::f64 i0  = besseli_scalar(0.0, arg, mr);
        return (xi / sg2) * keel::exp(-(xi * xi + s2) / (2.0 * sg2)) * i0;
    }, mr);
}

Value ricecdf(const Value &x, keel::f64 s, keel::f64 sigma, keel::Allocator *mr)
{
    if (sigma <= 0.0 || s < 0.0)
        return elementwise(x, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    return elementwise(x, [&](keel::f64 xi) {
        if (xi <= 0.0) return 0.0;
        return 1.0 - marcumq_scalar(s / sigma, xi / sigma, mr);
    }, mr);
}

Value riceinv(const Value &p, keel::f64 s, keel::f64 sigma, keel::Allocator *mr)
{
    if (sigma <= 0.0 || s < 0.0)
        return elementwise(p, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    return elementwise(p, [&](keel::f64 pi) {
        if (!(pi >= 0.0 && pi <= 1.0)) return keel::Limits<keel::f64>::quiet_nan();
        if (pi == 0.0) return 0.0;
        if (pi >= 1.0) return keel::Limits<keel::f64>::infinity();
        // Bracket then bisect: F(0) = 0, F(s + 8σ) ≈ 1 (Gaussian-tail bound).
        keel::f64 lo = 0.0;
        keel::f64 hi = s + 8.0 * sigma;
        // Expand `hi` if needed.
        for (keel::i32 iter = 0; iter < 50; ++iter) {
            const keel::f64 Fhi = 1.0 - marcumq_scalar(s / sigma, hi / sigma, mr);
            if (Fhi >= pi) break;
            hi *= 2.0;
        }
        // Bisection to ~1e-12 relative tolerance.
        for (keel::i32 iter = 0; iter < 80; ++iter) {
            const keel::f64 mid = 0.5 * (lo + hi);
            if (mid == lo || mid == hi) return mid;
            const keel::f64 F = 1.0 - marcumq_scalar(s / sigma, mid / sigma, mr);
            if (F < pi) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    }, mr);
}

Value ricernd(::numkit::ops::RngContext &rng, keel::f64 s, keel::f64 sigma, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (sigma <= 0.0 || s < 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 a = keel::random::normal_f64(gen, s, sigma);
        const keel::f64 b = keel::random::normal_f64(gen, 0.0, sigma);
        od[i] = keel::sqrt(a * a + b * b);
    }
    return out;
}

keel::Tuple<keel::f64, keel::f64> ricestat(keel::f64 s, keel::f64 sigma, keel::Allocator *mr)
{
    if (sigma <= 0.0 || s < 0.0) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return keel::make_tuple(nan, nan);
    }
    if (s == 0.0) {
        return keel::make_tuple(sigma * keel::sqrt(keel::kPi / 2.0),
                               sigma * sigma * (2.0 - keel::kPi / 2.0));
    }
    const keel::f64 s2 = s * s;
    const keel::f64 sg2 = sigma * sigma;
    const keel::f64 z = -s2 / (2.0 * sg2);
    const keel::f64 half = s2 / (4.0 * sg2);
    const keel::f64 i0 = besseli_scalar(0.0, half, mr);
    const keel::f64 i1 = besseli_scalar(1.0, half, mr);
    const keel::f64 L = keel::exp(z / 2.0) * ((1.0 - z) * i0 - z * i1);
    const keel::f64 mean = sigma * keel::sqrt(keel::kPi / 2.0) * L;
    const keel::f64 var  = 2.0 * sg2 + s2 - mean * mean;
    return keel::make_tuple(mean, var);
}

} // namespace numkit::stats
