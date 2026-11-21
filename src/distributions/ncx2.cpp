// toolboxes/stats/src/distributions/ncx2.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/ncx2.hpp>

#include <numkit/builtin/specfun.hpp>   // besseli, gammainc
#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <keel/random.h>

#include "ncx2_detail.hpp"

namespace numkit::stats {

namespace {
inline bool ncx2_argcheck(keel::f64 k, keel::f64 lambda) noexcept
{
    return k > 0.0 && lambda >= 0.0;
}
} // namespace

Value ncx2pdf(const Value &x, keel::f64 k, keel::f64 lambda, keel::Allocator *mr)
{
    if (!ncx2_argcheck(k, lambda))
        return elementwise(x, [](keel::f64){ return std::numeric_limits<keel::f64>::quiet_NaN(); }, mr);
    if (lambda == 0.0) {
        // Reduce to chi²(k).
        const keel::f64 C = 1.0 / (std::pow(2.0, k / 2.0) * std::tgamma(k / 2.0));
        return elementwise(x, [=](keel::f64 xi) {
            if (xi <= 0.0) return (k == 2.0 && xi == 0.0) ? 0.5 : 0.0;
            return C * std::pow(xi, k / 2.0 - 1.0) * std::exp(-xi / 2.0);
        }, mr);
    }
    const keel::f64 nu = (k - 2.0) / 2.0;
    return elementwise(x, [&](keel::f64 xi) {
        if (xi < 0.0) return 0.0;
        if (xi == 0.0) return 0.0;
        const keel::f64 sqrtLx = std::sqrt(lambda * xi);
        const keel::f64 iv = besseli_scalar(nu, sqrtLx, mr);
        return 0.5 * std::exp(-(xi + lambda) / 2.0)
             * std::pow(xi / lambda, (k - 2.0) / 4.0)
             * iv;
    }, mr);
}

Value ncx2cdf(const Value &x, keel::f64 k, keel::f64 lambda, keel::Allocator *mr)
{
    if (k <= 0.0 || lambda < 0.0)
        return elementwise(x, [](keel::f64){ return std::numeric_limits<keel::f64>::quiet_NaN(); }, mr);
    return elementwise(x, [&](keel::f64 xi) {
        return ncx2cdf_one(xi, k, lambda, mr);
    }, mr);
}

Value ncx2inv(const Value &p, keel::f64 k, keel::f64 lambda, keel::Allocator *mr)
{
    if (k <= 0.0 || lambda < 0.0)
        return elementwise(p, [](keel::f64){ return std::numeric_limits<keel::f64>::quiet_NaN(); }, mr);
    return elementwise(p, [&](keel::f64 pi) {
        if (!(pi >= 0.0 && pi <= 1.0)) return std::numeric_limits<keel::f64>::quiet_NaN();
        if (pi == 0.0) return 0.0;
        if (pi >= 1.0) return std::numeric_limits<keel::f64>::infinity();
        // Bracket on [0, mean + 10·sqrt(var)] then bisect.
        const keel::f64 mean = k + lambda;
        const keel::f64 var  = 2.0 * (k + 2.0 * lambda);
        keel::f64 lo = 0.0;
        keel::f64 hi = mean + 10.0 * std::sqrt(var) + 1.0;
        for (keel::i32 iter = 0; iter < 50; ++iter) {
            if (ncx2cdf_one(hi, k, lambda, mr) >= pi) break;
            hi *= 2.0;
        }
        for (keel::i32 iter = 0; iter < 80; ++iter) {
            const keel::f64 mid = 0.5 * (lo + hi);
            if (mid == lo || mid == hi) return mid;
            const keel::f64 F = ncx2cdf_one(mid, k, lambda, mr);
            if (F < pi) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    }, mr);
}

Value ncx2rnd(::numkit::ops::RngContext &rng, keel::f64 k, keel::f64 lambda, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (k <= 0.0 || lambda < 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    // Draw J ~ Poisson(λ/2), then X ~ chi²(k + 2J) = Gamma(shape=k/2+J, scale=2).
    for (keel::usize i = 0; i < n; ++i) {
        const keel::u64 J = (lambda == 0.0) ? 0ULL : keel::random::poisson_u64(gen, 0.5 * lambda);
        od[i] = keel::random::chi_squared_f64(gen, k + 2.0 * keel::as<keel::f64>(J));
    }
    return out;
}

std::tuple<keel::f64, keel::f64> ncx2stat(keel::f64 k, keel::f64 lambda)
{
    if (k <= 0.0 || lambda < 0.0) {
        const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(k + lambda, 2.0 * (k + 2.0 * lambda));
}

} // namespace numkit::stats
