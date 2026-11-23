// toolboxes/stats/src/distributions/binomial.cpp

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/binomial.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>

#include "binomial_detail.hpp"

namespace numkit::stats {

namespace {
inline bool bino_argcheck(keel::f64 n, keel::f64 p) noexcept
{
    return n >= 0.0 && keel::floor(n) == n && p >= 0.0 && p <= 1.0;
}
} // namespace

Value binopdf(const Value &k, keel::f64 n, keel::f64 p, keel::Allocator *mr)
{
    (void)bino_argcheck(n, p);
    return elementwise(k, [=](keel::f64 ki){ return keel::stats::bino_pdf(ki, n, p); }, mr);
}

Value binocdf(const Value &k, keel::f64 n, keel::f64 p, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki){ return keel::stats::bino_cdf(ki, n, p); }, mr);
}

Value binoinv(const Value &p_in, keel::f64 n, keel::f64 p, keel::Allocator *mr)
{
    return elementwise(p_in, [=](keel::f64 pi) { return keel::stats::bino_inv(pi, n, p); }, mr);
}

Value binornd(::numkit::ops::RngContext &rng, keel::f64 n, keel::f64 p, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (n < 0.0 || keel::floor(n) != n || p < 0.0 || p > 1.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize cnt = rows * cols;
    const keel::i32 ntrials = keel::as<keel::i32>(n);
    if (ntrials <= 1000000) {
        std::fill(od, od + cnt, 0.0);
        for (keel::i32 t = 0; t < ntrials; ++t) {
            for (keel::usize i = 0; i < cnt; ++i) {
                if (rng.nextUniform() < p)
                    od[i] += 1.0;
            }
        }
        return out;
    }
    for (keel::usize i = 0; i < cnt; ++i)
        od[i] = keel::as<keel::f64>(keel::random::binomial_u64(gen, keel::as<keel::u64>(ntrials), p));
    return out;
}

std::tuple<keel::f64, keel::f64> binostat(keel::f64 n, keel::f64 p)
{
    if (n < 0.0 || keel::floor(n) != n || p < 0.0 || p > 1.0) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(n * p, n * p * (1.0 - p));
}

} // namespace numkit::stats
