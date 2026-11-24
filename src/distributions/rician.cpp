// toolboxes/stats/src/distributions/rician.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/rician.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>
#include <keel/math.h>

#include "rician_detail.hpp"

namespace numkit::stats {

Value ricepdf(const Value &x, keel::f64 s, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::rice_pdf(xi, s, sigma); }, mr);
}

Value ricecdf(const Value &x, keel::f64 s, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::rice_cdf(xi, s, sigma); }, mr);
}

Value riceinv(const Value &p, keel::f64 s, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::rice_inv(pi, s, sigma); }, mr);
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

keel::Tuple<keel::f64, keel::f64> ricestat(keel::f64 s, keel::f64 sigma, keel::Allocator * /*mr*/)
{
    const auto m = keel::stats::rice_stat(s, sigma);
    return keel::make_tuple(m.mean, m.var);
}

} // namespace numkit::stats
