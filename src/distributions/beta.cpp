// toolboxes/stats/src/distributions/beta.cpp
//
// Beta distribution. pdf via log-form using lbeta = lgamma(a)+lgamma(b)-lgamma(a+b);
// cdf is betainc(x, a, b); icdf is betaincinv(p, a, b); rnd via two Gamma draws.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/beta.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <mutex>
#include <keel/random.h>

#include "beta_detail.hpp"

namespace numkit::stats {

namespace {
inline bool beta_argcheck(keel::f64 a, keel::f64 b) noexcept
{
    return a > 0.0 && b > 0.0;
}
} // namespace

Value betapdf(const Value &x, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    (void)beta_argcheck(a, b);
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::beta_pdf(xi, a, b); }, mr);
}

Value betacdf(const Value &x, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::beta_cdf(xi, a, b); }, mr);
}

Value betainv(const Value &p, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::beta_inv(pi, a, b); }, mr);
}

Value betarnd(::numkit::ops::RngContext &rng, keel::f64 a, keel::f64 b, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (a <= 0.0 || b <= 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        od[i] = keel::random::beta_f64(gen, a, b);
    }
    return out;
}

std::tuple<keel::f64, keel::f64> betastat(keel::f64 a, keel::f64 b)
{
    if (a <= 0.0 || b <= 0.0) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return std::make_tuple(nan, nan);
    }
    const keel::f64 s = a + b;
    const keel::f64 mean = a / s;
    const keel::f64 var = (a * b) / (s * s * (s + 1.0));
    return std::make_tuple(mean, var);
}

} // namespace numkit::stats
