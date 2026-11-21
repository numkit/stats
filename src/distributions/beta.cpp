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

#include <cmath>
#include <limits>
#include <mutex>
#include <keel/random.h>

#include "beta_detail.hpp"

namespace numkit::stats {

namespace {
inline bool beta_argcheck(double a, double b) noexcept
{
    return a > 0.0 && b > 0.0;
}
} // namespace

Value betapdf(const Value &x, double a, double b, keel::Allocator *mr)
{
    (void)beta_argcheck(a, b);
    return elementwise(x, [=](double xi) { return keel::stats::beta_pdf(xi, a, b); }, mr);
}

Value betacdf(const Value &x, double a, double b, keel::Allocator *mr)
{
    return elementwise(x, [=](double xi) { return keel::stats::beta_cdf(xi, a, b); }, mr);
}

Value betainv(const Value &p, double a, double b, keel::Allocator *mr)
{
    return elementwise(p, [=](double pi) { return keel::stats::beta_inv(pi, a, b); }, mr);
}

Value betarnd(::numkit::ops::RngContext &rng, double a, double b, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (a <= 0.0 || b <= 0.0 || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        od[i] = keel::random::beta_f64(gen, a, b);
    }
    return out;
}

std::tuple<double, double> betastat(double a, double b)
{
    if (a <= 0.0 || b <= 0.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    const double s = a + b;
    const double mean = a / s;
    const double var = (a * b) / (s * s * (s + 1.0));
    return std::make_tuple(mean, var);
}

} // namespace numkit::stats
