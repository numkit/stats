// toolboxes/stats/src/distributions/weibull.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/weibull.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <keel/random.h>

#include "weibull_detail.hpp"

namespace numkit::stats {

namespace {
inline bool wbl_argcheck(double a, double b) noexcept
{
    return a > 0.0 && b > 0.0;
}
} // namespace

Value wblpdf(const Value &x, double a, double b, keel::Allocator *mr)
{
    (void)wbl_argcheck(a, b);
    return elementwise(x, [=](double xi) { return keel::stats::wbl_pdf(xi, a, b); }, mr);
}

Value wblcdf(const Value &x, double a, double b, keel::Allocator *mr)
{
    return elementwise(x, [=](double xi) { return keel::stats::wbl_cdf(xi, a, b); }, mr);
}

Value wblinv(const Value &p, double a, double b, keel::Allocator *mr)
{
    return elementwise(p, [=](double pi) { return keel::stats::wbl_inv(pi, a, b); }, mr);
}

Value wblrnd(::numkit::ops::RngContext &rng, double a, double b, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (a <= 0.0 || b <= 0.0 || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    // keel::random::weibull_f64(rng, shape, scale) — note ORDER FLIP relative
    // to MATLAB (a=scale, b=shape).
    for (keel::usize i = 0; i < n; ++i) od[i] = keel::random::weibull_f64(gen, b, a);
    return out;
}

std::tuple<double, double> wblstat(double a, double b)
{
    if (a <= 0.0 || b <= 0.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    const double g1 = std::tgamma(1.0 + 1.0 / b);
    const double g2 = std::tgamma(1.0 + 2.0 / b);
    const double mean = a * g1;
    const double var  = a * a * (g2 - g1 * g1);
    return std::make_tuple(mean, var);
}

} // namespace numkit::stats
