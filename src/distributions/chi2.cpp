// toolboxes/stats/src/distributions/chi2.cpp
//
// Chi-squared distribution. Composes the existing gammainc /
// gammaincinv on (k/2, x/2) for cdf / icdf.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/chi2.hpp>

#include <numkit/builtin/datafun.hpp>  // RngContext + rand/randn/randi/randperm (session-scoped, no global/mutex)
#include <numkit/builtin/specfun.hpp> // gammainc, gammaincinv

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <cstring>
#include <mutex>
#include <keel/random.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "chi2_detail.hpp"

namespace numkit::stats {

namespace {
inline bool chi2_argcheck(double k) noexcept
{
    return k > 0.0;
}
} // namespace

Value chi2pdf(const Value &x, double k, keel::Allocator *mr)
{
    (void)chi2_argcheck(k);
    return elementwise(x, [=](double xi) { return keel::stats::chi2_pdf(xi, k); }, mr);
}

Value chi2cdf(const Value &x, double k, keel::Allocator *mr)
{
    return elementwise(x, [=](double xi) { return keel::stats::chi2_cdf(xi, k); }, mr);
}

Value chi2inv(const Value &p, double k, keel::Allocator *mr)
{
    return elementwise(p, [=](double pi) { return keel::stats::chi2_inv(pi, k); }, mr);
}

Value chi2rnd(::numkit::ops::RngContext &rng, double k, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (k <= 0.0 || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) od[i] = keel::random::chi_squared_f64(gen, k);
    return out;
}

std::tuple<double, double> chi2stat(double k)
{
    // MATLAB convention: k <= 0 ⇒ moments NaN (degenerate distribution).
    if (k <= 0.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(k, 2.0 * k);
}

} // namespace numkit::stats
