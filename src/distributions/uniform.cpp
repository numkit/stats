// toolboxes/stats/src/distributions/uniform.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/uniform.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <random>

#include "uniform_detail.hpp"

namespace numkit::stats {

namespace {
inline bool unif_argcheck(double a, double b) noexcept
{
    return b > a;
}
} // namespace

Value unifpdf(const Value &x, double a, double b, keel::Allocator *mr)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if (!unif_argcheck(a, b))
        return elementwise(x, [NaN](double){ return NaN; }, mr);
    const double inv = 1.0 / (b - a);
    return elementwise(x, [=](double xi) {
        if (std::isnan(xi)) return NaN;  // propagate NaN x — matches MATLAB
        return (xi >= a && xi <= b) ? inv : 0.0;
    }, mr);
}

Value unifcdf(const Value &x, double a, double b, keel::Allocator *mr)
{
    if (b <= a)
        return elementwise(x, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    const double inv = 1.0 / (b - a);
    return elementwise(x, [=](double xi) {
        if (xi <= a) return 0.0;
        if (xi >= b) return 1.0;
        return (xi - a) * inv;
    }, mr);
}

Value unifinv(const Value &p, double a, double b, keel::Allocator *mr)
{
    if (b <= a)
        return elementwise(p, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    const double w = b - a;
    return elementwise(p, [=](double pi) {
        if (pi < 0.0 || pi > 1.0) return std::numeric_limits<double>::quiet_NaN();
        return a + pi * w;
    }, mr);
}

Value unifrnd(::numkit::ops::RngContext &rng, double a, double b, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (b <= a || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    const double w = b - a;
    for (keel::usize i = 0; i < n; ++i) od[i] = a + w * rng.nextUniform();
    return out;
}

std::tuple<double, double> unifstat(double a, double b)
{
    if (b <= a) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    const double w = b - a;
    return std::make_tuple(0.5 * (a + b), (w * w) / 12.0);
}

} // namespace numkit::stats
