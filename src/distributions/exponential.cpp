// toolboxes/stats/src/distributions/exponential.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/exponential.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>

#include "exponential_detail.hpp"

namespace numkit::stats {

namespace {
inline bool exp_argcheck(double mu) noexcept
{
    return mu > 0.0;
}
} // namespace

Value exppdf(const Value &x, double mu, keel::Allocator *mr)
{
    return elementwise(x, [=](double xi) { return exppdfK(xi, mu); }, mr);
}

Value expcdf(const Value &x, double mu, keel::Allocator *mr)
{
    return elementwise(x, [=](double xi) { return expcdfK(xi, mu); }, mr);
}

Value expinv(const Value &p, double mu, keel::Allocator *mr)
{
    return elementwise(p, [=](double pi) { return expinvK(pi, mu); }, mr);
}

Value exprnd(::numkit::ops::RngContext &rng, double mu, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!exp_argcheck(mu) || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) od[i] = -mu * std::log(rng.nextUniform());
    return out;
}

std::tuple<double, double> expstat(double mu)
{
    if (mu <= 0.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(mu, mu * mu);
}

} // namespace numkit::stats
