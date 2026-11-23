// toolboxes/stats/src/distributions/exponential.cpp

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/exponential.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"


#include "exponential_detail.hpp"

namespace numkit::stats {

namespace {
inline bool exp_argcheck(keel::f64 mu) noexcept
{
    return mu > 0.0;
}
} // namespace

Value exppdf(const Value &x, keel::f64 mu, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return exppdfK(xi, mu); }, mr);
}

Value expcdf(const Value &x, keel::f64 mu, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return expcdfK(xi, mu); }, mr);
}

Value expinv(const Value &p, keel::f64 mu, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return expinvK(pi, mu); }, mr);
}

Value exprnd(::numkit::ops::RngContext &rng, keel::f64 mu, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!exp_argcheck(mu) || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) od[i] = -mu * keel::log(rng.nextUniform());
    return out;
}

std::tuple<keel::f64, keel::f64> expstat(keel::f64 mu)
{
    if (mu <= 0.0) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(mu, mu * mu);
}

} // namespace numkit::stats
