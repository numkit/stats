// toolboxes/stats/src/distributions/rayleigh.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <keel/math.h>
#include <numkit/stats/distributions/rayleigh.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"



#include "rayleigh_detail.hpp"

namespace numkit::stats {

namespace {
inline bool rayl_argcheck(keel::f64 b) noexcept
{
    return b > 0.0;
}
} // namespace

Value raylpdf(const Value &x, keel::f64 b, keel::Allocator *mr)
{
    (void)rayl_argcheck(b);
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::rayl_pdf(xi, b); }, mr);
}

Value raylcdf(const Value &x, keel::f64 b, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::rayl_cdf(xi, b); }, mr);
}

Value raylinv(const Value &p, keel::f64 b, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::rayl_inv(pi, b); }, mr);
}

Value raylrnd(::numkit::ops::RngContext &rng, keel::f64 b, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    const keel::usize n = rows * cols;
    if (n == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    if (!(b > 0.0)) {
        const keel::f64 fillVal = (b == 0.0) ? 0.0 : keel::Limits<keel::f64>::quiet_nan();
        for (keel::usize i = 0; i < n; ++i) od[i] = fillVal;
        return out;
    }
    // Reference engine formula: r = sqrt(randn(sizeOut).^2 + randn(sizeOut).^2) .* b
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 z1 = rng.nextNormal();
        od[i] = z1 * z1;
    }
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 z2 = rng.nextNormal();
        od[i] = keel::sqrt(od[i] + z2 * z2) * b;
    }
    return out;
}

std::tuple<keel::f64, keel::f64> raylstat(keel::f64 b)
{
    if (b <= 0.0) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return std::make_tuple(nan, nan);
    }
    const keel::f64 mean = b * keel::sqrt(0.5 * keel::kPi);
    const keel::f64 var  = (2.0 - 0.5 * keel::kPi) * b * b;
    return std::make_tuple(mean, var);
}

} // namespace numkit::stats
