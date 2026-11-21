// toolboxes/stats/src/distributions/gamma_dist.cpp
//
// Gamma distribution. Uses MATLAB convention: gampdf(x, a, b) interprets a
// as shape and b as scale (NOT rate). So f(x) = x^(a-1) exp(-x/b) /
// (b^a · Γ(a)). cdf composes gammainc on x/b; icdf uses gammaincinv;
// rnd uses MATLAB-exact Marsaglia-Tsang on RngContext.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/gamma_dist.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>

#include "gamma_dist_detail.hpp"

namespace numkit::stats {

namespace {

keel::f64 sampleGamma1(::numkit::ops::RngContext &rng, keel::f64 a)
{
    if (!(a > 0.0))
        return (a == 0.0) ? 0.0 : std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::f64 d = (a < 1.0 ? (a + 1.0) : a) - 1.0 / 3.0;
    const keel::f64 c = 1.0 / std::sqrt(9.0 * d);
    keel::f64 u = rng.nextUniform();
    const keel::f64 u_boost = (a < 1.0) ? rng.nextUniform() : 1.0;
    keel::f64 v = 0.0;
    for (;;) {
        const keel::f64 z = rng.nextNormal();
        v = 1.0 + c * z;
        if (v > 0.0) {
            v = v * v * v;
            const keel::f64 z2 = z * z;
            if (u < 1.0 - 0.0331 * z2 * z2 ||
                std::log(u) < 0.5 * z2 + d * (1.0 - v + std::log(v))) {
                break;
            }
        }
        u = rng.nextUniform();
    }
    return (a < 1.0) ? (d * v * std::pow(u_boost, 1.0 / a)) : (d * v);
}

} // namespace

namespace {
inline bool gam_argcheck(keel::f64 a, keel::f64 b) noexcept
{
    return a > 0.0 && b > 0.0;
}
} // namespace

Value gampdf(const Value &x, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    (void)gam_argcheck(a, b);
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::gam_pdf(xi, a, b); }, mr);
}

Value gamcdf(const Value &x, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::gam_cdf(xi, a, b); }, mr);
}

Value gaminv(const Value &p, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::gam_inv(pi, a, b); }, mr);
}

Value gamrnd(::numkit::ops::RngContext &rng, keel::f64 a, keel::f64 b, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (a <= 0.0 || b <= 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) od[i] = b * sampleGamma1(rng, a);
    return out;
}

Value randg(::numkit::ops::RngContext &rng, keel::f64 shape, keel::usize rows, keel::usize cols,
            keel::Allocator *mr)
{
    // randg(shape) ≡ gamrnd(shape, 1, rows, cols) — pure scale = 1.
    return gamrnd(rng, shape, 1.0, rows, cols, mr);
}

Value randg(::numkit::ops::RngContext &rng, const Value &shapeArray, keel::Allocator *mr)
{
    const keel::usize rows = shapeArray.dims().rows();
    const keel::usize cols = shapeArray.dims().cols();
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    const keel::usize n = rows * cols;
    if (n == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 a = shapeArray.elemAsDouble(i);
        od[i] = sampleGamma1(rng, a);
    }
    return out;
}

std::tuple<keel::f64, keel::f64> gamstat(keel::f64 a, keel::f64 b)
{
    if (a <= 0.0 || b <= 0.0) {
        const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(a * b, a * b * b);
}

} // namespace numkit::stats
