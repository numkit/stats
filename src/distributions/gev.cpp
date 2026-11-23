// toolboxes/stats/src/distributions/gev.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/gev.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"


#include "gev_detail.hpp"

namespace numkit::stats {

namespace {
inline bool gev_argcheck(keel::f64 sigma) noexcept
{
    return sigma > 0.0;
}
} // namespace

Value gevpdf(const Value &x, keel::f64 k, keel::f64 sigma, keel::f64 mu, keel::Allocator *mr)
{
    (void)gev_argcheck(sigma);
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::gev_pdf(xi, k, sigma, mu); }, mr);
}

Value gevcdf(const Value &x, keel::f64 k, keel::f64 sigma, keel::f64 mu, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::gev_cdf(xi, k, sigma, mu); }, mr);
}

Value gevinv(const Value &p, keel::f64 k, keel::f64 sigma, keel::f64 mu, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::gev_inv(pi, k, sigma, mu); }, mr);
}

Value gevrnd(::numkit::ops::RngContext &rng, keel::f64 k, keel::f64 sigma, keel::f64 mu, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (sigma <= 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        // genRes53 -- MATLAB-canonical 53-bit uniform; bypasses
        // std::uniform_real_distribution to preserve bit-parity.
        keel::f64 u = gen.genRes53();
        if (u <= 0.0) u = keel::Limits<keel::f64>::kMin;
        od[i] = gev_inv_one(u, k, sigma, mu);
    }
    return out;
}

keel::Tuple<keel::f64, keel::f64> gevstat(keel::f64 k, keel::f64 sigma, keel::f64 mu)
{
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    if (sigma <= 0.0) return keel::make_tuple(nan, nan);
    if (k == 0.0)
        return keel::make_tuple(mu + sigma * kEulerGamma, sigma * sigma * kPi2Over6);
    if (!(k < 1.0))
        return keel::make_tuple(keel::Limits<keel::f64>::infinity(), nan);
    const keel::f64 g1 = keel::tgamma(1.0 - k);
    const keel::f64 mean = mu + sigma * (g1 - 1.0) / k;
    if (!(k < 0.5))
        return keel::make_tuple(mean, keel::Limits<keel::f64>::infinity());
    const keel::f64 g2 = keel::tgamma(1.0 - 2.0 * k);
    const keel::f64 var = (sigma * sigma / (k * k)) * (g2 - g1 * g1);
    return keel::make_tuple(mean, var);
}

} // namespace numkit::stats
