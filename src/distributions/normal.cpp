// toolboxes/stats/src/distributions/normal.cpp
//
// Normal distribution. Standard formulas:
//   pdf:  f(x; μ, σ) = (1/(σ√(2π))) · exp(-(x-μ)² / (2σ²))
//   cdf:  F(x; μ, σ) = ½ · [1 + erf((x-μ) / (σ√2))]
//   icdf: F⁻¹(p; μ, σ) = μ + σ·√2·erfinv(2p - 1)
//   rnd:  μ + σ · randn
//   stat: [mean=μ, var=σ²]

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/normal.hpp>

#include <numkit/builtin/datafun.hpp>  // RngContext + rand/randn/randi/randperm (session-scoped, no global/mutex)
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"            // stripUpperFlag / applyUpperInPlace



#include "normal_detail.hpp"

namespace numkit::stats {

namespace {
inline bool norm_argcheck(keel::f64 sigma) noexcept
{
    return sigma > 0.0;
}
} // namespace

Value normpdf(const Value &x, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return normpdfK(xi, mu, sigma); }, mr);
}

Value normcdf(const Value &x, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return normcdfK(xi, mu, sigma); }, mr);
}

Value norminv(const Value &p, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return norminvK(pi, mu, sigma); }, mr);
}

Value normrnd(::numkit::ops::RngContext &rng, keel::f64 mu, keel::f64 sigma, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    if (sigma < 0.0) sigma = 0.0;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) od[i] = mu + sigma * rng.nextNormal();
    return out;
}

keel::Tuple<keel::f64, keel::f64> normstat(keel::f64 mu, keel::f64 sigma)
{
    if (!norm_argcheck(sigma)) {
        const keel::f64 NaN = keel::Limits<keel::f64>::quiet_nan();
        return keel::make_tuple(NaN, NaN);
    }
    return keel::make_tuple(mu, sigma * sigma);
}

} // namespace numkit::stats
