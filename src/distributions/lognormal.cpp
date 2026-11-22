// toolboxes/stats/src/distributions/lognormal.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/lognormal.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <keel/random.h>


#include "lognormal_detail.hpp"

namespace numkit::stats {

namespace {
inline bool logn_argcheck(keel::f64 sigma) noexcept
{
    return sigma > 0.0;
}
} // namespace

Value lognpdf(const Value &x, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    (void)logn_argcheck(sigma);
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::logn_pdf(xi, mu, sigma); }, mr);
}

Value logncdf(const Value &x, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::logn_cdf(xi, mu, sigma); }, mr);
}

Value logninv(const Value &p, keel::f64 mu, keel::f64 sigma, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::logn_inv(pi, mu, sigma); }, mr);
}

Value lognrnd(::numkit::ops::RngContext &rng, keel::f64 mu, keel::f64 sigma, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (sigma <= 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) od[i] = keel::random::lognormal_f64(gen, mu, sigma);
    return out;
}

std::tuple<keel::f64, keel::f64> lognstat(keel::f64 mu, keel::f64 sigma)
{
    if (sigma <= 0.0) {
        const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    const keel::f64 s2 = sigma * sigma;
    const keel::f64 mean = std::exp(mu + 0.5 * s2);
    const keel::f64 var = std::expm1(s2) * std::exp(2.0 * mu + s2);
    return std::make_tuple(mean, var);
}

} // namespace numkit::stats
