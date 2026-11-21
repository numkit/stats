// toolboxes/stats/src/distributions/poisson.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/poisson.hpp>
#include <numkit/stats/distributions/gamma_dist.hpp>
#include <numkit/stats/distributions/binomial.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>

#include "poisson_detail.hpp"

namespace numkit::stats {

namespace {

keel::f64 samplePoisson1(::numkit::ops::RngContext &rng, keel::f64 lam, keel::Allocator *mr)
{
    if (!(lam > 0.0))
        return (lam == 0.0) ? 0.0 : std::numeric_limits<keel::f64>::quiet_NaN();
    if (!std::isfinite(lam))
        return std::numeric_limits<keel::f64>::quiet_NaN();
    keel::f64 k = 0.0;
    while (lam >= 15.0) {
        const keel::f64 m = std::floor(7.0 * lam / 8.0);
        const keel::f64 x = randg(rng, m, 1, 1, mr).toScalar();
        if (x <= lam) {
            k += m;
            lam -= x;
        } else {
            if (m > 1.0) {
                k += binornd(rng, keel::as<keel::i32>(m - 1.0), lam / x, 1, 1, mr).toScalar();
            }
            return k;
        }
    }
    keel::f64 t = -std::log(rng.nextUniform());
    while (t <= lam) {
        k += 1.0;
        t -= std::log(rng.nextUniform());
    }
    return k;
}

} // namespace

namespace {
inline bool poiss_argcheck(keel::f64 lambda) noexcept
{
    return lambda >= 0.0 && !std::isnan(lambda);
}
} // namespace

Value poisspdf(const Value &k, keel::f64 lambda, keel::Allocator *mr)
{
    (void)poiss_argcheck(lambda);
    return elementwise(k, [=](keel::f64 ki){ return keel::stats::poiss_pdf(ki, lambda); }, mr);
}

Value poisscdf(const Value &k, keel::f64 lambda, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki){ return keel::stats::poiss_cdf(ki, lambda); }, mr);
}

Value poissinv(const Value &p, keel::f64 lambda, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi){ return keel::stats::poiss_inv(pi, lambda); }, mr);
}

Value poissrnd(::numkit::ops::RngContext &rng, keel::f64 lambda, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    if (lambda < 0.0 || std::isnan(lambda)) {
        const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
        for (keel::usize i = 0; i < n; ++i) od[i] = nan;
        return out;
    }
    if (lambda == 0.0) {
        for (keel::usize i = 0; i < n; ++i) od[i] = 0.0;
        return out;
    }
    for (keel::usize i = 0; i < n; ++i) od[i] = samplePoisson1(rng, lambda, mr);
    return out;
}

std::tuple<keel::f64, keel::f64> poisstat(keel::f64 lambda)
{
    if (lambda <= 0.0) {
        const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(lambda, lambda);
}

} // namespace numkit::stats
