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

double samplePoisson1(::numkit::ops::RngContext &rng, double lam, keel::Allocator *mr)
{
    if (!(lam > 0.0))
        return (lam == 0.0) ? 0.0 : std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(lam))
        return std::numeric_limits<double>::quiet_NaN();
    double k = 0.0;
    while (lam >= 15.0) {
        const double m = std::floor(7.0 * lam / 8.0);
        const double x = randg(rng, m, 1, 1, mr).toScalar();
        if (x <= lam) {
            k += m;
            lam -= x;
        } else {
            if (m > 1.0) {
                k += binornd(rng, keel::as<int>(m - 1.0), lam / x, 1, 1, mr).toScalar();
            }
            return k;
        }
    }
    double t = -std::log(rng.nextUniform());
    while (t <= lam) {
        k += 1.0;
        t -= std::log(rng.nextUniform());
    }
    return k;
}

} // namespace

namespace {
inline bool poiss_argcheck(double lambda) noexcept
{
    return lambda >= 0.0 && !std::isnan(lambda);
}
} // namespace

Value poisspdf(const Value &k, double lambda, keel::Allocator *mr)
{
    (void)poiss_argcheck(lambda);
    return elementwise(k, [=](double ki){ return keel::stats::poiss_pdf(ki, lambda); }, mr);
}

Value poisscdf(const Value &k, double lambda, keel::Allocator *mr)
{
    return elementwise(k, [=](double ki){ return keel::stats::poiss_cdf(ki, lambda); }, mr);
}

Value poissinv(const Value &p, double lambda, keel::Allocator *mr)
{
    return elementwise(p, [=](double pi){ return keel::stats::poiss_inv(pi, lambda); }, mr);
}

Value poissrnd(::numkit::ops::RngContext &rng, double lambda, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    if (lambda < 0.0 || std::isnan(lambda)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
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

std::tuple<double, double> poisstat(double lambda)
{
    if (lambda <= 0.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(lambda, lambda);
}

} // namespace numkit::stats
