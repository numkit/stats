// toolboxes/stats/src/distributions/nakagami.cpp

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/nakagami.hpp>

#include <numkit/builtin/specfun.hpp>   // gammainc, gammaincinv
#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <mutex>
#include <keel/random.h>

#include "nakagami_detail.hpp"

namespace numkit::stats {

namespace {
inline bool naka_argcheck(keel::f64 mu, keel::f64 omega) noexcept
{
    return mu > 0.0 && omega > 0.0;
}
} // namespace

Value nakapdf(const Value &x, keel::f64 mu, keel::f64 omega, keel::Allocator *mr)
{
    if (!naka_argcheck(mu, omega))
        return elementwise(x, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    // Pre-compute constants
    const keel::f64 C = 2.0 * keel::pow(mu, mu) / (keel::tgamma(mu) * keel::pow(omega, mu));
    const keel::f64 k = mu / omega;
    return elementwise(x, [=](keel::f64 xi) {
        if (xi < 0.0) return 0.0;
        if (xi == 0.0) return (mu < 0.5) ? keel::Limits<keel::f64>::infinity()
                                         : (mu == 0.5 ? C : 0.0);
        return C * keel::pow(xi, 2.0 * mu - 1.0) * keel::exp(-k * xi * xi);
    }, mr);
}

Value nakacdf(const Value &x, keel::f64 mu, keel::f64 omega, keel::Allocator *mr)
{
    if (mu <= 0.0 || omega <= 0.0)
        return elementwise(x, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    const keel::f64 k = mu / omega;
    return elementwise(x, [=](keel::f64 xi) {
        if (xi <= 0.0) return 0.0;
        return gammainc_scalar(k * xi * xi, mu, mr);
    }, mr);
}

Value nakainv(const Value &p, keel::f64 mu, keel::f64 omega, keel::Allocator *mr)
{
    if (mu <= 0.0 || omega <= 0.0)
        return elementwise(p, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    const keel::f64 s = keel::sqrt(omega / mu);
    return elementwise(p, [=](keel::f64 pi) {
        if (!(pi >= 0.0 && pi <= 1.0)) return keel::Limits<keel::f64>::quiet_nan();
        if (pi == 0.0) return 0.0;
        if (pi >= 1.0) return keel::Limits<keel::f64>::infinity();
        const keel::f64 q = gammaincinv_scalar(pi, mu, mr);
        return s * keel::sqrt(q);
    }, mr);
}

Value nakarnd(::numkit::ops::RngContext &rng, keel::f64 mu, keel::f64 omega, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (mu <= 0.0 || omega <= 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    // X² ~ Gamma(shape=mu, scale=omega/mu) ⇒ X = √Gamma.
    for (keel::usize i = 0; i < n; ++i) od[i] = keel::sqrt(keel::random::gamma_f64(gen, mu, omega / mu));
    return out;
}

std::tuple<keel::f64, keel::f64> nakastat(keel::f64 mu, keel::f64 omega)
{
    if (mu <= 0.0 || omega <= 0.0) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return std::make_tuple(nan, nan);
    }
    const keel::f64 r = keel::tgamma(mu + 0.5) / keel::tgamma(mu);
    const keel::f64 mean = keel::sqrt(omega / mu) * r;
    const keel::f64 var  = omega * (1.0 - r * r / mu);
    return std::make_tuple(mean, var);
}

} // namespace numkit::stats
