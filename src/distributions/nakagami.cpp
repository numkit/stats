// toolboxes/stats/src/distributions/nakagami.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/nakagami.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>

#include "nakagami_detail.hpp"

namespace numkit::stats {

Value nakapdf(const Value &x, keel::f64 mu, keel::f64 omega, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::naka_pdf(xi, mu, omega); }, mr);
}

Value nakacdf(const Value &x, keel::f64 mu, keel::f64 omega, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::naka_cdf(xi, mu, omega); }, mr);
}

Value nakainv(const Value &p, keel::f64 mu, keel::f64 omega, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::naka_inv(pi, mu, omega); }, mr);
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

keel::Tuple<keel::f64, keel::f64> nakastat(keel::f64 mu, keel::f64 omega)
{
    const auto m = keel::stats::naka_stat(mu, omega);
    return keel::make_tuple(m.mean, m.var);
}

} // namespace numkit::stats
