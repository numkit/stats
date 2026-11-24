// toolboxes/stats/src/distributions/negbin.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/negbin.hpp>
#include <numkit/stats/distributions/gamma_dist.hpp>
#include <numkit/stats/distributions/poisson.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include "negbin_detail.hpp"

namespace numkit::stats {

Value nbinpdf(const Value &k, keel::f64 r, keel::f64 p, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki) { return keel::stats::nbin_pdf(ki, r, p); }, mr);
}

Value nbincdf(const Value &k, keel::f64 r, keel::f64 p, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki) { return keel::stats::nbin_cdf(ki, r, p); }, mr);
}

Value nbininv(const Value &q, keel::f64 r, keel::f64 p, keel::Allocator *mr)
{
    return elementwise(q, [=](keel::f64 qi) { return keel::stats::nbin_inv(qi, r, p); }, mr);
}

Value nbinrnd(::numkit::ops::RngContext &rng, keel::f64 r, keel::f64 p, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (r <= 0.0 || p <= 0.0 || p > 1.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize cnt = rows * cols;
    if (p == 1.0) {
        for (keel::usize i = 0; i < cnt; ++i) od[i] = 0.0;
        return out;
    }
    // Sample via Gamma-Poisson mixture matching the reference engine's nbinrnd:
    // lambda ~ Gamma(r, (1-p)/p), K | lambda ~ Poisson(lambda).
    Value g = gamrnd(rng, r, (1.0 - p) / p, rows, cols, mr);
    const keel::f64 *gd = g.doubleData();
    for (keel::usize i = 0; i < cnt; ++i) {
        const keel::f64 lam = gd[i];
        od[i] = (lam <= 0.0) ? 0.0 : poissrnd(rng, lam, 1, 1, mr).toScalar();
    }
    return out;
}

keel::Tuple<keel::f64, keel::f64> nbinstat(keel::f64 r, keel::f64 p)
{
    const auto m = keel::stats::nbin_stat(r, p);
    return keel::make_tuple(m.mean, m.var);
}

} // namespace numkit::stats
