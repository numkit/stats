// toolboxes/stats/src/distributions/ncx2.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/ncx2.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>

#include "ncx2_detail.hpp"

namespace numkit::stats {

Value ncx2pdf(const Value &x, keel::f64 k, keel::f64 lambda, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::ncx2_pdf(xi, k, lambda); }, mr);
}

Value ncx2cdf(const Value &x, keel::f64 k, keel::f64 lambda, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::ncx2_cdf(xi, k, lambda); }, mr);
}

Value ncx2inv(const Value &p, keel::f64 k, keel::f64 lambda, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::ncx2_inv(pi, k, lambda); }, mr);
}

Value ncx2rnd(::numkit::ops::RngContext &rng, keel::f64 k, keel::f64 lambda, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (k <= 0.0 || lambda < 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    // Draw J ~ Poisson(λ/2), then X ~ chi²(k + 2J) = Gamma(shape=k/2+J, scale=2).
    for (keel::usize i = 0; i < n; ++i) {
        const keel::u64 J = (lambda == 0.0) ? 0ULL : keel::random::poisson_u64(gen, 0.5 * lambda);
        od[i] = keel::random::chi_squared_f64(gen, k + 2.0 * keel::as<keel::f64>(J));
    }
    return out;
}

keel::Tuple<keel::f64, keel::f64> ncx2stat(keel::f64 k, keel::f64 lambda)
{
    const auto m = keel::stats::ncx2_stat(k, lambda);
    return keel::make_tuple(m.mean, m.var);
}

} // namespace numkit::stats
