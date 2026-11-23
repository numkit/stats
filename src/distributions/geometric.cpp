// toolboxes/stats/src/distributions/geometric.cpp

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/geometric.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>

#include "geometric_detail.hpp"

namespace numkit::stats {

namespace {
inline bool geo_argcheck(keel::f64 p) noexcept
{
    return p > 0.0 && p <= 1.0;
}
} // namespace

Value geopdf(const Value &k, keel::f64 p, keel::Allocator *mr)
{
    if (!geo_argcheck(p))
        return elementwise(k, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    return elementwise(k, [=](keel::f64 ki) {
        if (ki < 0.0 || keel::floor(ki) != ki) return 0.0;
        if (p == 1.0) return ki == 0.0 ? 1.0 : 0.0;
        return keel::pow(1.0 - p, ki) * p;
    }, mr);
}

Value geocdf(const Value &k, keel::f64 p, keel::Allocator *mr)
{
    if (p <= 0.0 || p > 1.0)
        return elementwise(k, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    return elementwise(k, [=](keel::f64 ki) {
        if (ki < 0.0) return 0.0;
        if (p == 1.0) return ki >= 0.0 ? 1.0 : 0.0;
        // F(k) = 1 - (1-p)^(⌊k⌋ + 1)
        return -keel::expm1((keel::floor(ki) + 1.0) * keel::log1p(-p));
    }, mr);
}

Value geoinv(const Value &q, keel::f64 p, keel::Allocator *mr)
{
    if (p <= 0.0 || p > 1.0)
        return elementwise(q, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    return elementwise(q, [=](keel::f64 qi) {
        if (!(qi >= 0.0 && qi <= 1.0)) return keel::Limits<keel::f64>::quiet_nan();
        if (qi == 0.0) return 0.0;
        if (qi >= 1.0) return keel::Limits<keel::f64>::infinity();
        if (p == 1.0) return 0.0;
        // Smallest integer k such that 1 - (1-p)^(k+1) ≥ qi
        //   ⇔ (1-p)^(k+1) ≤ 1-qi
        //   ⇔ k+1 ≥ log(1-qi) / log(1-p)
        //   ⇔ k ≥ log(1-qi)/log(1-p) - 1
        const keel::f64 v = keel::log1p(-qi) / keel::log1p(-p) - 1.0;
        keel::f64 k = keel::ceil(v);
        if (k < 0.0) k = 0.0;
        // One-ULP tolerance: if v - (k - 1) < tol, we already had k-1 satisfying.
        if (k > 0.0) {
            const keel::f64 cdf_prev = -keel::expm1(k * keel::log1p(-p)); // F(k-1)
            const keel::f64 tol = std::max(1e-13, qi * 1e-13);
            if (cdf_prev >= qi - tol) k -= 1.0;
        }
        return k;
    }, mr);
}

Value geornd(::numkit::ops::RngContext &rng, keel::f64 p, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (p <= 0.0 || p > 1.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize cnt = rows * cols;
    for (keel::usize i = 0; i < cnt; ++i)
        od[i] = keel::as<keel::f64>(keel::random::geometric_u64(gen, p));
    return out;
}

std::tuple<keel::f64, keel::f64> geostat(keel::f64 p)
{
    if (p <= 0.0 || p > 1.0) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return std::make_tuple(nan, nan);
    }
    const keel::f64 q = 1.0 - p;
    return std::make_tuple(q / p, q / (p * p));
}

} // namespace numkit::stats
