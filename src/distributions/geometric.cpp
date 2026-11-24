// toolboxes/stats/src/distributions/geometric.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/geometric.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>

#include "geometric_detail.hpp"

namespace numkit::stats {

Value geopdf(const Value &k, keel::f64 p, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki) { return keel::stats::geo_pdf(ki, p); }, mr);
}

Value geocdf(const Value &k, keel::f64 p, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki) { return keel::stats::geo_cdf(ki, p); }, mr);
}

Value geoinv(const Value &q, keel::f64 p, keel::Allocator *mr)
{
    return elementwise(q, [=](keel::f64 qi) { return keel::stats::geo_inv(qi, p); }, mr);
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

keel::Tuple<keel::f64, keel::f64> geostat(keel::f64 p)
{
    const auto m = keel::stats::geo_stat(p);
    return keel::make_tuple(m.mean, m.var);
}

} // namespace numkit::stats
