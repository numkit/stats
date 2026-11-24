// toolboxes/stats/src/distributions/unid.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/unid.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>

#include "unid_detail.hpp"

namespace numkit::stats {

Value unidpdf(const Value &k, keel::f64 N, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki) { return keel::stats::unid_pdf(ki, N); }, mr);
}

Value unidcdf(const Value &k, keel::f64 N, keel::Allocator *mr)
{
    return elementwise(k, [=](keel::f64 ki) { return keel::stats::unid_cdf(ki, N); }, mr);
}

Value unidinv(const Value &p, keel::f64 N, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::unid_inv(pi, N); }, mr);
}

Value unidrnd(::numkit::ops::RngContext &rng, keel::f64 N, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (N < 1.0 || keel::floor(N) != N || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize cnt = rows * cols;
    const keel::i64 nMax = keel::as<keel::i64>(N);
    for (keel::usize i = 0; i < cnt; ++i)
        od[i] = keel::as<keel::f64>(keel::random::uniform_i64(gen, 1, nMax));
    return out;
}

keel::Tuple<keel::f64, keel::f64> unidstat(keel::f64 N)
{
    const auto m = keel::stats::unid_stat(N);
    return keel::make_tuple(m.mean, m.var);
}

} // namespace numkit::stats
