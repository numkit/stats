// toolboxes/stats/src/distributions/unid.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/unid.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <keel/random.h>

#include "unid_detail.hpp"

namespace numkit::stats {

namespace {
inline bool unid_argcheck(keel::f64 N) noexcept
{
    return N >= 1.0 && keel::floor(N) == N;
}
} // namespace

Value unidpdf(const Value &k, keel::f64 N, keel::Allocator *mr)
{
    if (!unid_argcheck(N))
        return elementwise(k, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    const keel::f64 inv = 1.0 / N;
    return elementwise(k, [=](keel::f64 ki) {
        if (ki < 1.0 || ki > N || keel::floor(ki) != ki) return 0.0;
        return inv;
    }, mr);
}

Value unidcdf(const Value &k, keel::f64 N, keel::Allocator *mr)
{
    if (N < 1.0 || keel::floor(N) != N)
        return elementwise(k, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    const keel::f64 inv = 1.0 / N;
    return elementwise(k, [=](keel::f64 ki) {
        if (ki < 1.0) return 0.0;
        if (ki >= N) return 1.0;
        return keel::floor(ki) * inv;
    }, mr);
}

Value unidinv(const Value &p, keel::f64 N, keel::Allocator *mr)
{
    if (!(N >= 1.0) || keel::floor(N) != N)  // also catches NaN N
        return elementwise(p, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    return elementwise(p, [=](keel::f64 pi) {
        // MATLAB convention: p outside (0, 1] -> NaN. p=0 has no
        // integer pre-image in the support, so it is also NaN.
        if (keel::is_nan(pi) || pi <= 0.0 || pi > 1.0)
            return keel::Limits<keel::f64>::quiet_nan();
        const keel::f64 r = keel::ceil(pi * N);
        return r < 1.0 ? 1.0 : (r > N ? N : r);
    }, mr);
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
    if (N < 1.0 || keel::floor(N) != N) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return keel::make_tuple(nan, nan);
    }
    return keel::make_tuple(0.5 * (N + 1.0), (N * N - 1.0) / 12.0);
}

} // namespace numkit::stats
