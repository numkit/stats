// toolboxes/stats/src/moments/moments.cpp
//
// skewness + kurtosis. Extracted from toolboxes/builtin/src/stats.cpp
// during the MATLAB-taxonomy refactor — they're Statistics Toolbox
// content (var/std/median stay in toolboxes/builtin as base MATLAB).

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <numkit/stats/moments/moments.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>

#include <cmath>
#include <utility>

namespace numkit::stats {

using ::numkit::ops::applyAlongDim;
using ::numkit::ops::resolveDim;
using ::numkit::createForDims;

namespace {

// Cast a DOUBLE result to SINGLE in place. Used to preserve SINGLE
// input type — arithmetic happens at double precision then narrows.
Value narrowToSingle(Value d, keel::Allocator *mr)
{
    if (d.type() != ValueType::DOUBLE) return d;
    Value r = createForDims(d.dims(), ValueType::SINGLE, mr);
    const keel::f64 *src = d.doubleData();
    keel::f32 *dst = r.singleDataMut();
    for (keel::usize i = 0; i < d.numel(); ++i)
        dst[i] = keel::as<keel::f32>(src[i]);
    return r;
}

// Per-slice skewness. n < 2: NaN (m2 == 0 by definition). normFlag=0
// requires n >= 3 for the bias correction; otherwise NaN.
keel::f64 skewnessFromSlice(const keel::f64 *data, keel::usize n, keel::i32 normFlag)
{
    if (n < 2) return std::nan("");
    keel::f64 mean = 0.0;
    for (keel::usize i = 0; i < n; ++i) mean += data[i];
    mean /= keel::as<keel::f64>(n);
    keel::f64 m2 = 0.0, m3 = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 d = data[i] - mean;
        const keel::f64 d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
    }
    m2 /= keel::as<keel::f64>(n);
    m3 /= keel::as<keel::f64>(n);
    if (m2 <= 0.0) return std::nan("");
    keel::f64 y = m3 / std::pow(m2, 1.5);
    if (normFlag == 0) {
        if (n < 3) return std::nan("");
        const keel::f64 nd = keel::as<keel::f64>(n);
        y *= std::sqrt(nd * (nd - 1.0)) / (nd - 2.0);
    }
    return y;
}

// Per-slice kurtosis (NON-excess; equals 3 for normal). normFlag=0
// requires n >= 4 for the bias correction; otherwise NaN.
keel::f64 kurtosisFromSlice(const keel::f64 *data, keel::usize n, keel::i32 normFlag)
{
    if (n < 2) return std::nan("");
    keel::f64 mean = 0.0;
    for (keel::usize i = 0; i < n; ++i) mean += data[i];
    mean /= keel::as<keel::f64>(n);
    keel::f64 m2 = 0.0, m4 = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 d = data[i] - mean;
        const keel::f64 d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= keel::as<keel::f64>(n);
    m4 /= keel::as<keel::f64>(n);
    if (m2 <= 0.0) return std::nan("");
    keel::f64 y = m4 / (m2 * m2);
    if (normFlag == 0) {
        if (n < 4) return std::nan("");
        const keel::f64 nd = keel::as<keel::f64>(n);
        y = ((nd - 1.0) / ((nd - 2.0) * (nd - 3.0)))
                * ((nd + 1.0) * y - 3.0 * (nd - 1.0))
            + 3.0;
    }
    return y;
}

void moments_argcheck(const Value &x, keel::i32 normFlag, const char *fn)
{
    if (normFlag != 0 && normFlag != 1)
        throw Error(keel::format("{}: normalization flag must be 0 or 1", fn),
                     0, 0, fn, "", keel::format("numkit:{}:badFlag", fn));
    if (x.type() == ValueType::COMPLEX)
        throw Error(keel::format("{}: complex inputs are not supported", fn),
                     0, 0, fn, "", keel::format("numkit:{}:complex", fn));
}

Value dispatchMomentReduction(const Value &x, keel::i32 dim, keel::i32 normFlag, const char *fn, keel::f64 (*fromSlice)(const keel::f64 *, keel::usize, keel::i32), keel::Allocator *mr)
{
    moments_argcheck(x, normFlag, fn);
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    const keel::i32 d = resolveDim(x, dim, fn);
    Value r = applyAlongDim(x, d,
        [normFlag, fromSlice](keel::usize, keel::f64 *slice, keel::usize n) {
            // MATLAB skewness/kurtosis treat NaN as missing and remove it.
            // Compact the non-NaN values to the front of the (mutable)
            // slice and reduce the count before the moment kernel runs.
            keel::usize k = 0;
            for (keel::usize i = 0; i < n; ++i)
                if (!std::isnan(slice[i])) slice[k++] = slice[i];
            return fromSlice(slice, k, normFlag);
        }, mr);
    if (x.type() == ValueType::SINGLE)
        r = narrowToSingle(std::move(r), mr);
    return r;
}

} // namespace

Value skewness(const Value &x, keel::i32 normFlag, keel::i32 dim, keel::Allocator *mr)
{
    return dispatchMomentReduction(x, dim, normFlag, "skewness", skewnessFromSlice, mr);
}

Value kurtosis(const Value &x, keel::i32 normFlag, keel::i32 dim, keel::Allocator *mr)
{
    return dispatchMomentReduction(x, dim, normFlag, "kurtosis", kurtosisFromSlice, mr);
}

} // namespace numkit::stats
