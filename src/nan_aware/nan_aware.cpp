// toolboxes/stats/src/nan_aware/nan_aware.cpp
//
// nansum / nanmean / nanmax / nanmin / nanvar / nanstdev / nanmedian.
// Extracted from toolboxes/builtin/src/stats.cpp during the
// MATLAB-taxonomy refactor — Statistics Toolbox content.

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/stats/nan_aware/nan_aware.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/ops/nan_reductions.hpp>

#include <algorithm>

#include "nan_aware_detail.hpp"

namespace numkit::stats {

using ::numkit::ops::applyAlongDim;
using ::numkit::ops::resolveDim;
using ::numkit::ops::nanSumScan;
using ::numkit::ops::nanSumCountScan;
using ::numkit::ops::nanMaxScan;
using ::numkit::ops::nanMinScan;
using ::numkit::ops::nanVarianceTwoPass;
using ::numkit::ops::compactNonNan;

namespace {
inline void nanvar_argcheck(keel::i32 normFlag, const char *fn)
{
    validateNormFlag(normFlag, fn);
}
} // namespace

Value nansum(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if ((x.dims().isVector() || x.isScalar()) && x.type() == ValueType::DOUBLE)
        return Value::scalar(nanSumScan(x.doubleData(), x.numel()), mr);

    const keel::i32 d = resolveDim(x, dim, "nansum");
    return applyAlongDim(x, d,
        [](keel::usize, keel::f64 *slice, keel::usize n) {
            return nanSumScan(slice, n); // all-NaN → 0
        }, mr);
}

Value nanmean(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if ((x.dims().isVector() || x.isScalar()) && x.type() == ValueType::DOUBLE) {
        const auto r = nanSumCountScan(x.doubleData(), x.numel());
        return Value::scalar(r.count > 0 ? r.sum / keel::as<keel::f64>(r.count)
                                          : keel::kNaN, mr);
    }

    const keel::i32 d = resolveDim(x, dim, "nanmean");
    return applyAlongDim(x, d,
        [](keel::usize, keel::f64 *slice, keel::usize n) {
            const auto r = nanSumCountScan(slice, n);
            return r.count > 0 ? r.sum / keel::as<keel::f64>(r.count)
                               : keel::kNaN;
        }, mr);
}

Value nanmax(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if ((x.dims().isVector() || x.isScalar()) && x.type() == ValueType::DOUBLE)
        return Value::scalar(nanMaxScan(x.doubleData(), x.numel()), mr);

    const keel::i32 d = resolveDim(x, dim, "nanmax");
    return applyAlongDim(x, d,
        [](keel::usize, keel::f64 *slice, keel::usize n) {
            return nanMaxScan(slice, n);
        }, mr);
}

Value nanmin(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if ((x.dims().isVector() || x.isScalar()) && x.type() == ValueType::DOUBLE)
        return Value::scalar(nanMinScan(x.doubleData(), x.numel()), mr);

    const keel::i32 d = resolveDim(x, dim, "nanmin");
    return applyAlongDim(x, d,
        [](keel::usize, keel::f64 *slice, keel::usize n) {
            return nanMinScan(slice, n);
        }, mr);
}

Value nanvar(const Value &x, keel::i32 normFlag, keel::i32 dim, keel::Allocator *mr)
{
    nanvar_argcheck(normFlag, "nanvar");
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if ((x.dims().isVector() || x.isScalar()) && x.type() == ValueType::DOUBLE)
        return Value::scalar(nanVarianceTwoPass(x.doubleData(), x.numel(), normFlag), mr);

    const keel::i32 d = resolveDim(x, dim, "nanvar");
    return applyAlongDim(x, d,
        [normFlag](keel::usize, keel::f64 *slice, keel::usize n) {
            return nanVarianceTwoPass(slice, n, normFlag);
        }, mr);
}

Value nanstdev(const Value &x, keel::i32 normFlag, keel::i32 dim, keel::Allocator *mr)
{
    validateNormFlag(normFlag, "nanstd");
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if ((x.dims().isVector() || x.isScalar()) && x.type() == ValueType::DOUBLE)
        return Value::scalar(keel::sqrt(nanVarianceTwoPass(x.doubleData(), x.numel(), normFlag)), mr);

    const keel::i32 d = resolveDim(x, dim, "nanstd");
    return applyAlongDim(x, d,
        [normFlag](keel::usize, keel::f64 *slice, keel::usize n) {
            return keel::sqrt(nanVarianceTwoPass(slice, n, normFlag));
        }, mr);
}

Value nanmedian(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = resolveDim(x, dim, "nanmedian");
    return applyAlongDim(x, d,
        [](keel::usize, keel::f64 *slice, keel::usize n) {
            const keel::usize k = compactNonNan(slice, n);
            return medianFromSlice(slice, k); // returns NaN at k==0
        }, mr);
}

// ── nancov ─────────────────────────────────────────────────────────
//
// `complete` mode only (== MATLAB default `cov(X, 'omitrows')`): drop
// every observation row that has at least one NaN, then standard
// covariance. `'pairwise'` mode is a documented v1 gap.


Value nancov(const Value &x, keel::i32 normFlag, keel::Allocator *mr)
{
    validateNormFlag(normFlag, "nancov");
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> raw(scratch.allocator());
    keel::usize n_orig = 0, p = 0;
    readObsMatrix(x, raw, n_orig, p);

    // Drop NaN-containing rows. We can do this in-place on `raw` since
    // dropNanRows compacts within each column down to row index w-1.
    const keel::usize n_kept = dropNanRows(raw.data(), n_orig, p);

    if (n_kept == 0) {
        // All rows dropped → MATLAB returns NaN-filled p×p (or scalar NaN for vector).
        if (p == 1) return Value::scalar(keel::kNaN, mr);
        auto out = Value::matrix(p, p, ValueType::DOUBLE, mr);
        keel::f64 *d = out.doubleDataMut();
        for (keel::usize k = 0; k < p * p; ++k) d[k] = keel::kNaN;
        return out;
    }

    // Pack the kept rows into a tight n_kept × p layout.
    keel::Array<keel::f64> tight(n_kept * p, scratch.allocator());
    packKeptRows(raw.data(), n_orig, n_kept, p, tight.data());

    centerColumnsLocal(tight.data(), n_kept, p);
    const keel::f64 divisor = (normFlag == 0)
        ? std::max(1.0, keel::as<keel::f64>(n_kept) - 1.0)
        : keel::as<keel::f64>(n_kept);

    if (p == 1) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < n_kept; ++i)
            s += tight[i] * tight[i];
        return Value::scalar(s / divisor, mr);
    }
    return covFromCenteredLocal(tight.data(), n_kept, p, divisor, mr);
}

Value nancov(const Value &x, const Value &y, keel::i32 normFlag,
             keel::Allocator *mr)
{
    validateNormFlag(normFlag, "nancov");
    if (!x.dims().isVector() || !y.dims().isVector())
        throw Error("nancov: two-input form requires vector arguments",
                    0, 0, "nancov", "", "numkit:nancov:notVector");
    if (x.numel() != y.numel())
        throw Error("nancov: x and y must have the same length",
                    0, 0, "nancov", "", "numkit:nancov:lengthMismatch");

    const keel::usize n = x.numel();
    if (n == 0)
        return Value::matrix(2, 2, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto data = keel::Array<keel::f64>(n * 2, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        data[i]       = x.elemAsDouble(i);      // column 0
        data[n + i]   = y.elemAsDouble(i);      // column 1
    }
    const keel::usize n_kept = dropNanRows(data.data(), n, 2);
    if (n_kept == 0) {
        auto out = Value::matrix(2, 2, ValueType::DOUBLE, mr);
        keel::f64 *d = out.doubleDataMut();
        for (keel::usize k = 0; k < 4; ++k) d[k] = keel::kNaN;
        return out;
    }
    keel::Array<keel::f64> tight(n_kept * 2, scratch.allocator());
    packKeptRows(data.data(), n, n_kept, 2, tight.data());
    centerColumnsLocal(tight.data(), n_kept, 2);
    const keel::f64 divisor = (normFlag == 0)
        ? std::max(1.0, keel::as<keel::f64>(n_kept) - 1.0)
        : keel::as<keel::f64>(n_kept);
    return covFromCenteredLocal(tight.data(), n_kept, 2, divisor, mr);
}

} // namespace numkit::stats
