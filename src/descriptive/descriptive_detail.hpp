// toolboxes/.../descriptive_detail.hpp â€” private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by descriptive.cpp + descriptive_reg.cpp.
// Phase 2b compute/register split â€” see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/helpers.hpp>            // createForDims/createMatrix/DimsArg (engine-free)
#include <numkit/ops/reductions.hpp>  // numkit::builtin::detail dim-infra (engine-free, ops re-export)
#include <numkit/builtin/detail/datafun/var_reduction.hpp> // varianceTwoPass (engine-free)

#include <keel/tuple.h>


namespace numkit::stats {

using namespace ::numkit::ops;
using ::numkit::builtin::varianceTwoPass;

// Named (external-linkage) so the file-scope worker quantileWithOpts can
// take it across the compute/register TU boundary.
enum class QMethod { Midpoint, Inclusive, Exclusive, Approximate };

namespace {

[[maybe_unused]] void validateNormFlag(keel::i32 w, const char *fn)
{
    if (w != 0 && w != 1)
        throw Error(keel::format("{}: normalization flag must be 0 or 1", fn),
                     0, 0, fn, "", keel::format("numkit:{}:badFlag", fn));
}

// Cast a DOUBLE result to SINGLE in place. Used to preserve SINGLE
// input type without writing parallel single-precision kernels â€”
// arithmetic happens at double precision (more precise than MATLAB)
// then narrows at the end.
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

// Cast a DOUBLE result back to an integer class: round half away from zero
// (std::round) and saturate to the type's range. Matches MATLAB R2025b, which
// preserves the integer class for median (median(int32([1 2 3 4]))=3 int32,
// median(int8([-1 -2]))=-2 int8) â€” the two-middle-element average is rounded
// half-away-from-zero and the class is retained.
[[maybe_unused]] Value narrowToInteger(const Value &d, ValueType vt, keel::Allocator *mr)
{
    if (d.type() != ValueType::DOUBLE) return d;
    Value r = createForDims(d.dims(), vt, mr);
    const keel::f64 *src = d.doubleData();
    const keel::usize n = d.numel();
    auto fill = [&](auto *dst) {
        using T = keel::RemovePointer<keel::Decay<decltype(dst)>>;
        const keel::f64 lo = keel::as<keel::f64>(keel::Limits<T>::kMin);
        const keel::f64 hi = keel::as<keel::f64>(keel::Limits<T>::kMax);
        for (keel::usize i = 0; i < n; ++i) {
            keel::f64 v = keel::round(src[i]);   // round half away from zero
            if (v < lo) v = lo;
            else if (v > hi) v = hi;          // saturate to class range
            dst[i] = static_cast<T>(v);
        }
    };
    switch (vt) {
    case ValueType::INT8:   fill(r.int8DataMut());   break;
    case ValueType::INT16:  fill(r.int16DataMut());  break;
    case ValueType::INT32:  fill(r.int32DataMut());  break;
    case ValueType::INT64:  fill(r.int64DataMut());  break;
    case ValueType::UINT8:  fill(r.uint8DataMut());  break;
    case ValueType::UINT16: fill(r.uint16DataMut()); break;
    case ValueType::UINT32: fill(r.uint32DataMut()); break;
    case ValueType::UINT64: fill(r.uint64DataMut()); break;
    default: return d;
    }
    return r;
}

// Complex variance: E[|x - mean|Â²]. Returns real-valued DOUBLE per
// MATLAB convention. With normFlag == 0 (default) divides by n-1;
// normFlag == 1 divides by n. When omitNan is true, NaN-complex
// elements (real or imag is NaN) are skipped before mean/variance.
inline bool isComplexNaNStats(keel::c64 c)
{
    return keel::is_nan(c.real()) || keel::is_nan(c.imag());
}

keel::f64 complexVarianceFromSlice(const keel::c64 *data, keel::usize n, keel::i32 normFlag,
                                bool omitNan = false)
{
    keel::c64 mean(0.0, 0.0);
    keel::usize k = 0;
    for (keel::usize i = 0; i < n; ++i) {
        if (omitNan && isComplexNaNStats(data[i])) continue;
        mean += data[i];
        ++k;
    }
    if (k == 0) return keel::kNaN;
    if (k == 1) return (normFlag == 0) ? keel::kNaN : 0.0;
    mean /= keel::as<keel::f64>(k);
    keel::f64 acc = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        if (omitNan && isComplexNaNStats(data[i])) continue;
        const keel::c64 d = data[i] - mean;
        acc += d.real() * d.real() + d.imag() * d.imag();
    }
    const keel::f64 divisor = keel::as<keel::f64>(k - (normFlag == 0 ? 1 : 0));
    return acc / divisor;
}

inline Value allocVarianceOutput(const Value &x, keel::i32 redDim, keel::Allocator *mr)
{
    if (x.dims().ndim() >= 4 && redDim >= 1 && redDim <= x.dims().ndim()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        auto shape = outShapeForDimND(x, redDim, scratch.allocator());
        return Value::matrixND(shape.data(), (keel::i32) shape.size(), ValueType::DOUBLE, mr);
    }
    auto outShape = outShapeForDim(x, redDim);
    return createMatrix(outShape, ValueType::DOUBLE, mr);
}

// MATLAB empty-input result for the NaN-returning descriptive reductions
// (var / std / median / mode), mirroring mean's empty rule:
//   * default (no explicit dim) of the 0x0 [] -> scalar NaN
//   * otherwise the operating dim (explicit dim when given, else the first
//     dim whose size != 1 â€” MATLAB treats a size-0 dim as non-singleton)
//     collapses to 1 and the result is NaN-filled.
// A size-0 sibling dim keeps the result empty, e.g. var(zeros(3,0))=1x0,
// median([],2)=0x1. nk previously returned a 0x0 empty for all of these.
inline Value emptyStatReductionFill(const Value &x, keel::i32 dim, keel::f64 fill,
                                    keel::Allocator *mr)
{
    const auto &dd = x.dims();
    if (dim < 1 && dd.ndim() == 2 && dd.rows() == 0 && dd.cols() == 0)
        return Value::scalar(fill, mr);
    keel::i32 opDim = dim;
    if (opDim < 1) {
        opDim = 1;
        const keel::i32 nd = dd.ndim();
        for (keel::i32 i = 0; i < nd; ++i)
            if (dd.dim(i) != 1) { opDim = i + 1; break; }
    }
    DimsArg o{dd.rows(), dd.cols(), dd.is3D() ? dd.pages() : 0};
    switch (opDim) {
        case 1: o.rows  = 1; break;
        case 2: o.cols  = 1; break;
        case 3: o.pages = (o.pages == 0) ? 0 : 1; break;
        default: break;
    }
    Value out = createMatrix(o, ValueType::DOUBLE, mr);
    keel::f64 *p = out.doubleDataMut();
    const keel::usize n = out.numel();
    for (keel::usize i = 0; i < n; ++i) p[i] = fill;
    return out;
}

inline Value emptyStatReductionNaN(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    return emptyStatReductionFill(x, dim, keel::kNaN, mr);
}

// Complex variance along dim: walks slices via stride math, gathers
// each slice into a Complex scratch buffer, computes per-slice complex
// variance, writes DOUBLE output.
void complexVarianceAlongDim(const Value &x, keel::i32 redDim, keel::f64 *dst, keel::i32 normFlag, bool omitNan, keel::Allocator *mr)
{
    const auto &d = x.dims();
    const keel::i32 redAxis = redDim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (keel::i32 i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (keel::i32 i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);

    const keel::c64 *src = x.complexData();
    keel::Array<keel::c64> scratch(sliceLen, mr);
    auto reduceSlice = [&](keel::usize outOff, keel::usize baseOff, keel::usize stride) {
        for (keel::usize k = 0; k < sliceLen; ++k)
            scratch[k] = src[baseOff + k * stride];
        dst[outOff] = complexVarianceFromSlice(scratch.data(), sliceLen, normFlag, omitNan);
    };
    if (B == 1) {
        for (keel::usize o = 0; o < O; ++o) reduceSlice(o, o * sliceLen, 1);
        return;
    }
    for (keel::usize o = 0; o < O; ++o)
        for (keel::usize b = 0; b < B; ++b)
            reduceSlice(o * B + b, o * sliceLen * B + b, B);
}

[[maybe_unused]] Value varianceComplex(const Value &x, keel::i32 normFlag, keel::i32 dim,
                       keel::Allocator *mr, bool sqrtIt, bool omitNan = false)
{
    if (x.isEmpty())
        return emptyStatReductionNaN(x, dim, mr);
    const keel::c64 *src = x.complexData();
    if (x.isScalar() || x.dims().isVector()) {
        keel::f64 v = complexVarianceFromSlice(src, x.numel(), normFlag, omitNan);
        if (sqrtIt && !keel::is_nan(v)) v = keel::sqrt(v);
        return Value::scalar(v, mr);
    }
    const keel::i32 d = (dim > 0) ? dim : firstNonSingletonDim(x);
    Value out = allocVarianceOutput(x, d, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    complexVarianceAlongDim(x, d, out.doubleDataMut(), normFlag, omitNan, scratch.allocator());
    if (sqrtIt) {
        keel::f64 *p = out.doubleDataMut();
        const keel::usize n = out.numel();
        for (keel::usize i = 0; i < n; ++i)
            if (!keel::is_nan(p[i])) p[i] = keel::sqrt(p[i]);
    }
    return out;
}

} // namespace
namespace {

[[maybe_unused]] keel::f64 medianFromSlice(keel::f64 *data, keel::usize n)
{
    if (n == 0) return keel::kNaN;
    if (n == 1) return data[0];
    // MATLAB R2025b default: NaN poisons (median(v) where v has NaN
    // returns NaN; 'omitnan' is opt-in via the explicit nanflag, which
    // routes to nanmedian instead of this kernel).
    for (keel::usize i = 0; i < n; ++i)
        if (keel::is_nan(data[i])) return keel::kNaN;
    const keel::usize mid = n / 2;
    keel::nth_element(data, data + mid, data + n);
    if (n % 2 == 1)
        return data[mid];
    const keel::f64 upper = data[mid];
    const keel::f64 lower = *keel::max_element(data, data + mid);
    return 0.5 * (lower + upper);
}

} // namespace
namespace {

// Complex ordering: by magnitude, ties broken by phase angle (matches nk
// sort/max and MATLAB's complex median ordering).
inline bool complexAbsAngleLess(const keel::c64 &a, const keel::c64 &b)
{
    const keel::f64 aa = keel::abs(a), ab = keel::abs(b);
    if (aa != ab) return aa < ab;
    return keel::arg(a) < keel::arg(b);
}

// Median of a complex slice (mutates s by sorting). Propagates NaN (MATLAB
// 'includenan' default). Even n â†’ mean of the two middle values.
keel::c64 complexMedianFromSlice(keel::c64 *s, keel::usize n)
{
    if (n == 0) return keel::c64(keel::kNaN, keel::kNaN);
    for (keel::usize i = 0; i < n; ++i)
        if (isComplexNaNStats(s[i])) return keel::c64(keel::kNaN, keel::kNaN);
    keel::sort(s, s + n, complexAbsAngleLess);
    if (n & 1u) return s[n / 2];
    return 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

// Per-dim complex median (mirrors complexVarianceAlongDim; COMPLEX output).
void complexMedianAlongDim(const Value &x, keel::i32 redDim, keel::c64 *dst,
                           keel::Allocator *mr)
{
    const auto &d = x.dims();
    const keel::i32 redAxis = redDim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (keel::i32 i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (keel::i32 i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);
    const keel::c64 *src = x.complexData();
    keel::Array<keel::c64> scratch(sliceLen, mr);
    auto reduceSlice = [&](keel::usize outOff, keel::usize baseOff, keel::usize stride) {
        for (keel::usize k = 0; k < sliceLen; ++k) scratch[k] = src[baseOff + k * stride];
        dst[outOff] = complexMedianFromSlice(scratch.data(), sliceLen);
    };
    if (B == 1) {
        for (keel::usize o = 0; o < O; ++o) reduceSlice(o, o * sliceLen, 1);
        return;
    }
    for (keel::usize o = 0; o < O; ++o)
        for (keel::usize b = 0; b < B; ++b)
            reduceSlice(o * B + b, o * sliceLen * B + b, B);
}

[[maybe_unused]] Value medianComplex(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (x.isEmpty()) return emptyStatReductionNaN(x, dim, mr);
    if (x.isScalar()) return Value::complexScalar(x.complexData()[0], mr);
    const keel::i32 d = resolveDim(x, dim, "median");
    Value out;
    if (x.dims().ndim() >= 4 && d >= 1 && d <= x.dims().ndim()) {
        keel::ArenaTemp sc = keel::get_scratch(mr);
        auto shape = outShapeForDimND(x, d, sc.allocator());
        out = Value::matrixND(shape.data(), keel::as<keel::i32>(shape.size()),
                              ValueType::COMPLEX, mr);
    } else {
        auto outShape = outShapeForDim(x, d);
        out = createMatrix(outShape, ValueType::COMPLEX, mr);
    }
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    complexMedianAlongDim(x, d, out.complexDataMut(), scratch.allocator());
    return out;
}

} // namespace
namespace {


keel::f64 quantileFromSortedSlice(const keel::f64 *sorted, keel::usize n, keel::f64 p,
                               QMethod method)
{
    if (n == 0) return keel::kNaN;
    if (n == 1) return sorted[0];
    if (!keel::is_finite(p)) return keel::kNaN;

    // Compute 1-based real-valued position q according to the method.
    keel::f64 q = 0.0;
    switch (method) {
        case QMethod::Midpoint:
        case QMethod::Approximate:                     // fallback
            q = p * keel::as<keel::f64>(n) + 0.5;
            break;
        case QMethod::Inclusive:
            q = p * keel::as<keel::f64>(n - 1) + 1.0;
            break;
        case QMethod::Exclusive:
            q = p * keel::as<keel::f64>(n + 1);
            break;
    }
    if (q <= 1.0) return sorted[0];
    if (q >= keel::as<keel::f64>(n)) return sorted[n - 1];
    const keel::usize lo = keel::as<keel::usize>(keel::floor(q)) - 1; // â†’ 0-based
    const keel::usize hi = lo + 1;
    const keel::f64 frac = q - keel::floor(q);
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

// quantile/prctile share the bulk of the logic. `pScale` converts the
// raw user input to a [0,1] probability (1.0 for quantile, 1/100 for
// prctile). `method` selects the interpolation rule (default Midpoint
// = MATLAB R2025b default).
[[maybe_unused]] Value quantileImpl(const Value &x, const Value &p, keel::i32 dim, keel::f64 pScale, QMethod method, const char *fn, keel::Allocator *mr)
{
    if (p.numel() == 0)
        throw Error(keel::format("{}: p must be non-empty", fn),
                     0, 0, fn, "", keel::format("numkit:{}:emptyP", fn));

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Normalize probabilities into a flat vector (so prctile sees /100)
    // then validate the post-scaling value lies in [0,1].
    auto probs = keel::Array<keel::f64>(p.numel(), scratch.allocator());
    for (keel::usize i = 0; i < p.numel(); ++i) {
        probs[i] = p.doubleData()[i] * pScale;
        if (!(probs[i] >= 0.0 && probs[i] <= 1.0))
            throw Error(keel::format("{}: probabilities out of range", fn),
                         0, 0, fn, "", keel::format("numkit:{}:badProb", fn));
    }

    const keel::usize k = probs.size();
    const keel::i32 d = resolveDim(x, dim, fn);

    // Scalar p â†’ standard one-output-per-slice path.
    if (k == 1) {
        const keel::f64 pp = probs[0];
        return applyAlongDim(x, d,
            [pp, method](keel::usize, keel::f64 *slice, keel::usize n) {
                keel::sort(slice, slice + n);
                return quantileFromSortedSlice(slice, n, pp, method);
            }, mr);
    }

    // Vector p: each slice produces k outputs along the reduced dim.
    // We need to allocate the output explicitly because applyAlongDim
    // assumes 1 output per slice.
    if (x.isEmpty()) {
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    }
    if (x.dims().isVector() || x.isScalar()) {
        // Output is 1Ã—k row vector.
        auto sorted = keel::Array<keel::f64>(x.numel(), scratch.allocator());
        keel::copy(x.doubleData(), x.doubleData() + x.numel(), sorted.data());
        keel::sort(sorted.begin(), sorted.end());
        auto out = Value::matrix(1, k, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < k; ++i)
            out.doubleDataMut()[i] = quantileFromSortedSlice(sorted.data(),
                                                              sorted.size(),
                                                              probs[i],
                                                              method);
        return out;
    }

    // Build output shape: same as input but the reduced dim has size k.
    const auto &dd = x.dims();
    const keel::i32 nd = keel::max(dd.ndim(), d);
    keel::Array<keel::usize> outShape(nd, scratch.allocator());
    for (keel::i32 i = 0; i < nd; ++i)
        outShape[i] = (i == d - 1) ? k : ((i < dd.ndim()) ? dd.dim(i) : 1);
    Value out = (nd <= 2) ? Value::matrix(outShape[0], outShape[1], ValueType::DOUBLE, mr)
              : (nd == 3) ? Value::matrix3d(outShape[0], outShape[1], outShape[2], ValueType::DOUBLE, mr)
                          : Value::matrixND(outShape.data(), nd, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();

    keel::usize B = 1;
    for (keel::i32 i = 0; i < d - 1; ++i)
        B *= (i < dd.ndim()) ? dd.dim(i) : 1;
    const keel::usize N = (d - 1 < dd.ndim()) ? dd.dim(d - 1) : 1;
    keel::usize O = 1;
    for (keel::i32 i = d; i < nd; ++i)
        O *= (i < dd.ndim()) ? dd.dim(i) : 1;

    auto sorted = keel::Array<keel::f64>(N, scratch.allocator());
    const keel::f64 *src = x.doubleData();

    for (keel::usize o = 0; o < O; ++o) {
        for (keel::usize b = 0; b < B; ++b) {
            const keel::f64 *srcBase = src + o * (B * N) + b;
            for (keel::usize s = 0; s < N; ++s)
                sorted[s] = srcBase[s * B];
            keel::sort(sorted.begin(), sorted.end());
            keel::f64 *dstBase = dst + o * (B * k) + b;
            for (keel::usize i = 0; i < k; ++i)
                dstBase[i * B] = quantileFromSortedSlice(sorted.data(), N, probs[i], method);
        }
    }
    return out;
}

} // namespace
namespace {

template <typename T>
inline T readSrcAsT(const Value &x, keel::usize i, bool typeMatch)
{
    if (typeMatch)
        return static_cast<const T *>(x.rawData())[i];
    if constexpr (keel::IsFloatingPoint<T>) {
        return static_cast<T>(x.elemAsDouble(i));
    } else {
        const keel::f64 d = x.elemAsDouble(i);
        return static_cast<T>(keel::clamp(keel::round(d),
            keel::as<keel::f64>(keel::Limits<T>::kLowest),
            keel::as<keel::f64>(keel::Limits<T>::kMax)));
    }
}

template <typename T>
inline Value makeScalarT(T v, ValueType outType, keel::Allocator *mr)
{
    if (outType == ValueType::DOUBLE)
        return Value::scalar(keel::as<keel::f64>(v), mr);
    if (outType == ValueType::LOGICAL)
        return Value::logicalScalar(v != 0, mr);
    auto r = Value::matrix(1, 1, outType, mr);
    static_cast<T *>(r.rawDataMut())[0] = v;
    return r;
}

template <typename T>
inline T modeEmptyValue()
{
    if constexpr (keel::IsFloatingPoint<T>)
        return keel::Limits<T>::quiet_nan();
    else
        return T{};
}

template <typename T>
inline keel::usize compactNonNanT(T *data, keel::usize n)
{
    if constexpr (keel::IsFloatingPoint<T>) {
        keel::usize w = 0;
        for (keel::usize i = 0; i < n; ++i)
            if (!keel::is_nan(data[i])) {
                if (w != i) data[w] = data[i];
                ++w;
            }
        return w;
    } else {
        (void) data;
        return n;
    }
}

template <typename T>
inline void modeFromSliceT(T *data, keel::usize n, T &outVal, keel::f64 &outCount)
{
    const keel::usize k = compactNonNanT<T>(data, n);
    if (k == 0) {
        outVal = modeEmptyValue<T>();
        outCount = 0.0;
        return;
    }
    keel::sort(data, data + k);
    T bestVal = data[0];
    keel::usize bestCount = 1;
    T curVal = data[0];
    keel::usize curCount = 1;
    for (keel::usize i = 1; i < k; ++i) {
        if (data[i] == curVal) {
            ++curCount;
        } else {
            if (curCount > bestCount) { bestCount = curCount; bestVal = curVal; }
            curVal = data[i];
            curCount = 1;
        }
    }
    if (curCount > bestCount) { bestCount = curCount; bestVal = curVal; }
    outVal = bestVal;
    outCount = keel::as<keel::f64>(bestCount);
}

// Walk every output cell along dim `redDim` (1-based). For each cell,
// gather the slice into `scratch`, run modeFromSliceT, and write the
// (value, count) pair. Handles 2D / 3D / ND uniformly via stride math.
// On empty slices (sliceLen == 0) fills outputs with NaN/0 (or 0/0 for
// integer T) â€” matches MATLAB's mode-of-empty-slice convention.
template <typename T>
void modeAlongDim(const Value &x, keel::i32 redDim, T *dst, keel::f64 *dstC, bool typeMatch, keel::Allocator *mr)
{
    const auto &d = x.dims();
    const keel::i32 redAxis = redDim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (keel::i32 i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (keel::i32 i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);

    if (sliceLen == 0) {
        const T defVal = modeEmptyValue<T>();
        const keel::usize total = B * O;
        for (keel::usize i = 0; i < total; ++i) { dst[i] = defVal; dstC[i] = 0.0; }
        return;
    }

    keel::Array<T> scratch(sliceLen, mr);
    auto runSlice = [&](keel::usize outIdx, keel::usize baseOff, keel::usize stride) {
        for (keel::usize k = 0; k < sliceLen; ++k)
            scratch[k] = readSrcAsT<T>(x, baseOff + k * stride, typeMatch);
        T v; keel::f64 c;
        modeFromSliceT<T>(scratch.data(), sliceLen, v, c);
        dst[outIdx] = v;
        dstC[outIdx] = c;
    };

    if (B == 1) {
        for (keel::usize o = 0; o < O; ++o) runSlice(o, o * sliceLen, 1);
        return;
    }
    for (keel::usize o = 0; o < O; ++o) {
        for (keel::usize b = 0; b < B; ++b) {
            const keel::usize base = o * sliceLen * B + b;
            const keel::usize outIdx = o * B + b;
            runSlice(outIdx, base, B);
        }
    }
}

inline keel::Pair<Value, Value>
allocModeOutputs(const Value &x, keel::i32 redDim, ValueType outType, keel::Allocator *mr)
{
    if (x.dims().ndim() >= 4 && redDim >= 1 && redDim <= x.dims().ndim()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        auto shape = outShapeForDimND(x, redDim, scratch.allocator());
        return {Value::matrixND(shape.data(), (keel::i32) shape.size(), outType, mr),
                Value::matrixND(shape.data(), (keel::i32) shape.size(), ValueType::DOUBLE, mr)};
    }
    auto outShape = outShapeForDim(x, redDim);
    return {createMatrix(outShape, outType, mr),
            createMatrix(outShape, ValueType::DOUBLE, mr)};
}

template <typename T>
keel::Tuple<Value, Value>
modeAllT(const Value &x, ValueType outType, keel::Allocator *mr)
{
    const bool typeMatch = (x.type() == outType);
    if (x.isEmpty() && x.dims().ndim() < 4) {
        return keel::make_tuple(Value::matrix(0, 0, outType, mr),
                               Value::matrix(0, 0, ValueType::DOUBLE, mr));
    }
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (x.isScalar() || x.dims().isVector()) {
        keel::Array<T> buf(x.numel(), scratch.allocator());
        for (keel::usize i = 0; i < x.numel(); ++i)
            buf[i] = readSrcAsT<T>(x, i, typeMatch);
        T v; keel::f64 c;
        modeFromSliceT<T>(buf.data(), x.numel(), v, c);
        return keel::make_tuple(makeScalarT<T>(v, outType, mr),
                               Value::scalar(c, mr));
    }
    const keel::i32 redDim = firstNonSingletonDim(x);
    auto [out, outC] = allocModeOutputs(x, redDim, outType, mr);
    modeAlongDim<T>(x, redDim, static_cast<T *>(out.rawDataMut()), outC.doubleDataMut(), typeMatch, scratch.allocator());
    return keel::make_tuple(keel::move(out), keel::move(outC));
}

template <typename T>
keel::Tuple<Value, Value>
modeAlongDimT(const Value &x, keel::i32 dim, ValueType outType, keel::Allocator *mr)
{
    const bool typeMatch = (x.type() == outType);
    if (x.isEmpty() && x.dims().ndim() < 4) {
        return keel::make_tuple(Value::matrix(0, 0, outType, mr),
                               Value::matrix(0, 0, ValueType::DOUBLE, mr));
    }
    if (x.isScalar() || x.dims().isVector()) {
        if (dim != firstNonSingletonDim(x)) {
            // Identity reduction: copy x as outType, frequencies = ones.
            const keel::usize n = x.numel();
            Value out, outC;
            if (x.dims().isVector()) {
                out  = createMatrix({x.dims().rows(), x.dims().cols(), 0}, outType, mr);
                outC = createMatrix({x.dims().rows(), x.dims().cols(), 0}, ValueType::DOUBLE, mr);
            } else {
                out  = makeScalarT<T>(readSrcAsT<T>(x, 0, typeMatch), outType, mr);
                outC = Value::scalar(1.0, mr);
                return keel::make_tuple(keel::move(out), keel::move(outC));
            }
            T *dst = static_cast<T *>(out.rawDataMut());
            keel::f64 *dstC = outC.doubleDataMut();
            for (keel::usize i = 0; i < n; ++i) {
                dst[i]  = readSrcAsT<T>(x, i, typeMatch);
                dstC[i] = 1.0;
            }
            return keel::make_tuple(keel::move(out), keel::move(outC));
        }
        return modeAllT<T>(x, outType, mr);
    }
    auto [out, outC] = allocModeOutputs(x, dim, outType, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    modeAlongDim<T>(x, dim, static_cast<T *>(out.rawDataMut()), outC.doubleDataMut(), typeMatch, scratch.allocator());
    return keel::make_tuple(keel::move(out), keel::move(outC));
}

// Dispatch on x.type(). LOGICAL maps to T=uint8_t (storage type) and
// outType=LOGICAL so the result preserves logical class. CHAR uses
// T=char. COMPLEX has no defined order â†’ throw.
[[maybe_unused]] keel::Tuple<Value, Value>
dispatchMode(const Value &x, keel::i32 dim, keel::Allocator *mr, const char *fn)
{
    const bool useDimReducer = (dim > 0);
    auto run = [&](auto tag, ValueType outT) {
        using T = decltype(tag);
        return useDimReducer
            ? modeAlongDimT<T>(x, dim, outT, mr)
            : modeAllT<T>(x, outT, mr);
    };
    switch (x.type()) {
    case ValueType::DOUBLE:  return run(keel::f64  {}, ValueType::DOUBLE);
    case ValueType::SINGLE:  return run(keel::f32   {}, ValueType::SINGLE);
    case ValueType::INT8:    return run(keel::i8  {}, ValueType::INT8);
    case ValueType::INT16:   return run(keel::i16 {}, ValueType::INT16);
    case ValueType::INT32:   return run(keel::i32 {}, ValueType::INT32);
    case ValueType::INT64:   return run(keel::i64 {}, ValueType::INT64);
    case ValueType::UINT8:   return run(keel::u8 {}, ValueType::UINT8);
    case ValueType::UINT16:  return run(keel::u16{}, ValueType::UINT16);
    case ValueType::UINT32:  return run(keel::u32{}, ValueType::UINT32);
    case ValueType::UINT64:  return run(keel::u64{}, ValueType::UINT64);
    case ValueType::LOGICAL: return run(keel::u8 {}, ValueType::LOGICAL);
    case ValueType::CHAR:    return run(char    {}, ValueType::CHAR);
    case ValueType::COMPLEX:
        throw Error(keel::format("{}: not defined for complex inputs", fn),
                     0, 0, fn, "", keel::format("numkit:{}:complex", fn));
    default:
        throw Error(keel::format("{}: unsupported input type", fn),
                     0, 0, fn, "", keel::format("numkit:{}:type", fn));
    }
}

} // namespace
namespace {

void validateCovInputs(const Value &x, const char *fn)
{
    if (x.type() == ValueType::COMPLEX)
        throw Error(keel::format("{}: complex inputs are not supported", fn),
                     0, 0, fn, "", keel::format("numkit:{}:complex", fn));
    if (x.dims().is3D() || x.dims().ndim() > 2)
        throw Error(keel::format("{}: only vector and 2D matrix inputs are supported", fn),
                     0, 0, fn, "", keel::format("numkit:{}:rank", fn));
}

void validateNormFlagCov(keel::i32 w, const char *fn)
{
    if (w != 0 && w != 1)
        throw Error(keel::format("{}: normalization flag must be 0 or 1", fn),
                     0, 0, fn, "", keel::format("numkit:{}:badFlag", fn));
}

// Build an nÃ—p column-major DOUBLE buffer from x. Vector input is
// treated as a single column. Returns the raw data, n, and p.
void readMatrix(const Value &x, keel::Array<keel::f64> &out,
                keel::usize &n, keel::usize &p)
{
    if (x.dims().isVector() || x.isScalar()) {
        n = x.numel();
        p = 1;
    } else {
        n = x.dims().rows();
        p = x.dims().cols();
    }
    out.assign(n * p, 0.0);
    if (x.type() == ValueType::DOUBLE && (x.dims().isVector() || x.isScalar()
                                       || (!x.dims().is3D() && x.dims().ndim() == 2))) {
        // Column-major source; for a single-column vector either
        // orientation already lays the elements contiguously.
        const keel::f64 *src = x.doubleData();
        keel::copy_n(out.data(), src, n * p);
        return;
    }
    for (keel::usize i = 0; i < n * p; ++i)
        out[i] = x.elemAsDouble(i);
}

// In-place: subtract per-column mean from a column-major nÃ—p buffer.
void centerColumns(keel::f64 *data, keel::usize n, keel::usize p)
{
    for (keel::usize c = 0; c < p; ++c) {
        keel::f64 s = 0.0;
        for (keel::usize r = 0; r < n; ++r)
            s += data[c * n + r];
        const keel::f64 m = s / keel::as<keel::f64>(n);
        for (keel::usize r = 0; r < n; ++r)
            data[c * n + r] -= m;
    }
}

// covImpl: take a centered nÃ—p buffer, compute X' * X / divisor â†’ pÃ—p.
Value covMatrixFromCentered(const keel::f64 *X, keel::usize n, keel::usize p, keel::f64 divisor, keel::Allocator *mr)
{
    auto out = Value::matrix(p, p, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize r = 0; r < n; ++r)
                s += X[i * n + r] * X[j * n + r];
            dst[j * p + i] = s / divisor;
        }
    return out;
}

} // namespace
namespace {

enum class CovNan { Include, Omitrows, Partialrows };

// Recognize a NaN-policy string ('includenan' | 'omitrows' | 'partialrows').
[[maybe_unused]] bool parseCovNanFlag(const Value &v, CovNan &mode)
{
    if (!v.isChar() && !v.isString()) return false;
    keel::String s = v.toString();
    s.to_ascii_lower_in_place();
    if (s.as_ref() == "includenan")  { mode = CovNan::Include;     return true; }
    if (s.as_ref() == "omitrows")    { mode = CovNan::Omitrows;    return true; }
    if (s.as_ref() == "partialrows") { mode = CovNan::Partialrows; return true; }
    return false;
}

// Assemble two equal-length vectors into an nÃ—2 column-major matrix Value
// [x(:) y(:)] for the two-vector NaN-aware path.
[[maybe_unused]] Value assembleXY(const Value &x, const Value &y, keel::Allocator *mr)
{
    validateCovInputs(x, "cov");
    validateCovInputs(y, "cov");
    if (!x.dims().isVector() || !y.dims().isVector())
        throw Error("cov: two-input form requires vector arguments",
                     0, 0, "cov", "", "numkit:cov:notVector");
    if (x.numel() != y.numel())
        throw Error("cov: x and y must have the same length",
                     0, 0, "cov", "", "numkit:cov:lengthMismatch");
    const keel::usize n = x.numel();
    auto out = Value::matrix(n, 2, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        o[i] = x.elemAsDouble(i);
        o[n + i] = y.elemAsDouble(i);
    }
    return out;
}

// cov with 'omitrows' (listwise deletion: drop any row containing a NaN) or
// 'partialrows' (pairwise deletion: each cov(i,j) uses rows where both
// columns are non-NaN, with the means taken over exactly those rows). The
// default 'includenan' is the NaN-poisoning behaviour of plain cov() and is
// routed there by the dispatcher.
[[maybe_unused]] Value covNanAware(const Value &x, keel::i32 normFlag, CovNan mode,
                  keel::Allocator *mr)
{
    validateNormFlagCov(normFlag, "cov");
    validateCovInputs(x, "cov");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> data(scratch.allocator());
    keel::usize n, p;
    readMatrix(x, data, n, p);

    auto divisorOf = [normFlag](keel::usize m) {
        return (normFlag == 0) ? keel::max(1.0, keel::as<keel::f64>(m) - 1.0)
                               : keel::as<keel::f64>(m);
    };

    // Vector input â†’ scalar variance over the non-NaN elements.
    if (p == 1) {
        keel::f64 s = 0.0;
        keel::usize m = 0;
        for (keel::usize i = 0; i < n; ++i)
            if (!keel::is_nan(data[i])) { s += data[i]; ++m; }
        if (m == 0) return Value::scalar(keel::kNaN, mr);
        const keel::f64 mean = s / keel::as<keel::f64>(m);
        keel::f64 sq = 0.0;
        for (keel::usize i = 0; i < n; ++i)
            if (!keel::is_nan(data[i])) { const keel::f64 d = data[i] - mean; sq += d * d; }
        return Value::scalar(sq / divisorOf(m), mr);
    }

    if (mode == CovNan::Omitrows) {
        auto rowOk = [&](keel::usize r) {
            for (keel::usize c = 0; c < p; ++c)
                if (keel::is_nan(data[c * n + r])) return false;
            return true;
        };
        keel::usize m = 0;
        for (keel::usize r = 0; r < n; ++r) if (rowOk(r)) ++m;
        if (m == 0) {
            auto out = Value::matrix(p, p, ValueType::DOUBLE, mr);
            keel::f64 *o = out.doubleDataMut();
            for (keel::usize i = 0; i < p * p; ++i) o[i] = keel::kNaN;
            return out;
        }
        keel::Array<keel::f64> kept(scratch.allocator());
        kept.assign(m * p, 0.0);
        keel::usize rr = 0;
        for (keel::usize r = 0; r < n; ++r) {
            if (!rowOk(r)) continue;
            for (keel::usize c = 0; c < p; ++c) kept[c * m + rr] = data[c * n + r];
            ++rr;
        }
        centerColumns(kept.data(), m, p);
        return covMatrixFromCentered(kept.data(), m, p, divisorOf(m), mr);
    }

    // Partialrows.
    auto out = Value::matrix(p, p, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = i; j < p; ++j) {
            keel::f64 si = 0.0, sj = 0.0;
            keel::usize m = 0;
            for (keel::usize r = 0; r < n; ++r) {
                const keel::f64 a = data[i * n + r], b = data[j * n + r];
                if (!keel::is_nan(a) && !keel::is_nan(b)) { si += a; sj += b; ++m; }
            }
            keel::f64 v;
            if (m == 0) {
                v = keel::kNaN;
            } else {
                const keel::f64 mi = si / keel::as<keel::f64>(m);
                const keel::f64 mj = sj / keel::as<keel::f64>(m);
                keel::f64 s = 0.0;
                for (keel::usize r = 0; r < n; ++r) {
                    const keel::f64 a = data[i * n + r], b = data[j * n + r];
                    if (!keel::is_nan(a) && !keel::is_nan(b))
                        s += (a - mi) * (b - mj);
                }
                v = s / divisorOf(m);
            }
            o[j * p + i] = v;
            o[i * p + j] = v;
        }
    return out;
}

// â”€â”€ corrcoef 'Rows' NaN policy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
enum class CorrcoefRows { All, Complete, Pairwise };

[[maybe_unused]] CorrcoefRows parseCorrcoefRows(keel::ArrayRef<Value> args, keel::usize start)
{
    // Probed R2025b: only 'alpha' and 'rows' are valid parameter names
    // (validatestring prefixes); anything else is invalidArgName, and a
    // lone unmatched value string is unmatchedPVPair.
    for (keel::usize i = start; i < args.size(); ++i) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        keel::String probe = args[i].toString();
        probe.to_ascii_lower_in_place();
        const keel::StringRef pr = probe.as_ref();
        const bool isName = pr == "rows" || pr == "alpha"
            || (pr.size() <= 4 && keel::StringRef("rows").substr(0, pr.size()) == pr)
            || (pr.size() <= 5 && keel::StringRef("alpha").substr(0, pr.size()) == pr);
        if (!isName) {
            if (i + 1 < args.size() && (args[i + 1].isChar() || args[i + 1].isString()))
                throw Error("corrcoef: Valid parameters are 'alpha' and 'rows'.",
                            0, 0, "corrcoef", "", "numkit:corrcoef:invalidArgName");
            throw Error("corrcoef: Unmatched parameter name/value pair.",
                        0, 0, "corrcoef", "", "numkit:corrcoef:unmatchedPVPair");
        }
        ++i;  // skip the value
    }
    for (keel::usize i = start; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        if (name.as_ref() == "rows") {
            keel::String v = args[i + 1].toString();
            v.to_ascii_lower_in_place();
            if (v.as_ref() == "all")      return CorrcoefRows::All;
            if (v.as_ref() == "complete") return CorrcoefRows::Complete;
            if (v.as_ref() == "pairwise") return CorrcoefRows::Pairwise;
            throw Error("corrcoef: Valid row choices are 'all', 'complete', "
                        "and 'pairwise'.",
                        0, 0, "corrcoef", "", "numkit:corrcoef:invalidRowChoice");
        }
    }
    return CorrcoefRows::All;
}

// corrcoef(...,'Alpha',a): significance level for the RL/RU confidence
// bounds (default 0.05). Must be in (0, 1).
[[maybe_unused]] keel::f64 parseCorrcoefAlpha(keel::ArrayRef<Value> args, keel::usize start)
{
    for (keel::usize i = start; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        if (name.as_ref() == "alpha") {
            const keel::f64 a = args[i + 1].toScalar();
            if (!(a > 0.0 && a < 1.0))
                throw Error("corrcoef: Alpha must be in the interval (0,1)",
                            0, 0, "corrcoef", "", "numkit:corrcoef:BadAlpha");
            return a;
        }
    }
    return 0.05;
}

[[maybe_unused]] keel::usize countCompleteRows(const Value &X)
{
    keel::usize n, p;
    {
        const bool vec = X.dims().isVector() || X.isScalar();
        n = vec ? X.numel() : X.dims().rows();
        p = vec ? 1 : X.dims().cols();
    }
    keel::usize m = 0;
    for (keel::usize r = 0; r < n; ++r) {
        bool ok = true;
        for (keel::usize c = 0; c < p && ok; ++c)
            if (keel::is_nan(X.elemAsDouble(r + c * n))) ok = false;
        if (ok) ++m;
    }
    return m;
}

// Pairwise Pearson correlation: each entry (i,j) uses the rows where both
// columns are non-NaN, with the means AND the standard deviations taken
// over exactly those common rows. Also optionally fills MCounts with the
// per-pair non-NaN sample count m_ij for pairwise p-values and CIs.
Value corrcoefPairwiseWithCounts(const Value &X, Value *MCounts,
                                 keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> data(scratch.allocator());
    keel::usize n, p;
    readMatrix(X, data, n, p);
    auto out = Value::matrix(p, p, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    keel::f64 *mc = nullptr;
    if (MCounts) {
        *MCounts = Value::matrix(p, p, ValueType::DOUBLE, mr);
        mc = MCounts->doubleDataMut();
    }
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j) {
            keel::f64 si = 0.0, sj = 0.0;
            keel::usize m = 0;
            for (keel::usize r = 0; r < n; ++r) {
                const keel::f64 a = data[i * n + r], b = data[j * n + r];
                if (!keel::is_nan(a) && !keel::is_nan(b)) { si += a; sj += b; ++m; }
            }
            if (mc) mc[i + j * p] = keel::as<keel::f64>(m);
            keel::f64 rij;
            if (m < 2) {
                rij = keel::Limits<keel::f64>::quiet_nan();
            } else {
                const keel::f64 mi = si / keel::as<keel::f64>(m);
                const keel::f64 mj = sj / keel::as<keel::f64>(m);
                keel::f64 sxy = 0.0, sxx = 0.0, syy = 0.0;
                for (keel::usize r = 0; r < n; ++r) {
                    const keel::f64 a = data[i * n + r], b = data[j * n + r];
                    if (keel::is_nan(a) || keel::is_nan(b)) continue;
                    const keel::f64 da = a - mi, db = b - mj;
                    sxy += da * db; sxx += da * da; syy += db * db;
                }
                const keel::f64 den = keel::sqrt(sxx * syy);
                rij = (den > 0.0) ? sxy / den
                                  : keel::Limits<keel::f64>::quiet_nan();
            }
            o[i + j * p] = rij;
        }
    return out;
}

[[maybe_unused]] Value corrcoefPairwise(const Value &X, keel::Allocator *mr)
{
    return corrcoefPairwiseWithCounts(X, nullptr, mr);
}

// Complex correlation matrix R for 1-arg X (n x p) or 2-arg (x, y).
// C(i,j) = sum_r conj(Z(r,i)) * Z(r,j) / (n - 1),
// R(i,j) = C(i,j) / sqrt(real(C(i,i)) * real(C(j,j))).
Value corrcoefComplexMatrix(const Value &X, keel::Allocator *mr)
{
    const bool isVec = X.dims().isVector() || X.isScalar();
    const keel::usize n = isVec ? X.numel() : X.dims().rows();
    const keel::usize p = isVec ? 1 : X.dims().cols();
    if (p == 0) return Value::complexMatrix(0, 0, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Z(n * p, scratch.allocator());
    const bool xCplx = X.isComplex();
    for (keel::usize c = 0; c < p; ++c) {
        for (keel::usize r = 0; r < n; ++r) {
            const keel::usize idx = c * n + r;
            Z[idx] = xCplx ? X.complexData()[idx]
                           : keel::c64(X.elemAsDouble(idx), 0.0);
        }
    }
    if (n <= 1) {
        Value R = Value::matrix(p, p, ValueType::DOUBLE, mr);
        for (keel::usize k = 0; k < p * p; ++k) R.doubleDataMut()[k] = keel::kNaN;
        return R;
    }
    for (keel::usize c = 0; c < p; ++c) {
        keel::c64 mean(0.0, 0.0);
        for (keel::usize r = 0; r < n; ++r) mean += Z[c * n + r];
        mean /= keel::as<keel::f64>(n);
        for (keel::usize r = 0; r < n; ++r) Z[c * n + r] -= mean;
    }
    keel::Array<keel::f64> stds(p, scratch.allocator());
    for (keel::usize c = 0; c < p; ++c) {
        keel::f64 ss = 0.0;
        for (keel::usize r = 0; r < n; ++r) ss += keel::norm(Z[c * n + r]);
        stds[c] = keel::sqrt(ss);
    }
    Value R = Value::complexMatrix(p, p, mr);
    keel::c64 *rd = R.complexDataMut();
    for (keel::usize i = 0; i < p; ++i) {
        for (keel::usize j = 0; j < p; ++j) {
            if (i == j) {
                rd[j * p + i] = (stds[i] > 0.0) ? keel::c64(1.0, 0.0)
                                                : keel::c64(keel::kNaN, keel::kNaN);
                continue;
            }
            keel::c64 s(0.0, 0.0);
            for (keel::usize r = 0; r < n; ++r)
                s += keel::conj(Z[i * n + r]) * Z[j * n + r];
            const keel::f64 den = stds[i] * stds[j];
            rd[j * p + i] = (den > 0.0) ? (s / den)
                                        : keel::c64(keel::kNaN, keel::kNaN);
        }
    }
    return R;
}

[[maybe_unused]] Value corrcoefComplexXY(const Value &x, const Value &y, keel::Allocator *mr)
{
    if (!x.dims().isVector() || !y.dims().isVector())
        throw Error("corrcoef: two-input form requires vector arguments",
                     0, 0, "corrcoef", "", "numkit:corrcoef:notVector");
    if (x.numel() != y.numel())
        throw Error("corrcoef: x and y must have the same length",
                     0, 0, "corrcoef", "", "numkit:corrcoef:lengthMismatch");
    const keel::usize n = x.numel();
    Value M = Value::complexMatrix(n, 2, mr);
    keel::c64 *md = M.complexDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        md[i]     = x.isComplex() ? x.complexData()[i] : keel::c64(x.elemAsDouble(i), 0.0);
        md[n + i] = y.isComplex() ? y.complexData()[i] : keel::c64(y.elemAsDouble(i), 0.0);
    }
    return corrcoefComplexMatrix(M, mr);
}

[[maybe_unused]] Value corrcoefScalarOne(keel::Allocator *mr)
{
    auto R = Value::matrix(1, 1, ValueType::DOUBLE, mr);
    R.doubleDataMut()[0] = 1.0;
    return R;
}

[[maybe_unused]] Value corrcoefFromCov(const Value &C, keel::Allocator *mr)
{
    if (C.dims().rows() != C.dims().cols())
        throw Error("corrcoef: covariance matrix must be square",
                     0, 0, "corrcoef", "", "numkit:corrcoef:internal");
    const keel::usize p = C.dims().rows();
    auto R = Value::matrix(p, p, ValueType::DOUBLE, mr);
    if (p == 0) return R;
    const keel::f64 *cd = C.doubleData();
    keel::f64 *rd = R.doubleDataMut();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto diag = keel::Array<keel::f64>(p, scratch.allocator());
    for (keel::usize i = 0; i < p; ++i)
        diag[i] = keel::sqrt(cd[i * p + i]);
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j) {
            const keel::f64 denom = diag[i] * diag[j];
            rd[j * p + i] = (denom == 0.0) ? keel::kNaN : cd[j * p + i] / denom;
        }
    return R;
}

// Two-sided p-values for a correlation matrix R computed from n observations.
// Diagonal is 1 (unless R(i,i) is NaN); off-diagonal P(i,j) = 2*tcdf(-|t|, n-2)
// with the test statistic t = r*sqrt((n-2)/(1-r^2)). With n <= 2 (df <= 0) the
// off-diagonal p-values are NaN.
[[maybe_unused]] Value corrcoefPValues(const Value &R, keel::f64 n, keel::Allocator *mr)
{
    const keel::usize p = R.dims().rows();
    auto P = Value::matrix(p, p, ValueType::DOUBLE, mr);
    if (p == 0) return P;
    const keel::f64 df = n - 2.0;
    const keel::f64 *rd = R.doubleData();
    keel::f64 *pd = P.doubleDataMut();

    // Build the matrix of -|t| values, then evaluate the t-CDF vectorized.
    auto T = Value::matrix(p, p, ValueType::DOUBLE, mr);
    keel::f64 *td = T.doubleDataMut();
    for (keel::usize k = 0; k < p * p; ++k) {
        const keel::f64 r = rd[k];
        if (keel::is_nan(r)) {
            td[k] = keel::kNaN;
            continue;
        }
        const keel::f64 oneMinus = 1.0 - r * r;
        keel::f64 t = keel::Limits<keel::f64>::infinity(); // |r|>=1 → extreme
        if (df > 0.0 && oneMinus > 0.0)
            t = keel::abs(r) * keel::sqrt(df / oneMinus);
        td[k] = -t;
    }
    Value cdf = (df > 0.0) ? tcdf(T, df, mr) : Value{};
    const keel::f64 *cd = (df > 0.0) ? cdf.doubleData() : nullptr;

    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j) {
            const keel::usize k = j * p + i;
            if (keel::is_nan(rd[k])) { pd[k] = keel::kNaN; continue; }
            if (i == j) { pd[k] = 1.0; continue; }
            if (df <= 0.0) { pd[k] = keel::kNaN; continue; }
            keel::f64 pv = 2.0 * cd[k];
            if (pv > 1.0) pv = 1.0;
            pd[k] = pv;
        }
    return P;
}

[[maybe_unused]] Value corrcoefPValuesPairwise(const Value &R, const Value &MCounts,
                              keel::Allocator *mr)
{
    const keel::usize p = R.dims().rows();
    auto P = Value::matrix(p, p, ValueType::DOUBLE, mr);
    if (p == 0) return P;
    const keel::f64 *rd = R.doubleData();
    const keel::f64 *mc = MCounts.doubleData();
    keel::f64 *pd = P.doubleDataMut();
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j) {
            const keel::usize k = j * p + i;
            const keel::f64 r = rd[k];
            if (keel::is_nan(r)) { pd[k] = keel::kNaN; continue; }
            if (i == j) { pd[k] = 1.0; continue; }
            const keel::f64 df = mc[k] - 2.0;
            if (df <= 0.0) { pd[k] = keel::kNaN; continue; }
            const keel::f64 oneMinus = 1.0 - r * r;
            if (oneMinus <= 0.0) { pd[k] = 0.0; continue; }
            const keel::f64 t = keel::abs(r) * keel::sqrt(df / oneMinus);
            keel::f64 pv = 2.0 * tcdf(Value::scalar(-t, mr), df, mr).toScalar();
            if (pv > 1.0) pv = 1.0;
            pd[k] = pv;
        }
    return P;
}

// Fisher z-transform confidence bounds for corrcoef (the 3rd/4th outputs
// RL/RU). For each correlation r: z = atanh(r), se = 1/sqrt(n-3),
// zc = norminv(1-alpha/2); RL = tanh(z - zc*se), RU = tanh(z + zc*se).
// |r| == 1 (incl. the diagonal) -> RL = RU = r. n <= 3 -> se infinite, so
// the bounds widen to the full [-1, 1]. Matches MATLAB R2025b.
[[maybe_unused]] void corrcoefConfBounds(const Value &R, keel::f64 n, keel::f64 alpha,
                        Value &RL, Value &RU, keel::Allocator *mr)
{
    const keel::usize p = R.dims().rows();
    RL = Value::matrix(p, p, ValueType::DOUBLE, mr);
    RU = Value::matrix(p, p, ValueType::DOUBLE, mr);
    if (p == 0) return;
    const keel::f64 *rd = R.doubleData();
    keel::f64 *rl = RL.doubleDataMut();
    keel::f64 *ru = RU.doubleDataMut();
    const keel::f64 zc =
        norminv(Value::scalar(1.0 - alpha / 2.0, mr), 0.0, 1.0, mr).toScalar();
    const keel::f64 se = (n > 3.0) ? 1.0 / keel::sqrt(n - 3.0)
                                : keel::Limits<keel::f64>::infinity();
    for (keel::usize k = 0; k < p * p; ++k) {
        const keel::f64 r = rd[k];
        if (keel::is_nan(r)) { rl[k] = keel::kNaN; ru[k] = keel::kNaN; continue; }
        if (keel::abs(r) >= 1.0) { rl[k] = r; ru[k] = r; continue; }
        const keel::f64 z = keel::atanh(r);
        rl[k] = keel::tanh(z - zc * se);
        ru[k] = keel::tanh(z + zc * se);
    }
}

[[maybe_unused]] void corrcoefConfBoundsPairwise(const Value &R, const Value &MCounts, keel::f64 alpha,
                                Value &RL, Value &RU, keel::Allocator *mr)
{
    const keel::usize p = R.dims().rows();
    RL = Value::matrix(p, p, ValueType::DOUBLE, mr);
    RU = Value::matrix(p, p, ValueType::DOUBLE, mr);
    if (p == 0) return;
    const keel::f64 *rd = R.doubleData();
    const keel::f64 *mc = MCounts.doubleData();
    keel::f64 *rl = RL.doubleDataMut();
    keel::f64 *ru = RU.doubleDataMut();
    const keel::f64 zc =
        norminv(Value::scalar(1.0 - alpha / 2.0, mr), 0.0, 1.0, mr).toScalar();
    for (keel::usize k = 0; k < p * p; ++k) {
        const keel::f64 r = rd[k];
        if (keel::is_nan(r)) { rl[k] = keel::kNaN; ru[k] = keel::kNaN; continue; }
        if (keel::abs(r) >= 1.0) { rl[k] = r; ru[k] = r; continue; }
        const keel::f64 n_ij = mc[k];
        const keel::f64 se = (n_ij > 3.0) ? 1.0 / keel::sqrt(n_ij - 3.0)
                                       : keel::Limits<keel::f64>::infinity();
        const keel::f64 z = keel::atanh(r);
        rl[k] = keel::tanh(z - zc * se);
        ru[k] = keel::tanh(z + zc * se);
    }
}

} // namespace
namespace {

// If the last positional arg is a 'omitnan'/'includenan' string,
// strip it from the count and return the omit flag. Throws on unknown
// trailing strings so user errors don't get silently ignored.
[[maybe_unused]] keel::usize stripNanFlag(keel::ArrayRef<Value> args, bool &omitNan, const char * /*fn*/)
{
    omitNan = false;
    if (args.empty()) return 0;
    const Value &last = args[args.size() - 1];
    if (last.type() != ValueType::CHAR) return args.size();
    keel::String s = last.toString();
    s.to_ascii_lower_in_place();
    if (s.as_ref() == "omitnan" || s.as_ref() == "omitmissing") { omitNan = true; return args.size() - 1; }
    if (s.as_ref() == "includenan" || s.as_ref() == "includemissing") return args.size() - 1;
    // Not a nan flag â€” leave it to the caller to handle (e.g. 'all').
    return args.size();
}

inline void rejectComplexOmitNan(const Value &x, const char *fn)
{
    if (x.type() == ValueType::COMPLEX)
        throw Error(keel::format("{}: 'omitnan' for complex input is not supported", fn),
                     0, 0, fn, "", keel::format("numkit:{}:complexOmitNan", fn));
}

} // namespace

Value quantileWithOpts(const Value &x, const Value &p, keel::i32 dim, bool flatten,
                       QMethod method, keel::f64 pScale, const char *fn,
                       keel::Allocator *mr);

} // namespace numkit::stats
