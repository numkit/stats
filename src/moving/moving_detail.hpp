// toolboxes/.../moving_detail.hpp — private compute substrate (anon-in-header,
// internal linkage per TU) shared by moving.cpp compute and moving_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
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
#include <numkit/ops/reductions.hpp>  // numkit::builtin::detail dim-infra (engine-free, ops re-export)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::stats {

namespace {

using ::numkit::ops::firstNonSingletonDim;
using ::numkit::ops::resolveDim;   // 0 -> first non-singleton, else validate
using ::numkit::ops::validateDim;

// Decode the `k` argument: scalar → reference-engine backward-leaning window
// [floor(k/2), floor((k-1)/2)] (even k centred on current+previous; odd k
// symmetric); 2-element vector → [kb, kf].
struct Window {
    long kb;   // samples back (inclusive of centre? no — i-kb is the start)
    long kf;   // samples forward
};

Window decodeWindow(keel::ArrayRef<keel::usize> k, const char *fn)
{
    auto badK = [&]() {
        throw Error(keel::format("{}: Window length must be a finite, "
                                 "positive, real scalar or 2-element vector of finite, "
                                 "nonnegative, real scalars.", fn),
                    0, 0, fn, "", keel::format("numkit:{}:badK", fn));
    };
    if (k.size() == 1) {
        const keel::usize n = k[0];
        if (n == 0) badK();
        // A scalar window of even length is centred about the CURRENT
        // and PREVIOUS elements — it leans BACKWARD: kb = floor(k/2),
        // kf = floor((k-1)/2). e.g. movsum([1 2 3 4],2) = [1 3 5 7] (window
        // [i-1, i]). Odd k is symmetric (kb == kf).
        return {keel::as<long>(n / 2), keel::as<long>((n - 1) / 2)};
    }
    if (k.size() == 2) {
        return {keel::as<long>(k[0]), keel::as<long>(k[1])};
    }
    badK();
    return {0, 0}; // unreachable
}

// Adapter / smoothdata path: extract k from a Value (scalar or 2-vec) into
// a ScratchVec<size_t>, with the same validation that decodeWindow used to
// do directly. Returns by value (live for the caller's scratch lifetime).
keel::Array<keel::usize> decodeWindowValueToScratch(const Value &k, const char *fn,
                                               const keel::ArenaTemp &scratch)
{
    auto badK = [&]() {
        throw Error(keel::format("{}: Window length must be a finite, "
                                 "positive, real scalar or 2-element vector of finite, "
                                 "nonnegative, real scalars.", fn),
                    0, 0, fn, "", keel::format("numkit:{}:badK", fn));
    };
    keel::Array<keel::usize> out(scratch.allocator());
    if (k.isScalar()) {
        const double v = k.toScalar();
        if (!std::isfinite(v) || v <= 0 || std::floor(v) != v) badK();
        out.push_back(keel::as<keel::usize>(v));
        return out;
    }
    if (k.numel() == 2) {
        const double a = k.elemAsDouble(0);
        const double b = k.elemAsDouble(1);
        if (!std::isfinite(a) || !std::isfinite(b) ||
            a < 0 || b < 0 || std::floor(a) != a || std::floor(b) != b) badK();
        out.push_back(keel::as<keel::usize>(a));
        out.push_back(keel::as<keel::usize>(b));
        return out;
    }
    badK();
    return out; // unreachable
}

// ── Mov-extras options (nanflag + Endpoints + SamplePoints) ───────────
// Parses the trailing args of mov*(x, k, ...). Supports:
//   * positional `dim` (numeric scalar, before any string)
//   * positional `nanflag` ∈ {"includemissing","includenan",
//                              "omitmissing","omitnan"}; default "includenan"
//   * Name-Value pairs: `Endpoints` ∈ {"shrink","discard","fill"} | scalar
//   * Name-Value pairs: `SamplePoints` (strictly increasing finite vector)
enum class EndpointMode { Shrink, Discard, Fill, Scalar };

struct MovOpts {
    int dim = 0;
    bool omit_nan = false;       // default = includenan
    EndpointMode ep = EndpointMode::Shrink;
    double ep_fill = std::numeric_limits<double>::quiet_NaN();
    bool has_sample_points = false;
    keel::Array<double> sample_points;
    double sp_kb = 0.0;
    double sp_kf = 0.0;
    const char *fn_name = "movmean";
};

inline keel::String toLower(keel::StringRef sv)
{
    keel::String s = keel::String::from(sv);
    for (auto &c : s)
        c = keel::to_ascii_lower(c);
    return s;
}

[[maybe_unused]] MovOpts parseMovExtras(keel::ArrayRef<Value> args, keel::usize start, const char *fn)
{
    MovOpts o;
    o.fn_name = fn;
    keel::usize i = start;

    // Optional positional `dim`: numeric scalar that's not a string and
    // appears before any string argument.
    if (i < args.size() && !args[i].isChar() && !args[i].isString()
        && !args[i].isEmpty() && args[i].isScalar()) {
        o.dim = keel::as<int>(args[i].toScalar());
        ++i;
    }

    // Optional positional `nanflag`.
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        const keel::String raw = args[i].toString();
        const keel::String s = toLower(raw.as_ref());
        if (s == "omitnan" || s == "omitmissing") {
            o.omit_nan = true; ++i;
        } else if (s == "includenan" || s == "includemissing") {
            o.omit_nan = false; ++i;
        }
        // else fall through — could be 'Endpoints' Name-Value pair.
    }

    // Name-Value pairs.
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error(keel::format("{}: expected a Name-Value pair", fn),
                        0, 0, fn, "", keel::format("numkit:{}:nv", fn));
        const keel::String rawName = args[i].toString();
        const keel::String name = toLower(rawName.as_ref());
        const Value &val = args[i + 1];
        if (name == "endpoints") {
            if (val.isChar() || val.isString()) {
                const keel::String rawVal = val.toString();
                const keel::String s = toLower(rawVal.as_ref());
                if      (s == "shrink")  o.ep = EndpointMode::Shrink;
                else if (s == "discard") o.ep = EndpointMode::Discard;
                else if (s == "fill")    { o.ep = EndpointMode::Fill;
                                           o.ep_fill = std::numeric_limits<double>::quiet_NaN(); }
                else
                    throw Error(keel::format("{}: Endpoints must be "
                                             "'shrink', 'discard', 'fill' or a scalar", fn),
                                0, 0, fn, "", keel::format("numkit:{}:ep", fn));
            } else if (val.isScalar()) {
                o.ep = EndpointMode::Scalar;
                o.ep_fill = val.toScalar();
            } else {
                throw Error(keel::format("{}: Endpoints must be a string "
                                         "or numeric scalar", fn),
                            0, 0, fn, "", keel::format("numkit:{}:ep", fn));
            }
        } else if (name == "samplepoints") {
            if (val.isEmpty() || (!val.dims().isVector() && !val.isScalar())
                || val.isChar() || val.isString() || val.isCell() || val.isStruct()
                || val.type() == ValueType::COMPLEX || val.type() == ValueType::LOGICAL) {
                throw Error(keel::format("{}: 'SamplePoints' must be a non-empty real numeric vector of strictly increasing finite values", fn),
                            0, 0, fn, "", keel::format("numkit:{}:samplePts", fn));
            }
            const keel::usize nt = val.numel();
            o.sample_points.clear();
            o.sample_points.reserve(nt);
            for (keel::usize idx = 0; idx < nt; ++idx) {
                const double tj = val.elemAsDouble(idx);
                if (!std::isfinite(tj) || (idx > 0 && tj <= o.sample_points.back())) {
                    throw Error(keel::format("{}: 'SamplePoints' must be strictly increasing and finite", fn),
                                0, 0, fn, "", keel::format("numkit:{}:samplePts", fn));
                }
                o.sample_points.push_back(tj);
            }
            o.has_sample_points = true;
        } else if (name == "datavariables" || name == "replacevalues") {
            throw Error(keel::format("{}: '{}' is for table/"
                                     "timetable inputs (nk does not implement those)", fn, name),
                        0, 0, fn, "", keel::format("numkit:{}:tableOnly", fn));
        } else {
            throw Error(keel::format("{}: unknown Name-Value '{}'", fn, name),
                        0, 0, fn, "", keel::format("numkit:{}:nv", fn));
        }
        i += 2;
    }
    if (i != args.size())
        throw Error(keel::format("{}: dangling argument at position {}", fn, i + 1),
                    0, 0, fn, "", keel::format("numkit:{}:nargin", fn));
    return o;
}

[[maybe_unused]] keel::Array<keel::usize> decodeWindowWithOpts(const Value &k, const char *fn,
                                        const keel::ArenaTemp &scratch, MovOpts &opt)
{
    if (!opt.has_sample_points)
        return decodeWindowValueToScratch(k, fn, scratch);

    auto badK = [&]() {
        throw Error(keel::format("{}: Window length must be a finite, "
                                 "positive, real scalar or 2-element vector of finite, "
                                 "nonnegative, real scalars.", fn),
                    0, 0, fn, "", keel::format("numkit:{}:badK", fn));
    };
    if (k.isEmpty() || k.isChar() || k.isString() || k.type() == ValueType::COMPLEX || k.type() == ValueType::LOGICAL)
        badK();
    keel::Array<keel::usize> out(scratch.allocator());
    if (k.isScalar()) {
        const double v = k.toScalar();
        if (!std::isfinite(v) || v <= 0.0) badK();
        opt.sp_kb = 0.5 * v;
        opt.sp_kf = 0.5 * v;
        out.push_back(1);
        return out;
    }
    if (k.numel() == 2) {
        const double a = k.elemAsDouble(0);
        const double b = k.elemAsDouble(1);
        if (!std::isfinite(a) || !std::isfinite(b) || a < 0.0 || b < 0.0) badK();
        opt.sp_kb = a;
        opt.sp_kf = b;
        out.push_back(1);
        return out;
    }
    badK();
    return out;
}

// Allocate a same-shape DOUBLE output via createLike.
[[maybe_unused]] Value allocSameShape(const Value &x, keel::Allocator *mr)
{
    return createLike(x, ValueType::DOUBLE, mr);
}

// Apply per-window reducer F over a 1-D run [src, src+n) with stride
// `step` (in elements). Output written to dst with the same stride.
// F has signature `double(const double *win, size_t winLen)`.
// `ep` controls how out-of-range window positions are filled:
//   Shrink — drop missing positions, the window shrinks at edges
//   Fill   — pad missing positions with NaN (window keeps full length)
//   Scalar — pad missing positions with `ep_fill`
//   Discard — handled in driver, never reaches here
// `omit_nan == true` filters NaN out of the (possibly already padded)
// window before the reducer; an empty filtered window collapses to NaN.
template <typename F>
void runMoving(const double *src, keel::usize n, keel::isize step,
               double *dst, const Window &w, const keel::ArenaTemp &scratch,
               bool omit_nan, EndpointMode ep, double ep_fill, F &&fn)
{
    if (n == 0) return;
    const long N = keel::as<long>(n);
    const long fullLen = w.kb + w.kf + 1;
    auto buf = keel::Array<double>(fullLen, scratch.allocator());
    for (long i = 0; i < N; ++i) {
        keel::usize bufN = 0;
        for (long j = -w.kb; j <= w.kf; ++j) {
            const long pos = i + j;
            double v;
            if (pos >= 0 && pos < N) {
                v = src[pos * step];
            } else if (ep == EndpointMode::Fill) {
                v = std::numeric_limits<double>::quiet_NaN();
            } else if (ep == EndpointMode::Scalar) {
                v = ep_fill;
            } else {
                continue;   // Shrink — drop the missing position
            }
            if (omit_nan && std::isnan(v)) continue;
            buf[bufN++] = v;
        }
        dst[i * step] = (bufN == 0) ? std::numeric_limits<double>::quiet_NaN()
                                    : fn(buf.data(), bufN);
    }
}

// Driver with explicit dim + MovOpts. Handles vector / 2-D / 3-D.
// For Endpoints == Discard, output is shorter along the operating dim by
// (kb + kf) elements. For Shrink/Fill/Scalar the output keeps the same
// shape as input.
template <typename F>
Value movingDriverDim(const Value &x, const Window &w, int dim, const MovOpts &opt, F &&fn, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (x.isScalar()) {
        if (opt.has_sample_points && opt.sample_points.size() != 1) {
            throw Error(keel::format("{}: 'SamplePoints' length must match the operating dimension length", opt.fn_name),
                        0, 0, opt.fn_name, "", keel::format("numkit:{}:samplePts", opt.fn_name));
        }
        auto out = Value::scalar(0.0, mr);
        const double v = x.toScalar();
        // Single-element window: 'Discard' would empty out — but a scalar
        // input is a degenerate case; mirror Shrink for consistency.
        out.doubleDataMut()[0] = (opt.omit_nan && std::isnan(v))
                                     ? std::numeric_limits<double>::quiet_NaN()
                                     : fn(&v, 1);
        return out;
    }
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const double *src = x.doubleData();

    auto runSamplePoints = [&](const double *colSrc, keel::usize n, keel::isize step,
                               double *colDst) {
        if (n == 0) return;
        const auto &t = opt.sample_points;
        const double kb = opt.sp_kb;
        const double kf = opt.sp_kf;
        const double span = (n > 1) ? std::fabs(t[n - 1] - t[0]) : 1.0;
        const double tol = 1e-12 * std::max(1.0, span);
        auto buf = keel::Array<double>(n + 2, scratch.allocator());
        keel::usize outI = 0;
        for (keel::usize i = 0; i < n; ++i) {
            const bool leftOut  = (t[i] - kb < t[0] - tol);
            const bool rightOut = (t[i] + kf > t[n - 1] + tol);
            if (opt.ep == EndpointMode::Discard && (leftOut || rightOut))
                continue;
            const double tLo = t[i] - kb - tol;
            const double tHi = t[i] + kf + tol;
            keel::usize bufN = 0;
            if (leftOut) {
                if (opt.ep == EndpointMode::Fill && !opt.omit_nan)
                    buf[bufN++] = std::numeric_limits<double>::quiet_NaN();
                else if (opt.ep == EndpointMode::Scalar && !(opt.omit_nan && std::isnan(opt.ep_fill)))
                    buf[bufN++] = opt.ep_fill;
            }
            for (keel::usize j = 0; j < n; ++j) {
                if (t[j] < tLo) continue;
                if (t[j] > tHi) break;
                const double v = colSrc[keel::as<keel::isize>(j) * step];
                if (opt.omit_nan && std::isnan(v)) continue;
                buf[bufN++] = v;
            }
            if (rightOut) {
                if (opt.ep == EndpointMode::Fill && !opt.omit_nan)
                    buf[bufN++] = std::numeric_limits<double>::quiet_NaN();
                else if (opt.ep == EndpointMode::Scalar && !(opt.omit_nan && std::isnan(opt.ep_fill)))
                    buf[bufN++] = opt.ep_fill;
            }
            colDst[keel::as<keel::isize>(outI) * step] =
                (bufN == 0) ? std::numeric_limits<double>::quiet_NaN()
                            : fn(buf.data(), bufN);
            ++outI;
        }
    };

    auto runDiscard = [&](const double *colSrc, keel::usize n, keel::isize step,
                          double *colDst) {
        // Discard: emit only positions i where the window is fully in-bounds,
        // i.e. i in [kb, n-1-kf]. Output length = n - (kb+kf) (or 0).
        const long kb = w.kb, kf = w.kf;
        const long fullLen = kb + kf + 1;
        if (keel::as<long>(n) < fullLen) return;
        auto buf = keel::Array<double>(fullLen, scratch.allocator());
        long outI = 0;
        for (long i = kb; i + kf < keel::as<long>(n); ++i) {
            keel::usize bufN = 0;
            if (opt.omit_nan) {
                for (long j = -kb; j <= kf; ++j) {
                    const double v = colSrc[(i + j) * step];
                    if (!std::isnan(v)) buf[bufN++] = v;
                }
            } else {
                for (long j = -kb; j <= kf; ++j)
                    buf[keel::as<keel::usize>(j + kb)] = colSrc[(i + j) * step];
                bufN = keel::as<keel::usize>(fullLen);
            }
            colDst[outI * step] = (bufN == 0)
                                      ? std::numeric_limits<double>::quiet_NaN()
                                      : fn(buf.data(), bufN);
            ++outI;
        }
    };

    // Compute output dims for Discard mode.
    auto reducedLen = [&](keel::usize L) -> keel::usize {
        if (opt.has_sample_points) {
            if (L == 0) return 0;
            const auto &t = opt.sample_points;
            const double span = (L > 1) ? std::fabs(t[L - 1] - t[0]) : 1.0;
            const double tol = 1e-12 * std::max(1.0, span);
            keel::usize cnt = 0;
            for (keel::usize i = 0; i < L; ++i) {
                if (t[i] - opt.sp_kb >= t[0] - tol && t[i] + opt.sp_kf <= t[L - 1] + tol)
                    ++cnt;
            }
            return cnt;
        }
        const long delta = w.kb + w.kf;
        if (keel::as<long>(L) <= delta) return 0;
        return L - keel::as<keel::usize>(delta);
    };

    if (x.dims().isVector()) {
        const keel::usize N = x.numel();
        if (opt.has_sample_points && opt.sample_points.size() != N) {
            throw Error(keel::format("{}: 'SamplePoints' length must match the operating dimension length", opt.fn_name),
                        0, 0, opt.fn_name, "", keel::format("numkit:{}:samplePts", opt.fn_name));
        }
        keel::usize outN = N;
        if (opt.ep == EndpointMode::Discard) outN = reducedLen(N);
        const keel::usize r = (x.dims().rows() > 1) ? outN : 1;
        const keel::usize c = (x.dims().rows() > 1) ? 1    : outN;
        Value out = Value::matrix(r, c, ValueType::DOUBLE, mr);
        if (outN == 0) return out;
        double *dst = out.doubleDataMut();
        if (opt.has_sample_points) {
            runSamplePoints(src, N, /*step=*/1, dst);
        } else if (opt.ep == EndpointMode::Discard) {
            runDiscard(src, N, /*step=*/1, dst);
        } else {
            runMoving(src, N, /*step=*/1, dst, w, scratch,
                      opt.omit_nan, opt.ep, opt.ep_fill, fn);
        }
        return out;
    }

    const auto &d = x.dims();
    const keel::usize R = d.rows(), C = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    const keel::usize pageStride = R * C;

    if (opt.has_sample_points) {
        const keel::usize expectedLen = (dim == 1) ? R : (dim == 2 ? C : (dim == 3 && d.is3D() ? P : 1));
        if (opt.sample_points.size() != expectedLen) {
            throw Error(keel::format("{}: 'SamplePoints' length must match the operating dimension length", opt.fn_name),
                        0, 0, opt.fn_name, "", keel::format("numkit:{}:samplePts", opt.fn_name));
        }
    }

    // Determine output dims.
    keel::usize outR = R, outC = C, outP = P;
    if (opt.ep == EndpointMode::Discard) {
        if      (dim == 1) outR = reducedLen(R);
        else if (dim == 2) outC = reducedLen(C);
        else if (dim == 3) outP = reducedLen(P);
    }
    Value out;
    if (d.is3D()) out = Value::matrix3d(outR, outC, outP, ValueType::DOUBLE, mr);
    else          out = Value::matrix(outR, outC, ValueType::DOUBLE, mr);
    if (out.numel() == 0) return out;
    double *dst = out.doubleDataMut();
    const keel::usize outPageStride = outR * outC;

    auto run = [&](const double *colSrc, keel::usize inLen, keel::isize step,
                   double *colDst) {
        if (opt.has_sample_points)           runSamplePoints(colSrc, inLen, step, colDst);
        else if (opt.ep == EndpointMode::Discard) runDiscard(colSrc, inLen, step, colDst);
        else                                 runMoving(colSrc, inLen, step, colDst, w, scratch,
                                                       opt.omit_nan, opt.ep, opt.ep_fill, fn);
    };

    if (dim == 1) {
        for (keel::usize p = 0; p < P; ++p)
            for (keel::usize c = 0; c < C; ++c) {
                const double *colSrc = src + p * pageStride + c * R;
                double       *colDst = dst + p * outPageStride + c * outR;
                run(colSrc, R, /*step=*/1, colDst);
            }
    } else if (dim == 2) {
        for (keel::usize p = 0; p < P; ++p)
            for (keel::usize r = 0; r < R; ++r) {
                const double *rowSrc = src + p * pageStride + r;
                double       *rowDst = dst + p * outPageStride + r;
                run(rowSrc, C, /*step=*/keel::as<keel::isize>(R), rowDst);
            }
    } else if (dim == 3 && d.is3D()) {
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize r = 0; r < R; ++r) {
                const double *pgSrc = src + c * R + r;
                double       *pgDst = dst + c * outR + r;
                run(pgSrc, P, /*step=*/keel::as<keel::isize>(pageStride), pgDst);
            }
    } else {
        // Trailing-singleton or out-of-rank: identity copy (legacy behaviour).
        std::copy(src, src + x.numel(), dst);
    }
    return out;
}

// ── per-window reducers ───────────────────────────────────────────────

double winMean(const double *w, keel::usize n)
{
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    double s = 0.0;
    for (keel::usize i = 0; i < n; ++i) s += w[i];
    return s / keel::as<double>(n);
}

double winSum(const double *w, keel::usize n)
{
    double s = 0.0;
    for (keel::usize i = 0; i < n; ++i) s += w[i];
    return s;
}

double winMin(const double *w, keel::usize n)
{
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    double m = w[0];
    for (keel::usize i = 1; i < n; ++i)
        if (w[i] < m) m = w[i];
    return m;
}

double winMax(const double *w, keel::usize n)
{
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    double m = w[0];
    for (keel::usize i = 1; i < n; ++i)
        if (w[i] > m) m = w[i];
    return m;
}

double winProd(const double *w, keel::usize n)
{
    double p = 1.0;
    for (keel::usize i = 0; i < n; ++i) p *= w[i];
    return p;
}

// nth_element on a scratch copy (window already lives in caller's buf).
double winMedianInPlace(double *w, keel::usize n)
{
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    if (n == 1) return w[0];
    const keel::usize mid = n / 2;
    std::nth_element(w, w + mid, w + n);
    if (n % 2 == 1)
        return w[mid];
    const double upper = w[mid];
    double lower = w[0];
    for (keel::usize i = 1; i < mid; ++i)
        if (w[i] > lower) lower = w[i];
    return 0.5 * (upper + lower);
}

double winVar(const double *w, keel::usize n, int normFlag)
{
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    // The reference engine defines std/var of a single value as 0 regardless of the
    // normalization (the degenerate 0/0 from N-1 is taken as 0). This is
    // hit at edge windows of length 1 and, under 'omitnan', at interior
    // windows that reduce to a single valid element.
    if (n == 1) return 0.0;
    double s = 0.0;
    for (keel::usize i = 0; i < n; ++i) s += w[i];
    const double mean = s / keel::as<double>(n);
    double sq = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const double d = w[i] - mean;
        sq += d * d;
    }
    const double denom = (normFlag == 1) ? keel::as<double>(n)
                                          : keel::as<double>(n - 1);
    return sq / denom;
}

// Median absolute deviation: median(|x - median(x)|).
double winMadInPlace(double *w, keel::usize n, const keel::ArenaTemp &scratch)
{
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    auto copy = keel::Array<double>(n, scratch.allocator());
    std::copy(w, w + n, copy.data());
    const double med = winMedianInPlace(copy.data(), n);
    auto devs = keel::Array<double>(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) devs[i] = keel::abs(w[i] - med);
    return winMedianInPlace(devs.data(), n);
}

} // namespace
namespace {

// Real / imaginary parts of a COMPLEX value as same-shape DOUBLE arrays, and
// the inverse combine (movmean is linear, so it commutes with this split).
Value cplxRealPart(const Value &f, keel::Allocator *mr)
{
    auto r = createLike(f, ValueType::DOUBLE, mr);
    double *d = r.doubleDataMut();
    const keel::c64 *c = f.complexData();
    const keel::usize n = f.numel();
    for (keel::usize i = 0; i < n; ++i) d[i] = c[i].real();
    return r;
}
Value cplxImagPart(const Value &f, keel::Allocator *mr)
{
    auto r = createLike(f, ValueType::DOUBLE, mr);
    double *d = r.doubleDataMut();
    const keel::c64 *c = f.complexData();
    const keel::usize n = f.numel();
    for (keel::usize i = 0; i < n; ++i) d[i] = c[i].imag();
    return r;
}
Value cplxCombine(const Value &re, const Value &im, keel::Allocator *mr)
{
    auto out = createLike(re, ValueType::COMPLEX, mr);
    keel::c64 *o = out.complexDataMut();
    const double *r = re.doubleData(), *m = im.doubleData();
    const keel::usize n = out.numel();
    for (keel::usize i = 0; i < n; ++i) o[i] = keel::c64(r[i], m[i]);
    return out;
}

// The reference engine promotes an integer/logical operand to DOUBLE for the arithmetic
// moving functions (movsum/movprod/movmean) — e.g. movsum(int8(...)) returns
// double, NOT int8. Promote element-wise here; double/single/complex pass
// through unchanged. char is NOT promoted, so the doubleData() path keeps throwing for it.
// bugs/builtin/movfun-typeclass.md.
static Value movPromoteIntLogical(const Value &x, keel::Allocator *mr)
{
    const ValueType t = x.type();
    if (!(isIntegerType(t) || t == ValueType::LOGICAL)) return x;
    Value r = createLike(x, ValueType::DOUBLE, mr);
    double *dst = r.doubleDataMut();
    const keel::usize n = x.numel();
    for (keel::usize i = 0; i < n; ++i) dst[i] = x.elemAsDouble(i);
    return r;
}

Value movmean_impl(const Value &x, keel::ArrayRef<keel::usize> k, const MovOpts &opt, keel::Allocator *mr)
{
    // Complex: moving-mean the real + imaginary parts separately, recombine
    // (window / endpoints / dim all carry through each recursive call).
    if (x.type() == ValueType::COMPLEX) {
        Value re = movmean_impl(cplxRealPart(x, mr), k, opt, mr);
        Value im = movmean_impl(cplxImagPart(x, mr), k, opt, mr);
        return cplxCombine(re, im, mr);
    }
    const Value xd = movPromoteIntLogical(x, mr);
    const auto w = decodeWindow(k, "movmean");
    const int d = resolveDim(xd, opt.dim, "movmean");
    return movingDriverDim(xd, w, d, opt, [](const double *win, keel::usize n) { return winMean(win, n); }, mr);
}

Value movsum_impl(const Value &x, keel::ArrayRef<keel::usize> k, const MovOpts &opt, keel::Allocator *mr)
{
    const Value xd = movPromoteIntLogical(x, mr);
    const auto w = decodeWindow(k, "movsum");
    const int d = resolveDim(xd, opt.dim, "movsum");
    return movingDriverDim(xd, w, d, opt, [](const double *win, keel::usize n) { return winSum(win, n); }, mr);
}

// movmax/movmin PRESERVE the input class (order statistics): integer -> integer
// (values are exact) and logical -> logical. movmedian preserves the INTEGER
// class with ROUND-half-away (a window median can be fractional), but on LOGICAL
// returns DOUBLE (a 0.5 median can't be logical). bugs/stats/movfun-order-stats.md.
static Value movNarrowToLogical(const Value &d, keel::Allocator *mr)
{
    if (d.isScalar()) return Value::logicalScalar(d.toScalar() != 0.0, mr);
    Value r = createLike(d, ValueType::LOGICAL, mr);
    keel::u8 *dst = r.logicalDataMut();
    const keel::usize n = d.numel();
    for (keel::usize i = 0; i < n; ++i) dst[i] = (d.elemAsDouble(i) != 0.0) ? 1 : 0;
    return r;
}

Value movmin_impl(const Value &x, keel::ArrayRef<keel::usize> k, const MovOpts &opt, keel::Allocator *mr)
{
    const ValueType ot = x.type();
    const Value xd = movPromoteIntLogical(x, mr);
    const auto w = decodeWindow(k, "movmin");
    const int d = resolveDim(xd, opt.dim, "movmin");
    Value out = movingDriverDim(xd, w, d, opt, [](const double *win, keel::usize n) { return winMin(win, n); }, mr);
    if (isIntegerType(ot)) return doubleToIntegerExact(out, ot, mr);   // exact (subset)
    if (ot == ValueType::LOGICAL) return movNarrowToLogical(out, mr);
    return out;
}

Value movmax_impl(const Value &x, keel::ArrayRef<keel::usize> k, const MovOpts &opt, keel::Allocator *mr)
{
    const ValueType ot = x.type();
    const Value xd = movPromoteIntLogical(x, mr);
    const auto w = decodeWindow(k, "movmax");
    const int d = resolveDim(xd, opt.dim, "movmax");
    Value out = movingDriverDim(xd, w, d, opt, [](const double *win, keel::usize n) { return winMax(win, n); }, mr);
    if (isIntegerType(ot)) return doubleToIntegerExact(out, ot, mr);   // exact (subset)
    if (ot == ValueType::LOGICAL) return movNarrowToLogical(out, mr);
    return out;
}

Value movprod_impl(const Value &x, keel::ArrayRef<keel::usize> k, const MovOpts &opt, keel::Allocator *mr)
{
    const Value xd = movPromoteIntLogical(x, mr);
    const auto w = decodeWindow(k, "movprod");
    const int d = resolveDim(xd, opt.dim, "movprod");
    return movingDriverDim(xd, w, d, opt, [](const double *win, keel::usize n) { return winProd(win, n); }, mr);
}

Value movmedian_impl(const Value &x, keel::ArrayRef<keel::usize> k, const MovOpts &opt, keel::Allocator *mr)
{
    const ValueType ot = x.type();
    const Value xd = movPromoteIntLogical(x, mr);
    const auto w = decodeWindow(k, "movmedian");
    const int d = resolveDim(xd, opt.dim, "movmedian");
    Value out = movingDriverDim(xd, w, d, opt, [](const double *win, keel::usize n) {
            double tmp[1024];   // typical k <= a few hundred
            if (n <= sizeof(tmp) / sizeof(tmp[0])) {
                std::copy(win, win + n, tmp);
                return winMedianInPlace(tmp, n);
            }
            keel::Array<double> heap(n);
            std::copy(win, win + n, heap.data());
            return winMedianInPlace(heap.data(), n);
        }, mr);
    if (isIntegerType(ot)) {
        // The reference engine rounds a fractional window median to the int class (round half
        // away from zero), unlike the truncating doubleToIntegerExact — so round
        // into the double buffer first. LOGICAL input returns DOUBLE (a 0.5
        // median has no logical representation), so no narrow there.
        double *p = out.doubleDataMut();
        const keel::usize n = out.numel();
        for (keel::usize i = 0; i < n; ++i) p[i] = std::round(p[i]);
        return doubleToIntegerExact(out, ot, mr);
    }
    return out;
}

Value movvar_impl(const Value &x, keel::ArrayRef<keel::usize> k, int normFlag, const MovOpts &opt, keel::Allocator *mr)
{
    if (normFlag != 0 && normFlag != 1)
        throw Error("movvar: normFlag must be 0 or 1",
                     0, 0, "movvar", "", "numkit:movvar:badNormFlag");
    const auto w = decodeWindow(k, "movvar");
    const int d = resolveDim(x, opt.dim, "movvar");
    return movingDriverDim(x, w, d, opt, [normFlag](const double *win, keel::usize n) { return winVar(win, n, normFlag); }, mr);
}

Value movstd_impl(const Value &x, keel::ArrayRef<keel::usize> k, int normFlag, const MovOpts &opt, keel::Allocator *mr)
{
    auto v = movvar_impl(x, k, normFlag, opt, mr);
    double *p = v.doubleDataMut();
    const keel::usize n = v.numel();
    for (keel::usize i = 0; i < n; ++i)
        p[i] = std::sqrt(p[i]);
    return v;
}

Value movmad_impl(const Value &x, keel::ArrayRef<keel::usize> k, const MovOpts &opt, keel::Allocator *mr)
{
    const auto w = decodeWindow(k, "movmad");
    const int d = resolveDim(x, opt.dim, "movmad");
    return movingDriverDim(x, w, d, opt, [mr](const double *win, keel::usize n) -> double {
            keel::ArenaTemp local = keel::get_scratch(mr);
            auto buf = keel::Array<double>(n, local.allocator());
            std::copy(win, win + n, buf.data());
            return winMadInPlace(buf.data(), n, local);
        }, mr);
}

} // anonymous

// hampel internals (def in moving.cpp, external): raw-buffer outlier core +
// the Value-level argument validator.
void hampelValidate(const Value &x, int k, double nsigmas);
void hampelCore(const double *src, keel::usize n, int k, double nsigmas,
                double *dst, keel::u8 *mask, double *median,
                double *sigma, keel::Allocator *mr);

} // namespace numkit::stats
