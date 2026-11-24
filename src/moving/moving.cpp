// toolboxes/stats/src/moving/moving.cpp
// Sliding-window statistics: movmean / movmedian / movsum / movmin /
// movmax / movstd / movvar / movmad / movprod, plus smoothdata + hampel.
// Strategy: a single per-slice driver walks the chosen dim; per-window
// reduction is supplied as a callable. O(N·k) — not the asymptotically
// optimal bound, but matches the rest of the library's "correctness
// first, optimise after benches" policy.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/stats.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/moving/moving.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>            // createLike, createForDims (toolboxes/builtin/src/)
#include <numkit/ops/reductions.hpp>  // numkit::ops::firstNonSingletonDim, validateDim


#include "moving_detail.hpp"

namespace numkit::stats {


// ── internal *_impl with full MovOpts ─────────────────────────────────


// ── public API (legacy dim-only sigs; defaults for extras) ────────────

Value movmean(const Value &x, keel::ArrayRef<keel::usize> k, keel::i32 dim, keel::Allocator *mr)
{ MovOpts o; o.dim = dim; return movmean_impl(x, k, o, mr); }
Value movsum(const Value &x, keel::ArrayRef<keel::usize> k, keel::i32 dim, keel::Allocator *mr)
{ MovOpts o; o.dim = dim; return movsum_impl(x, k, o, mr); }
Value movmin(const Value &x, keel::ArrayRef<keel::usize> k, keel::i32 dim, keel::Allocator *mr)
{ MovOpts o; o.dim = dim; return movmin_impl(x, k, o, mr); }
Value movmax(const Value &x, keel::ArrayRef<keel::usize> k, keel::i32 dim, keel::Allocator *mr)
{ MovOpts o; o.dim = dim; return movmax_impl(x, k, o, mr); }
Value movprod(const Value &x, keel::ArrayRef<keel::usize> k, keel::i32 dim, keel::Allocator *mr)
{ MovOpts o; o.dim = dim; return movprod_impl(x, k, o, mr); }
Value movmedian(const Value &x, keel::ArrayRef<keel::usize> k, keel::i32 dim, keel::Allocator *mr)
{ MovOpts o; o.dim = dim; return movmedian_impl(x, k, o, mr); }
Value movvar(const Value &x, keel::ArrayRef<keel::usize> k, keel::i32 normFlag, keel::i32 dim, keel::Allocator *mr)
{ MovOpts o; o.dim = dim; return movvar_impl(x, k, normFlag, o, mr); }
Value movstd(const Value &x, keel::ArrayRef<keel::usize> k, keel::i32 normFlag, keel::i32 dim, keel::Allocator *mr)
{ MovOpts o; o.dim = dim; return movstd_impl(x, k, normFlag, o, mr); }
Value movmad(const Value &x, keel::ArrayRef<keel::usize> k, keel::i32 dim, keel::Allocator *mr)
{ MovOpts o; o.dim = dim; return movmad_impl(x, k, o, mr); }

// Gaussian-weighted moving average for smoothdata 'gaussian'. MATLAB R2025b
// uses kernel exp(-d^2/(2*sigma^2)) with sigma = windowLength/5, CENTRED on
// the current sample, with the default 'shrink' endpoints (truncate the
// kernel at the array edges and renormalise). NaNs are omitted from the
// weighted sum when opt.omit_nan. dim 1/2/3 + vector. The generic moving
// driver cannot align a weighted kernel at truncated edges (its reducer only
// sees the shrunk window, not the centre position), so this path walks slices
// directly. Inputs are DOUBLE-backed (mirrors the rest of the moving driver).
static Value smoothGaussianDim(const Value &x, const Window &w, keel::f64 sigma,
                               keel::i32 dim, const MovOpts &opt,
                               keel::Allocator *mr)
{
    if (x.isEmpty()) return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (x.isScalar()) return Value::scalar(x.toScalar(), mr);

    const keel::i64 kb = w.kb, kf = w.kf;
    auto perSlice = [&](const keel::f64 *src, keel::usize n, keel::isize step, keel::f64 *dst) {
        keel::stats::smooth_gaussian_strided(src, n, step, kb, kf, sigma, opt.omit_nan, dst);
    };

    auto out = createLike(x, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    const keel::f64 *src = x.doubleData();
    const auto &d = x.dims();
    if (d.isVector()) { perSlice(src, x.numel(), 1, dst); return out; }

    const keel::usize R = d.rows(), C = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1, pg = R * C;
    if (dim == 1) {
        for (keel::usize p = 0; p < P; ++p)
            for (keel::usize c = 0; c < C; ++c)
                perSlice(src + p * pg + c * R, R, 1, dst + p * pg + c * R);
    } else if (dim == 2) {
        for (keel::usize p = 0; p < P; ++p)
            for (keel::usize r = 0; r < R; ++r)
                perSlice(src + p * pg + r, C, keel::as<keel::isize>(R), dst + p * pg + r);
    } else if (dim == 3 && d.is3D()) {
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize r = 0; r < R; ++r)
                perSlice(src + c * R + r, P, keel::as<keel::isize>(pg), dst + c * R + r);
    } else {
        keel::copy(src, src + x.numel(), dst);
    }
    return out;
}

static void buildSGMatrix(keel::i32 F, keel::i32 deg, keel::Array<keel::f64> &B)
{
    (void)keel::stats::smooth_sgolay_matrix(F, deg, B);
}

static void sgolaySlice(const keel::f64 *src, keel::usize n, keel::isize step, keel::f64 *dst,
                        keel::ArrayRef<keel::f64> B, keel::i32 F)
{
    keel::stats::smooth_sgolay_strided(src, n, step, B, F, dst);
}

// Savitzky-Golay smoothing for smoothdata 'sgolay' (degree 2). Walks slices
// along `dim`; reduces the window to fit short slices.
static Value smoothSGDim(const Value &x, keel::i32 F, keel::i32 deg, keel::i32 dim, keel::Allocator *mr)
{
    if (x.isEmpty()) return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (x.isScalar()) return Value::scalar(x.toScalar(), mr);

    auto          out = createLike(x, ValueType::DOUBLE, mr);
    keel::f64 *      dst = out.doubleDataMut();
    const keel::f64 *src = x.doubleData();
    const auto &  d   = x.dims();

    const keel::usize sliceLen = d.isVector() ? x.numel()
                                         : (dim == 2 ? d.cols() : (dim == 3 ? (d.is3D() ? d.pages() : 1) : d.rows()));
    keel::i32 Fe = F;
    if (Fe > keel::as<keel::i32>(sliceLen)) Fe = keel::as<keel::i32>(sliceLen);
    if (Fe % 2 == 0) --Fe;                       // SG window must be odd
    if (Fe < deg + 1) Fe = deg + 1 | 1;          // need > degree (odd)
    if (Fe < 1) Fe = 1;
    keel::Array<keel::f64> B;
    if (Fe >= deg + 1) buildSGMatrix(Fe, deg, B);

    auto perSlice = [&](const keel::f64 *s, keel::usize nlen, keel::isize step, keel::f64 *o) {
        if (Fe < deg + 1 || nlen < keel::as<keel::usize>(Fe)) {
            for (keel::usize i = 0; i < nlen; ++i) o[i * step] = s[i * step];  // too short → passthrough
            return;
        }
        sgolaySlice(s, nlen, step, o, B.as_ref(), Fe);
    };

    if (d.isVector()) { perSlice(src, x.numel(), 1, dst); return out; }
    const keel::usize R = d.rows(), C = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1, pg = R * C;
    if (dim == 1) {
        for (keel::usize p = 0; p < P; ++p)
            for (keel::usize c = 0; c < C; ++c)
                perSlice(src + p * pg + c * R, R, 1, dst + p * pg + c * R);
    } else if (dim == 2) {
        for (keel::usize p = 0; p < P; ++p)
            for (keel::usize r = 0; r < R; ++r)
                perSlice(src + p * pg + r, C, keel::as<keel::isize>(R), dst + p * pg + r);
    } else if (dim == 3 && d.is3D()) {
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize r = 0; r < R; ++r)
                perSlice(src + c * R + r, P, keel::as<keel::isize>(pg), dst + c * R + r);
    } else {
        keel::copy(src, src + x.numel(), dst);
    }
    return out;
}

static void localRegressSlice(const keel::f64 *src, keel::usize n, keel::isize step, keel::f64 *dst,
                              keel::i32 F, keel::i32 deg)
{
    keel::stats::smooth_local_regress_strided(src, n, step, F, deg, dst);
}

// smoothdata 'lowess'/'loess' driver — walks slices along `dim`.
static Value smoothLocalRegDim(const Value &x, keel::i32 F, keel::i32 deg, keel::i32 dim, keel::Allocator *mr)
{
    if (x.isEmpty()) return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (x.isScalar()) return Value::scalar(x.toScalar(), mr);

    auto          out = createLike(x, ValueType::DOUBLE, mr);
    keel::f64 *      dst = out.doubleDataMut();
    const keel::f64 *src = x.doubleData();
    const auto &  d   = x.dims();

    const keel::usize sliceLen = d.isVector() ? x.numel()
                                         : (dim == 2 ? d.cols() : (dim == 3 ? (d.is3D() ? d.pages() : 1) : d.rows()));
    keel::i32 Fe = F;
    if (Fe > keel::as<keel::i32>(sliceLen)) Fe = keel::as<keel::i32>(sliceLen);
    if (Fe < 1) Fe = 1;

    auto perSlice = [&](const keel::f64 *s, keel::usize nlen, keel::isize step, keel::f64 *o) {
        if (nlen < keel::as<keel::usize>(Fe) || Fe < 2) {
            for (keel::usize i = 0; i < nlen; ++i) o[i * step] = s[i * step];
            return;
        }
        localRegressSlice(s, nlen, step, o, Fe, deg);
    };

    if (d.isVector()) { perSlice(src, x.numel(), 1, dst); return out; }
    const keel::usize R = d.rows(), C = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1, pg = R * C;
    if (dim == 1) {
        for (keel::usize p = 0; p < P; ++p)
            for (keel::usize c = 0; c < C; ++c)
                perSlice(src + p * pg + c * R, R, 1, dst + p * pg + c * R);
    } else if (dim == 2) {
        for (keel::usize p = 0; p < P; ++p)
            for (keel::usize r = 0; r < R; ++r)
                perSlice(src + p * pg + r, C, keel::as<keel::isize>(R), dst + p * pg + r);
    } else if (dim == 3 && d.is3D()) {
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize r = 0; r < R; ++r)
                perSlice(src + c * R + r, P, keel::as<keel::isize>(pg), dst + c * R + r);
    } else {
        keel::copy(src, src + x.numel(), dst);
    }
    return out;
}

namespace {

void smoothdata_argcheck(keel::StringRef method)
{
    keel::String m = keel::String::from(method);
    m.to_ascii_lower_in_place();
    if (!m.empty() && m != "movmean" && m != "movmedian" && m != "gaussian"
        && m != "sgolay" && m != "lowess" && m != "loess") {
        throw Error(keel::format("smoothdata: method '{}' not supported "
                                 "(supported: 'movmean', 'movmedian', 'gaussian', 'sgolay', "
                                 "'lowess', 'loess')", method),
                     0, 0, "smoothdata", "", "numkit:smoothdata:unsupportedMethod");
    }
}

void hampel_argcheck(const Value &x, keel::i32 k, keel::f64 nsigmas)
{
    if (k < 0)
        throw Error("hampel: k must be >= 0",
                     0, 0, "hampel", "", "numkit:hampel:badK");
    if (nsigmas <= 0)
        throw Error("hampel: nsigmas must be positive",
                     0, 0, "hampel", "", "numkit:hampel:badSigmas");
    if (!x.dims().isVector() && !x.isScalar())
        throw Error("hampel: vector input only (matrix form deferred)",
                     0, 0, "hampel", "", "numkit:hampel:notVector");
}

} // namespace

// ── smoothdata ────────────────────────────────────────────────────────
Value smoothdata(const Value &x, keel::StringRef method, keel::i32 k, keel::i32 dim, keel::Allocator *mr)
{
    smoothdata_argcheck(method);
    keel::String m = keel::String::from(method);
    m.to_ascii_lower_in_place();// Heuristic default window — the reference engine picks ~max(min(round(0.1*n), 10), 3).
    if (k <= 0) {
        const keel::usize n = (x.dims().isVector() || x.isScalar())
                          ? x.numel()
                          : x.dims().rows();
        keel::i64 auto_k = keel::as<keel::i64>(keel::round(0.1 * keel::as<keel::f64>(n)));
        if (auto_k < 3) auto_k = 3;
        if (auto_k > keel::as<keel::i64>(n)) auto_k = keel::as<keel::i64>(n);
        k = keel::as<keel::i32>(auto_k);
        if (k < 1) k = 1;
    }
    const keel::usize kArr[1] = { keel::as<keel::usize>(k) };
    keel::ArrayRef<keel::usize> kSpan(kArr, 1);

    MovOpts opt;
    opt.dim = dim;
    if (m == "movmean" || m.empty())
        return movmean_impl(x, kSpan, opt, mr);
    if (m == "movmedian")
        return movmedian_impl(x, kSpan, opt, mr);
    if (m == "gaussian") {
        // Gaussian-weighted moving average using sigma =
        // windowLength/5 and centring the kernel on the CURRENT sample, with
        // 'shrink' endpoints.
        const auto w = decodeWindow(kSpan, "smoothdata");
        const keel::i32 d = resolveDim(x, dim, "smoothdata");
        const keel::f64 sigma = (k > 0) ? keel::as<keel::f64>(k) / 5.0 : 0.2;
        return smoothGaussianDim(x, w, sigma, d, opt, mr);
    }
    if (m == "sgolay") {
        // Savitzky-Golay (degree 2).
        const keel::i32 d = resolveDim(x, dim, "smoothdata");
        return smoothSGDim(x, k, /*deg=*/2, d, mr);
    }
    if (m == "lowess" || m == "loess") {
        // Local tricube-weighted regression — linear (lowess) / quadratic
        // (loess).
        const keel::i32 d = resolveDim(x, dim, "smoothdata");
        return smoothLocalRegDim(x, k, m == "loess" ? 2 : 1, d, mr);
    }
    throw Error(keel::format("smoothdata: method '{}' not supported "
                             "(supported: 'movmean', 'movmedian', 'gaussian', 'sgolay', "
                             "'lowess', 'loess')", method),
                 0, 0, "smoothdata", "", "numkit:smoothdata:unsupportedMethod");
}

// ── hampel ────────────────────────────────────────────────────────────

void hampelCore(const keel::f64 *src, keel::usize n, keel::i32 k, keel::f64 nsigmas,
                       keel::f64 *dst, keel::u8 *mask, keel::f64 *median,
                       keel::f64 *sigma, keel::Allocator *mr)
{
    (void)keel::stats::hampel_1d(
        keel::ArrayRef<keel::f64>(src, n), k, nsigmas,
        keel::MutArrayRef<keel::f64>(dst, n),
        mask ? keel::MutArrayRef<keel::u8>(mask, n) : keel::MutArrayRef<keel::u8>{},
        median ? keel::MutArrayRef<keel::f64>(median, n) : keel::MutArrayRef<keel::f64>{},
        sigma ? keel::MutArrayRef<keel::f64>(sigma, n) : keel::MutArrayRef<keel::f64>{},
        mr);
}

void hampelValidate(const Value &x, keel::i32 k, keel::f64 nsigmas)
{
    hampel_argcheck(x, k, nsigmas);
}

Value hampel(const Value &x, keel::i32 k, keel::f64 nsigmas, keel::Allocator *mr)
{
    hampel_argcheck(x, k, nsigmas);
    auto out = createLike(x, ValueType::DOUBLE, mr);
    const keel::usize n = x.numel();
    if (n == 0) return out;
    hampelCore(x.doubleData(), n, k, nsigmas, out.doubleDataMut(),
               nullptr, nullptr, nullptr, mr);
    return out;
}

} // namespace numkit::stats
