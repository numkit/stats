// toolboxes/stats/src/moving/moving.cpp
// Sliding-window statistics: movmean / movmedian / movsum / movmin /
// movmax / movstd / movvar / movmad / movprod, plus smoothdata + hampel.
// Strategy: a single per-slice driver walks the chosen dim; per-window
// reduction is supplied as a callable. O(N·k) — not the asymptotically
// optimal bound, but matches the rest of the library's "correctness
// first, optimise after benches" policy.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/moving/moving.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>            // createLike, createForDims (toolboxes/builtin/src/)
#include <numkit/ops/reductions.hpp>  // numkit::ops::firstNonSingletonDim, validateDim

#include <algorithm>
#include <cmath>
#include <limits>

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
    const keel::f64 inv2s2 = (sigma > 0.0) ? 1.0 / (2.0 * sigma * sigma) : 0.0;
    auto perSlice = [&](const keel::f64 *src, keel::usize n, keel::isize step, keel::f64 *dst) {
        const keel::i64 N = keel::as<keel::i64>(n);
        for (keel::i64 i = 0; i < N; ++i) {
            const keel::i64 lo = std::max(0L, i - kb), hi = std::min(N - 1, i + kf);
            keel::f64 sw = 0.0, ssum = 0.0;
            for (keel::i64 j = lo; j <= hi; ++j) {
                const keel::f64 v = src[j * step];
                if (opt.omit_nan && std::isnan(v)) continue;
                const keel::f64 dd = keel::as<keel::f64>(j - i);
                const keel::f64 wt = std::exp(-dd * dd * inv2s2);
                ssum += wt * v;
                sw += wt;
            }
            dst[i * step] = (sw > 0.0) ? ssum / sw
                                       : std::numeric_limits<keel::f64>::quiet_NaN();
        }
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
        std::copy(src, src + x.numel(), dst);
    }
    return out;
}

// Savitzky-Golay projection matrix B = A·(A'A)⁻¹·A' (F×F, row-major), where A
// is the Vandermonde of [-m..m] to degree `deg`. Row m is the steady-state
// smoothing kernel; the boundary rows handle the F-point edge windows. Matches
// MATLAB sgolay/sgolayfilt exactly.
static void buildSGMatrix(keel::i32 F, keel::i32 deg, keel::Array<keel::f64> &B)
{
    const keel::i32 p1 = deg + 1;
    const keel::i32 m  = (F - 1) / 2;
    keel::Array<keel::f64> AtA(keel::as<keel::usize>(p1) * p1, 0.0);
    for (keel::i32 a = 0; a < p1; ++a)
        for (keel::i32 b = 0; b < p1; ++b) {
            keel::f64 s = 0.0;
            for (keel::i32 i = 0; i < F; ++i) s += std::pow(keel::as<keel::f64>(i - m), a + b);
            AtA[a * p1 + b] = s;
        }
    // Invert AtA via Gauss-Jordan on [AtA | I].
    const keel::i32 W = 2 * p1;
    keel::Array<keel::f64> aug(keel::as<keel::usize>(p1) * W, 0.0);
    for (keel::i32 i = 0; i < p1; ++i) { for (keel::i32 j = 0; j < p1; ++j) aug[i * W + j] = AtA[i * p1 + j]; aug[i * W + p1 + i] = 1.0; }
    for (keel::i32 col = 0; col < p1; ++col) {
        keel::i32 piv = col; keel::f64 best = std::fabs(aug[col * W + col]);
        for (keel::i32 r = col + 1; r < p1; ++r) { keel::f64 v = std::fabs(aug[r * W + col]); if (v > best) { best = v; piv = r; } }
        if (piv != col) for (keel::i32 j = 0; j < W; ++j) std::swap(aug[col * W + j], aug[piv * W + j]);
        const keel::f64 d = aug[col * W + col];
        for (keel::i32 j = 0; j < W; ++j) aug[col * W + j] /= d;
        for (keel::i32 r = 0; r < p1; ++r) {
            if (r == col) continue;
            const keel::f64 f = aug[r * W + col];
            if (f == 0.0) continue;
            for (keel::i32 j = 0; j < W; ++j) aug[r * W + j] -= f * aug[col * W + j];
        }
    }
    B = keel::Array<keel::f64>(keel::as<keel::usize>(F) * F, 0.0);
    for (keel::i32 i = 0; i < F; ++i)
        for (keel::i32 j = 0; j < F; ++j) {
            keel::f64 s = 0.0;
            for (keel::i32 a = 0; a < p1; ++a) {
                const keel::f64 ia = std::pow(keel::as<keel::f64>(i - m), a);
                for (keel::i32 b = 0; b < p1; ++b)
                    s += ia * aug[a * W + p1 + b] * std::pow(keel::as<keel::f64>(j - m), b);
            }
            B[keel::as<keel::usize>(i) * F + j] = s;
        }
}

// Apply the SG projection over one slice (stride `step`): interior samples use
// the centre row; the first/last m samples use the boundary rows of the F-point
// edge window. Matches MATLAB sgolayfilt's edge handling.
static void sgolaySlice(const keel::f64 *src, keel::usize n, keel::isize step, keel::f64 *dst,
                        keel::ArrayRef<keel::f64> B, keel::i32 F)
{
    const keel::i32 N = keel::as<keel::i32>(n);
    const keel::i32 m = (F - 1) / 2;
    for (keel::i32 nn = 0; nn < N; ++nn) {
        keel::i32 row, base;
        if (nn < m)            { row = nn;            base = 0;     }   // leading edge
        else if (nn >= N - m)  { row = nn - (N - F);  base = N - F; }   // trailing edge
        else                   { row = m;             base = nn - m; }  // interior
        keel::f64 s = 0.0;
        for (keel::i32 j = 0; j < F; ++j) s += B[keel::as<keel::usize>(row) * F + j] * src[keel::as<keel::isize>(base + j) * step];
        dst[keel::as<keel::isize>(nn) * step] = s;
    }
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
        std::copy(src, src + x.numel(), dst);
    }
    return out;
}

// Local tricube-weighted polynomial regression over one slice for smoothdata
// 'lowess' (deg 1) / 'loess' (deg 2). For each sample the F-point window is
// shifted to stay in range; weights are tricube of the in-window distance
// (window edges → weight 0); the fitted value is the constant term (the fit at
// the query point). Matches MATLAB exactly. Rank-deficient windows fall back to
// the weighted mean. P = deg+1 ≤ 3.
static void localRegressSlice(const keel::f64 *src, keel::usize n, keel::isize step, keel::f64 *dst,
                              keel::i32 F, keel::i32 deg)
{
    const keel::i32 N = keel::as<keel::i32>(n);
    const keel::i32 m = (F - 1) / 2;
    const keel::i32 P = deg + 1;
    for (keel::i32 i = 0; i < N; ++i) {
        keel::i32 lo = i - m;
        if (lo < 0)      lo = 0;
        if (lo > N - F)  lo = N - F;
        keel::f64 dmax = 0.0;
        for (keel::i32 j = lo; j < lo + F; ++j) { const keel::f64 d = std::fabs(j - i); if (d > dmax) dmax = d; }
        if (dmax == 0.0) dmax = 1.0;

        keel::f64 AtA[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        keel::f64 Atb[3] = {0, 0, 0};
        for (keel::i32 j = lo; j < lo + F; ++j) {
            keel::f64 u = std::fabs(j - i) / dmax;
            if (u > 1.0) u = 1.0;
            const keel::f64 t1 = 1.0 - u * u * u;
            const keel::f64 w  = t1 * t1 * t1;             // tricube
            if (w == 0.0) continue;
            keel::f64 basis[3];
            basis[0] = 1.0;
            for (keel::i32 p = 1; p < P; ++p) basis[p] = basis[p - 1] * (j - i);
            const keel::f64 yj = src[keel::as<keel::isize>(j) * step];
            for (keel::i32 a = 0; a < P; ++a) {
                Atb[a] += w * basis[a] * yj;
                for (keel::i32 b = 0; b < P; ++b) AtA[a * P + b] += w * basis[a] * basis[b];
            }
        }
        // Solve the P×P system; the value is x[0]. Gaussian elimination with a
        // pivot threshold; on rank-deficiency fall back to the weighted mean.
        keel::f64 M[9], b[3], val;
        for (keel::i32 t = 0; t < P * P; ++t) M[t] = AtA[t];
        for (keel::i32 t = 0; t < P; ++t)     b[t] = Atb[t];
        bool ok = true;
        for (keel::i32 col = 0; col < P && ok; ++col) {
            keel::i32 piv = col; keel::f64 best = std::fabs(M[col * P + col]);
            for (keel::i32 r = col + 1; r < P; ++r) { const keel::f64 v = std::fabs(M[r * P + col]); if (v > best) { best = v; piv = r; } }
            if (best < 1e-12) { ok = false; break; }
            if (piv != col) { for (keel::i32 j = 0; j < P; ++j) std::swap(M[col * P + j], M[piv * P + j]); std::swap(b[col], b[piv]); }
            const keel::f64 d = M[col * P + col];
            for (keel::i32 j = 0; j < P; ++j) M[col * P + j] /= d;
            b[col] /= d;
            for (keel::i32 r = 0; r < P; ++r) { if (r == col) continue; const keel::f64 f = M[r * P + col]; for (keel::i32 j = 0; j < P; ++j) M[r * P + j] -= f * M[col * P + j]; b[r] -= f * b[col]; }
        }
        val = ok ? b[0] : (AtA[0] > 0.0 ? Atb[0] / AtA[0] : src[keel::as<keel::isize>(i) * step]);
        dst[keel::as<keel::isize>(i) * step] = val;
    }
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
        std::copy(src, src + x.numel(), dst);
    }
    return out;
}

namespace {

void smoothdata_argcheck(keel::StringRef method)
{
    keel::String m = keel::String::from(method);
    for (auto &c : m) c = keel::to_ascii_lower(c);
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
    for (auto &c : m) c = keel::to_ascii_lower(c);

    // Heuristic default window — the reference engine picks ~max(min(round(0.1*n), 10), 3).
    if (k <= 0) {
        const keel::usize n = (x.dims().isVector() || x.isScalar())
                          ? x.numel()
                          : x.dims().rows();
        keel::i64 auto_k = keel::as<keel::i64>(std::round(0.1 * keel::as<keel::f64>(n)));
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

// MAD→σ normal-consistency constant 1/norminv(0.75).
constexpr keel::f64 kHampelMadToStd = 1.4826022185056018;

// Core single-pass hampel filter over a length-`n` vector. Always fills
// `dst`; when non-null, also fills `mask` (1 where the point was replaced),
// `median` (local window median), and `sigma` (local 1.4826·MAD estimate).
void hampelCore(const keel::f64 *src, keel::usize n, keel::i32 k, keel::f64 nsigmas,
                       keel::f64 *dst, keel::u8 *mask, keel::f64 *median,
                       keel::f64 *sigma, keel::Allocator *mr)
{
    if (n == 0) return;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto buf = keel::Array<keel::f64>(2 * k + 1, scratch.allocator());
    for (keel::i64 i = 0; i < keel::as<keel::i64>(n); ++i) {
        const keel::i64 lo = std::max<keel::i64>(0, i - k);
        const keel::i64 hi = std::min<keel::i64>(keel::as<keel::i64>(n) - 1, i + k);
        const keel::i64 len = hi - lo + 1;
        for (keel::i64 j = 0; j < len; ++j)
            buf[keel::as<keel::usize>(j)] = src[lo + j];
        // Median + MAD on the same window.
        auto copy1 = keel::Array<keel::f64>(len, scratch.allocator());
        std::copy(buf.data(), buf.data() + len, copy1.data());
        const keel::f64 med = winMedianInPlace(copy1.data(), keel::as<keel::usize>(len));
        auto devs = keel::Array<keel::f64>(len, scratch.allocator());
        for (keel::i64 j = 0; j < len; ++j)
            devs[keel::as<keel::usize>(j)] = keel::abs(buf[keel::as<keel::usize>(j)] - med);
        const keel::f64 mad = winMedianInPlace(devs.data(), keel::as<keel::usize>(len));
        const keel::f64 sig = kHampelMadToStd * mad;
        const bool isOut = keel::abs(src[i] - med) > nsigmas * sig;
        dst[i] = isOut ? med : src[i];
        if (mask)   mask[keel::as<keel::usize>(i)]   = isOut ? 1 : 0;
        if (median) median[keel::as<keel::usize>(i)] = med;
        if (sigma)  sigma[keel::as<keel::usize>(i)]  = sig;
    }
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
