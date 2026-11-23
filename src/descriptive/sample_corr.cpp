// toolboxes/stats/src/descriptive/sample_corr.cpp
//
// Econometrics-Toolbox sample-correlation functions: autocorr (sample ACF)
// and crosscorr (sample CCF). These normalise to a correlation (lag-0 == 1 for
// the ACF) using the BIASED autocovariance estimator c(k) = (1/N) Σ (y_t-ȳ)
// (y_{t+k}-ȳ) — matching MATLAB R2025b — and report ±NumSTD/√N confidence
// bounds (default NumSTD = 2). Distinct from signal's xcorr (raw lags).
//
// parcorr (PACF) is intentionally NOT here: MATLAB's default parcorr Method is
// OLS regression (not the Durbin-Levinson recursion on the ACF), which needs a
// rank-robust least-squares path — deferred to its own change. See
// bugs/stats/autocorr.md.

#include <keel/array.h>
#include <keel/tuple.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::stats {

namespace {

// Default lag count: MATLAB uses min(20, N-1).
keel::usize defaultNumLags(keel::usize N)
{
    const keel::usize cap = (N >= 1) ? N - 1 : 0;
    return keel::min<keel::usize>(20, cap);
}

// Centred copy of a (vector) Value's data: out[i] = v[i] - mean(v).
void centred(const Value &v, keel::Array<keel::f64> &out, keel::f64 &c0)
{
    const keel::usize n = v.numel();
    out.resize(n);
    keel::f64 mean = 0.0;
    for (keel::usize i = 0; i < n; ++i) mean += v.elemAsDouble(i);
    mean = (n > 0) ? mean / keel::as<keel::f64>(n) : 0.0;
    c0 = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        out[i] = v.elemAsDouble(i) - mean;
        c0 += out[i] * out[i];
    }
    c0 /= keel::as<keel::f64>(n);   // biased variance
}

Value colVec(const keel::Array<keel::f64> &v, keel::Allocator *mr)
{
    Value out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    keel::f64 *d = out.doubleDataMut();
    for (keel::usize i = 0; i < v.size(); ++i) d[i] = v[i];
    return out;
}

// Solve the n×n system A·x = b in place (row-major A, Gaussian elimination with
// partial pivoting). Returns false on a near-singular pivot; on success the
// solution overwrites b. Used for the per-lag OLS normal equations in parcorr.
bool solveDense(keel::Array<keel::f64> &A, keel::Array<keel::f64> &b, keel::usize n)
{
    for (keel::usize col = 0; col < n; ++col) {
        keel::usize piv = col;
        keel::f64 best = keel::abs(A[col * n + col]);
        for (keel::usize r = col + 1; r < n; ++r) {
            const keel::f64 v = keel::abs(A[r * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-294) return false;
        if (piv != col) {
            for (keel::usize c = 0; c < n; ++c) keel::swap(A[col * n + c], A[piv * n + c]);
            keel::swap(b[col], b[piv]);
        }
        const keel::f64 d = A[col * n + col];
        for (keel::usize r = col + 1; r < n; ++r) {
            const keel::f64 f = A[r * n + col] / d;
            if (f == 0.0) continue;
            for (keel::usize c = col; c < n; ++c) A[r * n + c] -= f * A[col * n + c];
            b[r] -= f * b[col];
        }
    }
    for (keel::usize i = n; i-- > 0;) {
        keel::f64 s = b[i];
        for (keel::usize c = i + 1; c < n; ++c) s -= A[i * n + c] * b[c];
        b[i] = s / A[i * n + i];
    }
    return true;
}

void crosscorr_argcheck(const Value &y1, const Value &y2)
{
    if (y2.numel() != y1.numel())
        throw Error("crosscorr: y1 and y2 must have the same length",
                    0, 0, "crosscorr", "", "numkit:crosscorr:lengthMismatch");
}

} // namespace

keel::Tuple<Value, Value, Value>
autocorr(const Value &y, keel::i32 numLags, keel::f64 numSTD, keel::Allocator *mr)
{
    const keel::usize N = y.numel();
    const keel::usize K = (numLags >= 0) ? keel::as<keel::usize>(numLags) : defaultNumLags(N);

    keel::Array<keel::f64> yc;
    keel::f64 c0 = 0.0;
    centred(y, yc, c0);

    keel::Array<keel::f64> acf(K + 1), lags(K + 1);
    for (keel::usize k = 0; k <= K; ++k) {
        keel::f64 ck = 0.0;
        for (keel::usize t = 0; t + k < N; ++t)
            ck += yc[t] * yc[t + k];
        ck /= keel::as<keel::f64>(N);            // biased autocovariance
        acf[k]  = (c0 != 0.0) ? ck / c0 : (k == 0 ? 1.0 : 0.0);
        lags[k] = keel::as<keel::f64>(k);
    }

    const keel::f64 b = (N > 0) ? numSTD / keel::sqrt(keel::as<keel::f64>(N)) : 0.0;
    keel::Array<keel::f64> bounds = { b, -b };
    return keel::make_tuple(colVec(acf, mr), colVec(lags, mr), colVec(bounds, mr));
}

keel::Tuple<Value, Value, Value>
crosscorr(const Value &y1, const Value &y2, keel::i32 numLags, keel::f64 numSTD,
          keel::Allocator *mr)
{
    crosscorr_argcheck(y1, y2);
    const keel::usize N = y1.numel();
    const keel::usize K = (numLags >= 0) ? keel::as<keel::usize>(numLags) : defaultNumLags(N);

    keel::Array<keel::f64> c1, c2;
    keel::f64 v1 = 0.0, v2 = 0.0;
    centred(y1, c1, v1);
    centred(y2, c2, v2);
    const keel::f64 denom = keel::sqrt(v1 * v2);

    // xcf at lag k = (1/N) Σ_t c1[t]·c2[t+k] / √(c1(0)·c2(0)), k = -K..K.
    const keel::i64 Kl = keel::as<keel::i64>(K), Nl = keel::as<keel::i64>(N);
    keel::Array<keel::f64> xcf(2 * K + 1), lags(2 * K + 1);
    for (keel::i64 k = -Kl; k <= Kl; ++k) {
        keel::f64 s = 0.0;
        const keel::i64 t0 = keel::max(0L, -k), t1 = keel::min(Nl, Nl - k);
        for (keel::i64 t = t0; t < t1; ++t)
            s += c1[keel::as<keel::usize>(t)] * c2[keel::as<keel::usize>(t + k)];
        s /= keel::as<keel::f64>(N);
        const keel::usize idx = keel::as<keel::usize>(k + Kl);
        xcf[idx]  = (denom != 0.0) ? s / denom : 0.0;
        lags[idx] = keel::as<keel::f64>(k);
    }

    const keel::f64 b = (N > 0) ? numSTD / keel::sqrt(keel::as<keel::f64>(N)) : 0.0;
    keel::Array<keel::f64> bounds = { b, -b };
    return keel::make_tuple(colVec(xcf, mr), colVec(lags, mr), colVec(bounds, mr));
}

keel::Tuple<Value, Value, Value>
parcorr(const Value &y, keel::i32 numLags, keel::f64 numSTD, keel::Allocator *mr)
{
    const keel::usize N = y.numel();
    const keel::usize K = (numLags >= 0) ? keel::as<keel::usize>(numLags) : defaultNumLags(N);

    keel::Array<keel::f64> yv(N);
    for (keel::usize i = 0; i < N; ++i) yv[i] = y.elemAsDouble(i);

    keel::Array<keel::f64> pacf(K + 1, 0.0), lags(K + 1, 0.0);
    pacf[0] = 1.0;   // lag 0
    // MATLAB's default parcorr Method is OLS: for each lag k fit the AR(k)
    // model y_t = c + Σ_{j=1..k} φ_j·y_{t-j} on observations t = k+1..N and take
    // PACF(k) = φ_k (the deepest-lag coefficient). Normal equations reproduce
    // MATLAB's QR to full precision on well-conditioned lags (N-k ≥ k+1);
    // deeper, rank-deficient lags are unstable in both engines.
    keel::Array<keel::f64> xr;   // predictor row, reused
    for (keel::usize k = 1; k <= K; ++k) {
        lags[k] = keel::as<keel::f64>(k);
        if (N <= k) continue;                 // no observations → leave 0
        const keel::usize p = k + 1;               // intercept + k lags
        keel::Array<keel::f64> A(p * p, 0.0), bb(p, 0.0);
        xr.assign(p, 0.0);
        for (keel::usize t = k; t < N; ++t) {
            xr[0] = 1.0;
            for (keel::usize j = 1; j <= k; ++j) xr[j] = yv[t - j];
            for (keel::usize r = 0; r < p; ++r) {
                bb[r] += xr[r] * yv[t];
                for (keel::usize c = 0; c < p; ++c) A[r * p + c] += xr[r] * xr[c];
            }
        }
        if (solveDense(A, bb, p)) pacf[k] = bb[p - 1];   // φ_k; else leave 0
    }

    const keel::f64 bnd = (N > 0) ? numSTD / keel::sqrt(keel::as<keel::f64>(N)) : 0.0;
    keel::Array<keel::f64> bounds = { bnd, -bnd };
    return keel::make_tuple(colVec(pacf, mr), colVec(lags, mr), colVec(bounds, mr));
}

} // namespace numkit::stats
