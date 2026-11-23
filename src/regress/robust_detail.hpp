// toolboxes/.../robust_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by robust.cpp + robust_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)



namespace numkit::stats {

namespace {

// Solve (M · x = b) where M is p×p row-major, in-place Gauss-Jordan
// with partial pivoting. Returns false if singular.
bool gaussSolve(keel::f64 *M, keel::f64 *b, keel::usize p)
{
    for (keel::usize k = 0; k < p; ++k) {
        keel::usize piv = k;
        keel::f64 pmax = keel::abs(M[k * p + k]);
        for (keel::usize rr = k + 1; rr < p; ++rr) {
            const keel::f64 v = keel::abs(M[rr * p + k]);
            if (v > pmax) { pmax = v; piv = rr; }
        }
        if (pmax == 0.0) return false;
        if (piv != k) {
            for (keel::usize j = 0; j < p; ++j)
                keel::swap(M[k * p + j], M[piv * p + j]);
            keel::swap(b[k], b[piv]);
        }
        const keel::f64 pv = M[k * p + k];
        for (keel::usize rr = k + 1; rr < p; ++rr) {
            const keel::f64 f = M[rr * p + k] / pv;
            for (keel::usize j = k; j < p; ++j)
                M[rr * p + j] -= f * M[k * p + j];
            b[rr] -= f * b[k];
        }
    }
    for (keel::usize k = p; k-- > 0;) {
        keel::f64 s = b[k];
        for (keel::usize j = k + 1; j < p; ++j)
            s -= M[k * p + j] * b[j];
        b[k] = s / M[k * p + k];
    }
    return true;
}

// Weighted OLS: solve (X' · W · X) β = X' · W · y. X is n×p col-major,
// y and w are length n. Returns β as length-p vector.
[[maybe_unused]] keel::Array<keel::f64> weightedLS(const keel::f64 *X, keel::usize n, keel::usize p,
                                const keel::f64 *y, const keel::f64 *w)
{
    keel::Array<keel::f64> M(p * p, 0.0);
    keel::Array<keel::f64> b(p, 0.0);
    for (keel::usize i = 0; i < p; ++i) {
        for (keel::usize j = 0; j < p; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < n; ++k)
                s += X[i * n + k] * w[k] * X[j * n + k];
            M[i * p + j] = s;
        }
        keel::f64 s = 0.0;
        for (keel::usize k = 0; k < n; ++k)
            s += X[i * n + k] * w[k] * y[k];
        b[i] = s;
    }
    if (!gaussSolve(M.data(), b.data(), p))
        keel::fill(b.begin(), b.end(), keel::Limits<keel::f64>::quiet_nan());
    return b;
}


// MATLAB robustfit weight functions (statrobustwfun.m). Argument is the
// standardised, leverage-adjusted residual u = radj / (max(s,tiny_s)*tune).
constexpr keel::f64 kSqrtEps = 1.4901161193847656e-08;  // sqrt(eps)
#ifndef keel::kPi
#endif

[[maybe_unused]] keel::f64 robustWeight(RobustWeight wt, keel::f64 u)
{
    const keel::f64 a = keel::abs(u);
    switch (wt) {
        case RobustWeight::Andrews: {
            const keel::f64 r = keel::max(kSqrtEps, a);
            return (r < keel::kPi) ? keel::sin(r) / r : 0.0;
        }
        case RobustWeight::Bisquare: {
            if (a >= 1.0) return 0.0;
            const keel::f64 t = 1.0 - u * u;
            return t * t;
        }
        case RobustWeight::Cauchy:   return 1.0 / (1.0 + u * u);
        case RobustWeight::Fair:     return 1.0 / (1.0 + a);
        case RobustWeight::Huber:    return 1.0 / keel::max(1.0, a);
        case RobustWeight::Logistic: {
            const keel::f64 r = keel::max(kSqrtEps, a);
            return keel::tanh(r) / r;
        }
        case RobustWeight::Ols:      return 1.0;
        case RobustWeight::Talwar:   return (a < 1.0) ? 1.0 : 0.0;
        case RobustWeight::Welsch:   return keel::exp(-(u * u));
    }
    return 1.0;
}

[[maybe_unused]] keel::f64 defaultTune(RobustWeight wt)
{
    switch (wt) {
        case RobustWeight::Andrews:  return 1.339;
        case RobustWeight::Bisquare: return 4.685;
        case RobustWeight::Cauchy:   return 2.385;
        case RobustWeight::Fair:     return 1.400;
        case RobustWeight::Huber:    return 1.345;
        case RobustWeight::Logistic: return 1.205;
        case RobustWeight::Ols:      return 1.0;
        case RobustWeight::Talwar:   return 2.795;
        case RobustWeight::Welsch:   return 2.985;
    }
    return 1.0;
}

// MATLAB madsigma: sort |r| ascending, drop the smallest (p-1), take the
// median of the rest, divide by 0.6745.
[[maybe_unused]] keel::f64 madsigma(keel::ArrayRef<keel::f64> r_in, keel::usize p)
{
    const keel::usize n = r_in.size();
    keel::Array<keel::f64> r(n);
    for (keel::usize i = 0; i < n; ++i) r[i] = keel::abs(r_in[i]);
    keel::sort(r.begin(), r.end());
    const keel::usize start = (p >= 1) ? (p - 1) : 0;   // 0-based first kept
    const keel::usize m = (start < n) ? (n - start) : 0;
    if (m == 0) return 0.0;
    keel::f64 med;
    if (m & 1) med = r[start + m / 2];
    else       med = 0.5 * (r[start + m / 2 - 1] + r[start + m / 2]);
    return med / 0.6745;
}

// Sample standard deviation (N-1) used for tiny_s.
[[maybe_unused]] keel::f64 sampleStd(keel::ArrayRef<keel::f64> v)
{
    const keel::usize n = v.size();
    if (n < 2) return 0.0;
    keel::f64 s = 0.0;
    for (keel::f64 e : v) s += e;
    const keel::f64 m = s / keel::f64(n);
    keel::f64 ss = 0.0;
    for (keel::f64 e : v) ss += (e - m) * (e - m);
    return keel::sqrt(ss / keel::f64(n - 1));
}

// Hat-matrix (leverage) diagonal of X (col-major n×p) via (X'X)^-1.
// h_i = x_i' (X'X)^-1 x_i. Returns NaN-free leverages clamped at 0.9999.
[[maybe_unused]] keel::Array<keel::f64> leverage(const keel::f64 *X, keel::usize n, keel::usize p)
{
    keel::Array<keel::f64> XtX(p * p, 0.0);
    for (keel::usize j = 0; j < p; ++j)
        for (keel::usize k = 0; k < p; ++k) {
            keel::f64 s = 0.0;
            for (keel::usize i = 0; i < n; ++i)
                s += X[j * n + i] * X[k * n + i];
            XtX[j * p + k] = s;
        }
    // Gauss-Jordan inverse of the p×p matrix.
    keel::Array<keel::f64> inv(p * p, 0.0);
    for (keel::usize i = 0; i < p; ++i) inv[i * p + i] = 1.0;
    for (keel::usize col = 0; col < p; ++col) {
        keel::usize piv = col;
        keel::f64 best = keel::abs(XtX[col * p + col]);
        for (keel::usize r = col + 1; r < p; ++r) {
            const keel::f64 v = keel::abs(XtX[r * p + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col)
            for (keel::usize c = 0; c < p; ++c) {
                keel::swap(XtX[col * p + c], XtX[piv * p + c]);
                keel::swap(inv[col * p + c], inv[piv * p + c]);
            }
        const keel::f64 d = XtX[col * p + col];
        const keel::f64 invd = (d != 0.0) ? 1.0 / d : 0.0;
        for (keel::usize c = 0; c < p; ++c) {
            XtX[col * p + c] *= invd;
            inv[col * p + c] *= invd;
        }
        for (keel::usize r = 0; r < p; ++r) {
            if (r == col) continue;
            const keel::f64 f = XtX[r * p + col];
            if (f == 0.0) continue;
            for (keel::usize c = 0; c < p; ++c) {
                XtX[r * p + c] -= f * XtX[col * p + c];
                inv[r * p + c] -= f * inv[col * p + c];
            }
        }
    }
    keel::Array<keel::f64> h(n, 0.0);
    for (keel::usize i = 0; i < n; ++i) {
        keel::f64 s = 0.0;
        for (keel::usize j = 0; j < p; ++j) {
            keel::f64 t = 0.0;
            for (keel::usize k = 0; k < p; ++k)
                t += inv[j * p + k] * X[k * n + i];
            s += X[j * n + i] * t;
        }
        h[i] = keel::min(0.9999, s);
    }
    return h;
}

} // namespace
namespace {

// Compute Mahalanobis distances of rows of X (col-major n × d) w.r.t.
// mean `mu` and the inverse covariance `invC`. Returns length-n vector.
[[maybe_unused]] keel::Array<keel::f64> mahalanobisDist(const keel::f64 *X, keel::usize n,
                                     keel::usize d,
                                     const keel::f64 *mu, const keel::f64 *invC)
{
    keel::Array<keel::f64> out(n);
    keel::Array<keel::f64> diff(d);
    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < d; ++j)
            diff[j] = X[j * n + i] - mu[j];
        keel::f64 s = 0.0;
        for (keel::usize a = 0; a < d; ++a)
            for (keel::usize b = 0; b < d; ++b)
                s += diff[a] * invC[a * d + b] * diff[b];
        out[i] = s;
    }
    return out;
}

// Compute mean / cov of a column-major n × d matrix, restricted to
// rows in `keep` (sorted ascending). Returns mu (d), C (d × d row-major).
[[maybe_unused]] void meanCovSubset(const keel::f64 *X, keel::usize n_orig, keel::usize d,
                   keel::ArrayRef<keel::usize> keep,
                   keel::Array<keel::f64> &mu, keel::Array<keel::f64> &C)
{
    const keel::usize k = keep.size();
    mu = keel::Array<keel::f64>(d, 0.0);
    for (keel::usize i : keep)
        for (keel::usize j = 0; j < d; ++j)
            mu[j] += X[j * n_orig + i];
    for (keel::usize j = 0; j < d; ++j) mu[j] /= keel::as<keel::f64>(k);

    C = keel::Array<keel::f64>(d * d, 0.0);
    keel::Array<keel::f64> diff(d);
    for (keel::usize i : keep) {
        for (keel::usize j = 0; j < d; ++j) diff[j] = X[j * n_orig + i] - mu[j];
        for (keel::usize a = 0; a < d; ++a)
            for (keel::usize b = 0; b < d; ++b)
                C[a * d + b] += diff[a] * diff[b];
    }
    const keel::f64 denom = keel::as<keel::f64>(k > 1 ? k - 1 : 1);
    for (auto &c : C) c /= denom;
}

// Invert d×d row-major matrix via Gauss-Jordan. Returns inverse
// row-major in `out`. Returns false on singularity.
[[maybe_unused]] bool invertSPD(keel::ArrayRef<keel::f64> C, keel::usize d,
                keel::Array<keel::f64> &out)
{
    keel::Array<keel::f64> A(d * 2 * d, 0.0);
    for (keel::usize i = 0; i < d; ++i)
        for (keel::usize j = 0; j < d; ++j)
            A[i * 2 * d + j] = C[i * d + j];
    for (keel::usize i = 0; i < d; ++i)
        A[i * 2 * d + (d + i)] = 1.0;
    for (keel::usize k = 0; k < d; ++k) {
        keel::usize piv = k;
        keel::f64 pmax = keel::abs(A[k * 2 * d + k]);
        for (keel::usize rr = k + 1; rr < d; ++rr) {
            const keel::f64 v = keel::abs(A[rr * 2 * d + k]);
            if (v > pmax) { pmax = v; piv = rr; }
        }
        if (pmax == 0.0) return false;
        if (piv != k) {
            for (keel::usize j = 0; j < 2 * d; ++j)
                keel::swap(A[k * 2 * d + j], A[piv * 2 * d + j]);
        }
        const keel::f64 pv = A[k * 2 * d + k];
        for (keel::usize j = 0; j < 2 * d; ++j) A[k * 2 * d + j] /= pv;
        for (keel::usize rr = 0; rr < d; ++rr) {
            if (rr == k) continue;
            const keel::f64 f = A[rr * 2 * d + k];
            for (keel::usize j = 0; j < 2 * d; ++j)
                A[rr * 2 * d + j] -= f * A[k * 2 * d + j];
        }
    }
    out = keel::Array<keel::f64>(d * d, 0.0);
    for (keel::usize i = 0; i < d; ++i)
        for (keel::usize j = 0; j < d; ++j)
            out[i * d + j] = A[i * 2 * d + (d + j)];
    return true;
}

} // namespace

} // namespace numkit::stats
