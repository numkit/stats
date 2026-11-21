// toolboxes/stats/src/test/ad_dw.cpp
//
// Two classical hypothesis tests:
//   adtest  — Anderson-Darling test for normality (parameters estimated)
//   dwtest  — Durbin-Watson test for first-order autocorrelation in
//             regression residuals

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/stats/test/hypothesis.hpp>

#include <numkit/stats/distributions/normal.hpp>
#include <numkit/stats/distributions/beta.hpp>
#include <numkit/linalg/eig.hpp>   // eig_symmetric — exact DW p-value (Imhof)

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "ad_dw_detail.hpp"

namespace numkit::stats {

namespace {

// p-value formula from D'Agostino & Stephens (1986), Goodness-of-Fit
// Techniques, table 4.7 — parameters-estimated A² statistic.
keel::f64 adPvalueEstimatedParams(keel::f64 A2star)
{
    keel::f64 p;
    if (A2star < 0.200) {
        p = 1.0 - std::exp(-13.436 + 101.14 * A2star - 223.73 * A2star * A2star);
    } else if (A2star < 0.340) {
        p = 1.0 - std::exp(-8.318 + 42.796 * A2star - 59.938 * A2star * A2star);
    } else if (A2star < 0.600) {
        p = std::exp(0.9177 - 4.279 * A2star - 1.38 * A2star * A2star);
    } else if (A2star < 13.0) {
        p = std::exp(1.2937 - 5.709 * A2star + 0.0186 * A2star * A2star);
    } else {
        p = 0.0;
    }
    return std::max(0.0, std::min(1.0, p));
}

void adtest_argcheck(keel::f64 alpha)
{
    if (alpha <= 0.0 || alpha >= 1.0)
        throw Error("adtest: alpha must be in (0, 1)",
                    0, 0, "adtest", "", "numkit:adtest:badAlpha");
}

void dwtest_argcheck(const Value &r, const Value &X)
{
    const keel::usize n = r.numel();
    if (n < 3)
        throw Error("dwtest: need at least 3 residuals",
                    0, 0, "dwtest", "", "numkit:dwtest:tooFewObs");
    const keel::usize k = X.dims().cols();
    if (X.dims().rows() != n)
        throw Error("dwtest: rows(X) must equal length(r)",
                    0, 0, "dwtest", "", "numkit:dwtest:shapeMismatch");
    if (k >= n)
        throw Error("dwtest: design matrix has no degrees of freedom",
                    0, 0, "dwtest", "", "numkit:dwtest:noDOF");
}

} // namespace

std::tuple<Value, Value, Value, Value>
adtest(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    adtest_argcheck(alpha);

    // Pull data as DOUBLE vector, drop NaNs.
    const keel::usize N = x.numel();
    keel::Array<keel::f64> v;
    v.reserve(N);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (std::isnan(xi)) continue;
        v.push_back(xi);
    }
    const keel::usize n = v.size();
    if (n < 4)
        throw Error("adtest: need at least 4 non-NaN observations",
                    0, 0, "adtest", "", "numkit:adtest:tooFewObs");

    // Sample mean and unbiased std.
    keel::f64 mean = 0.0;
    for (keel::f64 xi : v) mean += xi;
    mean /= keel::as<keel::f64>(n);
    keel::f64 s2 = 0.0;
    for (keel::f64 xi : v) {
        const keel::f64 d = xi - mean;
        s2 += d * d;
    }
    s2 /= keel::as<keel::f64>(n - 1);
    const keel::f64 sd = std::sqrt(s2);
    if (!(sd > 0.0))
        throw Error("adtest: sample has zero variance",
                    0, 0, "adtest", "", "numkit:adtest:zeroVar");

    // Standardise + sort.
    keel::Array<keel::f64> z = v.clone();
    for (keel::f64 &zi : z) zi = (zi - mean) / sd;
    std::sort(z.begin(), z.end());

    // A² = -n - (1/n) Σ (2i-1) · [ln Φ(z_i) + ln(1 - Φ(z_{n+1-i}))]
    // Use stable: ln(1 - Φ(z)) for large z via the complementary form
    // — for our z values that come from a finite sample the standard
    // form is OK.
    keel::f64 sum = 0.0;
    constexpr keel::f64 SQRT2 = 1.41421356237309504880;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 Phi_lo = 0.5 * std::erfc(-z[i] / SQRT2);
        const keel::f64 Phi_hi = 0.5 * std::erfc(-z[n - 1 - i] / SQRT2);
        const keel::f64 lnLo = std::log(std::max(Phi_lo, 1e-300));
        const keel::f64 lnHi = std::log(std::max(1.0 - Phi_hi, 1e-300));
        sum += (2.0 * keel::as<keel::f64>(i) + 1.0) * (lnLo + lnHi);
    }
    const keel::f64 A2 = -keel::as<keel::f64>(n) - sum / keel::as<keel::f64>(n);

    // Stephens 1986 small-sample adjustment for parameters estimated.
    const keel::f64 nD = keel::as<keel::f64>(n);
    const keel::f64 A2star = A2 * (1.0 + 0.75 / nD + 2.25 / (nD * nD));

    // p-value via piecewise rational fit.
    const keel::f64 p = adPvalueEstimatedParams(A2star);

    // Critical value for parameters-estimated AD at alpha = 0.05 is
    // 0.752; we expose this as the 4th output regardless of `alpha`
    // (MATLAB does the same — it's the standard reference critical
    // for normality with estimated parameters).
    const keel::f64 cv = 0.752;

    const keel::i32 h = (p < alpha) ? 1 : 0;
    return {
        Value::scalar(keel::as<keel::f64>(h), mr),
        Value::scalar(p,       mr),
        Value::scalar(A2star,  mr),
        Value::scalar(cv,      mr),
    };
}

namespace {

constexpr keel::f64 kPi = 3.141592653589793;

// Invert a small D×D row-major matrix in place via Gauss-Jordan with partial
// pivoting. Throws on (near-)singularity.
void invertSmall(keel::Array<keel::f64> &A, keel::usize D)
{
    keel::Array<keel::f64> aug(D * 2 * D, 0.0);
    for (keel::usize r = 0; r < D; ++r) {
        for (keel::usize c = 0; c < D; ++c) aug[r * 2 * D + c] = A[r * D + c];
        aug[r * 2 * D + D + r] = 1.0;
    }
    for (keel::usize kk = 0; kk < D; ++kk) {
        keel::usize piv = kk; keel::f64 pmax = std::fabs(aug[kk * 2 * D + kk]);
        for (keel::usize r = kk + 1; r < D; ++r) {
            const keel::f64 v = std::fabs(aug[r * 2 * D + kk]);
            if (v > pmax) { pmax = v; piv = r; }
        }
        if (pmax < 1e-14)
            throw Error("dwtest: design matrix X is rank-deficient (X''X singular)",
                        0, 0, "dwtest", "", "numkit:dwtest:singular");
        if (piv != kk)
            for (keel::usize c = 0; c < 2 * D; ++c)
                std::swap(aug[kk * 2 * D + c], aug[piv * 2 * D + c]);
        const keel::f64 inv = 1.0 / aug[kk * 2 * D + kk];
        for (keel::usize c = 0; c < 2 * D; ++c) aug[kk * 2 * D + c] *= inv;
        for (keel::usize r = 0; r < D; ++r) {
            if (r == kk) continue;
            const keel::f64 f = aug[r * 2 * D + kk];
            if (f == 0.0) continue;
            for (keel::usize c = 0; c < 2 * D; ++c)
                aug[r * 2 * D + c] -= f * aug[kk * 2 * D + c];
        }
    }
    for (keel::usize r = 0; r < D; ++r)
        for (keel::usize c = 0; c < D; ++c)
            A[r * D + c] = aug[r * 2 * D + D + c];
}

// Imhof (1961) integrand g(u) = sin(θ(u)) / (u·ρ(u)) for Q = Σ h_j Z_j²
// (Z_j iid N(0,1), each central χ²₁):
//   θ(u) = ½ Σ atan(h_j u),  ρ(u) = Π (1 + h_j² u²)^{¼}.
// The u→0 limit is ½ Σ h_j.
keel::f64 imhofIntegrand(keel::f64 u, keel::ArrayRef<keel::f64> h)
{
    if (u <= 0.0) {
        keel::f64 s = 0.0; for (keel::f64 hj : h) s += hj; return 0.5 * s;
    }
    keel::f64 theta = 0.0, logrho = 0.0;
    for (keel::f64 hj : h) {
        const keel::f64 hu = hj * u;
        theta  += std::atan(hu);
        logrho += 0.25 * std::log1p(hu * hu);
    }
    theta *= 0.5;
    return std::sin(theta) / (u * std::exp(logrho));
}

// Exact P(Q < 0) for Q = Σ h_j Z_j² via Imhof's CF inversion:
//   P(Q < 0) = ½ − (1/π) ∫_0^∞ g(u) du.
// The infinite tail is mapped to [0,1) with u = t/(1−t) (du = dt/(1−t)²) and
// integrated with composite Simpson. Used as pLeft = P(DW < dw).
keel::f64 imhofPLeft(keel::ArrayRef<keel::f64> h)
{
    const keel::usize m = h.size();
    if (m == 0) return 0.5;
    if (m == 1) return (h[0] < 0.0) ? 1.0 : (h[0] > 0.0 ? 0.0 : 0.5);

    const keel::i32 N = 6000;                 // even panel count
    auto Ft = [&](keel::f64 t) -> keel::f64 {
        if (t >= 1.0) return 0.0;
        const keel::f64 u = t / (1.0 - t);
        const keel::f64 w = 1.0 / ((1.0 - t) * (1.0 - t));
        return imhofIntegrand(u, h) * w;
    };
    keel::f64 s = Ft(0.0) + Ft(1.0);
    const keel::f64 step = 1.0 / N;
    for (keel::i32 i = 1; i < N; ++i) {
        const keel::f64 t = i * step;
        s += ((i & 1) ? 4.0 : 2.0) * Ft(t);
    }
    const keel::f64 integral = s * step / 3.0;
    keel::f64 p = 0.5 - integral / kPi;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

// Exact left-tail Durbin–Watson p-value pLeft = P(DW < dw) under H0.
// DW = e'Ae / e'e; under H0 e = Mε (M = I − X(X'X)⁻¹X', ε ~ N(0,σ²I)), so
// DW < dw ⟺ ε'M(A−dw·I)Mε < 0. The nonzero eigenvalues λ_j of MAM give
// Q = Σ (λ_j − dw) Z_j²; Imhof inverts its CF. A is the first-difference
// tridiagonal (diag 1,2,…,2,1; off-diag −1).
keel::f64 dwExactPLeft(const Value &X, keel::usize n, keel::usize k, keel::f64 dw,
                    keel::Allocator *mr)
{
    // X'X (k×k) and its inverse.
    keel::Array<keel::f64> XtX(k * k, 0.0);
    for (keel::usize a = 0; a < k; ++a)
        for (keel::usize b = 0; b < k; ++b) {
            keel::f64 sab = 0.0;
            for (keel::usize i = 0; i < n; ++i)
                sab += X.elemAsDouble(a * n + i) * X.elemAsDouble(b * n + i);
            XtX[a * k + b] = sab;
        }
    invertSmall(XtX, k);

    // P = X · (X'X)⁻¹  (n×k, row-major): P[i,a] = Σ_b X[i,b] inv[b,a].
    keel::Array<keel::f64> P(n * k, 0.0);
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize a = 0; a < k; ++a) {
            keel::f64 s = 0.0;
            for (keel::usize b = 0; b < k; ++b)
                s += X.elemAsDouble(b * n + i) * XtX[b * k + a];
            P[i * k + a] = s;
        }
    // M = I − P·X'  (n×n symmetric, row-major): M[i,j] = δij − Σ_a P[i,a] X[j,a].
    keel::Array<keel::f64> M(n * n, 0.0);
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < n; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize a = 0; a < k; ++a)
                s += P[i * k + a] * X.elemAsDouble(a * n + j);
            M[i * n + j] = (i == j ? 1.0 : 0.0) - s;
        }
    // AM = A·M with A the DW tridiagonal applied row-wise (O(n²)).
    keel::Array<keel::f64> AM(n * n, 0.0);
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < n; ++j) {
            keel::f64 v = (i == 0 || i + 1 == n ? 1.0 : 2.0) * M[i * n + j];
            if (i > 0)     v -= M[(i - 1) * n + j];
            if (i + 1 < n) v -= M[(i + 1) * n + j];
            AM[i * n + j] = v;
        }
    // S = M·AM (n×n symmetric) as a Value for eig_symmetric (column-major).
    Value S = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::f64 *sd = S.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < n; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize l = 0; l < n; ++l) s += M[i * n + l] * AM[l * n + j];
            sd[j * n + i] = s;   // column-major store
        }
    auto [V, D] = ::numkit::linalg::eig_symmetric(S, mr);
    (void)V;
    keel::Array<keel::f64> eig(n);
    { const keel::f64 *dd = D.doubleData();
      for (keel::usize i = 0; i < n; ++i) eig[i] = dd[i + i * n]; }
    std::sort(eig.begin(), eig.end());
    // Drop the k structural zeros (col(X) directions); keep the n−k residual-
    // space eigenvalues, then h_j = λ_j − dw.
    keel::Array<keel::f64> h;
    h.reserve(n - k);
    for (keel::usize i = k; i < n; ++i) h.push_back(eig[i] - dw);
    return imhofPLeft(h.as_ref());
}

// Approximate left-tail p (Durbin & Watson 1971 beta-on-[0,4] moment fit) —
// the 'approximate' method and large-n fallback.
keel::f64 dwApproxPLeft(keel::f64 dw, keel::usize n, keel::Allocator *mr)
{
    const keel::f64 a = 0.5 * (keel::as<keel::f64>(n) - 1.0);
    Value pv = betacdf(Value::scalar(dw / 4.0, mr), a, a, mr);
    return pv.toScalar();
}

} // namespace

// Shared front-end: validates inputs, computes the DW statistic, and the
// left-tail p (P(DW < dw)) by the requested method. `exact==true` uses Imhof's
// exact distribution (MATLAB default); otherwise the beta approximation.
keel::f64 dwStatAndPLeft(const Value &r, const Value &X, bool exact,
                             keel::f64 &dwOut, keel::Allocator *mr)
{
    dwtest_argcheck(r, X);
    const keel::usize n = r.numel();
    const keel::usize k = X.dims().cols();

    keel::Array<keel::f64> rv(n);
    for (keel::usize i = 0; i < n; ++i) rv[i] = r.elemAsDouble(i);
    keel::f64 num = 0.0, den = 0.0;
    for (keel::usize i = 0; i < n; ++i) den += rv[i] * rv[i];
    for (keel::usize i = 1; i < n; ++i) { const keel::f64 d = rv[i] - rv[i - 1]; num += d * d; }
    if (!(den > 0.0))
        throw Error("dwtest: residuals are all zero",
                    0, 0, "dwtest", "", "numkit:dwtest:zeroResid");
    dwOut = num / den;
    return exact ? dwExactPLeft(X, n, k, dwOut, mr)
                 : dwApproxPLeft(dwOut, n, mr);
}

std::tuple<Value, Value>
dwtest(const Value &r, const Value &X, keel::Allocator *mr)
{
    // Default: exact (MATLAB) method, two-sided ('both') tail.
    keel::f64 dw = 0.0;
    const keel::f64 pLeft = dwStatAndPLeft(r, X, /*exact=*/true, dw, mr);
    const keel::f64 pBoth = std::min(1.0, 2.0 * std::min(pLeft, 1.0 - pLeft));
    return { Value::scalar(pBoth, mr), Value::scalar(dw, mr) };
}

} // namespace numkit::stats
