// toolboxes/stats/src/distributions/multivariate_detail.hpp
//
// Private (src-only) compute substrate for the multivariate distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in multivariate.cpp and
// its CallContext register half in multivariate_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/random.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>


namespace numkit::stats {

namespace {

// In-place Cholesky factorisation of an n×n symmetric positive-definite
// matrix. On return, the lower-triangular factor lives in the lower
// half of S; upper half is zeroed. Throws if S is not PD (a diagonal
// becomes ≤ 0). Algorithm: classical Cholesky-Banachiewicz.
[[maybe_unused]] void choleskyLowerInPlace(keel::f64 *S, keel::usize n)
{
    for (keel::usize j = 0; j < n; ++j) {
        // Diagonal entry: L[j][j] = sqrt(S[j][j] - sum_{k<j} L[j][k]^2)
        keel::f64 diag = S[j * n + j];
        for (keel::usize k = 0; k < j; ++k)
            diag -= S[j * n + k] * S[j * n + k];
        if (diag <= 0.0)
            throw Error(keel::format("mvnrnd: Sigma must be positive definite "
                        "(Cholesky failed at row {})", j + 1),
                         0, 0, "mvnrnd", "", "numkit:mvnrnd:notPD");
        const keel::f64 Ljj = std::sqrt(diag);
        S[j * n + j] = Ljj;
        // Below-diagonal entries: L[i][j] = (S[i][j] - sum_{k<j} L[i][k]*L[j][k]) / L[j][j]
        for (keel::usize i = j + 1; i < n; ++i) {
            keel::f64 s = S[i * n + j];
            for (keel::usize k = 0; k < j; ++k)
                s -= S[i * n + k] * S[j * n + k];
            S[i * n + j] = s / Ljj;
        }
        // Zero the strictly-upper triangle for clarity.
        for (keel::usize i = 0; i < j; ++i)
            S[i * n + j] = 0.0;
    }
}

} // namespace
namespace {

// Standard normal CDF.
keel::f64 phiCdf(keel::f64 x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// 16-point Gauss-Legendre quadrature on [0, 1] for the bivariate
// normal CDF via the parametric integral
//   Φ_2(h, k; ρ) = Φ(h) · Φ(k) + ∫_0^ρ φ_2(h, k; t) dt
// where φ_2 is the bivariate normal pdf.
[[maybe_unused]] keel::f64 bivariateCdf(keel::f64 h, keel::f64 k, keel::f64 rho)
{
    // Clamp ρ to avoid issues at the boundary.
    if (rho >  1.0 - 1e-12) rho =  1.0 - 1e-12;
    if (rho < -1.0 + 1e-12) rho = -1.0 + 1e-12;

    // Gauss-Legendre 16-point on [0, 1].
    static constexpr keel::f64 GL_X[16] = {
        0.0052995325041750, 0.0277124884633837, 0.0671843988060841,
        0.1222977958224985, 0.1910618777986781, 0.2709916111713863,
        0.3591982246103705, 0.4524937450811813, 0.5475062549188187,
        0.6408017753896295, 0.7290083888286137, 0.8089381222013219,
        0.8777022041775015, 0.9328156011939160, 0.9722875115366163,
        0.9947004674958250
    };
    static constexpr keel::f64 GL_W[16] = {
        0.0135762297058770, 0.0311267619693239, 0.0475792558412464,
        0.0623144856277781, 0.0747979944082885, 0.0845782596975013,
        0.0913017075224618, 0.0947253052275342, 0.0947253052275342,
        0.0913017075224618, 0.0845782596975013, 0.0747979944082885,
        0.0623144856277781, 0.0475792558412464, 0.0311267619693239,
        0.0135762297058770
    };

    keel::f64 integral = 0.0;
    for (keel::i32 i = 0; i < 16; ++i) {
        const keel::f64 t = rho * GL_X[i];   // map [0, 1] → [0, ρ]
        const keel::f64 r2 = 1.0 - t * t;
        // φ_2(h, k; t) at point (h, k) with corr t.
        const keel::f64 quad = (h * h - 2.0 * t * h * k + k * k) / (2.0 * r2);
        const keel::f64 pdf = std::exp(-quad) / (2.0 * keel::kPi * std::sqrt(r2));
        integral += GL_W[i] * pdf;
    }
    integral *= rho;
    return phiCdf(h) * phiCdf(k) + integral;
}

// Monte Carlo CDF (d ≥ 3): sample from N(mu, Sigma), count how many
// fall componentwise <= q. Antithetic sampling improves variance.
[[maybe_unused]] keel::f64 monteCarloCdf(const keel::f64 *L, keel::usize d,
                      const keel::f64 *mu, const keel::f64 *q,
                      keel::usize N, keel::random::Mt19937 &gen)
{
    keel::usize below = 0;
    keel::Array<keel::f64> z(d), sample(d);
    const keel::usize halfN = N / 2;
    for (keel::usize s = 0; s < halfN; ++s) {
        for (keel::usize k = 0; k < d; ++k) z[k] = keel::random::normal_f64(gen);
        // sample1 = mu + L * z, sample2 = mu - L * z  (antithetic).
        for (keel::i32 sign = 0; sign < 2; ++sign) {
            const keel::f64 sz = (sign == 0) ? 1.0 : -1.0;
            bool ok = true;
            for (keel::usize j = 0; j < d; ++j) {
                keel::f64 v = mu[j];
                for (keel::usize k = 0; k <= j; ++k)
                    v += sz * L[j * d + k] * z[k];
                if (v > q[j]) { ok = false; break; }
            }
            if (ok) ++below;
        }
    }
    const keel::usize total = halfN * 2;
    return keel::as<keel::f64>(below) / keel::as<keel::f64>(total);
}

// Acklam's full-precision inverse standard normal CDF Φ⁻¹(p).
inline keel::f64 phiInv(keel::f64 p)
{
    if (p <= 0.0) return -std::numeric_limits<keel::f64>::infinity();
    if (p >= 1.0) return  std::numeric_limits<keel::f64>::infinity();
    static constexpr keel::f64 a1 = -3.969683028665376e+01, a2 =  2.209460984245205e+02,
                            a3 = -2.759285104469687e+02, a4 =  1.383577518672690e+02,
                            a5 = -3.066479806614716e+01, a6 =  2.506628277459239e+00;
    static constexpr keel::f64 b1 = -5.447609879822406e+01, b2 =  1.615858368580409e+02,
                            b3 = -1.556989798598866e+02, b4 =  6.680131188771972e+01,
                            b5 = -1.328068155288572e+01;
    static constexpr keel::f64 c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01,
                            c3 = -2.400758277161838e+00, c4 = -2.549732539343734e+00,
                            c5 =  4.374664141464968e+00, c6 =  2.938163982698783e+00;
    static constexpr keel::f64 d1 =  7.784695709041462e-03, d2 =  3.224671290700398e-01,
                            d3 =  2.445134137142996e+00, d4 =  3.754408661907416e+00;
    static constexpr keel::f64 p_low  = 0.02425;
    static constexpr keel::f64 p_high = 1.0 - p_low;
    if (p < p_low) {
        const keel::f64 q = std::sqrt(-2.0 * std::log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    if (p <= p_high) {
        const keel::f64 q = p - 0.5;
        const keel::f64 r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    }
    const keel::f64 q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
}

// Genz (1992) Separation-of-Variables (SOV) quasi-Monte Carlo integrator
// for P(a <= X <= b) where X ~ N(0, L*L') and L is d×d row-major lower
// Cholesky. Uses antithetic Richtmyer irrational-lattice QMC shifts.
inline keel::f64 genzSovCdf(const keel::f64 *L, keel::usize d,
                         const keel::f64 *a, const keel::f64 *b,
                         keel::usize maxEvals = 10000,
                         keel::f64 tol = 1e-5)
{
    auto evalPhi = [](keel::f64 x) -> keel::f64 {
        if (std::isinf(x)) return x < 0.0 ? 0.0 : 1.0;
        return phiCdf(x);
    };
    const keel::f64 d0 = evalPhi(a[0] / L[0]);
    const keel::f64 e0 = evalPhi(b[0] / L[0]);
    const keel::f64 f0 = std::max(0.0, e0 - d0);
    if (f0 == 0.0 || d <= 1) return f0;

    // First 32 primes for Richtmyer quasi-random generators sqrt(p_k) mod 1.
    static constexpr keel::i32 kPrimes[32] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131
    };
    const keel::usize m = d - 1;
    keel::Array<keel::f64> alpha(m);
    for (keel::usize k = 0; k < m; ++k) {
        const keel::f64 pk = (k < 32) ? keel::as<keel::f64>(kPrimes[k])
                                   : keel::as<keel::f64>(137 + 2 * k);
        const keel::f64 rt = std::sqrt(pk);
        alpha[k] = rt - std::floor(rt);
    }

    keel::Array<keel::f64> y(m), w(m);
    auto evalIntegrand = [&](const keel::Array<keel::f64> &uvec) -> keel::f64 {
        keel::f64 f = f0;
        keel::f64 dp = d0, ep = e0;
        for (keel::usize i = 1; i < d; ++i) {
            keel::f64 u = dp + uvec[i - 1] * (ep - dp);
            if (u < 1e-16) u = 1e-16;
            if (u > 1.0 - 1e-16) u = 1.0 - 1e-16;
            y[i - 1] = phiInv(u);
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < i; ++k)
                s += L[i * d + k] * y[k];
            const keel::f64 di = evalPhi((a[i] - s) / L[i * d + i]);
            const keel::f64 ei = evalPhi((b[i] - s) / L[i * d + i]);
            const keel::f64 diff = std::max(0.0, ei - di);
            f *= diff;
            if (f == 0.0) return 0.0;
            dp = di;
            ep = ei;
        }
        return f;
    };

    const keel::usize nBatches = 8;
    const keel::usize pairsPerBatch = std::max<keel::usize>(64, maxEvals / (2 * nBatches));
    keel::random::Mt19937 shiftGen(5489u);
    keel::Array<keel::f64> shift(m);

    keel::f64 meanEst = 0.0;
    keel::f64 m2 = 0.0;
    keel::usize completedBatches = 0;
    for (keel::usize batch = 0; batch < nBatches; ++batch) {
        for (keel::usize k = 0; k < m; ++k) shift[k] = keel::random::uniform_f64(shiftGen);
        keel::f64 batchSum = 0.0;
        for (keel::usize p = 1; p <= pairsPerBatch; ++p) {
            const keel::f64 pd = keel::as<keel::f64>(p);
            for (keel::usize k = 0; k < m; ++k) {
                keel::f64 v = pd * alpha[k] + shift[k];
                w[k] = v - std::floor(v);
            }
            const keel::f64 f1 = evalIntegrand(w);
            for (keel::usize k = 0; k < m; ++k) w[k] = 1.0 - w[k];
            const keel::f64 f2 = evalIntegrand(w);
            batchSum += 0.5 * (f1 + f2);
        }
        const keel::f64 batchMean = batchSum / keel::as<keel::f64>(pairsPerBatch);
        ++completedBatches;
        const keel::f64 delta = batchMean - meanEst;
        meanEst += delta / keel::as<keel::f64>(completedBatches);
        m2 += delta * (batchMean - meanEst);
        if (completedBatches >= 4) {
            const keel::f64 se = std::sqrt(m2 / keel::as<keel::f64>(completedBatches * (completedBatches - 1)));
            if (se <= tol) break;
        }
    }
    return std::min(1.0, std::max(0.0, meanEst));
}

} // namespace
namespace {

// Validate C square, infer d. Return (d, n).
[[maybe_unused]] std::pair<keel::usize, keel::usize>
mvtcdf_shape_check(const Value &X, const Value &C, const char *fn)
{
    if (C.dims().rows() != C.dims().cols())
        throw Error(keel::format("{}: C must be square", fn),
                    0, 0, fn, "", keel::format("numkit:{}:notSquareC", fn));
    const keel::usize d = C.dims().rows();
    if (d == 0) return {0, 0};
    keel::usize n;
    if (X.isScalar()) {
        if (d != 1)
            throw Error(keel::format("{}: X is scalar but d > 1", fn),
                        0, 0, fn, "", keel::format("numkit:{}:shapeX", fn));
        n = 1;
    } else if (X.dims().rows() == 1 && X.dims().cols() == d) {
        n = 1;
    } else if (X.dims().cols() == d) {
        n = X.dims().rows();
    } else {
        throw Error(keel::format("{}: X must be 1×d or n×d", fn),
                    0, 0, fn, "", keel::format("numkit:{}:shapeX", fn));
    }
    return {d, n};
}

// Sample-count from tol: standard error of indicator MC ≈ 1/(2·sqrt(N))
// for p near 0.5; pick N = max(10000, 1/tol²).
[[maybe_unused]] keel::i32 mvtcdf_N_from_tol(keel::f64 tol)
{
    if (!(tol > 0.0)) tol = 0.01;
    const keel::f64 n_d = 1.0 / (tol * tol);
    keel::i32 N = (n_d > 1e7) ? 10000000 : keel::as<keel::i32>(std::ceil(n_d));
    if (N < 10000) N = 10000;
    if (N % 2) ++N;   // antithetic pairs
    return N;
}

// Core MC: hits when L[row] < y < U[row] componentwise, antithetic.
// Lvec / Uvec are length-d (-Inf / +Inf allowed). Returns probability.
template <typename FetchLU>
keel::f64 mvtcdf_mc(const keel::Array<keel::f64> &Lchol, keel::usize d,
                 keel::f64 df, keel::i32 N, keel::random::Xoshiro256PlusPlus &gen,
                 FetchLU &&fetch_LU /* (j) -> {Lj, Uj} */)
{
    keel::Array<keel::f64> z(d), y(d), Lvec(d), Uvec(d);
    for (keel::usize j = 0; j < d; ++j) {
        auto [lj, uj] = fetch_LU(j);
        Lvec[j] = lj; Uvec[j] = uj;
    }
    keel::i32 hits = 0;
    for (keel::i32 k = 0; k < N / 2; ++k) {
        for (keel::usize j = 0; j < d; ++j) z[j] = keel::random::normal_f64(gen);
        const keel::f64 scale = std::sqrt(df / keel::random::chi_squared_f64(gen, df));
        // Pass 1: z direct.
        for (keel::usize j = 0; j < d; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize k2 = 0; k2 <= j; ++k2)
                s += Lchol[j * d + k2] * z[k2];
            y[j] = s * scale;
        }
        bool inside = true;
        for (keel::usize j = 0; j < d; ++j) {
            if (y[j] < Lvec[j] || y[j] > Uvec[j]) { inside = false; break; }
        }
        if (inside) ++hits;
        // Pass 2: -z antithetic.
        for (keel::usize j = 0; j < d; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize k2 = 0; k2 <= j; ++k2)
                s -= Lchol[j * d + k2] * z[k2];
            y[j] = s * scale;
        }
        inside = true;
        for (keel::usize j = 0; j < d; ++j) {
            if (y[j] < Lvec[j] || y[j] > Uvec[j]) { inside = false; break; }
        }
        if (inside) ++hits;
    }
    return keel::as<keel::f64>(hits) / keel::as<keel::f64>(N);
}

} // anonymous

} // namespace numkit::stats
