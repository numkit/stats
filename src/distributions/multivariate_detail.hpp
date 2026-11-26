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
#include <keel/stats.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>



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
        const keel::f64 Ljj = keel::sqrt(diag);
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
[[maybe_unused]] inline keel::f64 phiCdf(keel::f64 x) { return keel::stats::norm_cdf(x, 0.0, 1.0); }

[[maybe_unused]] keel::f64 bivariateCdf(keel::f64 h, keel::f64 k, keel::f64 rho)
{
    return keel::stats::bivnorm_cdf(h, k, rho);
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

// Genz (1992) Separation-of-Variables (SOV) quasi-Monte Carlo integrator
inline keel::f64 genzSovCdf(const keel::f64 *L, keel::usize d,
                         const keel::f64 *a, const keel::f64 *b,
                         keel::usize maxEvals = 10000,
                         keel::f64 tol = 1e-5)
{
    return keel::stats::genz_sov_mvncdf(keel::ArrayRef<keel::f64>(L, d * d), d,
                                        keel::ArrayRef<keel::f64>(a, d),
                                        keel::ArrayRef<keel::f64>(b, d), maxEvals, tol);
}

} // namespace
namespace {

// Validate C square, infer d. Return (d, n).
[[maybe_unused]] keel::Pair<keel::usize, keel::usize>
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
    keel::i32 N = (n_d > 1e7) ? 10000000 : keel::as<keel::i32>(keel::ceil(n_d));
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
        const keel::f64 scale = keel::sqrt(df / keel::random::chi_squared_f64(gen, df));
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
