// toolboxes/.../lasso_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by lasso.cpp + lasso_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/array.h>
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

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

// Soft-threshold operator: sign(z) · max(|z| - g, 0).
keel::f64 softThreshold(keel::f64 z, keel::f64 g)
{
    if (z >  g) return z - g;
    if (z < -g) return z + g;
    return 0.0;
}

// Standardise X (n×p column-major) in-place: each column has mean 0,
// std 1 (population std, divides by sqrt(n)). Returns (mu, sd) vectors.
[[maybe_unused]] void standardise(keel::f64 *X, keel::usize n, keel::usize p,
                  keel::Array<keel::f64> &mu, keel::Array<keel::f64> &sd)
{
    mu = keel::Array<keel::f64>(p, 0.0);
    sd = keel::Array<keel::f64>(p, 1.0);
    for (keel::usize j = 0; j < p; ++j) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < n; ++i) s += X[j * n + i];
        mu[j] = s / keel::as<keel::f64>(n);
        keel::f64 s2 = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 d = X[j * n + i] - mu[j];
            s2 += d * d;
        }
        sd[j] = std::sqrt(s2 / keel::as<keel::f64>(n));
        if (sd[j] < 1e-12) sd[j] = 1.0;   // constant column → keep as 0-coef
        for (keel::usize i = 0; i < n; ++i)
            X[j * n + i] = (X[j * n + i] - mu[j]) / sd[j];
    }
}

// One-pass coordinate descent on standardised X. Updates beta in place.
// Returns true if max coefficient change < tol.
[[maybe_unused]] bool coordDescentSweep(const keel::f64 *Xs, keel::usize n, keel::usize p,
                        const keel::f64 * /*y*/,
                        keel::Array<keel::f64> &r, keel::Array<keel::f64> &beta,
                        keel::ArrayRef<keel::f64> XjNormSq,
                        keel::f64 lambda_eff, keel::f64 lambda_l2,
                        keel::f64 tol)
{
    keel::f64 maxChange = 0.0;
    for (keel::usize j = 0; j < p; ++j) {
        const keel::f64 oldB = beta[j];
        // z_j = (X_j · r + ||X_j||² · β_j) / n
        keel::f64 xr = 0.0;
        for (keel::usize i = 0; i < n; ++i)
            xr += Xs[j * n + i] * r[i];
        const keel::f64 z = xr + XjNormSq[j] * oldB;
        const keel::f64 newB = softThreshold(z / keel::as<keel::f64>(n), lambda_eff)
                            / (XjNormSq[j] / keel::as<keel::f64>(n) + lambda_l2);
        beta[j] = newB;
        const keel::f64 delta = newB - oldB;
        if (delta != 0.0) {
            for (keel::usize i = 0; i < n; ++i)
                r[i] -= delta * Xs[j * n + i];
            maxChange = std::max(maxChange, std::fabs(delta));
        }
    }
    return maxChange < tol;
}

} // namespace
namespace {

// Same link / variance machinery as glm.cpp (duplicated locally to
// keep the cpp self-contained; small enough to be acceptable).
keel::f64 invIdentity_g(keel::f64 e) { return e; }
keel::f64 dEtaIdentity_g(keel::f64) { return 1.0; }
keel::f64 invLogit_g(keel::f64 e) {
    if (e > 0) return 1.0 / (1.0 + std::exp(-e));
    const keel::f64 ex = std::exp(e); return ex / (1.0 + ex);
}
keel::f64 dEtaLogit_g(keel::f64 mu) {
    const keel::f64 m = std::min(std::max(mu, 1e-12), 1.0 - 1e-12);
    return 1.0 / (m * (1.0 - m));
}
keel::f64 invLog_g(keel::f64 e) { return std::exp(e); }
keel::f64 dEtaLog_g(keel::f64 mu) { return 1.0 / std::max(mu, 1e-12); }

struct LinkPair { keel::f64 (*inv)(keel::f64); keel::f64 (*dEta)(keel::f64); };
[[maybe_unused]] LinkPair pickLink(GlmDistribution d) {
    switch (d) {
    case GlmDistribution::Normal:   return { invIdentity_g, dEtaIdentity_g };
    case GlmDistribution::Binomial: return { invLogit_g,    dEtaLogit_g };
    case GlmDistribution::Poisson:  return { invLog_g,      dEtaLog_g };
    default: return { invIdentity_g, dEtaIdentity_g };  // gamma/IG fall back
    }
}

[[maybe_unused]] keel::f64 varG(keel::f64 mu, GlmDistribution d) {
    switch (d) {
    case GlmDistribution::Normal:   return 1.0;
    case GlmDistribution::Binomial: {
        const keel::f64 m = std::min(std::max(mu, 1e-12), 1.0 - 1e-12);
        return m * (1.0 - m);
    }
    case GlmDistribution::Poisson:  return std::max(mu, 1e-12);
    default: return 1.0;
    }
}

[[maybe_unused]] keel::f64 initMu(keel::f64 yi, GlmDistribution d) {
    switch (d) {
    case GlmDistribution::Normal:   return yi;
    case GlmDistribution::Binomial: return (yi + 0.5) * 0.5;
    case GlmDistribution::Poisson:  return std::max(yi + 0.1, 0.1);
    default: return std::max(yi, 0.1);
    }
}

} // namespace

} // namespace numkit::stats
