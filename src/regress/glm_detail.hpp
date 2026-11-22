// toolboxes/.../glm_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by glm.cpp + glm_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>


namespace numkit::stats {

namespace {

// In-place Gauss elimination on row-major p×p `M`, p-length `b`.
[[maybe_unused]] bool gaussSolve(keel::f64 *M, keel::f64 *b, keel::usize p)
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
                std::swap(M[k * p + j], M[piv * p + j]);
            std::swap(b[k], b[piv]);
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

// Link function evaluator. Computes (eta, dmu/deta, V(mu)) given mu.
//
// Returned: linkFwd(mu) -> eta, invLink(eta) -> mu, deta_dmu(mu),
// variance(mu).
struct LinkFn {
    keel::f64 (*invLink)(keel::f64);          // mu = g^{-1}(eta)
    keel::f64 (*deta_dmu)(keel::f64);         // g'(mu)
};

keel::f64 sigmoid(keel::f64 x) {
    if (x > 0)  return 1.0 / (1.0 + keel::exp(-x));
    const keel::f64 e = keel::exp(x);
    return e / (1.0 + e);
}

keel::f64 invIdentity(keel::f64 e) { return e; }
keel::f64 dEtaIdentity(keel::f64)  { return 1.0; }

keel::f64 invLogit(keel::f64 e)   { return sigmoid(e); }
keel::f64 dEtaLogit(keel::f64 mu) {
    const keel::f64 m = std::min(std::max(mu, 1e-12), 1.0 - 1e-12);
    return 1.0 / (m * (1.0 - m));
}

keel::f64 invLog(keel::f64 e)   { return keel::exp(e); }
keel::f64 dEtaLog(keel::f64 mu) {
    const keel::f64 m = std::max(mu, 1e-12);
    return 1.0 / m;
}

keel::f64 invRecip(keel::f64 e) {
    const keel::f64 r = (keel::abs(e) > 1e-300) ? e : ((e < 0) ? -1e-300 : 1e-300);
    return 1.0 / r;
}
keel::f64 dEtaRecip(keel::f64 mu) {
    const keel::f64 m = (keel::abs(mu) > 1e-300) ? mu : 1e-300;
    return -1.0 / (m * m);
}

// Probit: g(mu) = norminv(mu), inverse = normcdf(eta).
// dEta/dMu = 1 / phi(norminv(mu)) where phi is the standard normal pdf.
keel::f64 invProbit(keel::f64 e) {
    // normcdf(e) = 0.5 * erfc(-e / sqrt(2))
    return 0.5 * keel::erfc(-e / keel::sqrt(2.0));
}
keel::f64 dEtaProbit(keel::f64 mu) {
    const keel::f64 m = std::min(std::max(mu, 1e-12), 1.0 - 1e-12);
    // z = norminv(m) — use std::erfinv? not standard. Fall back to a
    // numerical approximation via Beasley-Springer-Moro? For v1 we
    // rely on the inverse-erf approach.
    // norminv(m) = sqrt(2) * erfinv(2m - 1). MSVC doesn't ship erfinv,
    // so use a rational approximation (Beasley-Springer / Wichura):
    const keel::f64 pp = 2.0 * m - 1.0;
    // Cheap fallback: iterate one Newton step from a logit guess.
    keel::f64 z = keel::log(m / (1.0 - m));
    for (keel::i32 i = 0; i < 5; ++i) {
        const keel::f64 F = 0.5 * keel::erfc(-z / keel::sqrt(2.0));
        const keel::f64 f = (1.0 / keel::sqrt(2.0 * keel::kPi)) * keel::exp(-0.5 * z * z);
        if (f < 1e-300) break;
        z -= (F - m) / f;
    }
    (void)pp;
    const keel::f64 phi = (1.0 / keel::sqrt(2.0 * keel::kPi)) * keel::exp(-0.5 * z * z);
    return (phi > 1e-300) ? 1.0 / phi : 1e300;
}

[[maybe_unused]] LinkFn makeLink(GlmLink l) {
    switch (l) {
    case GlmLink::Identity:   return { invIdentity, dEtaIdentity };
    case GlmLink::Logit:      return { invLogit,    dEtaLogit };
    case GlmLink::Log:        return { invLog,      dEtaLog };
    case GlmLink::Reciprocal: return { invRecip,    dEtaRecip };
    case GlmLink::Probit:     return { invProbit,   dEtaProbit };
    }
    return { invIdentity, dEtaIdentity };
}

// Variance function V(mu) for each family.
[[maybe_unused]] keel::f64 varianceFn(keel::f64 mu, GlmDistribution d) {
    switch (d) {
    case GlmDistribution::Normal:           return 1.0;
    case GlmDistribution::Binomial: {
        const keel::f64 m = std::min(std::max(mu, 1e-12), 1.0 - 1e-12);
        return m * (1.0 - m);
    }
    case GlmDistribution::Poisson:          return std::max(mu, 1e-12);
    case GlmDistribution::Gamma: {
        const keel::f64 m = std::max(mu, 1e-12);
        return m * m;
    }
    case GlmDistribution::InverseGaussian: {
        const keel::f64 m = std::max(mu, 1e-12);
        return m * m * m;
    }
    }
    return 1.0;
}

// Initial mu estimate for IRLS.
[[maybe_unused]] keel::f64 initialMu(keel::f64 yi, GlmDistribution d) {
    switch (d) {
    case GlmDistribution::Normal:          return yi;
    case GlmDistribution::Binomial:        return (yi + 0.5) / 2.0;  // shrink toward 0.5
    case GlmDistribution::Poisson:         return std::max(yi + 0.1, 0.1);
    case GlmDistribution::Gamma:           return std::max(yi, 0.1);
    case GlmDistribution::InverseGaussian: return std::max(yi, 0.1);
    }
    return yi;
}

// Deviance term per observation (a multiplied by 2 in the total).
[[maybe_unused]] keel::f64 devTerm(keel::f64 y, keel::f64 mu, GlmDistribution d) {
    switch (d) {
    case GlmDistribution::Normal: {
        const keel::f64 r = y - mu;
        return r * r;
    }
    case GlmDistribution::Binomial: {
        const keel::f64 m = std::min(std::max(mu, 1e-12), 1.0 - 1e-12);
        const keel::f64 yy = std::min(std::max(y,  0.0), 1.0);
        keel::f64 t = 0.0;
        if (yy > 0.0)     t += yy * keel::log(yy / m);
        if (yy < 1.0)     t += (1.0 - yy) * keel::log((1.0 - yy) / (1.0 - m));
        return 2.0 * t;
    }
    case GlmDistribution::Poisson: {
        const keel::f64 m = std::max(mu, 1e-12);
        const keel::f64 yy = std::max(y, 0.0);
        const keel::f64 termY = (yy > 0.0) ? yy * keel::log(yy / m) : 0.0;
        return 2.0 * (termY - (yy - m));
    }
    case GlmDistribution::Gamma: {
        const keel::f64 m = std::max(mu, 1e-12);
        const keel::f64 yy = std::max(y, 1e-12);
        return 2.0 * (-keel::log(yy / m) + (yy - m) / m);
    }
    case GlmDistribution::InverseGaussian: {
        const keel::f64 m = std::max(mu, 1e-12);
        const keel::f64 yy = std::max(y, 1e-12);
        const keel::f64 r = (yy - m);
        return r * r / (yy * m * m);
    }
    }
    return 0.0;
}

// Pick the canonical link if Identity was passed (which we use as
// "default" sentinel).
[[maybe_unused]] GlmLink canonicalLink(GlmDistribution d) {
    switch (d) {
    case GlmDistribution::Normal:           return GlmLink::Identity;
    case GlmDistribution::Binomial:         return GlmLink::Logit;
    case GlmDistribution::Poisson:          return GlmLink::Log;
    case GlmDistribution::Gamma:            return GlmLink::Reciprocal;
    case GlmDistribution::InverseGaussian:  return GlmLink::Reciprocal;
    }
    return GlmLink::Identity;
}

} // namespace

} // namespace numkit::stats
