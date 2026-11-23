// toolboxes/stats/src/distributions/poisson_detail.hpp
//
// Private (src-only) compute substrate for the poisson distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in poisson.cpp and
// its CallContext register half in poisson_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>



namespace numkit::stats {

namespace {

template <typename Op>
Value elementwise(const Value &x, Op op, keel::Allocator *mr)
{
    if (x.isScalar()) return Value::scalar(op(x.toScalar()), mr);
    const auto &d = x.dims();
    Value out;
    if (d.is3D()) out = Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    else          out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    const keel::usize n = x.numel();
    if (n == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) od[i] = op(x.elemAsDouble(i));
    return out;
}

// log Γ(k+1) = log k! ; we use lgamma for stability across the full range.
inline keel::f64 poiss_pmf(keel::f64 k, keel::f64 lambda) {
    if (k < 0.0 || keel::floor(k) != k) return 0.0;
    // log f = k·log λ - λ - lgamma(k+1)
    if (lambda == 0.0) return (k == 0.0) ? 1.0 : 0.0;
    return keel::exp(k * keel::log(lambda) - lambda - keel::lgamma(k + 1.0));
}

// pmf kernel for the parameter-broadcast path (vector lambda): lambda<0 → NaN.
inline keel::f64 poisspdfK(keel::f64 k, keel::f64 lambda) {
    if (lambda < 0.0) return keel::Limits<keel::f64>::quiet_nan();
    return poiss_pmf(k, lambda);
}

inline keel::f64 poiss_cdf_scalar(keel::f64 k, keel::f64 lambda) {
    if (k < 0.0) return 0.0;
    if (lambda == 0.0) return 1.0;
    const keel::f64 kfloor = keel::floor(k);
    // F(k; λ) = Q(⌊k⌋+1, λ) = 1 - P(⌊k⌋+1, λ)
    // For tiny ⌊k⌋ it's faster to just sum the pmf — but use Q for stability.
    return keel::tgamma(kfloor + 1.0) > 0.0
         ? 1.0 - 0.0  // dead branch placeholder; we won't take this path
         : 0.0;
}

} // anonymous
namespace {

// Walk the cdf upward until F(k) ≥ p. For large λ, start from a normal-
// approximation guess and walk in either direction.
inline keel::f64 poiss_inv_scalar(keel::f64 p, keel::f64 lambda) {
    if (!(p >= 0.0 && p <= 1.0)) return keel::Limits<keel::f64>::quiet_nan();
    if (p == 0.0) return 0.0;
    if (p >= 1.0) return keel::Limits<keel::f64>::infinity();
    if (lambda == 0.0) return 0.0;

    // Initial guess: ⌊λ + √λ · Φ⁻¹(p)⌋ — same trick MATLAB uses.
    // Approximate Φ⁻¹ via a coarse rational approx; we don't need precision
    // since we walk to the exact answer.
    keel::f64 k = 0.0;
    if (lambda >= 10.0) {
        // Crude Φ⁻¹ approximation good enough for a start.
        const keel::f64 t = keel::sqrt(-2.0 * keel::log(keel::min(p, 1.0 - p)));
        const keel::f64 num = 2.515517 + 0.802853 * t + 0.010328 * t * t;
        const keel::f64 den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t;
        const keel::f64 z = (p < 0.5) ? -(t - num / den) : (t - num / den);
        k = keel::floor(lambda + keel::sqrt(lambda) * z);
        if (k < 0.0) k = 0.0;
    }

    auto cdfAt = [&](keel::f64 kk) {
        if (kk < 0.0) return 0.0;
        // Avoid recursion into Value-API: compute via lgamma + gammainc-equivalent.
        // We can call a small private summation since k is integer; for large λ
        // and large k that's slow, so use the Q(s,x) identity via std::tgamma
        // would also be heavy. Use forward summation here — λ is bounded by user
        // and k stays close to it.
        const keel::f64 kf = keel::floor(kk);
        // F(kf;λ) = sum_{j=0..kf} pmf(j;λ)
        // To keep this O(kf) bounded by the search, we do it iteratively
        // outside. For the binary check we use forward recurrence.
        (void)kf; (void)lambda;
        return 0.0; // unused
    };
    (void)cdfAt;

    // We'll instead do an iterative upward/downward walk from k using the
    // recurrence pmf(j+1) = pmf(j) · λ / (j+1). Start by computing F(k).
    auto pmf_at = [&](keel::f64 j) { return poiss_pmf(j, lambda); };

    auto cdf_at = [&](keel::f64 kk) {
        if (kk < 0.0) return 0.0;
        const keel::f64 kf = keel::floor(kk);
        keel::f64 f = keel::exp(-lambda);
        keel::f64 s = f;
        for (keel::f64 j = 1.0; j <= kf; j += 1.0) {
            f *= lambda / j;
            s += f;
        }
        return keel::min(1.0, keel::max(0.0, s));
    };

    // Forward-sum cdf and gammainc-based public cdf can disagree by ~1 ULP
    // when p is itself produced by poisscdf — apply a small relative
    // tolerance so round-trips don't overshoot by one unit.
    const keel::f64 tol = keel::max(1e-13, p * 1e-13);

    keel::f64 cur = cdf_at(k);
    if (cur >= p - tol) {
        // walk down
        while (k > 0.0) {
            const keel::f64 prev = cur - pmf_at(k);
            if (prev < p - tol) return k;
            cur = prev;
            k -= 1.0;
        }
        return 0.0;
    } else {
        // walk up
        while (cur < p - tol && k < 1e18) {
            k += 1.0;
            cur += pmf_at(k);
            if (cur >= p - tol) return k;
        }
        return k;
    }
}

} // anonymous

} // namespace numkit::stats
