// toolboxes/stats/src/distributions/students_t_detail.hpp
//
// Private (src-only) compute substrate for the students_t distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in students_t.cpp and
// its CallContext register half in students_t_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>


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

// Read a Value's first element as scalar (for inputs we know are
// scalar Values produced by the helpers). Avoids extra copies.
[[maybe_unused]] keel::f64 scalarOf(const Value &v) { return v.toScalar(); }

// ── Central-t scalar helpers for the parameter-broadcast path (vector nu) ──
// tpdf is closed-form; tcdf/tinv compose betainc/betaincinv so they use a
// per-element fixup loop in the adapter, but the nu→∞ Gaussian limit (which
// the public scalar fns special-case) is handled per element via these.
inline keel::f64 tpdfK(keel::f64 x, keel::f64 nu)
{
    if (!(nu > 0.0)) return keel::Limits<keel::f64>::quiet_nan();   // nu<=0 / NaN
    if (keel::is_inf(nu)) {
        const keel::f64 inv = 1.0 / keel::sqrt(2.0 * keel::kPi);
        return inv * keel::exp(-0.5 * x * x);
    }
    const keel::f64 log_norm = keel::lgamma(0.5 * (nu + 1.0))
                          - keel::lgamma(0.5 * nu)
                          - 0.5 * keel::log(nu * keel::kPi);
    return keel::exp(log_norm - 0.5 * (nu + 1.0) * keel::log1p(x * x / nu));
}

inline keel::f64 tNormCdf(keel::f64 x) { return 0.5 * keel::erfc(-x / keel::sqrt(2.0)); }

// norminv via Winitzki erfinv + 2 Newton steps (same construction as the
// public tinv's nu==Inf branch).
inline keel::f64 tNormInv(keel::f64 p)
{
    const keel::f64 y = 2.0 * p - 1.0;
    if (y >= 1.0) return keel::Limits<keel::f64>::infinity();
    if (y <= -1.0) return -keel::Limits<keel::f64>::infinity();
    const keel::f64 a = 0.147;
    const keel::f64 ln1m = keel::log(1.0 - y * y);
    const keel::f64 term = 2.0 / (keel::kPi * a) + 0.5 * ln1m;
    keel::f64 xv = keel::copysign(keel::sqrt(keel::sqrt(term * term - ln1m / a) - term), y);
    for (keel::i32 it = 0; it < 2; ++it) {
        const keel::f64 e  = keel::erf(xv) - y;
        const keel::f64 de = 2.0 * keel::exp(-xv * xv) / keel::sqrt(keel::kPi);
        xv -= e / de;
    }
    return keel::sqrt(2.0) * xv;
}

} // anonymous
namespace {

// Standard normal CDF.
inline keel::f64 phiCdf(keel::f64 z)
{
    return 0.5 * (1.0 + keel::erf(z / keel::sqrt(2.0)));
}

inline keel::f64 betainc_scalar(keel::f64 y, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    Value yv = Value::scalar(y, mr);
    Value av = Value::scalar(a, mr);
    Value bv = Value::scalar(b, mr);
    return ::numkit::builtin::betainc(yv, av, bv, mr).toScalar();
}

keel::f64 nctpdf_one(keel::f64 x, keel::f64 nu, keel::f64 delta,
                  keel::Allocator * /*mr*/)
{
    if (!(nu > 0.0))
        return keel::Limits<keel::f64>::quiet_nan();
    // δ = 0 → central tpdf.
    if (delta == 0.0) {
        const keel::f64 log_norm = keel::lgamma(0.5 * (nu + 1.0))
                              - keel::lgamma(0.5 * nu)
                              - 0.5 * keel::log(nu * keel::kPi);
        return keel::exp(log_norm - 0.5 * (nu + 1.0) * keel::log1p(x * x / nu));
    }
    if (x == 0.0) {
        // f(0; ν, δ) = Γ((ν+1)/2) / [√(νπ) · Γ(ν/2)] · exp(-δ²/2)
        const keel::f64 log_v = keel::lgamma(0.5 * (nu + 1.0))
                           - 0.5 * keel::log(nu * keel::kPi)
                           - keel::lgamma(0.5 * nu)
                           - 0.5 * delta * delta;
        return keel::exp(log_v);
    }
    const keel::f64 npx2 = nu + x * x;
    // log-prefactor: (ν/2)·log(ν) - δ²/2 - 0.5·log(π) - lgamma(ν/2)
    //              - (ν+1)/2 · log(ν + x²)
    const keel::f64 log_pref = 0.5 * nu * keel::log(nu)
                          - 0.5 * delta * delta
                          - 0.5 * keel::log(keel::kPi)
                          - keel::lgamma(0.5 * nu)
                          - 0.5 * (nu + 1.0) * keel::log(npx2);
    const keel::f64 xd = x * delta;
    const keel::i32 sign_factor = (xd >= 0.0) ? 1 : -1;
    const keel::f64 log_t = keel::log(keel::abs(xd) * keel::sqrt(2.0));   // log(|xδ|√2)
    const keel::f64 half_log_npx2 = 0.5 * keel::log(npx2);

    keel::f64 sum = 0.0, abs_sum = 0.0;
    keel::i32 sign_k = 1;
    constexpr keel::i32 kMax = 2000;
    for (keel::i32 k = 0; k < kMax; ++k) {
        const keel::f64 log_termk = keel::lgamma(0.5 * (nu + keel::f64(k) + 1.0))
                              + keel::f64(k) * log_t
                              - keel::f64(k) * half_log_npx2
                              - keel::lgamma(keel::f64(k) + 1.0);
        const keel::f64 abs_term = keel::exp(log_termk);
        const keel::f64 contrib = sign_k * abs_term;
        sum += contrib;
        abs_sum += abs_term;
        if (k > 10 && abs_term < 1e-16 * abs_sum) break;
        sign_k *= sign_factor;
    }
    keel::f64 v = keel::exp(log_pref) * sum;
    if (!keel::is_finite(v) || v < 0.0) v = std::max(v, 0.0);
    return v;
}

keel::f64 nctcdf_one(keel::f64 x, keel::f64 nu, keel::f64 delta,
                  keel::Allocator *mr)
{
    if (!(nu > 0.0))
        return keel::Limits<keel::f64>::quiet_nan();
    if (delta == 0.0) {
        // Central tcdf branch.
        const keel::f64 z = nu / (nu + x * x);
        const keel::f64 I = betainc_scalar(z, 0.5 * nu, 0.5, mr);
        return (x >= 0.0) ? 1.0 - 0.5 * I : 0.5 * I;
    }
    // Symmetry for negative x.
    if (x < 0.0) return 1.0 - nctcdf_one(-x, nu, -delta, mr);
    // At x = 0: F = Φ(-δ).
    if (x == 0.0) return phiCdf(-delta);

    const keel::f64 y = (x * x) / (x * x + nu);
    const keel::f64 z = 0.5 * delta * delta;
    const keel::f64 phi_neg_d = phiCdf(-delta);

    // Series 1 (P_k coefficient): Poisson pmf with mean z.
    // Series 2 (Q_k coefficient): (δ/(2√(2π))) · (z^k / Γ(k+3/2)) · e^{-z}.
    const keel::f64 e_neg_z = keel::exp(-z);
    keel::f64 alpha = e_neg_z;                                    // P_0
    keel::f64 beta  = (2.0 / keel::sqrt(keel::kPi)) * e_neg_z;          // (z^0 / Γ(3/2)) · e^{-z}

    keel::f64 sum1 = 0.0, sum2 = 0.0;
    constexpr keel::i32 kMax = 2000;
    for (keel::i32 k = 0; k < kMax; ++k) {
        const keel::f64 I1 = betainc_scalar(y, keel::f64(k) + 0.5, 0.5 * nu, mr);
        const keel::f64 I2 = betainc_scalar(y, keel::f64(k) + 1.0, 0.5 * nu, mr);
        const keel::f64 t1 = alpha * I1;
        const keel::f64 t2 = beta  * I2;
        sum1 += t1;
        sum2 += t2;
        if (k > 10 && t1 < 1e-16 * (sum1 + 1e-300)
                  && t2 < 1e-16 * (sum2 + 1e-300)) break;
        // Recurrences.
        alpha *= z / keel::f64(k + 1);
        beta  *= z / (keel::f64(k) + 1.5);
    }

    keel::f64 F = phi_neg_d + 0.5 * sum1 + (delta / (2.0 * keel::sqrt(2.0))) * sum2;
    if (F < 0.0) F = 0.0;
    if (F > 1.0) F = 1.0;
    return F;
}

} // anonymous
namespace {

// Central tinv (scalar): inverse of central t-cdf via betaincinv.
keel::f64 tinv_scalar(keel::f64 p, keel::f64 nu, keel::Allocator *mr)
{
    if (p <= 0.0) return -keel::Limits<keel::f64>::infinity();
    if (p >= 1.0) return  keel::Limits<keel::f64>::infinity();
    if (p == 0.5) return 0.0;
    // betaincinv(2·min(p,1-p), ν/2, ½) → y; then x = sign·sqrt(ν(1/y - 1)).
    const bool lower_half = (p < 0.5);
    const keel::f64 tail = lower_half ? 2.0 * p : 2.0 * (1.0 - p);
    Value tv = Value::scalar(tail, mr);
    Value av = Value::scalar(0.5 * nu, mr);
    Value bv = Value::scalar(0.5, mr);
    const keel::f64 y = ::numkit::builtin::betaincinv(tv, av, bv, mr).toScalar();
    if (y <= 0.0) return lower_half ? -keel::Limits<keel::f64>::infinity()
                                     :  keel::Limits<keel::f64>::infinity();
    if (y >= 1.0) return 0.0;
    const keel::f64 x = keel::sqrt(nu * (1.0 / y - 1.0));
    return lower_half ? -x : x;
}

[[maybe_unused]] keel::f64 nctinv_one(keel::f64 p, keel::f64 nu, keel::f64 delta,
                  keel::Allocator *mr)
{
    if (!(nu > 0.0)) return keel::Limits<keel::f64>::quiet_nan();
    if (keel::is_nan(p) || p < 0.0 || p > 1.0)
        return keel::Limits<keel::f64>::quiet_nan();
    if (p == 0.0) return -keel::Limits<keel::f64>::infinity();
    if (p == 1.0) return  keel::Limits<keel::f64>::infinity();
    if (delta == 0.0) return tinv_scalar(p, nu, mr);

    // Initial guess: central tinv shifted by δ.
    keel::f64 x = tinv_scalar(p, nu, mr) + delta;
    if (!keel::is_finite(x)) x = delta;

    // Establish a bracket [lo, hi] for fallback bisection.
    // Find any x_lo with cdf < p and any x_hi with cdf > p.
    keel::f64 lo = std::min(x - 1.0, delta - 50.0);
    keel::f64 hi = std::max(x + 1.0, delta + 50.0);
    while (nctcdf_one(lo, nu, delta, mr) > p) {
        lo -= std::max(1.0, keel::abs(lo));
        if (!keel::is_finite(lo)) break;
    }
    while (nctcdf_one(hi, nu, delta, mr) < p) {
        hi += std::max(1.0, keel::abs(hi));
        if (!keel::is_finite(hi)) break;
    }

    // Newton + bisection guard, ≤ 60 iterations.
    for (keel::i32 it = 0; it < 60; ++it) {
        const keel::f64 F  = nctcdf_one(x, nu, delta, mr);
        const keel::f64 f  = nctpdf_one(x, nu, delta, mr);
        const keel::f64 err = F - p;
        if (keel::abs(err) < 1e-14) return x;
        // Bisect endpoint update.
        if (err > 0.0) hi = x; else lo = x;
        keel::f64 x_new;
        if (f > 1e-300) {
            x_new = x - err / f;
            // Reject Newton step outside bracket — bisect instead.
            if (!keel::is_finite(x_new) || x_new <= lo || x_new >= hi)
                x_new = 0.5 * (lo + hi);
        } else {
            x_new = 0.5 * (lo + hi);
        }
        if (keel::abs(x_new - x) < 1e-14 * std::max(1.0, keel::abs(x_new)))
            return x_new;
        x = x_new;
    }
    return x;
}

} // anonymous

} // namespace numkit::stats
