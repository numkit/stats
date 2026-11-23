// toolboxes/stats/src/distributions/fisher_f_detail.hpp
//
// Private (src-only) compute substrate for the fisher_f distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in fisher_f.cpp and
// its CallContext register half in fisher_f_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <tuple>
#include <utility>


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

// Scalar pdf kernel for the parameter-broadcast path (vector v1/v2). Owns its
// per-element domain (v1<=0 or v2<=0 → NaN) and the x==0 boundary regimes.
// Mirrors the public fpdf bit-identically.
inline keel::f64 fpdfK(keel::f64 x, keel::f64 v1, keel::f64 v2)
{
    if (v1 <= 0.0 || v2 <= 0.0) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 a = 0.5 * v1;
    const keel::f64 b = 0.5 * v2;
    const keel::f64 lbeta = keel::lgamma(a) + keel::lgamma(b) - keel::lgamma(a + b);
    const keel::f64 log_v1 = keel::log(v1);
    const keel::f64 log_v2 = keel::log(v2);
    if (x < 0.0) return 0.0;
    if (x == 0.0) {
        if (v1 < 2.0) return keel::Limits<keel::f64>::infinity();
        if (v1 == 2.0)
            return keel::exp(a * log_v1 + b * log_v2 - (a + b) * keel::log(v2) - lbeta);
        return 0.0;
    }
    return keel::exp(a * log_v1 + b * log_v2 + (a - 1.0) * keel::log(x)
                    - (a + b) * keel::log(v2 + v1 * x) - lbeta);
}

} // anonymous
namespace {

keel::f64 ncfpdf_one(keel::f64 x, keel::f64 nu1, keel::f64 nu2, keel::f64 delta)
{
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || delta < 0.0)
        return keel::Limits<keel::f64>::quiet_nan();
    if (x <= 0.0) return 0.0;
    if (delta == 0.0) {
        // Central F pdf in log-space.
        const keel::f64 log_norm = keel::lgamma(0.5 * (nu1 + nu2))
                              - keel::lgamma(0.5 * nu1)
                              - keel::lgamma(0.5 * nu2)
                              + 0.5 * nu1 * (keel::log(nu1) - keel::log(nu2));
        return keel::exp(log_norm + (0.5 * nu1 - 1.0) * keel::log(x)
                       - 0.5 * (nu1 + nu2) * keel::log1p(nu1 * x / nu2));
    }
    const keel::f64 L = 0.5 * delta;
    const keel::f64 log_L = keel::log(L);
    const keel::f64 log_r = keel::log(nu1 / nu2);
    const keel::f64 log_x = keel::log(x);
    const keel::f64 log_one_plus_rx = keel::log1p(nu1 * x / nu2);

    keel::f64 sum = 0.0, abs_sum = 0.0;
    constexpr keel::i32 kMax = 2000;
    for (keel::i32 k = 0; k < kMax; ++k) {
        // log term_k
        const keel::f64 a = 0.5 * nu1 + keel::f64(k);
        const keel::f64 b = 0.5 * nu2;
        const keel::f64 log_beta = keel::lgamma(a) + keel::lgamma(b) - keel::lgamma(a + b);
        const keel::f64 log_term = -L + keel::f64(k) * log_L - keel::lgamma(keel::f64(k) + 1.0)
                              + a * log_r
                              + (a - 1.0) * log_x
                              - (a + b) * log_one_plus_rx
                              - log_beta;
        const keel::f64 t = keel::exp(log_term);
        sum += t;
        abs_sum += t;
        if (k > 10 && t < 1e-16 * abs_sum) break;
    }
    return std::max(0.0, sum);
}

} // anonymous
namespace {

inline keel::f64 betainc_scalar(keel::f64 y, keel::f64 a, keel::f64 b,
                             keel::Allocator *mr)
{
    Value yv = Value::scalar(y, mr);
    Value av = Value::scalar(a, mr);
    Value bv = Value::scalar(b, mr);
    return ::numkit::builtin::betainc(yv, av, bv, mr).toScalar();
}

keel::f64 ncfcdf_one(keel::f64 x, keel::f64 nu1, keel::f64 nu2, keel::f64 delta,
                  keel::Allocator *mr)
{
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || delta < 0.0)
        return keel::Limits<keel::f64>::quiet_nan();
    if (x <= 0.0) return 0.0;
    const keel::f64 y = (nu1 * x) / (nu1 * x + nu2);
    if (delta == 0.0)
        return betainc_scalar(y, 0.5 * nu1, 0.5 * nu2, mr);

    const keel::f64 L = 0.5 * delta;
    keel::f64 Pj = keel::exp(-L);
    keel::f64 sum = 0.0;
    constexpr keel::i32 kMax = 2000;
    for (keel::i32 k = 0; k < kMax; ++k) {
        const keel::f64 I = betainc_scalar(y, 0.5 * nu1 + keel::f64(k), 0.5 * nu2, mr);
        const keel::f64 t = Pj * I;
        sum += t;
        if (k > 5 && t < 1e-16 * (sum + 1e-300)) break;
        Pj *= L / keel::f64(k + 1);
    }
    if (sum > 1.0) sum = 1.0;
    if (sum < 0.0) sum = 0.0;
    return sum;
}

[[maybe_unused]] keel::f64 ncfinv_one(keel::f64 p, keel::f64 nu1, keel::f64 nu2, keel::f64 delta,
                  keel::Allocator *mr)
{
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || delta < 0.0
        || keel::is_nan(p) || p < 0.0 || p > 1.0)
        return keel::Limits<keel::f64>::quiet_nan();
    if (p == 0.0) return 0.0;
    if (p == 1.0) return keel::Limits<keel::f64>::infinity();

    // Initial guess: central finv (or rough fallback for δ > 0).
    keel::f64 x;
    {
        Value pv = Value::scalar(p, mr);
        x = finv(pv, nu1, nu2, mr).toScalar();
        if (!(x > 0.0) || !keel::is_finite(x)) x = 1.0;
    }
    // Heuristic right-shift for δ > 0: noncentral F has larger mean.
    if (delta > 0.0) x *= (1.0 + delta / nu1);

    keel::f64 lo = 0.0;
    keel::f64 hi = std::max(x + 1.0, 50.0 * x + 50.0);
    while (ncfcdf_one(hi, nu1, nu2, delta, mr) < p) {
        hi *= 2.0;
        if (!keel::is_finite(hi)) break;
    }

    for (keel::i32 it = 0; it < 80; ++it) {
        const keel::f64 F  = ncfcdf_one(x, nu1, nu2, delta, mr);
        const keel::f64 f  = ncfpdf_one(x, nu1, nu2, delta);
        const keel::f64 err = F - p;
        if (keel::abs(err) < 1e-14) return x;
        if (err > 0.0) hi = x; else lo = x;
        keel::f64 x_new;
        if (f > 1e-300) {
            x_new = x - err / f;
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
