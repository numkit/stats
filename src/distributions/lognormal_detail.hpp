// toolboxes/stats/src/distributions/lognormal_detail.hpp
//
// Private (src-only) compute substrate for the lognormal distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in lognormal.cpp and
// its CallContext register half in lognormal_reg.cpp. Kept in an anonymous
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
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
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

constexpr keel::f64 kSqrt2 = 1.41421356237309504880;
constexpr keel::f64 kSqrt2Pi = keel::kSqrt2Pi;

// Φ(z) via complementary error fn (avoids cancellation in tail).
inline keel::f64 phi(keel::f64 z) { return 0.5 * std::erfc(-z / kSqrt2); }

// Φ⁻¹(p) — Beasley-Springer-Moro / Acklam-style approximation; refined by
// one Newton step against erfc. Same construction as norminv in normal.cpp.
inline keel::f64 phiInv(keel::f64 p)
{
    if (!(p > 0.0) || !(p < 1.0)) {
        if (p == 0.0) return -std::numeric_limits<keel::f64>::infinity();
        if (p == 1.0) return  std::numeric_limits<keel::f64>::infinity();
        return std::numeric_limits<keel::f64>::quiet_NaN();
    }
    // Acklam coefficients
    static const keel::f64 a[] = { -3.969683028665376e+01,  2.209460984245205e+02,
                                 -2.759285104469687e+02,  1.383577518672690e+02,
                                 -3.066479806614716e+01,  2.506628277459239e+00 };
    static const keel::f64 b[] = { -5.447609879822406e+01,  1.615858368580409e+02,
                                 -1.556989798598866e+02,  6.680131188771972e+01,
                                 -1.328068155288572e+01 };
    static const keel::f64 c[] = { -7.784894002430293e-03, -3.223964580411365e-01,
                                 -2.400758277161838e+00, -2.549732539343734e+00,
                                  4.374664141464968e+00,  2.938163982698783e+00 };
    static const keel::f64 d[] = {  7.784695709041462e-03,  3.224671290700398e-01,
                                  2.445134137142996e+00,  3.754408661907416e+00 };
    const keel::f64 pl = 0.02425, ph = 1.0 - pl;
    keel::f64 q, r, x;
    if (p < pl) {
        q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= ph) {
        q = p - 0.5;
        r = q*q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    // One Newton refinement: x ← x - (Φ(x) - p) / φ(x)
    const keel::f64 e = phi(x) - p;
    const keel::f64 u = e * kSqrt2Pi * std::exp(0.5 * x * x);
    return x - u;
}

// Scalar kernels for the parameter-broadcast path (vector mu/sigma). Own
// their per-element domain (sigma<=0 → NaN). Mirror the public fns
// bit-identically (cdf/inv via phi / phiInv).
inline keel::f64 lognpdfK(keel::f64 x, keel::f64 mu, keel::f64 sigma)
{
    if (sigma <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::f64 inv_sig = 1.0 / sigma;
    const keel::f64 inv_sqrt2pi = 1.0 / kSqrt2Pi;
    if (x <= 0.0) return 0.0;
    const keel::f64 z = (std::log(x) - mu) * inv_sig;
    return inv_sqrt2pi * inv_sig * std::exp(-0.5 * z * z) / x;
}

inline keel::f64 logncdfK(keel::f64 x, keel::f64 mu, keel::f64 sigma)
{
    if (sigma <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::f64 inv_sig = 1.0 / sigma;
    if (x <= 0.0) return 0.0;
    return phi((std::log(x) - mu) * inv_sig);
}

inline keel::f64 logninvK(keel::f64 p, keel::f64 mu, keel::f64 sigma)
{
    if (sigma <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
    if (std::isnan(p) || p < 0.0 || p > 1.0)
        return std::numeric_limits<keel::f64>::quiet_NaN();
    if (p == 0.0) return 0.0;
    if (p == 1.0) return std::numeric_limits<keel::f64>::infinity();
    return std::exp(mu + sigma * phiInv(p));
}

} // anonymous

} // namespace numkit::stats
