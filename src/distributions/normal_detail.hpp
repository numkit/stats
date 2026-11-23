// toolboxes/stats/src/distributions/normal_detail.hpp
//
// Private (src-only) compute substrate for the normal distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in normal.cpp and
// its CallContext register half in normal_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/stats.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>



namespace numkit::stats {

namespace {

constexpr keel::f64 kSqrt2  = 1.41421356237309504880;
constexpr keel::f64 kSqrt2Pi = keel::kSqrt2Pi;

// Apply f(x[i]) elementwise into a fresh DOUBLE Value of same shape.
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

// MATLAB's erfinv uses the Beasley-Springer-Moro initial + 1 Newton
// step. For our purposes the existing numkit::builtin::erfinv is sufficient
// (already tested at ULP-10 in the parity bench).

[[maybe_unused]] keel::f64 erfinvScalar(keel::f64 y)
{
    if (y >= 1.0) return keel::Limits<keel::f64>::infinity();
    if (y <= -1.0) return -keel::Limits<keel::f64>::infinity();
    if (y == 0.0) return 0.0;
    const keel::f64 ay = keel::abs(y);
    const keel::f64 q = 1.0 - ay; // tail probability erfc(|x|) = q
    const keel::f64 a = 0.147;
    const keel::f64 ln1m = keel::log((1.0 + ay) * q);
    const keel::f64 term = 2.0 / (keel::kPi * a) + 0.5 * ln1m;
    keel::f64 x = keel::copysign(keel::sqrt(keel::sqrt(term * term - ln1m / a) - term), y);
    const keel::f64 invSqrtPi2 = 2.0 / keel::sqrt(keel::kPi);
    // 4 Halley iterations for full 53-bit precision across central and tail regions.
    for (keel::i32 i = 0; i < 4; ++i) {
        keel::f64 err;
        if (ay <= 0.5) {
            err = keel::erf(x) - y;
        } else {
            const keel::f64 tailErr = (q - keel::erfc(keel::abs(x)));
            err = (y > 0.0) ? tailErr : -tailErr;
        }
        const keel::f64 de = invSqrtPi2 * keel::exp(-x * x);
        if (de == 0.0) break;
        const keel::f64 u = err / de;
        x -= u / (1.0 + x * u);
    }
    return x;
}

// ── Scalar kernels (single source of truth) ──────────────────────────
// Each owns its per-element domain handling so they can be broadcast over
// vector parameters: sigma<=0 (or NaN) → NaN, matching MATLAB R2025b.

inline keel::f64 normpdfK(keel::f64 x, keel::f64 mu, keel::f64 sigma)
{
    return keel::stats::norm_pdf(x, mu, sigma);
}

inline keel::f64 normcdfK(keel::f64 x, keel::f64 mu, keel::f64 sigma)
{
    return keel::stats::norm_cdf(x, mu, sigma);
}

inline keel::f64 norminvK(keel::f64 p, keel::f64 mu, keel::f64 sigma)
{
    return keel::stats::norm_inv(p, mu, sigma);
}

} // anonymous

} // namespace numkit::stats
