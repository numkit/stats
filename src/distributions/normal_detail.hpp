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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::stats {

namespace {

constexpr double kSqrt2  = 1.41421356237309504880;
constexpr double kSqrt2Pi = 2.50662827463100050241;

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
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) od[i] = op(x.elemAsDouble(i));
    return out;
}

// MATLAB's erfinv uses the Beasley-Springer-Moro initial + 1 Newton
// step. For our purposes the existing numkit::builtin::erfinv is sufficient
// (already tested at ULP-10 in the parity bench).

[[maybe_unused]] double erfinvScalar(double y)
{
    if (y >= 1.0) return std::numeric_limits<double>::infinity();
    if (y <= -1.0) return -std::numeric_limits<double>::infinity();
    if (y == 0.0) return 0.0;
    const double ay = keel::abs(y);
    const double q = 1.0 - ay; // tail probability erfc(|x|) = q
    const double a = 0.147;
    const double ln1m = std::log((1.0 + ay) * q);
    const double term = 2.0 / (M_PI * a) + 0.5 * ln1m;
    double x = std::copysign(std::sqrt(std::sqrt(term * term - ln1m / a) - term), y);
    const double invSqrtPi2 = 2.0 / std::sqrt(M_PI);
    // 4 Halley iterations for full 53-bit precision across central and tail regions.
    for (int i = 0; i < 4; ++i) {
        double err;
        if (ay <= 0.5) {
            err = std::erf(x) - y;
        } else {
            const double tailErr = (q - std::erfc(keel::abs(x)));
            err = (y > 0.0) ? tailErr : -tailErr;
        }
        const double de = invSqrtPi2 * std::exp(-x * x);
        if (de == 0.0) break;
        const double u = err / de;
        x -= u / (1.0 + x * u);
    }
    return x;
}

// ── Scalar kernels (single source of truth) ──────────────────────────
// Each owns its per-element domain handling so they can be broadcast over
// vector parameters: sigma<=0 (or NaN) → NaN, matching MATLAB R2025b.

inline double normpdfK(double x, double mu, double sigma)
{
    return keel::stats::norm_pdf(x, mu, sigma);
}

inline double normcdfK(double x, double mu, double sigma)
{
    return keel::stats::norm_cdf(x, mu, sigma);
}

inline double norminvK(double p, double mu, double sigma)
{
    return keel::stats::norm_inv(p, mu, sigma);
}

} // anonymous

} // namespace numkit::stats
