// toolboxes/stats/src/distributions/rayleigh_detail.hpp
//
// Private (src-only) compute substrate for the rayleigh distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in rayleigh.cpp and
// its CallContext register half in rayleigh_reg.cpp. Kept in an anonymous
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
#include <cstddef>
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

// Scalar kernels for the parameter-broadcast path (vector b). Own their
// per-element domain (b<=0 → NaN). Mirror the public fns bit-identically.
inline keel::f64 raylpdfK(keel::f64 x, keel::f64 b)
{
    if (b <= 0.0) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 inv_b2 = 1.0 / (b * b);
    if (x < 0.0) return 0.0;
    return x * inv_b2 * keel::exp(-0.5 * x * x * inv_b2);
}

inline keel::f64 raylcdfK(keel::f64 x, keel::f64 b)
{
    if (b <= 0.0) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 inv_b2 = 1.0 / (b * b);
    if (x <= 0.0) return 0.0;
    return -keel::expm1(-0.5 * x * x * inv_b2);
}

inline keel::f64 raylinvK(keel::f64 p, keel::f64 b)
{
    if (b <= 0.0) return keel::Limits<keel::f64>::quiet_nan();
    if (p < 0.0 || p > 1.0) return keel::Limits<keel::f64>::quiet_nan();
    if (p == 0.0) return 0.0;
    if (p >= 1.0) return keel::Limits<keel::f64>::infinity();
    return b * keel::sqrt(-2.0 * keel::log1p(-p));
}

} // anonymous

} // namespace numkit::stats
