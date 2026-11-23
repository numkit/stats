// toolboxes/stats/src/distributions/geometric_detail.hpp
//
// Private (src-only) compute substrate for the geometric distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in geometric.cpp and
// its CallContext register half in geometric_reg.cpp. Kept in an anonymous
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

// Scalar kernels for the parameter-broadcast path (vector p). Own their
// per-element domain (p<=0 or p>1 → NaN). Mirror the public fns. MATLAB's
// geometric starts at k=0.
inline keel::f64 geopdfK(keel::f64 k, keel::f64 p) {
    if (p <= 0.0 || p > 1.0) return keel::Limits<keel::f64>::quiet_nan();
    if (k < 0.0 || keel::floor(k) != k) return 0.0;
    if (p == 1.0) return k == 0.0 ? 1.0 : 0.0;
    return keel::pow(1.0 - p, k) * p;
}

inline keel::f64 geocdfK(keel::f64 k, keel::f64 p) {
    if (p <= 0.0 || p > 1.0) return keel::Limits<keel::f64>::quiet_nan();
    if (k < 0.0) return 0.0;
    if (p == 1.0) return k >= 0.0 ? 1.0 : 0.0;
    return -keel::expm1((keel::floor(k) + 1.0) * keel::log1p(-p));   // 1 - (1-p)^(⌊k⌋+1)
}

inline keel::f64 geoinvK(keel::f64 q, keel::f64 p) {
    if (p <= 0.0 || p > 1.0) return keel::Limits<keel::f64>::quiet_nan();
    if (!(q >= 0.0 && q <= 1.0)) return keel::Limits<keel::f64>::quiet_nan();
    if (q == 0.0) return 0.0;
    if (q >= 1.0) return keel::Limits<keel::f64>::infinity();
    if (p == 1.0) return 0.0;
    const keel::f64 v = keel::log1p(-q) / keel::log1p(-p) - 1.0;
    keel::f64 k = keel::ceil(v);
    if (k < 0.0) k = 0.0;
    if (k > 0.0) {
        const keel::f64 cdf_prev = -keel::expm1(k * keel::log1p(-p));   // F(k-1)
        const keel::f64 tol = std::max(1e-13, q * 1e-13);
        if (cdf_prev >= q - tol) k -= 1.0;
    }
    return k;
}

} // anonymous

} // namespace numkit::stats
