// toolboxes/stats/src/distributions/gamma_dist_detail.hpp
//
// Private (src-only) compute substrate for the gamma_dist distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in gamma_dist.cpp and
// its CallContext register half in gamma_dist_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
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

// Scalar pdf kernel for the parameter-broadcast path (vector a/b). Owns its
// per-element domain: a<0 or b<=0 → NaN, a==0 → 0. Mirrors public gampdf.
inline keel::f64 gampdfK(keel::f64 x, keel::f64 a, keel::f64 b)
{
    if (a < 0.0 || b <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
    if (a == 0.0) return 0.0;
    if (x < 0.0) return 0.0;
    const keel::f64 log_b = std::log(b);
    const keel::f64 lga   = std::lgamma(a);
    if (x == 0.0) {
        if (a < 1.0) return std::numeric_limits<keel::f64>::infinity();
        if (a > 1.0) return 0.0;
        return std::exp(-lga - log_b);   // a == 1 → 1/b
    }
    return std::exp((a - 1.0) * std::log(x) - x / b - a * log_b - lga);
}

} // anonymous

} // namespace numkit::stats
