// toolboxes/stats/src/distributions/gp_detail.hpp
//
// Private (src-only) compute substrate for the gp distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in gp.cpp and
// its CallContext register half in gp_reg.cpp. Kept in an anonymous
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

// Sample inverse-cdf: x = theta + sigma · ((1-u)^(-k) - 1)/k for k≠0,
// x = theta − sigma·log1p(−u) for k=0.
inline keel::f64 gp_inv_one(keel::f64 u, keel::f64 k, keel::f64 sigma, keel::f64 theta) {
    if (!(u >= 0.0 && u <= 1.0)) return keel::Limits<keel::f64>::quiet_nan();
    if (u == 0.0) return theta;
    if (u == 1.0) return (k < 0) ? theta - sigma / k
                                 :  keel::Limits<keel::f64>::infinity();
    if (k == 0.0) return theta - sigma * keel::log1p(-u);
    return theta + sigma * (keel::pow(1.0 - u, -k) - 1.0) / k;
}

} // anonymous

} // namespace numkit::stats
