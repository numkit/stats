// toolboxes/stats/src/distributions/beta_detail.hpp
//
// Private (src-only) compute substrate for the beta distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in beta.cpp and
// its CallContext register half in beta_reg.cpp. Kept in an anonymous
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

// Scalar pdf kernel for the parameter-broadcast path (vector a/b). Owns its
// per-element domain: a<=0 or b<=0 → NaN. Mirrors public betapdf exactly,
// including the x∈{0,1} boundary cases.
inline keel::f64 betapdfK(keel::f64 x, keel::f64 a, keel::f64 b)
{
    if (a <= 0.0 || b <= 0.0) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 lbeta = keel::lgamma(a) + keel::lgamma(b) - keel::lgamma(a + b);
    if (x < 0.0 || x > 1.0) return 0.0;
    if (x == 0.0)
        return (a == 1.0) ? keel::exp(-lbeta) * (b == 1.0 ? 1.0 : keel::pow(1.0, b - 1.0))
                          : (a > 1.0 ? 0.0 : keel::Limits<keel::f64>::infinity());
    if (x == 1.0)
        return (b == 1.0) ? keel::exp(-lbeta) * (a == 1.0 ? 1.0 : keel::pow(1.0, a - 1.0))
                          : (b > 1.0 ? 0.0 : keel::Limits<keel::f64>::infinity());
    return keel::exp((a - 1.0) * keel::log(x) + (b - 1.0) * keel::log1p(-x) - lbeta);
}

} // anonymous

} // namespace numkit::stats
