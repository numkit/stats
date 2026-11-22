// toolboxes/stats/src/distributions/chi2_detail.hpp
//
// Private (src-only) compute substrate for the chi2 distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in chi2.cpp and
// its CallContext register half in chi2_reg.cpp. Kept in an anonymous
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

// Scalar pdf kernel for the parameter-broadcast path (vector k). Owns its
// per-element domain: k<0 → NaN, k==0 → 0 (degenerate Chi²(0)). Mirrors the
// public chi2pdf formula exactly.
inline keel::f64 chi2pdfK(keel::f64 x, keel::f64 k)
{
    if (k < 0.0) return keel::Limits<keel::f64>::quiet_nan();
    if (k == 0.0) return 0.0;
    if (x < 0.0) return 0.0;
    const keel::f64 half_k = 0.5 * k;
    const keel::f64 log_norm = -half_k * keel::log(2.0) - keel::lgamma(half_k);
    if (x == 0.0)
        return (k == 2.0) ? 0.5 : (k > 2.0 ? 0.0 : keel::Limits<keel::f64>::infinity());
    return keel::exp(log_norm + (half_k - 1.0) * keel::log(x) - 0.5 * x);
}

} // anonymous

} // namespace numkit::stats
