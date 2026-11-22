// toolboxes/stats/src/distributions/exponential_detail.hpp
//
// Private (src-only) compute substrate for the exponential distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in exponential.cpp and
// its CallContext register half in exponential_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
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

// ── Scalar kernels (single source of truth) ──────────────────────────
// Each owns its per-element domain handling so they can be broadcast over a
// vector mu: mu<=0 (or NaN) → NaN, matching MATLAB R2025b.

inline keel::f64 exppdfK(keel::f64 x, keel::f64 mu)
{
    return keel::stats::exp_pdf(x, mu);
}

inline keel::f64 expcdfK(keel::f64 x, keel::f64 mu)
{
    return keel::stats::exp_cdf(x, mu);
}

inline keel::f64 expinvK(keel::f64 p, keel::f64 mu)
{
    return keel::stats::exp_inv(p, mu);
}

} // anonymous

} // namespace numkit::stats
