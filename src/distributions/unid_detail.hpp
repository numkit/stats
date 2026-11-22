// toolboxes/stats/src/distributions/unid_detail.hpp
//
// Private (src-only) compute substrate for the unid distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in unid.cpp and
// its CallContext register half in unid_reg.cpp. Kept in an anonymous
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

// Scalar kernels for the parameter-broadcast path (vector N). Own their
// per-element domain (N<1 or noninteger N → NaN). Mirror the public fns.
inline keel::f64 unidpdfK(keel::f64 k, keel::f64 N) {
    if (N < 1.0 || keel::floor(N) != N) return keel::Limits<keel::f64>::quiet_nan();
    if (k < 1.0 || k > N || keel::floor(k) != k) return 0.0;
    return 1.0 / N;
}

inline keel::f64 unidcdfK(keel::f64 k, keel::f64 N) {
    if (N < 1.0 || keel::floor(N) != N) return keel::Limits<keel::f64>::quiet_nan();
    if (k < 1.0) return 0.0;
    if (k >= N) return 1.0;
    return keel::floor(k) / N;
}

inline keel::f64 unidinvK(keel::f64 p, keel::f64 N) {
    if (!(N >= 1.0) || keel::floor(N) != N) return keel::Limits<keel::f64>::quiet_nan();
    if (keel::is_nan(p) || p <= 0.0 || p > 1.0) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 r = keel::ceil(p * N);
    return r < 1.0 ? 1.0 : (r > N ? N : r);
}

} // anonymous

} // namespace numkit::stats
