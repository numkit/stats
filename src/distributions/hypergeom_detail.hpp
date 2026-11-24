// toolboxes/stats/src/distributions/hypergeom_detail.hpp
//
// Private (src-only) compute substrate for the hypergeom distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in hypergeom.cpp and
// its CallContext register half in hypergeom_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

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

inline bool params_valid(keel::f64 M, keel::f64 K, keel::f64 N) {
    return M >= 0.0 && K >= 0.0 && N >= 0.0
        && keel::floor(M) == M && keel::floor(K) == K && keel::floor(N) == N
        && K <= M && N <= M;
}
inline keel::f64 hyge_pmf(keel::f64 k, keel::f64 M, keel::f64 K, keel::f64 N) {
    return keel::stats::hyge_pdf(k, M, K, N);
}
inline keel::f64 hyge_cdf_scalar(keel::f64 k, keel::f64 M, keel::f64 K, keel::f64 N) {
    return keel::stats::hyge_cdf(k, M, K, N);
}
inline keel::f64 hyge_inv_scalar(keel::f64 q, keel::f64 M, keel::f64 K, keel::f64 N) {
    return keel::stats::hyge_inv(q, M, K, N);
}
inline keel::f64 hygepdfK(keel::f64 k, keel::f64 M, keel::f64 K, keel::f64 N) {
    return keel::stats::hyge_pdf(k, M, K, N);
}
inline keel::f64 hygecdfK(keel::f64 k, keel::f64 M, keel::f64 K, keel::f64 N) {
    return keel::stats::hyge_cdf(k, M, K, N);
}
inline keel::f64 hygeinvK(keel::f64 q, keel::f64 M, keel::f64 K, keel::f64 N) {
    return keel::stats::hyge_inv(q, M, K, N);
}

} // anonymous

} // namespace numkit::stats
