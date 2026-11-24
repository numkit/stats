// toolboxes/stats/src/distributions/negbin_detail.hpp
//
// Private (src-only) compute substrate for the negbin distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in negbin.cpp and
// its CallContext register half in negbin_reg.cpp. Kept in an anonymous
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

inline bool nbin_params_ok(keel::f64 r, keel::f64 p) { return r > 0.0 && p > 0.0 && p <= 1.0; }
inline keel::f64 nbin_pmf(keel::f64 k, keel::f64 r, keel::f64 p) { return keel::stats::nbin_pdf(k, r, p); }
inline keel::f64 nbin_cdf_scalar(keel::f64 k, keel::f64 r, keel::f64 p, keel::Allocator * /*mr*/) {
    return keel::stats::nbin_cdf(k, r, p);
}
inline keel::f64 nbinpdfK(keel::f64 k, keel::f64 r, keel::f64 p) { return keel::stats::nbin_pdf(k, r, p); }
inline keel::f64 nbin_inv_scalar(keel::f64 qival, keel::f64 r, keel::f64 p) {
    return keel::stats::nbin_inv(qival, r, p);
}

} // anonymous

} // namespace numkit::stats
