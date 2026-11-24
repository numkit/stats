// toolboxes/stats/src/distributions/nakagami_detail.hpp
//
// Private (src-only) compute substrate for the nakagami distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in nakagami.cpp and
// its CallContext register half in nakagami_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/specfun.h>
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

inline keel::f64 gammainc_scalar(keel::f64 xx, keel::f64 a, keel::Allocator * /*mr*/)
{
    return keel::specfun::gammainc(xx, a);
}

inline keel::f64 gammaincinv_scalar(keel::f64 p, keel::f64 a, keel::Allocator * /*mr*/)
{
    return keel::specfun::gammaincinv(p, a);
}

} // anonymous

} // namespace numkit::stats
