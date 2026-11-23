// toolboxes/stats/src/distributions/rician_detail.hpp
//
// Private (src-only) compute substrate for the rician distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in rician.cpp and
// its CallContext register half in rician_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>


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

inline keel::f64 besseli_scalar(keel::f64 nu, keel::f64 xx, keel::Allocator *mr)
{
    Value nv = Value::scalar(nu, mr);
    Value xv = Value::scalar(xx, mr);
    return ::numkit::builtin::besseli(nv, xv, mr).toScalar();
}

inline keel::f64 marcumq_scalar(keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    Value av = Value::scalar(a, mr);
    Value bv = Value::scalar(b, mr);
    return ::numkit::builtin::marcumq(av, bv, 1, mr).toScalar();
}

} // anonymous

} // namespace numkit::stats
