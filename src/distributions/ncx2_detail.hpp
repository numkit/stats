// toolboxes/stats/src/distributions/ncx2_detail.hpp
//
// Private (src-only) compute substrate for the ncx2 distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in ncx2.cpp and
// its CallContext register half in ncx2_reg.cpp. Kept in an anonymous
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

[[maybe_unused]] keel::f64 ncx2cdf_one(keel::f64 x, keel::f64 k, keel::f64 lambda, keel::Allocator * /*mr*/)
{
    return keel::stats::ncx2_cdf(x, k, lambda);
}

} // anonymous

} // namespace numkit::stats
