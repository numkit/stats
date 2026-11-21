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
#include <keel/memory.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

inline keel::f64 gammainc_scalar(keel::f64 xx, keel::f64 a, keel::Allocator *mr)
{
    Value xv = Value::scalar(xx, mr);
    Value av = Value::scalar(a,  mr);
    return ::numkit::builtin::gammainc(xv, av, mr).toScalar();
}

[[maybe_unused]] keel::f64 ncx2cdf_one(keel::f64 x, keel::f64 k, keel::f64 lambda, keel::Allocator *mr)
{
    if (x <= 0.0) return 0.0;
    if (lambda == 0.0) return gammainc_scalar(x / 2.0, k / 2.0, mr);
    const keel::f64 halfL = lambda / 2.0;
    const keel::f64 halfX = x / 2.0;
    keel::f64 Pj = std::exp(-halfL);
    keel::f64 sum = 0.0;
    const int maxIter = 2000;
    for (int j = 0; j < maxIter; ++j) {
        const keel::f64 cdfTerm = gammainc_scalar(halfX, k / 2.0 + keel::f64(j), mr);
        const keel::f64 contrib = Pj * cdfTerm;
        sum += contrib;
        if (j > 5 && contrib < 1e-16 * (sum + 1e-300)) break;
        Pj *= halfL / keel::f64(j + 1);
    }
    return std::min(1.0, sum);
}

} // anonymous

} // namespace numkit::stats
