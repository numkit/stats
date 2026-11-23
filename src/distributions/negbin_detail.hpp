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

// log f(k; r, p) = lgamma(k + r) - lgamma(k+1) - lgamma(r)
//                + r·log(p) + k·log(1-p)
inline keel::f64 nbin_pmf(keel::f64 k, keel::f64 r, keel::f64 p) {
    if (k < 0.0 || keel::floor(k) != k) return 0.0;
    if (p == 1.0) return (k == 0.0) ? 1.0 : 0.0;
    const keel::f64 lc = keel::lgamma(k + r) - keel::lgamma(k + 1.0) - keel::lgamma(r);
    return keel::exp(lc + r * keel::log(p) + k * keel::log1p(-p));
}

inline keel::f64 nbin_cdf_scalar(keel::f64 k, keel::f64 r, keel::f64 p, keel::Allocator *mr) {
    if (k < 0.0) return 0.0;
    if (p == 1.0) return 1.0;
    const keel::f64 kf = keel::floor(k);
    // F(k; r, p) = I_p(r, ⌊k⌋ + 1)
    Value xv = Value::scalar(p, mr);
    Value av = Value::scalar(r, mr);
    Value bv = Value::scalar(kf + 1.0, mr);
    Value rv = ::numkit::builtin::betainc(xv, av, bv, mr);
    return rv.toScalar();
}

// (r, p) valid iff r>0 (r may be non-integer) and p ∈ (0, 1]. Else → NaN.
inline bool nbin_params_ok(keel::f64 r, keel::f64 p) {
    return r > 0.0 && p > 0.0 && p <= 1.0;
}

inline keel::f64 nbinpdfK(keel::f64 k, keel::f64 r, keel::f64 p) {
    if (!nbin_params_ok(r, p)) return keel::Limits<keel::f64>::quiet_nan();
    return nbin_pmf(k, r, p);
}

// Smallest integer k with F(k; r, p) ≥ qival, via pmf-recurrence
// pmf(k+1)/pmf(k) = (k+r)/(k+1)·(1-p). (r, p) assumed valid.
inline keel::f64 nbin_inv_scalar(keel::f64 qival, keel::f64 r, keel::f64 p) {
    if (!(qival >= 0.0 && qival <= 1.0)) return keel::Limits<keel::f64>::quiet_nan();
    if (qival == 0.0) return 0.0;
    if (qival >= 1.0) return keel::Limits<keel::f64>::infinity();
    if (p == 1.0) return 0.0;
    const keel::f64 q1m = 1.0 - p;
    keel::f64 pmf = keel::pow(p, r);   // pmf(0)
    keel::f64 cdf = pmf;
    const keel::f64 tol = std::max(1e-13, qival * 1e-13);
    if (cdf >= qival - tol) return 0.0;
    for (keel::f64 k = 0.0; k < 1e12; k += 1.0) {
        pmf *= (k + r) / (k + 1.0) * q1m;
        cdf += pmf;
        if (cdf >= qival - tol) return k + 1.0;
    }
    return keel::Limits<keel::f64>::infinity();
}

} // anonymous

} // namespace numkit::stats
