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

inline keel::f64 log_choose(keel::f64 a, keel::f64 b) {
    if (b < 0.0 || b > a) return -std::numeric_limits<keel::f64>::infinity();
    return std::lgamma(a + 1.0) - std::lgamma(b + 1.0) - std::lgamma(a - b + 1.0);
}

inline bool params_valid(keel::f64 M, keel::f64 K, keel::f64 N) {
    return M >= 0.0 && K >= 0.0 && N >= 0.0
        && std::floor(M) == M && std::floor(K) == K && std::floor(N) == N
        && K <= M && N <= M;
}

inline keel::f64 hyge_pmf(keel::f64 k, keel::f64 M, keel::f64 K, keel::f64 N) {
    const keel::f64 k_min = std::max(0.0, N - (M - K));
    const keel::f64 k_max = std::min(N, K);
    if (k < k_min || k > k_max || std::floor(k) != k) return 0.0;
    const keel::f64 lp = log_choose(K, k) + log_choose(M - K, N - k) - log_choose(M, N);
    return std::exp(lp);
}

// Forward sum from k_min, with one-ULP tolerance.
inline keel::f64 hyge_cdf_scalar(keel::f64 k, keel::f64 M, keel::f64 K, keel::f64 N) {
    if (k < 0.0) return 0.0;
    const keel::f64 k_max = std::min(N, K);
    if (k >= k_max) return 1.0;
    const keel::f64 k_min = std::max(0.0, N - (M - K));
    if (k < k_min) return 0.0;
    // pmf(j+1) / pmf(j) = (K - j)/(j + 1) · (N - j)/(M - K - N + j + 1)
    keel::f64 f = hyge_pmf(k_min, M, K, N);
    keel::f64 s = f;
    const keel::f64 kf = std::floor(k);
    for (keel::f64 j = k_min; j < kf; j += 1.0) {
        const keel::f64 num = (K - j) * (N - j);
        const keel::f64 den = (j + 1.0) * (M - K - N + j + 1.0);
        if (den == 0.0) { f = 0.0; }
        else            { f *= num / den; }
        s += f;
    }
    return std::min(1.0, std::max(0.0, s));
}

inline keel::f64 hyge_inv_scalar(keel::f64 q, keel::f64 M, keel::f64 K, keel::f64 N) {
    if (!(q >= 0.0 && q <= 1.0)) return std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::f64 k_min = std::max(0.0, N - (M - K));
    const keel::f64 k_max = std::min(N, K);
    if (q == 0.0) return k_min;
    if (q >= 1.0) return k_max;
    const keel::f64 tol = std::max(1e-13, q * 1e-13);
    keel::f64 f = hyge_pmf(k_min, M, K, N);
    keel::f64 s = f;
    if (s >= q - tol) return k_min;
    for (keel::f64 j = k_min; j < k_max; j += 1.0) {
        const keel::f64 num = (K - j) * (N - j);
        const keel::f64 den = (j + 1.0) * (M - K - N + j + 1.0);
        if (den == 0.0) { f = 0.0; }
        else            { f *= num / den; }
        s += f;
        if (s >= q - tol) return j + 1.0;
    }
    return k_max;
}

// Kernels for the parameter-broadcast path (vector M/K/N). Each wraps
// params_valid + the scalar helper above (all pure double, no mr needed).
inline keel::f64 hygepdfK(keel::f64 k, keel::f64 M, keel::f64 K, keel::f64 N) {
    return params_valid(M, K, N) ? hyge_pmf(k, M, K, N)
                                 : std::numeric_limits<keel::f64>::quiet_NaN();
}
inline keel::f64 hygecdfK(keel::f64 k, keel::f64 M, keel::f64 K, keel::f64 N) {
    return params_valid(M, K, N) ? hyge_cdf_scalar(k, M, K, N)
                                 : std::numeric_limits<keel::f64>::quiet_NaN();
}
inline keel::f64 hygeinvK(keel::f64 q, keel::f64 M, keel::f64 K, keel::f64 N) {
    return params_valid(M, K, N) ? hyge_inv_scalar(q, M, K, N)
                                 : std::numeric_limits<keel::f64>::quiet_NaN();
}

} // anonymous

} // namespace numkit::stats
