// toolboxes/stats/src/descriptive/cholcov.cpp
//
// MATLAB cholcov(SIGMA): Cholesky-like factor of a (possibly singular)
// covariance matrix.
//
//   [T, p] = cholcov(SIGMA)
//
// Returns T such that T'*T == SIGMA (within rounding) and the
// "non-PD count" p:
//   - SIGMA positive-definite      -> T = upper-tri n×n (chol),  p = 0
//   - SIGMA PSD, rank r < n        -> T = r×n,                   p = 0
//   - SIGMA has any negative eig   -> T = empty 0×0,             p = #(eig <= tol)
//
// Algorithm:
//   1. Try in-place upper-tri Cholesky. If sqrt(s) succeeds at every
//      step, return the resulting upper-triangular R.
//   2. Else eigendecompose SIGMA (treated as symmetric).
//   3. Tolerance tol = eps^(3/4) * max(|d|).
//   4. p = sum(d <= -tol) (count of "negative" eigenvalues).
//   5. If p > 0, return empty T.
//   6. Else (PSD), keep eigvals d > tol and rebuild T = sqrt(d_keep) .* V_keep'.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/builtin/elmat.hpp>
#include <numkit/linalg/decompositions.hpp>   // chol (migrated)
#include <numkit/linalg/eig.hpp>              // eig_symmetric (migrated)
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::stats {

namespace {

void cholcov_argcheck(const Value &SIGMA)
{
    if (SIGMA.dims().rows() != SIGMA.dims().cols())
        throw Error("cholcov: SIGMA must be square",
                    0, 0, "cholcov", "", "numkit:cholcov:NotSquare");
}

// In-place upper-tri Cholesky attempt. Returns true if SIGMA is PD;
// fills `out` with R such that R'*R = SIGMA. On any non-positive
// pivot, returns false.
bool tryCholesky(const Value &SIGMA, keel::Array<keel::f64> &out, keel::usize n)
{
    const keel::f64 *a = SIGMA.doubleData();
    out.assign(n * n, 0.0);
    keel::f64 *r = out.data();
    for (keel::usize j = 0; j < n; ++j) {
        keel::f64 s = a[j + j * n];
        for (keel::usize k = 0; k < j; ++k)
            s -= r[k + j * n] * r[k + j * n];
        if (s <= 0.0) return false;
        r[j + j * n] = keel::sqrt(s);
        const keel::f64 inv_diag = 1.0 / r[j + j * n];
        for (keel::usize i = j + 1; i < n; ++i) {
            keel::f64 t = a[j + i * n];
            for (keel::usize k = 0; k < j; ++k)
                t -= r[k + j * n] * r[k + i * n];
            r[j + i * n] = t * inv_diag;
        }
    }
    return true;
}

} // namespace

keel::Pair<Value, Value>
cholcov(const Value &SIGMA, keel::Allocator *mr)
{
    cholcov_argcheck(SIGMA);
    const keel::usize R = SIGMA.dims().rows();
    if (R == 0) {
        Value T = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return {keel::move(T), Value::scalar(0.0, mr)};
    }

    // 1. Try positive-definite Cholesky.
    keel::Array<keel::f64> chol_out;
    if (tryCholesky(SIGMA, chol_out, R)) {
        Value T = Value::matrix(R, R, ValueType::DOUBLE, mr);
        std::copy(chol_out.begin(), chol_out.end(), T.doubleDataMut());
        return {keel::move(T), Value::scalar(0.0, mr)};
    }

    // 2. Fall back to eigendecomposition. Treat SIGMA as symmetric;
    //    eig_symmetric throws if it's not symmetric.
    auto [V, D] = ::numkit::linalg::eig_symmetric(SIGMA, mr);

    // D is n×n diagonal; extract eigvals into a flat vector.
    keel::Array<keel::f64> d(R);
    {
        const keel::f64 *dd = D.doubleData();
        for (keel::usize i = 0; i < R; ++i) d[i] = dd[i + i * R];
    }
    keel::f64 maxAbsD = 0.0;
    for (keel::f64 v : d) maxAbsD = std::max(maxAbsD, keel::abs(v));
    const keel::f64 tol = keel::pow(keel::Limits<keel::f64>::kEpsilon,
                                0.75) * maxAbsD;

    // 3. Count "negative" eigenvalues beyond tolerance.
    keel::usize p = 0;
    for (keel::f64 v : d) if (v <= -tol) ++p;
    if (p > 0) {
        Value T = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return {keel::move(T), Value::scalar(keel::as<keel::f64>(p), mr)};
    }

    // 4. PSD: keep eigvals > tol, rebuild T = sqrt(d_keep) .* V_keep'.
    keel::Array<keel::usize> keep;
    keep.reserve(R);
    for (keel::usize i = 0; i < R; ++i) if (d[i] > tol) keep.push_back(i);
    const keel::usize r = keep.size();
    Value T = Value::matrix(r, R, ValueType::DOUBLE, mr);
    keel::f64 *o = T.doubleDataMut();
    const keel::f64 *vd = V.doubleData();
    for (keel::usize k = 0; k < r; ++k) {
        const keel::usize col = keep[k];
        const keel::f64 s = keel::sqrt(d[col]);
        // T(k, j) = s * V(j, col)
        for (keel::usize j = 0; j < R; ++j)
            o[k + j * r] = s * vd[j + col * R];
    }
    return {keel::move(T), Value::scalar(0.0, mr)};
}

} // namespace numkit::stats
