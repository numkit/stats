// toolboxes/stats/src/descriptive/corrcov.cpp
//
// MATLAB corrcov(C) — correlation matrix derived from a covariance C.
//
//   R = corrcov(C)
//   [R, sigma] = corrcov(C)
//
// R(i, j) = C(i, j) / sqrt(C(i, i) * C(j, j))
// sigma(i) = sqrt(C(i, i))     (standard deviations, returned as a row)
//
// C must be square. Entries on the diagonal must be non-negative
// (variances); a negative diagonal entry is rejected.
// Off-diagonal divisions where the denominator is zero produce
// NaN, matching MATLAB R2025b.

#include <keel/array.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::stats {

namespace {

void corrcov_argcheck(const Value &C)
{
    if (C.dims().rows() != C.dims().cols())
        throw Error("corrcov: input must be a square matrix",
                    0, 0, "corrcov", "", "numkit:corrcov:NotSquare");
    if (C.dims().rows() == 0)
        throw Error("corrcov: input must be non-empty",
                    0, 0, "corrcov", "", "numkit:corrcov:Empty");
}

} // namespace

std::pair<Value, Value>
corrcov(const Value &C, keel::Allocator *mr)
{
    corrcov_argcheck(C);
    const keel::usize R = C.dims().rows();

    // sigma(i) = sqrt(C(i, i)).
    keel::Array<keel::f64> sigma(R);
    for (keel::usize i = 0; i < R; ++i) {
        const keel::f64 v = C.elemAsDouble(i + i * R);   // column-major
        if (v < 0.0)
            throw Error("corrcov: negative variance on diagonal",
                        0, 0, "corrcov", "",
                        "numkit:corrcov:NegativeVar");
        sigma[i] = keel::sqrt(v);
    }

    Value Rv = Value::matrix(R, R, ValueType::DOUBLE, mr);
    keel::f64 *o = Rv.doubleDataMut();
    for (keel::usize j = 0; j < R; ++j) {
        for (keel::usize i = 0; i < R; ++i) {
            const keel::f64 cij = C.elemAsDouble(i + j * R);
            const keel::f64 denom = sigma[i] * sigma[j];
            o[i + j * R] = (denom == 0.0)
                               ? keel::Limits<keel::f64>::quiet_nan()
                               : cij / denom;
        }
    }
    // Sigma as row vector (MATLAB convention).
    Value Sv = Value::matrix(1, R, ValueType::DOUBLE, mr);
    keel::f64 *s = Sv.doubleDataMut();
    for (keel::usize i = 0; i < R; ++i) s[i] = sigma[i];

    return {std::move(Rv), std::move(Sv)};
}

} // namespace numkit::stats
