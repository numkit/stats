// toolboxes/.../regress_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by regress.cpp + regress_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>


namespace numkit::stats {

namespace {

// Cholesky factor of d×d symmetric PSD matrix M (column-major)
// into L (lower-triangular, column-major). Returns true on success.
[[maybe_unused]] bool cholesky(const keel::f64 *M, keel::f64 *L, keel::usize d)
{
    for (keel::usize i = 0; i < d * d; ++i) L[i] = 0.0;
    for (keel::usize j = 0; j < d; ++j) {
        keel::f64 s = M[j + j * d];
        for (keel::usize k = 0; k < j; ++k) s -= L[j + k * d] * L[j + k * d];
        if (s <= 0.0) return false;
        const keel::f64 Ljj = keel::sqrt(s);
        L[j + j * d] = Ljj;
        for (keel::usize i = j + 1; i < d; ++i) {
            keel::f64 t = M[i + j * d];
            for (keel::usize k = 0; k < j; ++k) t -= L[i + k * d] * L[j + k * d];
            L[i + j * d] = t / Ljj;
        }
    }
    return true;
}

// Solve L · z = b (forward substitution).
[[maybe_unused]] void fwd_solve(const keel::f64 *L, keel::f64 *z, const keel::f64 *b, keel::usize d)
{
    for (keel::usize i = 0; i < d; ++i) {
        keel::f64 s = b[i];
        for (keel::usize k = 0; k < i; ++k) s -= L[i + k * d] * z[k];
        z[i] = s / L[i + i * d];
    }
}

// Solve L^T · x = z (backward substitution).
[[maybe_unused]] void back_solve(const keel::f64 *L, keel::f64 *x, const keel::f64 *z, keel::usize d)
{
    for (keel::usize i = d; i-- > 0;) {
        keel::f64 s = z[i];
        for (keel::usize k = i + 1; k < d; ++k) s -= L[k + i * d] * x[k];
        x[i] = s / L[i + i * d];
    }
}

} // anonymous

// regress_full: full regress() worker -> [b, bint, r, rint, stats]. Def in
// regress.cpp (external).
std::tuple<Value, Value, Value, Value, Value>
regress_full(const Value &y, const Value &X, keel::f64 alpha,
             keel::Allocator *mr);

} // namespace numkit::stats
