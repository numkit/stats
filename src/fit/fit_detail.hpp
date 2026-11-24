// toolboxes/.../fit_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by fit.cpp + fit_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/math.h>
#include <keel/stats.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

namespace numkit::stats {

namespace {

[[maybe_unused]] Value rowCI(keel::f64 lo, keel::f64 hi, keel::Allocator *mr) {
    Value v = Value::matrix(2, 1, ValueType::DOUBLE, mr);
    keel::f64 *d = v.doubleDataMut();
    d[0] = lo;
    d[1] = hi;
    return v;
}

} // anonymous

// MLE negative-log-likelihood workers (def in fit.cpp, external). lognlike_full/
// explike_full fill an optional 2x2 asymptotic-covariance via double *avarOut.
keel::f64 lognlike_full(keel::f64 mu, keel::f64 sigma, const Value &x,
                     const Value &cens, const Value &freq, keel::f64 *avarOut);
keel::f64 explike_full(keel::f64 mu, const Value &x,
                    const Value &cens, const Value &freq, keel::f64 *avarOut);
keel::f64 wbllike_full(keel::f64 scale, keel::f64 shape, const Value &x,
                    const Value &cens, const Value &freq,
                    keel::Allocator *mr);
keel::f64 evlike_full(keel::f64 mu, keel::f64 sigma, const Value &x,
                   const Value &cens, const Value &freq,
                   keel::Allocator *mr);

} // namespace numkit::stats
