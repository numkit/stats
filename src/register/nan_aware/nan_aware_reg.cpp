// toolboxes/signal/src/nan_aware/nan_aware_reg.cpp
//
// CallContext register half of nan_aware/nan_aware.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/array.h>
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/nan_aware/nan_aware.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/ops/nan_reductions.hpp>
#include <numkit/ops/helpers.hpp>
#include "nan_aware/nan_aware_detail.hpp"
#include <numkit/ops/reductions.hpp>


#include "support/reduce_vecdim.hpp"

namespace numkit::stats {

namespace detail {

#define NUMKIT_NAN_REDUCTION_ADAPTER(name, fn)                                      \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,                \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 1, 2, #name);                         \
        keel::i32 dim = 0;                                                             \
        keel::Array<keel::i32> vecdimSet;                                              \
        if (args.size() >= 2)                                                    \
            vecdim::parseDimSlot(args[0], args[1], #name, dim, vecdimSet);       \
        auto *mr = ctx.engine->allocator();                                       \
        if (!vecdimSet.empty()) {                                                \
            Value c = vecdim::collapseDims(args[0], vecdimSet.as_ref(), mr);     \
            outs[0] = vecdim::expandReduced(                                     \
                fn(c, keel::as<keel::i32>(c.dims().ndim()), mr),                    \
                args[0], vecdimSet.as_ref(), mr);                                \
        } else {                                                                 \
            outs[0] = fn(args[0], dim, mr);                                      \
        }                                                                        \
        /* single contagion (probed R2025b: nansum(single) -> single) */         \
        if (args[0].type() == ValueType::SINGLE && !outs[0].isComplex()) {       \
            Value sg = Value::matrix(outs[0].dims().rows(),                      \
                                    outs[0].dims().cols(),                       \
                                    ValueType::SINGLE, mr);                      \
            for (keel::usize i = 0; i < outs[0].numel(); ++i)                    \
                sg.singleDataMut()[i] =                                          \
                    keel::as<keel::f32>(outs[0].elemAsDouble(i));                 \
            outs[0] = keel::move(sg);                                            \
        }                                                                        \
    }

NUMKIT_NAN_REDUCTION_ADAPTER(nansum,    nansum)
NUMKIT_NAN_REDUCTION_ADAPTER(nanmean,   nanmean)
NUMKIT_NAN_REDUCTION_ADAPTER(nanmedian, nanmedian)

#undef NUMKIT_NAN_REDUCTION_ADAPTER

// nanmax / nanmin accept both signatures:
//   nanmax(A)         — reduce over first non-singleton
//   nanmax(A, dim)    — legacy/nk form (dim in arg 1)
//   nanmax(A, [], d)  — MATLAB-style 3-arg form (dim in arg 2; arg 1 = [])
#define NUMKIT_NAN_MAXMIN_ADAPTER(name, fn)                                          \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,                 \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                         \
    {                                                                             \
        argcheck::checkNargin(args.size(), 1, #name);  /* no max: flag/option tail in the reference */ \
        keel::i32 dim = 0;                                                              \
        keel::Array<keel::i32> vecdimSet;                                               \
        if (args.size() == 2 && !args[1].isEmpty())                               \
            vecdim::parseDimSlot(args[0], args[1], #name, dim, vecdimSet);        \
        else if (args.size() >= 3 && !args[2].isEmpty())                          \
            vecdim::parseDimSlot(args[0], args[2], #name, dim, vecdimSet);        \
        auto *mr = ctx.engine->allocator();                                        \
        if (!vecdimSet.empty()) {                                                 \
            Value c = vecdim::collapseDims(args[0], vecdimSet.as_ref(), mr);      \
            outs[0] = vecdim::expandReduced(                                      \
                fn(c, keel::as<keel::i32>(c.dims().ndim()), mr),                     \
                args[0], vecdimSet.as_ref(), mr);                                 \
            return;                                                               \
        }                                                                         \
        outs[0] = fn(args[0], dim, mr);                                           \
    }

NUMKIT_NAN_MAXMIN_ADAPTER(nanmax, nanmax)
NUMKIT_NAN_MAXMIN_ADAPTER(nanmin, nanmin)

#undef NUMKIT_NAN_MAXMIN_ADAPTER

void nanvar_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "nanvar");
    keel::i32 w = 0, dim = 0;
    keel::Array<keel::i32> vecdimSet;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const keel::f64 wv = argcheck::expectedReal(args[1], 2, "w", "nanvar");
        argcheck::expectedFlag01(wv, 2, "w", "nanvar");
        w = keel::as<keel::i32>(wv);
    }
    if (args.size() >= 3)
        vecdim::parseDimSlot(args[0], args[2], "nanvar", dim, vecdimSet);
    auto *mr = ctx.engine->allocator();
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet.as_ref(), mr);
        outs[0] = vecdim::expandReduced(
            nanvar(c, w, keel::as<keel::i32>(c.dims().ndim()), mr),
            args[0], vecdimSet.as_ref(), mr);
        return;
    }
    outs[0] = nanvar(args[0], w, dim, mr);
}

void nanstd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "nanstd");
    keel::i32 w = 0, dim = 0;
    keel::Array<keel::i32> vecdimSet;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const keel::f64 wv = argcheck::expectedReal(args[1], 2, "w", "nanstd");
        argcheck::expectedFlag01(wv, 2, "w", "nanstd");
        w = keel::as<keel::i32>(wv);
    }
    if (args.size() >= 3)
        vecdim::parseDimSlot(args[0], args[2], "nanstd", dim, vecdimSet);
    auto *mr = ctx.engine->allocator();
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet.as_ref(), mr);
        outs[0] = vecdim::expandReduced(
            nanstdev(c, w, keel::as<keel::i32>(c.dims().ndim()), mr),
            args[0], vecdimSet.as_ref(), mr);
        return;
    }
    outs[0] = nanstdev(args[0], w, dim, mr);
}

// nancov has two MATLAB call patterns:
//   nancov(X)                   — covariance matrix of X
//   nancov(X, normFlag)         — normalization 0 (n-1) or 1 (n)
//   nancov(X, Y)                — between two vectors → 2×2
//   nancov(X, Y, normFlag)      — same + normalization
// We disambiguate by arg-2 type: scalar → normFlag; non-scalar → Y.
void nancov_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "nancov");
    auto *mr = ctx.engine->allocator();
    if (args.size() == 1) {
        outs[0] = nancov(args[0], 0, mr);
        return;
    }
    // args.size() >= 2
    const bool secondIsScalar = (args[1].numel() == 1) && !args[1].isEmpty();
    if (secondIsScalar) {
        const keel::f64 wv = argcheck::expectedReal(args[1], 2, "w", "nancov");
        argcheck::expectedFlag01(wv, 2, "w", "nancov");
        outs[0] = nancov(args[0], keel::as<keel::i32>(wv), mr);
    } else {
        keel::i32 w = 0;
        if (args.size() >= 3 && !args[2].isEmpty()) {
            const keel::f64 wv = argcheck::expectedReal(args[2], 3, "w", "nancov");
            argcheck::expectedFlag01(wv, 3, "w", "nancov");
            w = keel::as<keel::i32>(wv);
        }
        outs[0] = nancov(args[0], args[1], w, mr);
    }
}

} // namespace detail

} // namespace numkit::stats
