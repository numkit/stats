// toolboxes/signal/src/nan_aware/nan_aware_reg.cpp
//
// CallContext register half of nan_aware/nan_aware.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/nan_aware/nan_aware.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include <numkit/ops/nan_reductions.hpp>
#include <numkit/ops/helpers.hpp>
#include "nan_aware/nan_aware_detail.hpp"
#include <numkit/ops/reductions.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "support/reduce_vecdim.hpp"

namespace numkit::stats {

namespace detail {

#define NUMKIT_NAN_REDUCTION_ADAPTER(name, fn)                                      \
    void name##_reg(Span<const Value> args, size_t /*nargout*/,                \
                    Span<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 1, 2, #name);                         \
        int dim = 0;                                                             \
        std::vector<int> vecdimSet;                                              \
        if (args.size() >= 2)                                                    \
            vecdim::parseDimSlot(args[0], args[1], #name, dim, vecdimSet);       \
        auto *mr = ctx.engine->resource();                                       \
        if (!vecdimSet.empty()) {                                                \
            Value c = vecdim::collapseDims(args[0], vecdimSet, mr);              \
            outs[0] = vecdim::expandReduced(                                     \
                fn(c, static_cast<int>(c.dims().ndim()), mr),                    \
                args[0], vecdimSet, mr);                                         \
        } else {                                                                 \
            outs[0] = fn(args[0], dim, mr);                                      \
        }                                                                        \
        /* single contagion (probed R2025b: nansum(single) -> single) */         \
        if (args[0].type() == ValueType::SINGLE && !outs[0].isComplex()) {       \
            Value sg = Value::matrix(outs[0].dims().rows(),                      \
                                    outs[0].dims().cols(),                       \
                                    ValueType::SINGLE, mr);                      \
            for (std::size_t i = 0; i < outs[0].numel(); ++i)                    \
                sg.singleDataMut()[i] =                                          \
                    static_cast<float>(outs[0].elemAsDouble(i));                 \
            outs[0] = std::move(sg);                                             \
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
    void name##_reg(Span<const Value> args, size_t /*nargout*/,                 \
                    Span<Value> outs, CallContext &ctx)                         \
    {                                                                             \
        argcheck::checkNargin(args.size(), 1, #name);  /* no max: flag/option tail in the reference */ \
        int dim = 0;                                                              \
        std::vector<int> vecdimSet;                                               \
        if (args.size() == 2 && !args[1].isEmpty())                               \
            vecdim::parseDimSlot(args[0], args[1], #name, dim, vecdimSet);        \
        else if (args.size() >= 3 && !args[2].isEmpty())                          \
            vecdim::parseDimSlot(args[0], args[2], #name, dim, vecdimSet);        \
        auto *mr = ctx.engine->resource();                                        \
        if (!vecdimSet.empty()) {                                                 \
            Value c = vecdim::collapseDims(args[0], vecdimSet, mr);               \
            outs[0] = vecdim::expandReduced(                                      \
                fn(c, static_cast<int>(c.dims().ndim()), mr),                     \
                args[0], vecdimSet, mr);                                          \
            return;                                                               \
        }                                                                         \
        outs[0] = fn(args[0], dim, mr);                                           \
    }

NUMKIT_NAN_MAXMIN_ADAPTER(nanmax, nanmax)
NUMKIT_NAN_MAXMIN_ADAPTER(nanmin, nanmin)

#undef NUMKIT_NAN_MAXMIN_ADAPTER

void nanvar_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "nanvar");
    int w = 0, dim = 0;
    std::vector<int> vecdimSet;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const double wv = argcheck::expectedReal(args[1], 2, "w", "nanvar");
        argcheck::expectedFlag01(wv, 2, "w", "nanvar");
        w = static_cast<int>(wv);
    }
    if (args.size() >= 3)
        vecdim::parseDimSlot(args[0], args[2], "nanvar", dim, vecdimSet);
    auto *mr = ctx.engine->resource();
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        outs[0] = vecdim::expandReduced(
            nanvar(c, w, static_cast<int>(c.dims().ndim()), mr),
            args[0], vecdimSet, mr);
        return;
    }
    outs[0] = nanvar(args[0], w, dim, mr);
}

void nanstd_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "nanstd");
    int w = 0, dim = 0;
    std::vector<int> vecdimSet;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const double wv = argcheck::expectedReal(args[1], 2, "w", "nanstd");
        argcheck::expectedFlag01(wv, 2, "w", "nanstd");
        w = static_cast<int>(wv);
    }
    if (args.size() >= 3)
        vecdim::parseDimSlot(args[0], args[2], "nanstd", dim, vecdimSet);
    auto *mr = ctx.engine->resource();
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        outs[0] = vecdim::expandReduced(
            nanstdev(c, w, static_cast<int>(c.dims().ndim()), mr),
            args[0], vecdimSet, mr);
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
void nancov_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "nancov");
    auto *mr = ctx.engine->resource();
    if (args.size() == 1) {
        outs[0] = nancov(args[0], 0, mr);
        return;
    }
    // args.size() >= 2
    const bool secondIsScalar = (args[1].numel() == 1) && !args[1].isEmpty();
    if (secondIsScalar) {
        const double wv = argcheck::expectedReal(args[1], 2, "w", "nancov");
        argcheck::expectedFlag01(wv, 2, "w", "nancov");
        outs[0] = nancov(args[0], static_cast<int>(wv), mr);
    } else {
        int w = 0;
        if (args.size() >= 3 && !args[2].isEmpty()) {
            const double wv = argcheck::expectedReal(args[2], 3, "w", "nancov");
            argcheck::expectedFlag01(wv, 3, "w", "nancov");
            w = static_cast<int>(wv);
        }
        outs[0] = nancov(args[0], args[1], w, mr);
    }
}

} // namespace detail

} // namespace numkit::stats
