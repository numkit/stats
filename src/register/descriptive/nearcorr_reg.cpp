// toolboxes/signal/src/descriptive/nearcorr_reg.cpp
//
// CallContext register half of descriptive/nearcorr.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include "descriptive/descriptive_extras_detail.hpp"   // nearcorr_full
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/argcheck.hpp>

namespace numkit::stats {

namespace detail {

void nearcorr_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "nearcorr");  // no max: name-value tail in the reference
    // Marshalling only — the input gate and the Tolerance / MaxIterations /
    // Method / Weights name-value contract live in nearcorr_full.
    outs[0] = nearcorr_full(
        args[0], Span<const Value>(args.data() + 1, args.size() - 1),
        ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::stats
