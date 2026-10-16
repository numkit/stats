// toolboxes/signal/src/cluster/distance_reg.cpp
//
// CallContext register half of cluster/distance.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/cluster/distance.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include "cluster/distance_detail.hpp"
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

namespace numkit::stats {

namespace detail {

namespace {
} // anonymous

void pdist_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "pdist");  // no max: metric/p/C tail in the reference
    // Marshalling only — the metric/p/covariance contract lives in
    // pdist_full.
    outs[0] = pdist_full(args[0],
                         Span<const Value>(args.data() + 1, args.size() - 1),
                         ctx.engine->resource());
}

void pdist2_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "pdist2");  // no max: metric/option tail in the reference
    // Marshalling only — the metric tail plus the top-k extension live
    // in pdist2_full; the 2nd output is the top-k index matrix.
    auto r = pdist2_full(args[0], args[1],
                         Span<const Value>(args.data() + 2, args.size() - 2),
                         ctx.engine->resource());
    outs[0] = std::move(r.D);
    if (nargout >= 2) outs[1] = std::move(r.I);
}

void squareform_reg(Span<const Value> args, size_t /*nargout*/,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "squareform");
    outs[0] = squareform(args[0], ctx.engine->resource());
}

void mahal_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "mahal");
    outs[0] = mahal(args[0], args[1], ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::stats
