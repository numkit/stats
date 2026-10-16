// toolboxes/signal/src/cluster/silhouette_reg.cpp
//
// CallContext register half of cluster/silhouette.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/cluster/distance.hpp>
#include <numkit/stats/cluster/silhouette.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/argcheck.hpp>

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

void silhouette_reg(Span<const Value> args, size_t /*nargout*/,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "silhouette");  // no max: metric tail in the reference
    // Marshalling only — the metric contract lives in silhouette_full.
    outs[0] = silhouette_full(args[0], args[1],
                              Span<const Value>(args.data() + 2, args.size() - 2),
                              ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::stats
