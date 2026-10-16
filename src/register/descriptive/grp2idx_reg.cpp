// toolboxes/signal/src/descriptive/grp2idx_reg.cpp
//
// CallContext register half of descriptive/grp2idx.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
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

void grp2idx_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "grp2idx");
    Grp2idxResult r = grp2idx(args[0], ctx.engine->resource());
    outs[0] = std::move(r.G);
    if (nargout >= 2) outs[1] = std::move(r.GN);
    if (nargout >= 3) outs[2] = std::move(r.GL);
}

} // namespace detail

} // namespace numkit::stats
