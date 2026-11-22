// toolboxes/signal/src/descriptive/grp2idx_reg.cpp
//
// CallContext register half of descriptive/grp2idx.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void grp2idx_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "grp2idx");
    Grp2idxResult r = grp2idx(args[0], ctx.engine->allocator());
    outs[0] = std::move(r.G);
    if (nargout >= 2) outs[1] = std::move(r.GN);
    if (nargout >= 3) outs[2] = std::move(r.GL);
}

} // namespace detail

} // namespace numkit::stats
