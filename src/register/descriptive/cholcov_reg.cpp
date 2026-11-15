// toolboxes/signal/src/descriptive/cholcov_reg.cpp
//
// CallContext register half of descriptive/cholcov.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/elmat.hpp>
#include <numkit/linalg/decompositions.hpp>   // chol (migrated)
#include <numkit/linalg/eig.hpp>              // eig_symmetric (migrated)
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::stats {

namespace detail {

void cholcov_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "cholcov");
    auto *mr = ctx.engine->allocator();
    auto [T, p] = cholcov(args[0], mr);
    outs[0] = std::move(T);
    if (nargout > 1) outs[1] = std::move(p);
}

} // namespace detail

} // namespace numkit::stats
