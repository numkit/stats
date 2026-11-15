// toolboxes/signal/src/distributions/exponential_reg.cpp
//
// CallContext register half of distributions/exponential.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/exponential.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/exponential_detail.hpp"
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

void exppdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "exppdf");
    auto *mr = ctx.engine->allocator();
    // MATLAB default: exppdf(x) ≡ exppdf(x, 1).
    Value hmu;
    const Value &mu = dist_param(args, 1, 1.0, mr, hmu);
    outs[0] = broadcast_dist2(args[0], mu, mr, "exppdf", exppdfK);
}

void expcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "expcdf");
    bool upper = false;
    const keel::ArrayRef<Value> a = args.slice(0, stripUpperFlag(args, upper));
    if (a.empty())
        throw Error("expcdf: requires (x[, mu][, 'upper'])", 0, 0, "expcdf", "", "numkit:expcdf:nargin");
    auto *mr = ctx.engine->allocator();
    // MATLAB default: expcdf(x) ≡ expcdf(x, 1).
    Value hmu;
    const Value &mu = dist_param(a, 1, 1.0, mr, hmu);
    Value v = broadcast_dist2(a[0], mu, mr, "expcdf", expcdfK);
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void expinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "expinv");
    auto *mr = ctx.engine->allocator();
    // MATLAB default: expinv(p) ≡ expinv(p, 1).
    Value hmu;
    const Value &mu = dist_param(args, 1, 1.0, mr, hmu);
    outs[0] = broadcast_dist2(args[0], mu, mr, "expinv", expinvK);
}

void exprnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "exprnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double mu = args[0].toScalar();
    keel::usize rows, cols;
    parse_rng_size(args, 1, rows, cols, "exprnd");
    outs[0] = reshape_rng_nd(exprnd(ctx.engine->rng(), mu, rows, cols, ctx.engine->allocator()), args, 1, ctx.engine->allocator(), "exprnd");
}

void expstat_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "expstat");
    emit_vec_stat_1arg(args, nargout, outs, ctx.engine->allocator(), "expstat",
                       [](double mu) { return expstat(mu); });
}

} // namespace detail

} // namespace numkit::stats
