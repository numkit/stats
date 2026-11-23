// toolboxes/signal/src/distributions/uniform_reg.cpp
//
// CallContext register half of distributions/uniform.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/uniform.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/uniform_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>

namespace numkit::stats {

namespace detail {

namespace {
// MATLAB defaults: a = 0, b = 1 when omitted.
inline keel::f64 argA(keel::ArrayRef<Value> args, keel::usize i) {
    return (args.size() > i && !args[i].isEmpty()) ? args[i].toScalar() : 0.0;
}
inline keel::f64 argB(keel::ArrayRef<Value> args, keel::usize i) {
    return (args.size() > i && !args[i].isEmpty()) ? args[i].toScalar() : 1.0;
}
}

void unifpdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "unifpdf");
    outs[0] = unifpdf(args[0], argA(args, 1), argB(args, 2), ctx.engine->allocator());
}

void unifcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "unifcdf");
    bool upper = false;
    const keel::ArrayRef<Value> stripped = args.slice(0, stripUpperFlag(args, upper));
    if (stripped.empty())
        throw Error("unifcdf: requires (x[, a, b][, 'upper'])", 0, 0, "unifcdf", "", "numkit:unifcdf:nargin");
    Value v = unifcdf(stripped[0], argA(stripped, 1), argB(stripped, 2), ctx.engine->allocator());
    if (upper) applyUpperInPlace(v);
    outs[0] = keel::move(v);
}

void unifinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "unifinv");
    outs[0] = unifinv(args[0], argA(args, 1), argB(args, 2), ctx.engine->allocator());
}

void unifrnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "unifrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const keel::f64 a = args[0].toScalar();
    const keel::f64 b = args[1].toScalar();
    keel::usize rows, cols;
    parse_rng_size(args, 2, rows, cols, "unifrnd");
    outs[0] = reshape_rng_nd(unifrnd(ctx.engine->rng(), a, b, rows, cols, ctx.engine->allocator()), args, 2, ctx.engine->allocator(), "unifrnd");
}

void unifstat_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "unifstat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->allocator(), "unifstat",
                       [](keel::f64 a, keel::f64 b) { return unifstat(a, b); });
}

} // namespace detail

} // namespace numkit::stats
