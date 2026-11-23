// toolboxes/signal/src/distributions/weibull_reg.cpp
//
// CallContext register half of distributions/weibull.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/weibull.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/weibull_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>


namespace numkit::stats {

namespace detail {

namespace {
inline keel::f64 argA(keel::ArrayRef<Value> args, keel::usize i) {
    return (args.size() > i && !args[i].isEmpty()) ? args[i].toScalar() : 1.0;
}
inline keel::f64 argB(keel::ArrayRef<Value> args, keel::usize i) {
    return (args.size() > i && !args[i].isEmpty()) ? args[i].toScalar() : 1.0;
}
}

void wblpdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "wblpdf");
    auto *mr = ctx.engine->allocator();
    Value ha, hb;
    const Value &a = dist_param(args, 1, 1.0, mr, ha);
    const Value &b = dist_param(args, 2, 1.0, mr, hb);
    if (a.isScalar() && b.isScalar())
        outs[0] = wblpdf(args[0], a.toScalar(), b.toScalar(), mr);
    else
        outs[0] = broadcast_dist3(args[0], a, b, mr, "wblpdf", wblpdfK);
}

void wblcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "wblcdf");
    bool upper = false;
    const keel::ArrayRef<Value> s = args.slice(0, stripUpperFlag(args, upper));
    if (s.empty())
        throw Error("wblcdf: requires (x[, a, b][, 'upper'])", 0, 0, "wblcdf", "", "numkit:wblcdf:nargin");
    auto *mr = ctx.engine->allocator();
    Value ha, hb;
    const Value &a = dist_param(s, 1, 1.0, mr, ha);
    const Value &b = dist_param(s, 2, 1.0, mr, hb);
    Value v = (a.isScalar() && b.isScalar())
                  ? wblcdf(s[0], a.toScalar(), b.toScalar(), mr)
                  : broadcast_dist3(s[0], a, b, mr, "wblcdf", wblcdfK);
    if (upper) applyUpperInPlace(v);
    outs[0] = keel::move(v);
}

void wblinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "wblinv");
    auto *mr = ctx.engine->allocator();
    Value ha, hb;
    const Value &a = dist_param(args, 1, 1.0, mr, ha);
    const Value &b = dist_param(args, 2, 1.0, mr, hb);
    if (a.isScalar() && b.isScalar())
        outs[0] = wblinv(args[0], a.toScalar(), b.toScalar(), mr);
    else
        outs[0] = broadcast_dist3(args[0], a, b, mr, "wblinv", wblinvK);
}

void wblrnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "wblrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const keel::f64 a = argA(args, 0);
    const keel::f64 b = argB(args, 1);
    keel::usize rows, cols;
    parse_rng_size(args, 2, rows, cols, "wblrnd");
    outs[0] = reshape_rng_nd(wblrnd(ctx.engine->rng(), a, b, rows, cols, ctx.engine->allocator()), args, 2, ctx.engine->allocator(), "wblrnd");
}

void wblstat_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "wblstat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->allocator(), "wblstat",
                       [](keel::f64 a, keel::f64 b) { return wblstat(a, b); });
}

} // namespace detail

} // namespace numkit::stats
