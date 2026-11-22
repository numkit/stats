// toolboxes/signal/src/distributions/rayleigh_reg.cpp
//
// CallContext register half of distributions/rayleigh.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/rayleigh.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/rayleigh_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void raylpdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "raylpdf");
    auto *mr = ctx.engine->allocator();
    const Value &b = args[1];
    if (b.isScalar())
        outs[0] = raylpdf(args[0], b.toScalar(), mr);
    else
        outs[0] = broadcast_dist2(args[0], b, mr, "raylpdf", raylpdfK);
}

void raylcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "raylcdf");
    bool upper = false;
    const keel::ArrayRef<Value> a = args.slice(0, stripUpperFlag(args, upper));
    if (a.size() < 2)
        throw Error("raylcdf: requires (x, b[, 'upper'])", 0, 0, "raylcdf", "", "numkit:raylcdf:nargin");
    auto *mr = ctx.engine->allocator();
    const Value &b = a[1];
    Value v = b.isScalar() ? raylcdf(a[0], b.toScalar(), mr)
                           : broadcast_dist2(a[0], b, mr, "raylcdf", raylcdfK);
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void raylinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "raylinv");
    auto *mr = ctx.engine->allocator();
    const Value &b = args[1];
    if (b.isScalar())
        outs[0] = raylinv(args[0], b.toScalar(), mr);
    else
        outs[0] = broadcast_dist2(args[0], b, mr, "raylinv", raylinvK);
}

void raylrnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "raylrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const keel::f64 b = args[0].toScalar();
    keel::usize rows, cols;
    parse_rng_size(args, 1, rows, cols, "raylrnd");
    outs[0] = reshape_rng_nd(raylrnd(ctx.engine->rng(), b, rows, cols, ctx.engine->allocator()), args, 1, ctx.engine->allocator(), "raylrnd");
}

void raylstat_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "raylstat");
    emit_vec_stat_1arg(args, nargout, outs, ctx.engine->allocator(), "raylstat",
                       [](keel::f64 b) { return raylstat(b); });
}

} // namespace detail

} // namespace numkit::stats
