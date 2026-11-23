// toolboxes/signal/src/distributions/hypergeom_reg.cpp
//
// CallContext register half of distributions/hypergeom.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/hypergeom.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/hypergeom_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

// Helper: are all three params (args[i1..i1+2]) scalar?
inline bool hyge_params_scalar(keel::ArrayRef<Value> a, keel::usize i) {
    return a[i].isScalar() && a[i + 1].isScalar() && a[i + 2].isScalar();
}

void hygepdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "hygepdf");
    auto *mr = ctx.engine->allocator();
    if (hyge_params_scalar(args, 1))
        outs[0] = hygepdf(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), mr);
    else
        outs[0] = broadcast_dist4(args[0], args[1], args[2], args[3], mr, "hygepdf", hygepdfK);
}

void hygecdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "hygecdf");
    bool upper = false;
    const keel::ArrayRef<Value> a = args.slice(0, stripUpperFlag(args, upper));
    if (a.size() < 4)
        throw Error("hygecdf: requires (k, M, K, N[, 'upper'])", 0, 0, "hygecdf", "", "numkit:hygecdf:nargin");
    auto *mr = ctx.engine->allocator();
    Value v = hyge_params_scalar(a, 1)
                  ? hygecdf(a[0], a[1].toScalar(), a[2].toScalar(), a[3].toScalar(), mr)
                  : broadcast_dist4(a[0], a[1], a[2], a[3], mr, "hygecdf", hygecdfK);
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void hygeinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "hygeinv");
    auto *mr = ctx.engine->allocator();
    if (hyge_params_scalar(args, 1))
        outs[0] = hygeinv(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), mr);
    else
        outs[0] = broadcast_dist4(args[0], args[1], args[2], args[3], mr, "hygeinv", hygeinvK);
}

void hygernd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "hygernd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const keel::f64 M = args[0].toScalar();
    const keel::f64 K = args[1].toScalar();
    const keel::f64 N = args[2].toScalar();
    keel::usize rows = 1, cols = 1;
    parse_rng_size(args, 3, rows, cols, "hygernd");
    outs[0] = reshape_rng_nd(hygernd(ctx.engine->rng(), M, K, N, rows, cols, ctx.engine->allocator()), args, 3, ctx.engine->allocator(), "hygernd");
}

void hygestat_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "hygestat");
    emit_vec_stat_3arg(args, nargout, outs, ctx.engine->allocator(), "hygestat",
                       [](keel::f64 M, keel::f64 K, keel::f64 N) {
                           return hygestat(M, K, N);
                       });
}

} // namespace detail

} // namespace numkit::stats
