// toolboxes/signal/src/distributions/gp_reg.cpp
//
// CallContext register half of distributions/gp.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/gp.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/gp_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>


namespace numkit::stats {

namespace detail {

void gppdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "gppdf");
    outs[0] = gppdf(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), ctx.engine->allocator());
}

void gpcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "gpcdf");
    bool upper = false;
    const keel::usize n = stripUpperFlag(args, upper);
    if (n != 4)
        throw Error("gpcdf: requires (x, k, sigma, theta[, 'upper'])",
                    0, 0, "gpcdf", "", "numkit:gpcdf:nargin");
    Value v = gpcdf(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), ctx.engine->allocator());
    if (upper) applyUpperInPlace(v);
    outs[0] = keel::move(v);
}

void gpinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "gpinv");
    outs[0] = gpinv(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), ctx.engine->allocator());
}

void gprnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "gprnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const keel::f64 k     = args[0].toScalar();
    const keel::f64 sigma = args[1].toScalar();
    const keel::f64 theta = args[2].toScalar();
    keel::usize rows = 1, cols = 1;
    parse_rng_size(args, 3, rows, cols, "gprnd");
    outs[0] = reshape_rng_nd(gprnd(ctx.engine->rng(), k, sigma, theta, rows, cols, ctx.engine->allocator()), args, 3, ctx.engine->allocator(), "gprnd");
}

void gpstat_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "gpstat");
    emit_vec_stat_3arg(args, nargout, outs, ctx.engine->allocator(), "gpstat",
                       [](keel::f64 k, keel::f64 sigma, keel::f64 theta) {
                           return gpstat(k, sigma, theta);
                       });
}

} // namespace detail

} // namespace numkit::stats
