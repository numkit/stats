// toolboxes/signal/src/distributions/ncx2_reg.cpp
//
// CallContext register half of distributions/ncx2.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>   // besseli, gammainc
#include <numkit/stats/distributions/ncx2.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/ncx2_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>


namespace numkit::stats {

namespace detail {

void ncx2pdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "ncx2pdf");
    outs[0] = ncx2pdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->allocator());
}

void ncx2cdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "ncx2cdf");
    bool upper = false;
    const keel::usize n = stripUpperFlag(args, upper);
    if (n < 3)
        throw Error("ncx2cdf: requires (x, k, lambda[, 'upper'])",
                    0, 0, "ncx2cdf", "", "numkit:ncx2cdf:nargin");
    Value v = ncx2cdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->allocator());
    if (upper) applyUpperInPlace(v);
    outs[0] = keel::move(v);
}

void ncx2inv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "ncx2inv");
    outs[0] = ncx2inv(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->allocator());
}

void ncx2rnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "ncx2rnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const keel::f64 k      = args[0].toScalar();
    const keel::f64 lambda = args[1].toScalar();
    keel::usize rows = 1, cols = 1;
    parse_rng_size(args, 2, rows, cols, "ncx2rnd");
    outs[0] = reshape_rng_nd(ncx2rnd(ctx.engine->rng(), k, lambda, rows, cols, ctx.engine->allocator()), args, 2, ctx.engine->allocator(), "ncx2rnd");
}

void ncx2stat_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "ncx2stat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->allocator(), "ncx2stat",
                       [](keel::f64 k, keel::f64 lambda) { return ncx2stat(k, lambda); });
}

} // namespace detail

} // namespace numkit::stats
