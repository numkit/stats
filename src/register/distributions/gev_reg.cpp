// toolboxes/signal/src/distributions/gev_reg.cpp
//
// CallContext register half of distributions/gev.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/gev.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/gev_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void gevpdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "gevpdf");
    outs[0] = gevpdf(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), ctx.engine->allocator());
}

void gevcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "gevcdf");
    bool upper = false;
    const keel::usize n = stripUpperFlag(args, upper);
    if (n < 4)
        throw Error("gevcdf: requires (x, k, sigma, mu[, 'upper'])",
                    0, 0, "gevcdf", "", "numkit:gevcdf:nargin");
    Value v = gevcdf(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), ctx.engine->allocator());
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void gevinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "gevinv");
    outs[0] = gevinv(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), ctx.engine->allocator());
}

void gevrnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "gevrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double k     = args[0].toScalar();
    const double sigma = args[1].toScalar();
    const double mu    = args[2].toScalar();
    keel::usize rows = 1, cols = 1;
    parse_rng_size(args, 3, rows, cols, "gevrnd");
    outs[0] = reshape_rng_nd(gevrnd(ctx.engine->rng(), k, sigma, mu, rows, cols, ctx.engine->allocator()), args, 3, ctx.engine->allocator(), "gevrnd");
}

void gevstat_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "gevstat");
    emit_vec_stat_3arg(args, nargout, outs, ctx.engine->allocator(), "gevstat",
                       [](double k, double sigma, double mu) {
                           return gevstat(k, sigma, mu);
                       });
    return;
}

} // namespace detail

} // namespace numkit::stats
