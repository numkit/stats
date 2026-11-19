// toolboxes/signal/src/distributions/extreme_value_reg.cpp
//
// CallContext register half of distributions/extreme_value.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/extreme_value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/extreme_value_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void evpdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "evpdf");
    const double mu    = (args.size() >= 2) ? args[1].toScalar() : 0.0;
    const double sigma = (args.size() >= 3) ? args[2].toScalar() : 1.0;
    outs[0] = evpdf(args[0], mu, sigma, ctx.engine->allocator());
}

void evcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "evcdf");
    bool upper = false;
    const keel::usize n = stripUpperFlag(args, upper);
    if (n < 1)
        throw Error("evcdf: requires x[, mu, sigma][, 'upper']",
                    0, 0, "evcdf", "", "numkit:evcdf:nargin");
    const double mu    = (n >= 2) ? args[1].toScalar() : 0.0;
    const double sigma = (n >= 3) ? args[2].toScalar() : 1.0;
    Value v = evcdf(args[0], mu, sigma, ctx.engine->allocator());
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void evinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "evinv");
    const double mu    = (args.size() >= 2) ? args[1].toScalar() : 0.0;
    const double sigma = (args.size() >= 3) ? args[2].toScalar() : 1.0;
    outs[0] = evinv(args[0], mu, sigma, ctx.engine->allocator());
}

void evrnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "evrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double mu    = args[0].toScalar();
    const double sigma = args[1].toScalar();
    keel::usize rows = 1, cols = 1;
    parse_rng_size(args, 2, rows, cols, "evrnd");
    outs[0] = reshape_rng_nd(evrnd(ctx.engine->rng(), mu, sigma, rows, cols, ctx.engine->allocator()), args, 2, ctx.engine->allocator(), "evrnd");
}

void evstat_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "evstat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->allocator(), "evstat",
                       [](double mu, double sigma) { return evstat(mu, sigma); });
}

} // namespace detail

} // namespace numkit::stats
