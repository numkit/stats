// toolboxes/signal/src/distributions/extreme_value_reg.cpp
//
// CallContext register half of distributions/extreme_value.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/extreme_value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/extreme_value_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::stats {

namespace detail {

void evpdf_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "evpdf");
    const double mu    = (args.size() >= 2) ? args[1].toScalar() : 0.0;
    const double sigma = (args.size() >= 3) ? args[2].toScalar() : 1.0;
    outs[0] = evpdf(args[0], mu, sigma, ctx.engine->resource());
}

void evcdf_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "evcdf");
    bool upper = false;
    const size_t n = stripUpperFlag(args, upper);
    if (n < 1)
        throw Error("evcdf: requires x[, mu, sigma][, 'upper']",
                    0, 0, "evcdf", "", "numkit:evcdf:nargin");
    const double mu    = (n >= 2) ? args[1].toScalar() : 0.0;
    const double sigma = (n >= 3) ? args[2].toScalar() : 1.0;
    Value v = evcdf(args[0], mu, sigma, ctx.engine->resource());
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void evinv_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "evinv");
    const double mu    = (args.size() >= 2) ? args[1].toScalar() : 0.0;
    const double sigma = (args.size() >= 3) ? args[2].toScalar() : 1.0;
    outs[0] = evinv(args[0], mu, sigma, ctx.engine->resource());
}

void evrnd_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "evrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double mu    = args[0].toScalar();
    const double sigma = args[1].toScalar();
    size_t rows = 1, cols = 1;
    parse_rng_size(args, 2, rows, cols, "evrnd");
    outs[0] = reshape_rng_nd(evrnd(ctx.engine->rng(), mu, sigma, rows, cols, ctx.engine->resource()), args, 2, ctx.engine->resource(), "evrnd");
}

void evstat_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "evstat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->resource(), "evstat",
                       [](double mu, double sigma) { return evstat(mu, sigma); });
}

} // namespace detail

} // namespace numkit::stats
