// toolboxes/signal/src/distributions/gp_reg.cpp
//
// CallContext register half of distributions/gp.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/gp.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/gp_detail.hpp"
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

void gppdf_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "gppdf");
    outs[0] = gppdf(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), ctx.engine->resource());
}

void gpcdf_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "gpcdf");
    bool upper = false;
    const size_t n = stripUpperFlag(args, upper);
    if (n < 4)
        throw Error("gpcdf: requires (x, k, sigma, theta[, 'upper'])",
                    0, 0, "gpcdf", "", "nk:gpcdf:nargin");
    Value v = gpcdf(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), ctx.engine->resource());
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void gpinv_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "gpinv");
    outs[0] = gpinv(args[0], args[1].toScalar(), args[2].toScalar(), args[3].toScalar(), ctx.engine->resource());
}

void gprnd_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "gprnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double k     = args[0].toScalar();
    const double sigma = args[1].toScalar();
    const double theta = args[2].toScalar();
    size_t rows = 1, cols = 1;
    parse_rng_size(args, 3, rows, cols, "gprnd");
    outs[0] = gprnd(ctx.engine->rng(), k, sigma, theta, rows, cols, ctx.engine->resource());
}

void gpstat_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "gpstat");
    emit_vec_stat_3arg(args, nargout, outs, ctx.engine->resource(), "gpstat",
                       [](double k, double sigma, double theta) {
                           return gpstat(k, sigma, theta);
                       });
}

} // namespace detail

} // namespace numkit::stats
