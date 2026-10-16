// toolboxes/signal/src/distributions/ncx2_reg.cpp
//
// CallContext register half of distributions/ncx2.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>   // besseli, gammainc
#include <numkit/stats/distributions/ncx2.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/ncx2_detail.hpp"
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

void ncx2pdf_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "ncx2pdf");
    outs[0] = ncx2pdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->resource());
}

void ncx2cdf_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "ncx2cdf");
    bool upper = false;
    const size_t n = stripUpperFlag(args, upper);
    if (n < 3)
        throw Error("ncx2cdf: requires (x, k, lambda[, 'upper'])",
                    0, 0, "ncx2cdf", "", "numkit:ncx2cdf:nargin");
    Value v = ncx2cdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->resource());
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void ncx2inv_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "ncx2inv");
    outs[0] = ncx2inv(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->resource());
}

void ncx2rnd_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "ncx2rnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double k      = args[0].toScalar();
    const double lambda = args[1].toScalar();
    size_t rows = 1, cols = 1;
    parse_rng_size(args, 2, rows, cols, "ncx2rnd");
    outs[0] = ncx2rnd(ctx.engine->rng(), k, lambda, rows, cols, ctx.engine->resource());
}

void ncx2stat_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "ncx2stat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->resource(), "ncx2stat",
                       [](double k, double lambda) { return ncx2stat(k, lambda); });
}

} // namespace detail

} // namespace numkit::stats
