// toolboxes/signal/src/distributions/exponential_reg.cpp
//
// CallContext register half of distributions/exponential.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/exponential.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/exponential_detail.hpp"
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

void exppdf_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "exppdf");
    auto *mr = ctx.engine->resource();
    // MATLAB default: exppdf(x) ≡ exppdf(x, 1).
    Value hmu;
    const Value &mu = dist_param(args, 1, 1.0, mr, hmu);
    outs[0] = broadcast_dist2(args[0], mu, mr, "exppdf", exppdfK);
}

void expcdf_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "expcdf");
    bool upper = false;
    const Span<const Value> a = args.subspan(0, stripUpperFlag(args, upper));
    if (a.empty())
        throw Error("expcdf: requires (x[, mu][, 'upper'])", 0, 0, "expcdf", "", "nk:expcdf:nargin");
    auto *mr = ctx.engine->resource();
    // MATLAB default: expcdf(x) ≡ expcdf(x, 1).
    Value hmu;
    const Value &mu = dist_param(a, 1, 1.0, mr, hmu);
    Value v = broadcast_dist2(a[0], mu, mr, "expcdf", expcdfK);
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void expinv_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "expinv");
    auto *mr = ctx.engine->resource();
    // MATLAB default: expinv(p) ≡ expinv(p, 1).
    Value hmu;
    const Value &mu = dist_param(args, 1, 1.0, mr, hmu);
    outs[0] = broadcast_dist2(args[0], mu, mr, "expinv", expinvK);
}

void exprnd_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "exprnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double mu = args[0].toScalar();
    size_t rows, cols;
    parse_rng_size(args, 1, rows, cols, "exprnd");
    outs[0] = exprnd(ctx.engine->rng(), mu, rows, cols, ctx.engine->resource());
}

void expstat_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "expstat");
    emit_vec_stat_1arg(args, nargout, outs, ctx.engine->resource(), "expstat",
                       [](double mu) { return expstat(mu); });
}

} // namespace detail

} // namespace numkit::stats
