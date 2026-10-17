// toolboxes/signal/src/distributions/normal_reg.cpp
//
// CallContext register half of distributions/normal.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>  // RngContext + rand/randn/randi/randperm (session-scoped, no global/mutex)
#include <numkit/stats/distributions/normal.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"            // stripUpperFlag / applyUpperInPlace
#include "distributions/normal_detail.hpp"
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

void normpdf_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "normpdf");
    auto *mr = ctx.engine->resource();
    Value hmu, hsig;
    const Value &mu  = dist_param(args, 1, 0.0, mr, hmu);
    const Value &sig = dist_param(args, 2, 1.0, mr, hsig);
    outs[0] = broadcast_dist3(args[0], mu, sig, mr, "normpdf", normpdfK);
}

void normcdf_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "normcdf");
    auto *mr = ctx.engine->resource();
    bool upper = false;
    const Span<const Value> a = args.subspan(0, stripUpperFlag(args, upper));
    Value hmu, hsig;
    const Value &mu  = dist_param(a, 1, 0.0, mr, hmu);
    const Value &sig = dist_param(a, 2, 1.0, mr, hsig);
    Value v = broadcast_dist3(a[0], mu, sig, mr, "normcdf", normcdfK);
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void norminv_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "norminv");
    auto *mr = ctx.engine->resource();
    Value hmu, hsig;
    const Value &mu  = dist_param(args, 1, 0.0, mr, hmu);
    const Value &sig = dist_param(args, 2, 1.0, mr, hsig);
    outs[0] = broadcast_dist3(args[0], mu, sig, mr, "norminv", norminvK);
}

void normrnd_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "normrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double mu = args[0].toScalar();
    const double sigma = args[1].toScalar();
    size_t rows, cols;
    parse_rng_size(args, 2, rows, cols, "normrnd");
    outs[0] = reshape_rng_nd(normrnd(ctx.engine->rng(), mu, sigma, rows, cols, ctx.engine->resource()), args, 2, ctx.engine->resource(), "normrnd");
}

void normstat_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "normstat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->resource(), "normstat",
                       [](double mu, double sigma) { return normstat(mu, sigma); });
}

} // namespace detail

} // namespace numkit::stats
