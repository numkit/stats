// toolboxes/signal/src/distributions/weibull_reg.cpp
//
// CallContext register half of distributions/weibull.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/weibull.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/weibull_detail.hpp"
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

namespace {
inline double argA(Span<const Value> args, size_t i) {
    return (args.size() > i && !args[i].isEmpty()) ? args[i].toScalar() : 1.0;
}
inline double argB(Span<const Value> args, size_t i) {
    return (args.size() > i && !args[i].isEmpty()) ? args[i].toScalar() : 1.0;
}
}

void wblpdf_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "wblpdf");
    auto *mr = ctx.engine->resource();
    Value ha, hb;
    const Value &a = dist_param(args, 1, 1.0, mr, ha);
    const Value &b = dist_param(args, 2, 1.0, mr, hb);
    if (a.isScalar() && b.isScalar())
        outs[0] = wblpdf(args[0], a.toScalar(), b.toScalar(), mr);
    else
        outs[0] = broadcast_dist3(args[0], a, b, mr, "wblpdf", wblpdfK);
}

void wblcdf_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "wblcdf");
    bool upper = false;
    const Span<const Value> s = args.subspan(0, stripUpperFlag(args, upper));
    if (s.empty())
        throw Error("wblcdf: requires (x[, a, b][, 'upper'])", 0, 0, "wblcdf", "", "nk:wblcdf:nargin");
    auto *mr = ctx.engine->resource();
    Value ha, hb;
    const Value &a = dist_param(s, 1, 1.0, mr, ha);
    const Value &b = dist_param(s, 2, 1.0, mr, hb);
    Value v = (a.isScalar() && b.isScalar())
                  ? wblcdf(s[0], a.toScalar(), b.toScalar(), mr)
                  : broadcast_dist3(s[0], a, b, mr, "wblcdf", wblcdfK);
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void wblinv_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "wblinv");
    auto *mr = ctx.engine->resource();
    Value ha, hb;
    const Value &a = dist_param(args, 1, 1.0, mr, ha);
    const Value &b = dist_param(args, 2, 1.0, mr, hb);
    if (a.isScalar() && b.isScalar())
        outs[0] = wblinv(args[0], a.toScalar(), b.toScalar(), mr);
    else
        outs[0] = broadcast_dist3(args[0], a, b, mr, "wblinv", wblinvK);
}

void wblrnd_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "wblrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double a = argA(args, 0);
    const double b = argB(args, 1);
    size_t rows, cols;
    parse_rng_size(args, 2, rows, cols, "wblrnd");
    outs[0] = wblrnd(ctx.engine->rng(), a, b, rows, cols, ctx.engine->resource());
}

void wblstat_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "wblstat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->resource(), "wblstat",
                       [](double a, double b) { return wblstat(a, b); });
}

} // namespace detail

} // namespace numkit::stats
