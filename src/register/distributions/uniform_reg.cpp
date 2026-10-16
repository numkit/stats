// toolboxes/signal/src/distributions/uniform_reg.cpp
//
// CallContext register half of distributions/uniform.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/uniform.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/uniform_detail.hpp"
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
// MATLAB defaults: a = 0, b = 1 when omitted.
inline double argA(Span<const Value> args, size_t i) {
    return (args.size() > i && !args[i].isEmpty()) ? args[i].toScalar() : 0.0;
}
inline double argB(Span<const Value> args, size_t i) {
    return (args.size() > i && !args[i].isEmpty()) ? args[i].toScalar() : 1.0;
}
}

void unifpdf_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "unifpdf");
    outs[0] = unifpdf(args[0], argA(args, 1), argB(args, 2), ctx.engine->resource());
}

void unifcdf_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "unifcdf");
    bool upper = false;
    const Span<const Value> stripped = args.subspan(0, stripUpperFlag(args, upper));
    if (stripped.empty())
        throw Error("unifcdf: requires (x[, a, b][, 'upper'])", 0, 0, "unifcdf", "", "nk:unifcdf:nargin");
    Value v = unifcdf(stripped[0], argA(stripped, 1), argB(stripped, 2), ctx.engine->resource());
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void unifinv_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "unifinv");
    outs[0] = unifinv(args[0], argA(args, 1), argB(args, 2), ctx.engine->resource());
}

void unifrnd_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "unifrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const double a = args[0].toScalar();
    const double b = args[1].toScalar();
    size_t rows, cols;
    parse_rng_size(args, 2, rows, cols, "unifrnd");
    outs[0] = unifrnd(ctx.engine->rng(), a, b, rows, cols, ctx.engine->resource());
}

void unifstat_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "unifstat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->resource(), "unifstat",
                       [](double a, double b) { return unifstat(a, b); });
}

} // namespace detail

} // namespace numkit::stats
