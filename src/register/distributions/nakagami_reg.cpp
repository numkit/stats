// toolboxes/signal/src/distributions/nakagami_reg.cpp
//
// CallContext register half of distributions/nakagami.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>   // gammainc, gammaincinv
#include <numkit/stats/distributions/nakagami.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/nakagami_detail.hpp"
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

void nakapdf_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "nakapdf");  // no max: absent from the R2025b reference build
    outs[0] = nakapdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->resource());
}

void nakacdf_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "nakacdf");  // no max: absent from the R2025b reference build
    bool upper = false;
    const size_t n = stripUpperFlag(args, upper);
    if (n < 3)
        throw Error("nakacdf: requires (x, mu, omega[, 'upper'])",
                    0, 0, "nakacdf", "", "numkit:nakacdf:nargin");
    Value v = nakacdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->resource());
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void nakainv_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "nakainv");  // no max: absent from the R2025b reference build
    outs[0] = nakainv(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->resource());
}

void nakarnd_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "nakarnd");  // no max: absent from the R2025b reference build
    const double mu    = args[0].toScalar();
    const double omega = args[1].toScalar();
    size_t rows = 1, cols = 1;
    parse_rng_size(args, 2, rows, cols, "nakarnd");
    outs[0] = nakarnd(ctx.engine->rng(), mu, omega, rows, cols, ctx.engine->resource());
}

void nakastat_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "nakastat");  // no max: absent from the R2025b reference build
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->resource(), "nakastat",
                       [](double mu, double omega) { return nakastat(mu, omega); });
}

} // namespace detail

} // namespace numkit::stats
