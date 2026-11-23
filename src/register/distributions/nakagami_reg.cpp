// toolboxes/signal/src/distributions/nakagami_reg.cpp
//
// CallContext register half of distributions/nakagami.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>   // gammainc, gammaincinv
#include <numkit/stats/distributions/nakagami.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/nakagami_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>

namespace numkit::stats {

namespace detail {

void nakapdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "nakapdf");  // no max: absent from the R2025b reference build
    outs[0] = nakapdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->allocator());
}

void nakacdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "nakacdf");  // no max: absent from the R2025b reference build
    bool upper = false;
    const keel::usize n = stripUpperFlag(args, upper);
    if (n < 3)
        throw Error("nakacdf: requires (x, mu, omega[, 'upper'])",
                    0, 0, "nakacdf", "", "numkit:nakacdf:nargin");
    Value v = nakacdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->allocator());
    if (upper) applyUpperInPlace(v);
    outs[0] = keel::move(v);
}

void nakainv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "nakainv");  // no max: absent from the R2025b reference build
    outs[0] = nakainv(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->allocator());
}

void nakarnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "nakarnd");  // no max: absent from the R2025b reference build
    const keel::f64 mu    = args[0].toScalar();
    const keel::f64 omega = args[1].toScalar();
    keel::usize rows = 1, cols = 1;
    parse_rng_size(args, 2, rows, cols, "nakarnd");
    outs[0] = reshape_rng_nd(nakarnd(ctx.engine->rng(), mu, omega, rows, cols, ctx.engine->allocator()), args, 2, ctx.engine->allocator(), "nakarnd");
}

void nakastat_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "nakastat");  // no max: absent from the R2025b reference build
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->allocator(), "nakastat",
                       [](keel::f64 mu, keel::f64 omega) { return nakastat(mu, omega); });
}

} // namespace detail

} // namespace numkit::stats
