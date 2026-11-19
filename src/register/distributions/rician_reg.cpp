// toolboxes/signal/src/distributions/rician_reg.cpp
//
// CallContext register half of distributions/rician.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>   // besseli, marcumq
#include <numkit/stats/distributions/rician.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/rician_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void ricepdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "ricepdf");  // no max: absent from the R2025b reference build
    outs[0] = ricepdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->allocator());
}

void ricecdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "ricecdf");  // no max: absent from the R2025b reference build
    bool upper = false;
    const keel::usize n = stripUpperFlag(args, upper);
    if (n < 3)
        throw Error("ricecdf: requires (x, s, sigma[, 'upper'])",
                    0, 0, "ricecdf", "", "numkit:ricecdf:nargin");
    Value v = ricecdf(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->allocator());
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void riceinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "riceinv");  // no max: absent from the R2025b reference build
    outs[0] = riceinv(args[0], args[1].toScalar(), args[2].toScalar(), ctx.engine->allocator());
}

void ricernd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "ricernd");  // no max: absent from the R2025b reference build
    const double s     = args[0].toScalar();
    const double sigma = args[1].toScalar();
    keel::usize rows = 1, cols = 1;
    parse_rng_size(args, 2, rows, cols, "ricernd");
    outs[0] = reshape_rng_nd(ricernd(ctx.engine->rng(), s, sigma, rows, cols, ctx.engine->allocator()), args, 2, ctx.engine->allocator(), "ricernd");
}

void ricestat_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "ricestat");  // no max: absent from the R2025b reference build
    auto *mr = ctx.engine->allocator();
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->allocator(), "ricestat",
                       [mr](double s, double sigma) { return ricestat(s, sigma, mr); });
}

} // namespace detail

} // namespace numkit::stats
