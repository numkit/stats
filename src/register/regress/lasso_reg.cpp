// toolboxes/signal/src/regress/lasso_reg.cpp
//
// CallContext register half of regress/lasso.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/regress/regress.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "regress/lasso_detail.hpp"
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void lasso_reg(keel::ArrayRef<Value> args, keel::usize nargout,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "lasso");  // no max: name-value tail in the reference
    keel::f64 alpha = 1.0;
    if (args.size() >= 4 && !args[3].isEmpty())
        alpha = args[3].toScalar();
    auto r = lasso(args[0], args[1], args[2], alpha, ctx.engine->allocator());
    outs[0] = std::move(r.B);
    if (nargout > 1) outs[1] = std::move(r.Intercept);
    if (nargout > 2) outs[2] = std::move(r.Lambda);
}

void lassoglm_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, "lassoglm");  // no max: name-value tail in the reference
    if (!args[2].isChar())
        throw Error("lassoglm: distr must be a string",
                    0, 0, "lassoglm", "", "numkit:lassoglm:badDistr");
    const keel::String s = args[2].toString();
    GlmDistribution d;
    if (s == "normal")        d = GlmDistribution::Normal;
    else if (s == "binomial") d = GlmDistribution::Binomial;
    else if (s == "poisson")  d = GlmDistribution::Poisson;
    else
        throw Error(keel::format("lassoglm: unsupported distribution '{}' "
                                 "(v1: normal, binomial, poisson)", s),
                    0, 0, "lassoglm", "", "numkit:lassoglm:badDistr");
    keel::f64 alpha = 1.0;
    if (args.size() >= 5 && !args[4].isEmpty())
        alpha = args[4].toScalar();
    auto r = lassoglm(args[0], args[1], d, args[3], alpha, ctx.engine->allocator());
    outs[0] = std::move(r.B);
    if (nargout > 1) outs[1] = std::move(r.Intercept);
    if (nargout > 2) outs[2] = std::move(r.Lambda);
}

} // namespace detail

} // namespace numkit::stats
