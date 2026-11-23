// toolboxes/signal/src/regress/glm_reg.cpp
//
// CallContext register half of regress/glm.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/distributions/normal.hpp>     // for probit via norminv/normcdf
#include <numkit/stats/regress/regress.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "regress/glm_detail.hpp"
#include <numkit/value/argcheck.hpp>


namespace numkit::stats {

namespace detail {

static GlmDistribution parseDistr(keel::StringRef s, const char *fn)
{
    if (s == "normal")             return GlmDistribution::Normal;
    if (s == "binomial")           return GlmDistribution::Binomial;
    if (s == "poisson")            return GlmDistribution::Poisson;
    if (s == "gamma")              return GlmDistribution::Gamma;
    if (s == "inverse gaussian"
        || s == "inversegaussian") return GlmDistribution::InverseGaussian;
    throw Error(keel::format("{}: unknown distribution '{}'", fn, s),
                0, 0, fn, "", keel::format("numkit:{}:badDistr", fn));
}

static GlmLink parseLink(keel::StringRef s, const char *fn)
{
    if (s.empty() || s == "canonical") return GlmLink::Identity;
    if (s == "identity")    return GlmLink::Identity;
    if (s == "logit")       return GlmLink::Logit;
    if (s == "log")         return GlmLink::Log;
    if (s == "reciprocal")  return GlmLink::Reciprocal;
    if (s == "probit")      return GlmLink::Probit;
    throw Error(keel::format("{}: unknown link '{}'", fn, s),
                0, 0, fn, "", keel::format("numkit:{}:badLink", fn));
}

void glmfit_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "glmfit");  // no max: name-value tail in the reference
    if (!args[2].isChar() && !args[2].isString())
        throw Error("glmfit: distr must be a string",
                    0, 0, "glmfit", "", "numkit:glmfit:badDistr");
    const keel::String distrStr = args[2].toString();
    const GlmDistribution d = parseDistr(distrStr.as_ref(), "glmfit");
    GlmLink link = GlmLink::Identity;
    if (args.size() >= 4 && (args[3].isChar() || args[3].isString())) {
        const keel::String linkStr = args[3].toString();
        link = parseLink(linkStr.as_ref(), "glmfit");
    }
    auto r = glmfit(args[0], args[1], d, link, ctx.engine->allocator());
    outs[0] = keel::move(r.b);
    if (nargout > 1) outs[1] = keel::move(r.dev);
    if (nargout > 2) outs[2] = keel::move(r.stats);
}

void glmval_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "glmval");  // no max: name-value tail in the reference
    Value invHandle;
    if (args[2].isStruct()) {
        if (args[2].hasField("Inverse")) invHandle = args[2].field("Inverse");
        else if (args[2].hasField("inverse")) invHandle = args[2].field("inverse");
    } else if (args[2].isCell() && args[2].numel() >= 3) {
        invHandle = args[2].cellAt(2);
    }
    auto customInvCb = [&ctx, &invHandle](keel::ArrayRef<Value> a, keel::MutArrayRef<Value> o,
                                          keel::Allocator * /*mr*/) {
        Value r = ctx.engine->callFunctionHandle(invHandle, a);
        if (!o.empty()) o[0] = keel::move(r);
    };
    FnHandle customInvLink;
    if (invHandle.isFuncHandle())
        customInvLink = customInvCb;
    auto res = glmval_full_bounds(
        args[0], args[1], args[2],
        keel::ArrayRef<Value>(args.data() + 3, args.size() - 3),
        nargout >= 2, customInvLink, ctx.engine->allocator());
    outs[0] = keel::move(res.yhat);
    if (nargout > 1) outs[1] = keel::move(res.dylo);
    if (nargout > 2) outs[2] = keel::move(res.dyhi);
}

} // namespace detail

} // namespace numkit::stats
