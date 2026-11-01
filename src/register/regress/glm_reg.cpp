// toolboxes/signal/src/regress/glm_reg.cpp
//
// CallContext register half of regress/glm.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/distributions/normal.hpp>     // for probit via norminv/normcdf
#include <numkit/stats/regress/regress.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include "regress/glm_detail.hpp"
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

static GlmDistribution parseDistr(const std::string &s, const char *fn)
{
    if (s == "normal")             return GlmDistribution::Normal;
    if (s == "binomial")           return GlmDistribution::Binomial;
    if (s == "poisson")            return GlmDistribution::Poisson;
    if (s == "gamma")              return GlmDistribution::Gamma;
    if (s == "inverse gaussian"
        || s == "inversegaussian") return GlmDistribution::InverseGaussian;
    throw Error(std::string(fn) + ": unknown distribution '" + s + "'",
                0, 0, fn, "", std::string("numkit:") + fn + ":badDistr");
}

static GlmLink parseLink(const std::string &s, const char *fn)
{
    if (s.empty() || s == "canonical") return GlmLink::Identity;
    if (s == "identity")    return GlmLink::Identity;
    if (s == "logit")       return GlmLink::Logit;
    if (s == "log")         return GlmLink::Log;
    if (s == "reciprocal")  return GlmLink::Reciprocal;
    if (s == "probit")      return GlmLink::Probit;
    throw Error(std::string(fn) + ": unknown link '" + s + "'",
                0, 0, fn, "", std::string("numkit:") + fn + ":badLink");
}

void glmfit_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "glmfit");  // no max: name-value tail in the reference
    if (!args[2].isChar() && !args[2].isString())
        throw Error("glmfit: distr must be a string",
                    0, 0, "glmfit", "", "numkit:glmfit:badDistr");
    const GlmDistribution d = parseDistr(args[2].toString(), "glmfit");
    GlmLink link = GlmLink::Identity;
    if (args.size() >= 4 && (args[3].isChar() || args[3].isString()))
        link = parseLink(args[3].toString(), "glmfit");
    auto r = glmfit(args[0], args[1], d, link, ctx.engine->resource());
    outs[0] = std::move(r.b);
    if (nargout > 1) outs[1] = std::move(r.dev);
    if (nargout > 2) outs[2] = std::move(r.stats);
}

void glmval_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "glmval");  // no max: name-value tail in the reference
    Value invHandle;
    if (args[2].isStruct()) {
        if (args[2].hasField("Inverse")) invHandle = args[2].field("Inverse");
        else if (args[2].hasField("inverse")) invHandle = args[2].field("inverse");
    } else if (args[2].isCell() && args[2].numel() >= 3) {
        invHandle = args[2].cellAt(2);
    }
    auto customInvCb = [&ctx, &invHandle](Span<const Value> a, Span<Value> o,
                                          std::pmr::memory_resource * /*mr*/) {
        Value r = ctx.engine->callFunctionHandle(invHandle, a);
        if (!o.empty()) o[0] = std::move(r);
    };
    FnHandle customInvLink;
    if (invHandle.isFuncHandle())
        customInvLink = customInvCb;
    auto res = glmval_full_bounds(
        args[0], args[1], args[2],
        Span<const Value>(args.data() + 3, args.size() - 3),
        nargout >= 2, customInvLink, ctx.engine->resource());
    outs[0] = std::move(res.yhat);
    if (nargout > 1) outs[1] = std::move(res.dylo);
    if (nargout > 2) outs[2] = std::move(res.dyhi);
}

} // namespace detail

} // namespace numkit::stats
