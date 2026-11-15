// toolboxes/signal/src/sampling/lhs_reg.cpp
//
// CallContext register half of sampling/lhs.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/distributions/normal.hpp>
#include <numkit/stats/sampling/lhs.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::stats {

namespace detail {

void lhsdesign_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "lhsdesign");  // no max: name-value tail in the reference
    const keel::usize n = argcheck::arrayDim(args[0], "lhsdesign");
    const keel::usize p = argcheck::arrayDim(args[1], "lhsdesign");
    bool smooth = true;
    LhsCriterion crit = LhsCriterion::Maximin;
    keel::usize iters = 5;
    // Parse name-value pairs from args[2..].
    for (keel::usize k = 2; k + 1 < args.size(); k += 2) {
        if (!args[k].isChar() && !args[k].isString())
            throw Error("lhsdesign: name-value arguments expected",
                        0, 0, "lhsdesign", "", "numkit:lhsdesign:badNameValue");
        std::string name = args[k].toString();
        for (auto &c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (name == "smooth") {
            std::string v = args[k + 1].toString();
            for (auto &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            smooth = (v != "off");
        } else if (name == "criterion") {
            static constexpr LhsCriterion kCrit[] = {LhsCriterion::None,
                                                     LhsCriterion::Maximin,
                                                     LhsCriterion::Correlation};
            crit = kCrit[argcheck::expectedChoice(
                args[k + 1], {"none", "maximin", "correlation"},
                static_cast<int>(k) + 3, "criterion", "lhsdesign")];
        } else if (name == "iterations") {
            iters = argcheck::roundLength(args[k + 1], static_cast<int>(k) + 3, "Iterations", "lhsdesign");
            if (iters < 1) iters = 1;
        } else {
            throw Error("lhsdesign: unknown option '" + name + "'",
                        0, 0, "lhsdesign", "", "numkit:lhsdesign:badOption");
        }
    }
    outs[0] = lhsdesign(ctx.engine->rng(), n, p, smooth, crit, iters, ctx.engine->allocator());
}

void lhsnorm_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "lhsnorm");
    const keel::usize n = argcheck::arrayDim(args[2], "lhsnorm");
    outs[0] = lhsnorm(ctx.engine->rng(), args[0], args[1], n, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::stats
