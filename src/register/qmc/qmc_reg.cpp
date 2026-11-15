// toolboxes/signal/src/qmc/qmc_reg.cpp
//
// CallContext register half of qmc/qmc.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/qmc/qmc.hpp>
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

void haltonset_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, "haltonset");  // no max: name-value tail in the reference
    const int d = args.empty() ? 1 : argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[0], 1, "d", "haltonset"), 1, "d", "haltonset");
    // MATLAB's default Skip is 0 — the trivial origin [0…0] is the first
    // returned point. 'Skip', 1 starts at index 1 = [0.5, 1/3, …].
    long long skip = 0, leap = 0;
    for (keel::usize i = 1; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) break;
        std::string name = args[i].toString();
        for (auto &c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const Value &v = args[i + 1];
        if      (name == "skip") skip = static_cast<long long>(v.toScalar());
        else if (name == "leap") leap = static_cast<long long>(v.toScalar());
    }
    outs[0] = haltonset(d, skip, leap, ctx.engine->allocator());
}

void net_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
             keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "net");  // no max: absent from the R2025b reference build
    const long long n = static_cast<long long>(args[1].toScalar());
    outs[0] = net(args[0], n, ctx.engine->allocator());
}

void scramble_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "scramble");
    std::string type = (args.size() >= 2 && (args[1].isChar() || args[1].isString()))
                     ? args[1].toString() : "RR2";
    outs[0] = scramble(args[0], type, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::stats
