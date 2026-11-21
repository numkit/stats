// toolboxes/signal/src/qmc/qmc_reg.cpp
//
// CallContext register half of qmc/qmc.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/qmc/qmc.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void haltonset_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, "haltonset");  // no max: name-value tail in the reference
    const keel::i32 d = args.empty() ? 1 : argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[0], 1, "d", "haltonset"), 1, "d", "haltonset");
    // MATLAB's default Skip is 0 — the trivial origin [0…0] is the first
    // returned point. 'Skip', 1 starts at index 1 = [0.5, 1/3, …].
    keel::i64 skip = 0, leap = 0;
    for (keel::usize i = 1; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) break;
        keel::String name = args[i].toString();
        for (auto &c : name) c = keel::to_ascii_lower(c);
        const Value &v = args[i + 1];
        if      (name == "skip") skip = keel::as<keel::i64>(v.toScalar());
        else if (name == "leap") leap = keel::as<keel::i64>(v.toScalar());
    }
    outs[0] = haltonset(d, skip, leap, ctx.engine->allocator());
}

void net_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
             keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "net");  // no max: absent from the R2025b reference build
    const keel::i64 n = keel::as<keel::i64>(args[1].toScalar());
    outs[0] = net(args[0], n, ctx.engine->allocator());
}

void scramble_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "scramble");
    keel::String type = (args.size() >= 2 && (args[1].isChar() || args[1].isString()))
                      ? args[1].toString() : keel::String::from("RR2");
    outs[0] = scramble(args[0], type.as_ref(), ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::stats
