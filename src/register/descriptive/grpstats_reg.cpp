// toolboxes/signal/src/descriptive/grpstats_reg.cpp
//
// CallContext register half of descriptive/grpstats.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/array.h>
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void grpstats_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "grpstats");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();

    keel::Array<keel::String> fn_names;
    if (args.size() >= 3) {
        const Value &fn_arg = args[2];
        if (fn_arg.isChar() || fn_arg.isString()) {
            fn_names.push_back(fn_arg.toString());
        } else if (fn_arg.isCell()) {
            for (keel::usize i = 0; i < fn_arg.numel(); ++i) {
                const Value &el = fn_arg.cellAt(i);
                if (!(el.isChar() || el.isString()))
                    throw Error("grpstats: cell fn names must be strings",
                                0, 0, "grpstats", "",
                                "numkit:grpstats:BadCell");
                fn_names.push_back(el.toString());
            }
        } else {
            throw Error("grpstats: fn arg must be a string or cell of strings",
                        0, 0, "grpstats", "", "numkit:grpstats:BadFnArg");
        }
    }

    // With no whichstats given, MATLAB's default outputs are
    // [means, sem, counts] — emit as many as the caller requested.
    if (fn_names.empty()) {
        static const char *kDefault[] = {"mean", "sem", "numel"};
        const keel::usize k = std::min<keel::usize>(std::max<keel::usize>(nargout, 1), 3);
        for (keel::usize i = 0; i < k; ++i) fn_names.push_back(keel::String::from(kDefault[i]));
    }

    auto results = grpstats(args[0], args[1], fn_names.as_ref(), mr);
    // Distribute results across outputs (1-output gets the first;
    // multi-fn cell-of-fn fills successive outputs).
    const keel::usize emit = std::min<keel::usize>(results.size(),
                                         std::max<keel::usize>(nargout, 1));
    for (keel::usize i = 0; i < emit; ++i)
        outs[i] = std::move(results[i]);
}

} // namespace detail

} // namespace numkit::stats
