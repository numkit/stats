// toolboxes/signal/src/lda/lda_reg.cpp
//
// CallContext register half of lda/lda.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/lda/lda.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>


namespace numkit::stats {

namespace detail {

void classify_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "classify");
    keel::String type;
    if (args.size() >= 4 && (args[3].isChar() || args[3].isString()))
        type = args[3].toString();
    auto [c, err, post, logp] = classify(args[0], args[1], args[2], type, ctx.engine->allocator());
    outs[0] = keel::move(c);
    if (nargout > 1) outs[1] = keel::move(err);
    if (nargout > 2) outs[2] = keel::move(post);
    if (nargout > 3) outs[3] = keel::move(logp);
}

} // namespace detail

} // namespace numkit::stats
