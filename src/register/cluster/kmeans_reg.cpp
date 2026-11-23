// toolboxes/signal/src/cluster/kmeans_reg.cpp
//
// CallContext register half of cluster/kmeans.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/cluster/kmeans.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "cluster/kmeans_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void kmeans_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "kmeans");  // no max: name-value tail in the reference
    const keel::i32 K = (keel::i32)args[1].toScalar();
    keel::i32 max_iter  = 100;
    keel::i32 replicates = 1;
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        s.to_ascii_lower_in_place();
        return s;
    };
    for (keel::usize i = 2; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        const keel::String key = lower(args[i].toString());
        const Value &v = args[i + 1];
        if      (key == "maxiter")    max_iter   = (keel::i32)v.toScalar();
        else if (key == "replicates") replicates = (keel::i32)v.toScalar();
        else if (key == "distance") {
            const keel::String dn = lower(v.toString());
            if (dn != "sqeuclidean" && dn != "squaredeuclidean")
                throw Error(keel::format("kmeans: only Distance = 'sqeuclidean' is "
                            "implemented (got '{}')", dn),
                            0, 0, "kmeans", "", "numkit:kmeans:distance");
        }
        else if (key == "start") {
            const keel::String sn = lower(v.toString());
            if (sn != "plus")
                throw Error(keel::format("kmeans: only Start = 'plus' is implemented "
                            "(got '{}')", sn),
                            0, 0, "kmeans", "", "numkit:kmeans:start");
        }
        // 'Display' / 'EmptyAction' / 'OnlinePhase' / 'Options' silently
        // accepted (no-op).
    }
    auto R = kmeans_full(ctx.engine->rng(), args[0], K, max_iter, replicates, ctx.engine->allocator());
    outs[0] = std::move(R.idx);
    if (nargout > 1) outs[1] = std::move(R.C);
    if (nargout > 2) outs[2] = std::move(R.sumd);
    if (nargout > 3) outs[3] = std::move(R.D);
}

} // namespace detail

} // namespace numkit::stats
