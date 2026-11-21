// toolboxes/signal/src/cluster/linkage_reg.cpp
//
// CallContext register half of cluster/linkage.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/cluster/distance.hpp>
#include <numkit/stats/cluster/linkage.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "cluster/linkage_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void linkage_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "linkage");  // no max: method/metric tail in the reference
    // Marshalling only — the METHOD/metric/p contract and its value
    // classes live in linkage_full.
    outs[0] = linkage_full(args[0],
                           keel::ArrayRef<Value>(args.data() + 1, args.size() - 1),
                           ctx.engine->allocator());
}

void cluster_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "cluster");  // no max: name-value tail in the reference
    // Marshalling only — the MaxClust/Cutoff/Criterion/Depth contract
    // lives in cluster_full.
    outs[0] = cluster_full(args[0],
                           keel::ArrayRef<Value>(args.data() + 1, args.size() - 1),
                           ctx.engine->allocator());
}

void inconsistent_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "inconsistent");
    // Marshalling only — the depth gate lives in inconsistent_full.
    outs[0] = inconsistent_full(args[0],
                                keel::ArrayRef<Value>(args.data() + 1, args.size() - 1),
                                ctx.engine->allocator());
}

void clusterdata_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "clusterdata");  // no max: name-value tail in the reference
    keel::i32 maxclust = -1;
    keel::f64 cutoff = -1.0;
    keel::i32 depth = 2;
    keel::f64 p = 2.0;
    keel::String method     = keel::String::from("single");
    keel::String criterion  = keel::String::from("inconsistent");
    keel::String distance_metric = keel::String::from("euclidean");
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        for (auto &c : s) c = keel::to_ascii_lower(c);
        return s;
    };
    // MATLAB scalar shortcut: clusterdata(X, c) where c is numeric.
    //   c >= 2  → maxclust (rounded toward zero)
    //   0 < c < 2 → cutoff (inconsistency criterion)
    // Verified via R2025b probe.
    if (args.size() >= 2 && args[1].numel() == 1
        && !(args[1].isChar() || args[1].isString())) {
        const keel::f64 c = args[1].toScalar();
        if (c >= 2.0) maxclust = (keel::i32)c;
        else if (c > 0.0) cutoff = c;
    }
    for (keel::usize i = 1; i + 1 < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            const auto s = lower(args[i].toString());
            if      (s == "maxclust")  maxclust  = (keel::i32)args[i + 1].toScalar();
            else if (s == "cutoff")    cutoff    = args[i + 1].toScalar();
            else if (s == "linkage")   method    = lower(args[i + 1].toString());
            else if (s == "criterion") criterion = lower(args[i + 1].toString());
            else if (s == "depth")     depth     = (keel::i32)args[i + 1].toScalar();
            else if (s == "distance")  distance_metric = lower(args[i + 1].toString());
            else if (s == "p")         p         = args[i + 1].toScalar();
            // 'savememory' and other doc'd N-V silently ignored.
        }
    }
    outs[0] = clusterdata(args[0], maxclust, cutoff, method, criterion, depth, distance_metric, p, ctx.engine->allocator());
}

void cophenet_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "cophenet");
    auto [c, d] = cophenet_full(args[0], args[1], ctx.engine->allocator());
    outs[0] = std::move(c);
    if (nargout > 1) outs[1] = std::move(d);
}

} // namespace detail

} // namespace numkit::stats
