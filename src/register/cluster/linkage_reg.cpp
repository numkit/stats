// toolboxes/signal/src/cluster/linkage_reg.cpp
//
// CallContext register half of cluster/linkage.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/cluster/distance.hpp>
#include <numkit/stats/cluster/linkage.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include "cluster/linkage_detail.hpp"
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

void linkage_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "linkage");  // no max: method/metric tail in the reference
    // Marshalling only — the METHOD/metric/p contract and its value
    // classes live in linkage_full.
    outs[0] = linkage_full(args[0],
                           Span<const Value>(args.data() + 1, args.size() - 1),
                           ctx.engine->resource());
}

void cluster_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "cluster");  // no max: name-value tail in the reference
    // Marshalling only — the MaxClust/Cutoff/Criterion/Depth contract
    // lives in cluster_full.
    outs[0] = cluster_full(args[0],
                           Span<const Value>(args.data() + 1, args.size() - 1),
                           ctx.engine->resource());
}

void inconsistent_reg(Span<const Value> args, size_t /*nargout*/,
                      Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "inconsistent");
    // Marshalling only — the depth gate lives in inconsistent_full.
    outs[0] = inconsistent_full(args[0],
                                Span<const Value>(args.data() + 1, args.size() - 1),
                                ctx.engine->resource());
}

void clusterdata_reg(Span<const Value> args, size_t /*nargout*/,
                     Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "clusterdata");  // no max: name-value tail in the reference
    int maxclust = -1;
    double cutoff = -1.0;
    int depth = 2;
    double p = 2.0;
    std::string method     = "single";
    std::string criterion  = "inconsistent";
    std::string distance_metric = "euclidean";
    auto lower = [](std::string s) {
        for (auto &c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    // MATLAB scalar shortcut: clusterdata(X, c) where c is numeric.
    //   c >= 2  → maxclust (rounded toward zero)
    //   0 < c < 2 → cutoff (inconsistency criterion)
    // Verified via R2025b probe.
    if (args.size() >= 2 && args[1].numel() == 1
        && !(args[1].isChar() || args[1].isString())) {
        const double c = args[1].toScalar();
        if (c >= 2.0) maxclust = (int)c;
        else if (c > 0.0) cutoff = c;
    }
    for (size_t i = 1; i + 1 < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            const auto s = lower(args[i].toString());
            if      (s == "maxclust")  maxclust  = (int)args[i + 1].toScalar();
            else if (s == "cutoff")    cutoff    = args[i + 1].toScalar();
            else if (s == "linkage")   method    = lower(args[i + 1].toString());
            else if (s == "criterion") criterion = lower(args[i + 1].toString());
            else if (s == "depth")     depth     = (int)args[i + 1].toScalar();
            else if (s == "distance")  distance_metric = lower(args[i + 1].toString());
            else if (s == "p")         p         = args[i + 1].toScalar();
            // 'savememory' and other doc'd N-V silently ignored.
        }
    }
    outs[0] = clusterdata(args[0], maxclust, cutoff, method, criterion, depth, distance_metric, p, ctx.engine->resource());
}

void cophenet_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "cophenet");
    auto [c, d] = cophenet_full(args[0], args[1], ctx.engine->resource());
    outs[0] = std::move(c);
    if (nargout > 1) outs[1] = std::move(d);
}

} // namespace detail

} // namespace numkit::stats
