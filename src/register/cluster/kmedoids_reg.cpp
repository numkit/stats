// toolboxes/signal/src/cluster/kmedoids_reg.cpp
//
// CallContext register half of cluster/kmedoids.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/cluster/kmedoids.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "cluster/kmedoids_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void kmedoids_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "kmedoids");  // no max: name-value tail in the reference
    const int K = (int)args[1].toScalar();
    int max_iter  = 100;
    int replicates = 1;
    // MATLAB R2025b kmedoids default 'Distance' is 'sqeuclidean'.
    keel::String metric    = keel::String::from("sqeuclidean");
    keel::String algorithm = keel::String::from("pam");
    keel::String start     = keel::String::from("plus");
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        for (auto &c : s) c = keel::to_ascii_lower(c);
        return s;
    };
    for (keel::usize i = 2; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        const keel::String key = lower(args[i].toString());
        const Value &v = args[i + 1];
        if      (key == "maxiter")    max_iter   = (int)v.toScalar();
        else if (key == "replicates") replicates = (int)v.toScalar();
        else if (key == "distance")   metric     = lower(v.toString());
        else if (key == "algorithm")  algorithm  = lower(v.toString());
        else if (key == "start")      start      = lower(v.toString());
        // 'OnlinePhase' / 'Options' / 'PercentNeighbors' silently accepted.
    }
    auto R = kmedoids_full(ctx.engine->rng(), args[0], K, max_iter, replicates, metric, ctx.engine->allocator());
    outs[0] = std::move(R.idx);
    if (nargout > 1) outs[1] = std::move(R.C);
    if (nargout > 2) outs[2] = std::move(R.sumd);
    if (nargout > 3) outs[3] = std::move(R.D);
    if (nargout > 4) outs[4] = std::move(R.midx);
    if (nargout > 5) {
        keel::Allocator *mr = ctx.engine->allocator();
        Value info = Value::structure(mr);
        info.field("algorithm")     = Value::fromString(algorithm, mr);
        info.field("start")         = Value::fromString(start, mr);
        info.field("distance")      = Value::fromString(metric, mr);
        info.field("iterations")    = Value::scalar(double(R.iters), mr);
        info.field("bestReplicate") = Value::scalar(double(R.best_rep), mr);
        outs[5] = std::move(info);
    }
}

void dbscan_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "dbscan");  // no max: name-value tail in the reference
    const double eps    = args[1].toScalar();
    const int    minpts = (int)args[2].toScalar();
    keel::String metric  = keel::String::from("euclidean");
    double p = 2.0;
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        for (auto &c : s) c = keel::to_ascii_lower(c);
        return s;
    };
    // MATLAB requires N-V form; we also tolerate a 4th positional metric
    // string for backward compat with prior nk usage.
    keel::usize i = 3;
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        keel::String s = lower(args[i].toString());
        if (s != "distance" && s != "p" && s != "cov" && s != "scale") {
            metric = std::move(s);
            ++i;
        }
    }
    for (; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        const keel::String key = lower(args[i].toString());
        const Value &val = args[i + 1];
        if      (key == "distance") metric = lower(val.toString());
        else if (key == "p")        p      = val.toScalar();
        // 'Cov' and 'Scale' silently accepted but ignored — only matter
        // for mahalanobis / seuclidean which dbscan() doesn't yet wire up.
    }
    auto [idx, core] = dbscan(args[0], eps, minpts, metric, p, ctx.engine->allocator());
    outs[0] = std::move(idx);
    if (nargout > 1) outs[1] = std::move(core);
}

} // namespace detail

} // namespace numkit::stats
