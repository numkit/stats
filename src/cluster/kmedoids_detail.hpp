// toolboxes/stats/src/cluster/kmedoids_detail.hpp — private compute/register shared
// surface: the kmedoids worker result struct + its forward declaration (def in
// kmedoids.cpp, external). Phase 2b compute/register split.
#pragma once

#include <keel/memory.h>
#include <numkit/value/value.hpp>

#include <string>

namespace numkit { namespace ops { class RngContext; } }

namespace numkit::stats {

struct KmedoidsResult {
    Value idx;     // N×1
    Value C;       // K×D — coordinates of medoid points
    Value sumd;    // K×1
    Value D;       // N×K distances point-to-medoid
    Value midx;    // K×1 — 1-based row indices of medoids in X
    int   iters;   // best replicate iteration count
    int   best_rep;
};

KmedoidsResult kmedoids_full(::numkit::ops::RngContext &rng, const Value &X, int K, int max_iter, int replicates,
                             const std::string &metric_name,
                             keel::Allocator *mr);

} // namespace numkit::stats
