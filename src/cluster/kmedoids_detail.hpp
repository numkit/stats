// toolboxes/stats/src/cluster/kmedoids_detail.hpp — private compute/register shared
// surface: the kmedoids worker result struct + its forward declaration (def in
// kmedoids.cpp, external). Phase 2b compute/register split.
#pragma once

#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

namespace numkit { namespace ops { class RngContext; } }

namespace numkit::stats {

struct KmedoidsResult {
    Value idx;     // N×1
    Value C;       // K×D — coordinates of medoid points
    Value sumd;    // K×1
    Value D;       // N×K distances point-to-medoid
    Value midx;    // K×1 — 1-based row indices of medoids in X
    keel::i32   iters;   // best replicate iteration count
    keel::i32   best_rep;
};

KmedoidsResult kmedoids_full(::numkit::ops::RngContext &rng, const Value &X, keel::i32 K, keel::i32 max_iter, keel::i32 replicates,
                             keel::StringRef metric_name,
                             keel::Allocator *mr);

} // namespace numkit::stats
