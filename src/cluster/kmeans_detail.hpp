// toolboxes/.../kmeans_detail.hpp — private compute substrate (anon-in-header,
// internal linkage per TU) shared by kmeans.cpp compute and kmeans_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/random.h>
#include <keel/stats.h>
#include <numkit/value/value.hpp>
#include <numkit/ops/matlab_mt19937.hpp>
#include <numkit/value/error.hpp>



namespace numkit { namespace ops { class RngContext; } }

namespace numkit::stats {

namespace {

inline keel::f64 sq_dist(const keel::f64 *a, const keel::f64 *b, keel::usize D) {
    keel::f64 s = 0.0;
    for (keel::usize k = 0; k < D; ++k) { keel::f64 d = a[k] - b[k]; s += d * d; }
    return s;
}

// k-means++ centroid initialisation. Returns K rows of D values each
// (flattened K*D-length buffer).
[[maybe_unused]] keel::Array<keel::f64> kmeanspp_init(const keel::Array<keel::f64> &X,
                                 keel::usize N, keel::usize D, keel::i32 K,
                                 numkit::ops::MatlabMT19937 &gen,
                                 keel::Allocator *scratch_mr)
{
    keel::Array<keel::f64> C((keel::usize)K * D, scratch_mr);
    const keel::usize i0 = keel::random::uniform_usize(gen, 0, N - 1);
    keel::copy(X.begin() + i0 * D, X.begin() + (i0 + 1) * D, C.begin());

    keel::Array<keel::f64> dist(N, keel::Limits<keel::f64>::infinity(),
                            scratch_mr);
    for (keel::i32 k = 1; k < K; ++k) {
        // Update each point's nearest-centroid distance against the new
        // centroid (k-1).
        const keel::f64 *cprev = &C[(keel::usize)(k - 1) * D];
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 d = sq_dist(&X[i * D], cprev, D);
            if (d < dist[i]) dist[i] = d;
        }
        // Sample next centroid with probability proportional to dist.
        keel::f64 total = 0.0;
        for (auto v : dist) total += v;
        if (total <= 0.0) {
            // All points coincide with existing centroids; pick uniformly.
            const keel::usize pick = keel::random::uniform_usize(gen, 0, N - 1);
            keel::copy(X.begin() + pick * D, X.begin() + (pick + 1) * D,
                      C.begin() + (keel::usize)k * D);
            continue;
        }
        const keel::f64 r = keel::random::uniform_f64(gen, 0.0, total);
        keel::f64 acc = 0.0;
        keel::usize pick = N - 1;
        for (keel::usize i = 0; i < N; ++i) {
            acc += dist[i];
            if (acc >= r) { pick = i; break; }
        }
        keel::copy(X.begin() + pick * D, X.begin() + (pick + 1) * D,
                  C.begin() + (keel::usize)k * D);
    }
    return C;
}

// Lloyd iteration. Returns (assignments, centroids, sum-of-squared-dist
// per cluster, total-WCSS). Stops when no assignment changes or when
// max_iter reached.
struct LloydResult {
    keel::Array<keel::i32>    idx;
    keel::Array<keel::f64> C;
    keel::Array<keel::f64> sumd;
    keel::f64             total{};

    LloydResult(keel::Allocator *scratch_mr)
        : idx(scratch_mr), C(scratch_mr), sumd(scratch_mr) {}
};

[[maybe_unused]] LloydResult lloyd(const keel::Array<keel::f64> &X, keel::usize N, keel::usize D, keel::i32 K,
                  keel::Array<keel::f64> &&C0, keel::i32 max_iter,
                  keel::Allocator *scratch_mr)
{
    LloydResult res(scratch_mr);
    res.idx.assign(N, 0);
    res.C    = keel::move(C0);
    res.sumd.assign(keel::as<keel::usize>(K), 0.0);
    res.total = 0.0;
    (void)keel::stats::kmeans_lloyd(
        X.as_ref(), N, D, K,
        keel::MutArrayRef<keel::f64>(res.C.data(), res.C.size()), max_iter,
        keel::MutArrayRef<keel::i32>(res.idx.data(), res.idx.size()),
        keel::MutArrayRef<keel::f64>(res.sumd.data(), res.sumd.size()),
        res.total, scratch_mr);
    return res;
}

} // anonymous


struct KmeansResult {
    Value idx;
    Value C;
    Value sumd;
    Value D;     // optional N×K
};
KmeansResult kmeans_full(::numkit::ops::RngContext &rng, const Value &X, keel::i32 K, keel::i32 max_iter, keel::i32 replicates,
                         keel::Allocator *mr);

Value evalclusters_compute(::numkit::ops::RngContext &rng,
                           const Value &X,
                           const Value &clustArg,
                           keel::StringRef criterionRaw,
                           keel::ArrayRef<keel::i32> klist,
                           keel::Allocator *mr);

} // namespace numkit::stats
