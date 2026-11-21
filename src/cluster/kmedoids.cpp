// toolboxes/stats/src/cluster/kmedoids.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/cluster/kmedoids.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/ops/rng_context.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <keel/random.h>
#include <limits>
#include <mutex>

#include "kmedoids_detail.hpp"

namespace numkit::stats {

namespace {

enum class Metric { Euclidean, SqEuclidean, Cityblock, Chebychev,
                    Minkowski, Cosine, Hamming, Jaccard, Precomputed };

Metric parse_metric(keel::StringRef s) {
    keel::String sl; sl.reserve(s.size());
    for (char c : s) sl.push_back(keel::to_ascii_lower(c));
    if (sl == "euclidean")        return Metric::Euclidean;
    if (sl == "squaredeuclidean" ||
        sl == "sqeuclidean")      return Metric::SqEuclidean;
    if (sl == "cityblock")        return Metric::Cityblock;
    if (sl == "chebychev")        return Metric::Chebychev;
    if (sl == "minkowski")        return Metric::Minkowski;
    if (sl == "cosine")           return Metric::Cosine;
    if (sl == "hamming")          return Metric::Hamming;
    if (sl == "jaccard")          return Metric::Jaccard;
    if (sl == "precomputed")      return Metric::Precomputed;
    return Metric::Euclidean;
}

inline keel::f64 dist_pair(const keel::f64 *a, const keel::f64 *b, keel::usize D, Metric m,
                        keel::f64 p = 2.0) {
    switch (m) {
        case Metric::Euclidean: {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < D; ++k) { keel::f64 d = a[k] - b[k]; s += d * d; }
            return std::sqrt(s);
        }
        case Metric::SqEuclidean: {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < D; ++k) { keel::f64 d = a[k] - b[k]; s += d * d; }
            return s;
        }
        case Metric::Cityblock: {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < D; ++k) s += std::fabs(a[k] - b[k]);
            return s;
        }
        case Metric::Chebychev: {
            keel::f64 m_ = 0.0;
            for (keel::usize k = 0; k < D; ++k) {
                keel::f64 d = std::fabs(a[k] - b[k]);
                if (d > m_) m_ = d;
            }
            return m_;
        }
        case Metric::Minkowski: {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < D; ++k) {
                keel::f64 d = std::fabs(a[k] - b[k]);
                s += std::pow(d, p);
            }
            return std::pow(s, 1.0 / p);
        }
        case Metric::Cosine: {
            keel::f64 xy = 0, xx = 0, yy = 0;
            for (keel::usize k = 0; k < D; ++k) {
                xy += a[k] * b[k];
                xx += a[k] * a[k];
                yy += b[k] * b[k];
            }
            const keel::f64 denom = std::sqrt(xx) * std::sqrt(yy);
            return (denom > 0.0) ? (1.0 - xy / denom) : 1.0;
        }
        case Metric::Hamming: {
            int diff = 0;
            for (keel::usize k = 0; k < D; ++k) if (a[k] != b[k]) ++diff;
            return keel::f64(diff) / keel::f64(D);
        }
        case Metric::Jaccard: {
            int diff = 0, considered = 0;
            for (keel::usize k = 0; k < D; ++k) {
                const bool xi = a[k] != 0.0, yi = b[k] != 0.0;
                if (xi || yi) {
                    ++considered;
                    if (xi != yi) ++diff;
                }
            }
            return considered > 0 ? keel::f64(diff) / keel::f64(considered) : 0.0;
        }
        case Metric::Precomputed:
            return 0.0;  // never used through this dispatch
    }
    return 0.0;
}

// Read X (N×D, column-major) into a flat row-major ScratchVec.
keel::Array<keel::f64> read_rows(const Value &X, keel::Allocator *scratch_mr) {
    const keel::usize N = X.dims().rows();
    const keel::usize D = X.dims().cols();
    keel::Array<keel::f64> out(N * D, scratch_mr);
    for (keel::usize r = 0; r < N; ++r)
        for (keel::usize c = 0; c < D; ++c)
            out[r * D + c] = X.elemAsDouble(c * N + r);
    return out;
}

void kmedoids_argcheck(const Value &X, int K)
{
    const keel::usize N = X.dims().rows();
    if (K < 1 || keel::as<keel::usize>(K) > N)
        throw Error("kmedoids: K must be in 1..N", 0, 0, "kmedoids", "",
                    "numkit:kmedoids:badK");
}

void dbscan_argcheck(keel::f64 eps)
{
    if (eps <= 0.0)
        throw Error("dbscan: eps must be positive",
                    0, 0, "dbscan", "", "numkit:dbscan:badeps");
}

} // anonymous

// Full kmedoids result: (idx, C, sumd, D, midx).
// `info` (the 6th output) is built by the adapter via Value::structure.

KmedoidsResult
kmedoids_full(::numkit::ops::RngContext &rng, const Value &X, int K, int max_iter, int replicates, keel::StringRef metric_name, keel::Allocator *mr)
{
    kmedoids_argcheck(X, K);
    if (max_iter <= 0)   max_iter = 100;
    if (replicates <= 0) replicates = 1;
    const Metric m = parse_metric(metric_name);

    const keel::usize N = X.dims().rows();
    const keel::usize D = X.dims().cols();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> Xv = read_rows(X, scratch.allocator());
    auto &gen = rng;

    keel::Array<int>    best_med((keel::usize)K, scratch.allocator());
    keel::Array<int>    best_idx(N, scratch.allocator());
    keel::Array<keel::f64> best_sumd((keel::usize)K, scratch.allocator());
    keel::f64 best_total = std::numeric_limits<keel::f64>::infinity();
    int    best_iters = 0;
    int    best_rep   = 0;

    keel::Array<int>    med((keel::usize)K, scratch.allocator());
    keel::Array<int>    idx(N, scratch.allocator());
    keel::Array<keel::f64> sumd((keel::usize)K, scratch.allocator());
    keel::Array<int>    members(scratch.allocator());
    members.reserve(N);

    for (int rep = 0; rep < replicates; ++rep) {
        // Random initial medoids — sample K distinct rows.
        {
            keel::Array<int> all(N, scratch.allocator());
            for (keel::usize i = 0; i < N; ++i) all[i] = (int)i;
            keel::random::shuffle(gen, keel::MutArrayRef<int>(all.data(), all.size()));
            for (int k = 0; k < K; ++k) med[k] = all[k];
        }

        int rep_iters = 0;
        for (int iter = 0; iter < max_iter; ++iter) {
            ++rep_iters;
            // Assign each point to nearest medoid.
            std::fill(sumd.begin(), sumd.end(), 0.0);
            for (keel::usize i = 0; i < N; ++i) {
                int best = 0;
                keel::f64 bd = std::numeric_limits<keel::f64>::infinity();
                for (int k = 0; k < K; ++k) {
                    const keel::f64 d = dist_pair(&Xv[i * D],
                                                &Xv[(keel::usize)med[k] * D], D, m);
                    if (d < bd) { bd = d; best = k; }
                }
                idx[i] = best;
                sumd[best] += bd;
            }

            // Update each medoid: among its members, pick the one whose
            // sum of distances to the rest is minimal.
            bool changed = false;
            for (int k = 0; k < K; ++k) {
                members.clear();
                for (keel::usize i = 0; i < N; ++i)
                    if (idx[i] == k) members.push_back((int)i);
                if (members.empty()) continue;

                int new_med = med[k];
                keel::f64 best_sum = std::numeric_limits<keel::f64>::infinity();
                for (int cand : members) {
                    keel::f64 s = 0.0;
                    for (int p : members) {
                        s += dist_pair(&Xv[(keel::usize)cand * D],
                                       &Xv[(keel::usize)p * D], D, m);
                    }
                    if (s < best_sum) { best_sum = s; new_med = cand; }
                }
                if (new_med != med[k]) { med[k] = new_med; changed = true; }
            }
            if (!changed) break;
        }

        // Re-evaluate total against final medoids.
        keel::f64 total = 0.0;
        std::fill(sumd.begin(), sumd.end(), 0.0);
        for (keel::usize i = 0; i < N; ++i) {
            const int k = idx[i];
            const keel::f64 d = dist_pair(&Xv[i * D], &Xv[(keel::usize)med[k] * D], D, m);
            sumd[k] += d;
            total += d;
        }

        if (total < best_total) {
            best_total = total;
            std::copy(med.begin(),  med.end(),  best_med.begin());
            std::copy(idx.begin(),  idx.end(),  best_idx.begin());
            std::copy(sumd.begin(), sumd.end(), best_sumd.begin());
            best_iters = rep_iters;
            best_rep   = rep + 1;  // 1-based for MATLAB parity
        }
    }

    KmedoidsResult R{};

    R.idx = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    {
        keel::f64 *ip = R.idx.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) ip[i] = keel::f64(best_idx[i] + 1);
    }

    R.C = Value::matrix(K, D, ValueType::DOUBLE, mr);
    {
        keel::f64 *mp = R.C.doubleDataMut();
        for (int k = 0; k < K; ++k)
            for (keel::usize d = 0; d < D; ++d)
                mp[d * K + k] = Xv[(keel::usize)best_med[k] * D + d];
    }

    R.sumd = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    {
        keel::f64 *sp = R.sumd.doubleDataMut();
        for (int k = 0; k < K; ++k) sp[k] = best_sumd[k];
    }

    // 4th output D — N×K distance from each point to each medoid.
    R.D = Value::matrix(N, K, ValueType::DOUBLE, mr);
    {
        keel::f64 *dp = R.D.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i)
            for (int k = 0; k < K; ++k)
                dp[k * N + i] = dist_pair(&Xv[i * D],
                                          &Xv[(keel::usize)best_med[k] * D], D, m);
    }

    // 5th output midx — K×1 row index (1-based) of each medoid in X.
    R.midx = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    {
        keel::f64 *xp = R.midx.doubleDataMut();
        for (int k = 0; k < K; ++k) xp[k] = keel::f64(best_med[k] + 1);
    }

    R.iters    = best_iters;
    R.best_rep = best_rep;
    return R;
}

// Backward-compat 3-output wrapper.
std::tuple<Value, Value, Value>
kmedoids(::numkit::ops::RngContext &rng, const Value &X, int K, int max_iter, int replicates, keel::StringRef metric_name, keel::Allocator *mr)
{
    auto R = kmedoids_full(rng, X, K, max_iter, replicates, metric_name, mr);
    return std::make_tuple(std::move(R.idx), std::move(R.C),
                           std::move(R.sumd));
}

// ════════════════════════════════════════════════════════════════════
// dbscan
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value>
dbscan(const Value &X, keel::f64 eps, int minpts, keel::StringRef metric_name, keel::f64 p, keel::Allocator *mr)
{
    dbscan_argcheck(eps);
    if (minpts <= 0) minpts = 1;
    const Metric m = parse_metric(metric_name);

    // For 'precomputed' the input is the N×N pairwise distance matrix
    // directly. Otherwise X is N×D feature matrix.
    const bool precomputed = (m == Metric::Precomputed);
    const keel::usize N = X.dims().rows();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> Xv(scratch.allocator());
    keel::Array<keel::f64> Dmat(scratch.allocator());  // N×N row-major (precomputed only)
    keel::usize D = 0;
    if (precomputed) {
        if (X.dims().cols() != N)
            throw Error("dbscan: precomputed distance matrix must be N×N",
                        0, 0, "dbscan", "", "numkit:dbscan:badprecomp");
        Dmat.assign(N * N, 0.0);
        for (keel::usize i = 0; i < N; ++i)
            for (keel::usize j = 0; j < N; ++j)
                Dmat[i * N + j] = X.elemAsDouble(j * N + i);
    } else {
        D = X.dims().cols();
        Xv = read_rows(X, scratch.allocator());
    }

    // Neighbour lookup. Returns count and writes indices into `out`.
    auto neighbours = [&](keel::usize i, keel::Array<int> &out) {
        out.clear();
        for (keel::usize j = 0; j < N; ++j) {
            if (j == i) { out.push_back((int)j); continue; }
            keel::f64 d;
            if (precomputed) d = Dmat[i * N + j];
            else             d = dist_pair(&Xv[i * D], &Xv[j * D], D, m, p);
            if (d <= eps) out.push_back((int)j);
        }
    };

    keel::Array<int>     labels(N, 0, scratch.allocator());   // 0 = unclassified
    keel::Array<keel::u8> core(N, 0, scratch.allocator());
    int cluster = 0;

    keel::Array<int> nbrs(scratch.allocator());
    keel::Array<int> qn(scratch.allocator());
    keel::Array<int> seeds(scratch.allocator());
    nbrs.reserve(64); qn.reserve(64); seeds.reserve(N);

    for (keel::usize i = 0; i < N; ++i) {
        if (labels[i] != 0) continue;
        neighbours(i, nbrs);
        if ((int)nbrs.size() < minpts) {
            labels[i] = -1;  // noise (MATLAB R2025b convention)
            continue;
        }
        ++cluster;
        labels[i] = cluster;
        core[i] = 1;

        // Expand seed set (FIFO via index).
        seeds.clear();
        seeds.append(keel::ArrayRef<int>(nbrs.data(), nbrs.size()));
        for (keel::usize s = 0; s < seeds.size(); ++s) {
            const int q = seeds[s];
            if ((keel::usize)q == i) continue;
            if (labels[q] == -1) labels[q] = cluster;  // noise → border
            if (labels[q] != 0) continue;              // already assigned
            labels[q] = cluster;
            neighbours((keel::usize)q, qn);
            if ((int)qn.size() >= minpts) {
                core[q] = 1;
                for (int n : qn) {
                    if (labels[n] == 0 || labels[n] == -1) seeds.push_back(n);
                }
            }
        }
    }

    Value idx = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *ip = idx.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) ip[i] = keel::f64(labels[i]);

    Value core_v = Value::matrix(N, 1, ValueType::LOGICAL, mr);
    keel::u8 *cp = core_v.logicalDataMut();
    for (keel::usize i = 0; i < N; ++i) cp[i] = core[i];

    return std::make_tuple(std::move(idx), std::move(core_v));
}

// Backward-compat without p.
std::tuple<Value, Value>
dbscan(const Value &X, keel::f64 eps, int minpts, keel::StringRef metric_name, keel::Allocator *mr)
{
    return dbscan(X, eps, minpts, metric_name, 2.0, mr);
}

} // namespace numkit::stats
