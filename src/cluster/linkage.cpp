// toolboxes/stats/src/cluster/linkage.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/ref.h>
#include <numkit/stats/cluster/linkage.hpp>

#include <numkit/stats/cluster/distance.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "linkage_detail.hpp"

namespace numkit::stats {

namespace {

enum class LinkMethod {
    Single, Complete, Average, Weighted, Centroid, Median, Ward
};

LinkMethod parse_link(keel::StringRef s) {
    if (s == "single")   return LinkMethod::Single;
    if (s == "complete") return LinkMethod::Complete;
    if (s == "average")  return LinkMethod::Average;
    if (s == "weighted") return LinkMethod::Weighted;
    if (s == "centroid") return LinkMethod::Centroid;
    if (s == "median")   return LinkMethod::Median;
    if (s == "ward")     return LinkMethod::Ward;
    return LinkMethod::Single;
}

// Read pdist row vector into a square matrix D (size N×N).
// Y has length N(N-1)/2; D is size N (column-major).
keel::usize solve_N_from_pdist(keel::usize k) {
    const keel::f64 Nf = (1.0 + std::sqrt(1.0 + 8.0 * keel::f64(k))) / 2.0;
    return (keel::usize)std::round(Nf);
}

// Lance-Williams update: d(c, k) where c = merge(a, b).
inline keel::f64 lance_williams(LinkMethod m, keel::f64 d_ak, keel::f64 d_bk, keel::f64 d_ab,
                             int n_a, int n_b, int n_k) {
    switch (m) {
        case LinkMethod::Single:   return std::min(d_ak, d_bk);
        case LinkMethod::Complete: return std::max(d_ak, d_bk);
        case LinkMethod::Average:
            return (n_a * d_ak + n_b * d_bk) / keel::f64(n_a + n_b);
        case LinkMethod::Weighted: return 0.5 * (d_ak + d_bk);
        case LinkMethod::Centroid: {
            const keel::f64 na = n_a, nb = n_b;
            return std::sqrt(
                (na * d_ak * d_ak + nb * d_bk * d_bk) / (na + nb)
                - na * nb * d_ab * d_ab / ((na + nb) * (na + nb)));
        }
        case LinkMethod::Median:
            return std::sqrt(0.5 * d_ak * d_ak + 0.5 * d_bk * d_bk
                             - 0.25 * d_ab * d_ab);
        case LinkMethod::Ward: {
            const keel::f64 na = n_a, nb = n_b, nk = n_k;
            const keel::f64 tot = na + nb + nk;
            return std::sqrt(
                ((na + nk) * d_ak * d_ak + (nb + nk) * d_bk * d_bk
                 - nk * d_ab * d_ab) / tot);
        }
    }
    return 0.0;
}

} // anonymous

Value linkage(const Value &Y, keel::StringRef method, keel::StringRef metric, keel::f64 p, keel::Allocator *mr)
{
    const LinkMethod m = parse_link(method);
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Determine N. Y is either pdist row (1×n*(n-1)/2) or NxD raw data.
    const keel::usize Yrows = Y.dims().rows();
    const keel::usize Ycols = Y.dims().cols();
    keel::usize N;
    keel::Array<keel::f64> D(scratch.allocator());  // upper-tri row-major

    if (Yrows == 1 || Ycols == 1) {
        const keel::usize n = Y.numel();
        N = solve_N_from_pdist(n);
        if (N * (N - 1) / 2 != n)
            throw Error("linkage: pdist input has invalid length",
                        0, 0, "linkage", "", "numkit:linkage:size");
        D.resize(n);
        for (keel::usize i = 0; i < n; ++i) D[i] = Y.elemAsDouble(i);
    } else {
        // Treat as raw data: compute pdist with the requested metric.
        Value Yp = pdist(Y, metric, p, mr);
        const keel::usize n = Yp.numel();
        N = Yrows;
        if (N * (N - 1) / 2 != n)
            throw Error("linkage: bad pdist intermediate", 0, 0, "linkage", "",
                        "numkit:linkage:internal");
        D.resize(n);
        for (keel::usize i = 0; i < n; ++i) D[i] = Yp.elemAsDouble(i);
    }

    if (N < 2) return Value::matrix(0, 3, ValueType::DOUBLE, mr);

    // Build a dense N×N distance matrix as flat row-major buffer.
    auto pdist_idx = [&](keel::usize i, keel::usize j) {
        if (i > j) std::swap(i, j);
        // pdist convention: index of pair (i, j) i<j among rows of an N×N
        // matrix in column-major order matching MATLAB's pdist:
        //   k = (i-1)*(N - i/2) + (j - i) - 1   (1-indexed)
        // 0-based: k = i*(N - 1) - i*(i+1)/2 + (j - i - 1)
        return i * N - (i * (i + 1)) / 2 + (j - i - 1);
    };

    // dm is N×N flat row-major; we keep it that way and just track an
    // active-row mask + count to avoid per-step erase shuffles.
    keel::Array<keel::f64> dm(N * N, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        for (keel::usize j = i + 1; j < N; ++j) {
            const keel::f64 d = D[pdist_idx(i, j)];
            dm[i * N + j] = d;
            dm[j * N + i] = d;
        }

    // Active cluster set.
    keel::Array<int> ids(N, scratch.allocator());
    keel::Array<int> sizes(N, 1, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) ids[i] = (int)i;  // 0-based; emit 1-based later

    keel::Array<keel::f64> Zflat(scratch.allocator());
    Zflat.reserve(3 * (N - 1));
    keel::Array<keel::f64> newrow(N, scratch.allocator());  // re-used per step

    int next_id = (int)N;
    for (keel::usize step = 0; step + 1 < N; ++step) {
        // Find smallest distance among active rows. On ties, prefer the
        // pair with the largest (i, j) lex-order — MATLAB R2025b's
        // linkage() picks ties this way (verified by probe), so we use
        // `<=` instead of `<` to keep the last-scanned minimum.
        const keel::usize M = ids.size();
        keel::usize bi = 0, bj = 1;
        keel::f64 bd = std::numeric_limits<keel::f64>::infinity();
        for (keel::usize i = 0; i < M; ++i)
            for (keel::usize j = i + 1; j < M; ++j)
                if (dm[i * N + j] <= bd) { bd = dm[i * N + j]; bi = i; bj = j; }

        // Record merge: lower id first (per MATLAB convention).
        int a_id = ids[bi];
        int b_id = ids[bj];
        if (a_id > b_id) std::swap(a_id, b_id);
        Zflat.push_back(keel::f64(a_id + 1));
        Zflat.push_back(keel::f64(b_id + 1));
        Zflat.push_back(bd);

        // Build new row: distances from new cluster to every other.
        const int n_a = sizes[bi], n_b = sizes[bj];
        for (keel::usize k = 0; k < M; ++k) {
            if (k == bi || k == bj) { newrow[k] = 0.0; continue; }
            const keel::f64 d_ak = dm[bi * N + k];
            const keel::f64 d_bk = dm[bj * N + k];
            newrow[k] = lance_williams(m, d_ak, d_bk, bd, n_a, n_b, sizes[k]);
        }

        // Replace bi with new cluster and remove bj.
        ids[bi] = next_id++;
        sizes[bi] = n_a + n_b;
        for (keel::usize k = 0; k < M; ++k) {
            dm[bi * N + k] = newrow[k];
            dm[k * N + bi] = newrow[k];
        }
        // Erase bj from ids/sizes and compact row/col bj of dm.
        ids.erase(bj);
        sizes.erase(bj);
        // Shift columns left for rows < M, then shift rows up.
        for (keel::usize r = 0; r < M; ++r)
            for (keel::usize c = bj; c + 1 < M; ++c)
                dm[r * N + c] = dm[r * N + c + 1];
        for (keel::usize r = bj; r + 1 < M; ++r)
            for (keel::usize c = 0; c < M - 1; ++c)
                dm[r * N + c] = dm[(r + 1) * N + c];
    }

    // Pack as (N-1)×3 column-major.
    const keel::usize rows = N - 1;
    Value Z = Value::matrix(rows, 3, ValueType::DOUBLE, mr);
    keel::f64 *zd = Z.doubleDataMut();
    for (keel::usize i = 0; i < rows; ++i) {
        zd[0 * rows + i] = Zflat[3 * i + 0];
        zd[1 * rows + i] = Zflat[3 * i + 1];
        zd[2 * rows + i] = Zflat[3 * i + 2];
    }
    return Z;
}

// Backward-compat 2-arg wrapper: defaults to euclidean.
Value linkage(const Value &Y, keel::StringRef method, keel::Allocator *mr)
{
    return linkage(Y, method, "euclidean", 2.0, mr);
}

// ════════════════════════════════════════════════════════════════════
// cluster — flatten a linkage tree
// ════════════════════════════════════════════════════════════════════

// Forward decl for inconsistency cutoff.
Value inconsistent(const Value &Z, int depth, keel::Allocator *mr);

namespace {

void cluster_from_linkage_argcheck(const Value &Z)
{
    if (Z.dims().cols() != 3)
        throw Error("cluster: Z must be (N-1)×3", 0, 0, "cluster", "",
                    "numkit:cluster:size");
}

void cophenet_full_argcheck(const Value &Z, const Value &Y)
{
    const keel::usize M = Z.dims().rows();
    const keel::usize N = M + 1;
    if (Z.dims().cols() != 3)
        throw Error("cophenet: Z must be (N-1)×3", 0, 0, "cophenet", "",
                    "numkit:cophenet:size");
    if (N * (N - 1) / 2 != Y.numel())
        throw Error("cophenet: Y / Z size mismatch", 0, 0, "cophenet", "",
                    "numkit:cophenet:size");
}

void inconsistent_argcheck(const Value &Z)
{
    if (Z.dims().cols() != 3)
        throw Error("inconsistent: Z must be (N-1)×3", 0, 0, "inconsistent", "",
                    "numkit:inconsistent:size");
}

} // namespace

Value cluster_from_linkage(const Value &Z, int maxclust, keel::f64 cutoff, keel::StringRef criterion, int depth, keel::Allocator *mr)
{
    cluster_from_linkage_argcheck(Z);
    const keel::usize M = Z.dims().rows();
    const keel::usize N = M + 1;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<int>    a(M, scratch.allocator()), b(M, scratch.allocator());
    keel::Array<keel::f64> d(M, scratch.allocator());
    for (keel::usize i = 0; i < M; ++i) {
        a[i] = (int)Z.doubleData()[0 * M + i] - 1;
        b[i] = (int)Z.doubleData()[1 * M + i] - 1;
        d[i] = Z.doubleData()[2 * M + i];
    }

    // Determine cluster assignment via tree-walk for the inconsistency
    // criterion (MATLAB default for 'cutoff'); use distance threshold for
    // 'distance'; use prefix merges for maxclust.
    keel::Array<int> labels(N, 0, scratch.allocator());

    if (maxclust > 0 && maxclust >= 1 && (keel::usize)maxclust <= N) {
        const int merges = (int)N - maxclust;
        // Union-find.
        keel::Array<int> parent(N + M, scratch.allocator());
        for (keel::usize i = 0; i < parent.size(); ++i) parent[i] = (int)i;
        auto find = [&](int x) -> int {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        for (int i = 0; i < merges; ++i) {
            const int newid = (int)N + i;
            int x = find(a[i]), y = find(newid);
            if (x != y) parent[x] = y;
            x = find(b[i]); y = find(newid);
            if (x != y) parent[x] = y;
        }
        keel::Array<int> root2lab(scratch.allocator());
        for (keel::usize i = 0; i < N; ++i) {
            const int r = find((int)i);
            int lab = -1;
            for (keel::usize j = 0; j < root2lab.size(); ++j)
                if (root2lab[j] == r) { lab = (int)j; break; }
            if (lab < 0) { lab = (int)root2lab.size(); root2lab.push_back(r); }
            labels[i] = lab + 1;
        }
    } else if (cutoff > 0.0 && criterion == "distance") {
        // A non-leaf node id (= N + i) is "kept" iff its merge distance d[i]
        // is <= cutoff. Walk top-down: at each node, if kept then collect
        // all leaves under it into one cluster; else recurse into children.
        int next_label = 0;
        auto assign = [&](auto &&self, int id, int lab) -> void {
            if ((keel::usize)id < N) { labels[(keel::usize)id] = lab; return; }
            self(self, a[(keel::usize)id - N], lab);
            self(self, b[(keel::usize)id - N], lab);
        };
        auto walk = [&](auto &&self, int id) -> void {
            if ((keel::usize)id < N) {
                labels[(keel::usize)id] = ++next_label;
                return;
            }
            const keel::usize i = (keel::usize)id - N;
            if (d[i] <= cutoff) {
                ++next_label;
                assign(assign, id, next_label);
            } else {
                self(self, a[i]);
                self(self, b[i]);
            }
        };
        walk(walk, (int)N + (int)M - 1);
    } else if (cutoff > 0.0) {
        // Inconsistency criterion (MATLAB default).
        Value Yinc = inconsistent(Z, depth, mr);
        const keel::f64 *yi = Yinc.doubleData();
        // inc per non-leaf node: column 4 (= 3 in 0-based), row layout col-major M×4.
        keel::Array<keel::f64> inc(M, scratch.allocator());
        for (keel::usize i = 0; i < M; ++i) inc[i] = yi[3 * M + i];
        int next_label = 0;
        auto assign = [&](auto &&self, int id, int lab) -> void {
            if ((keel::usize)id < N) { labels[(keel::usize)id] = lab; return; }
            self(self, a[(keel::usize)id - N], lab);
            self(self, b[(keel::usize)id - N], lab);
        };
        auto walk = [&](auto &&self, int id) -> void {
            if ((keel::usize)id < N) {
                labels[(keel::usize)id] = ++next_label;
                return;
            }
            const keel::usize i = (keel::usize)id - N;
            if (inc[i] <= cutoff) {
                ++next_label;
                assign(assign, id, next_label);
            } else {
                self(self, a[i]);
                self(self, b[i]);
            }
        };
        walk(walk, (int)N + (int)M - 1);
    } else {
        // No criterion — every leaf is its own cluster.
        for (keel::usize i = 0; i < N; ++i) labels[i] = (int)i + 1;
    }

    // Compact labels to 1..K in first-encountered order.
    keel::Array<int> remap(N + 1, 0, scratch.allocator());
    int next = 0;
    keel::Array<int> compact(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        if (remap[(keel::usize)labels[i]] == 0) remap[(keel::usize)labels[i]] = ++next;
        compact[i] = remap[(keel::usize)labels[i]];
    }

    Value out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = keel::f64(compact[i]);
    return out;
}

// Backward-compat wrapper without depth.
Value cluster_from_linkage(const Value &Z, int maxclust, keel::f64 cutoff, keel::StringRef criterion, keel::Allocator *mr)
{
    return cluster_from_linkage(Z, maxclust, cutoff, criterion, 2, mr);
}

Value clusterdata(const Value &X, int maxclust, keel::f64 cutoff, keel::StringRef linkage_method, keel::StringRef criterion, int depth, keel::StringRef distance_metric, keel::f64 p, keel::Allocator *mr)
{
    Value Y = pdist(X, distance_metric, p, mr);
    Value Z = linkage(Y, linkage_method, mr);
    return cluster_from_linkage(Z, maxclust, cutoff, criterion, depth, mr);
}

Value clusterdata(const Value &X, int maxclust, keel::f64 cutoff, keel::StringRef linkage_method, keel::StringRef criterion, int depth, keel::Allocator *mr)
{
    return clusterdata(X, maxclust, cutoff, linkage_method, criterion, depth, "euclidean", 2.0, mr);
}

Value clusterdata(const Value &X, int maxclust, keel::f64 cutoff, keel::StringRef linkage_method, keel::StringRef criterion, keel::Allocator *mr)
{
    return clusterdata(X, maxclust, cutoff, linkage_method, criterion, 2, mr);
}

// ════════════════════════════════════════════════════════════════════
// cophenet — cophenetic correlation coefficient
// ════════════════════════════════════════════════════════════════════

// Returns (correlation, dcoph). The caller uses the scalar c for the
// 1-output case or both for the 2-output case.
//
// Bug fix 2026-05-08: previously returned only the scalar correlation;
// MATLAB's `[c, d] = cophenet(Z, Y)` 2-output form was throwing
// "Undefined function or variable 'd'". Now both outputs are produced.
std::tuple<Value, Value> cophenet_full(const Value &Z, const Value &Y, keel::Allocator *mr)
{
    cophenet_full_argcheck(Z, Y);
    const keel::usize M = Z.dims().rows();
    const keel::usize N = M + 1;
    const keel::usize Yn = Y.numel();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<int>    a(M, scratch.allocator()), b(M, scratch.allocator());
    keel::Array<keel::f64> d(M, scratch.allocator());
    for (keel::usize i = 0; i < M; ++i) {
        a[i] = (int)Z.doubleData()[0 * M + i] - 1;
        b[i] = (int)Z.doubleData()[1 * M + i] - 1;
        d[i] = Z.doubleData()[2 * M + i];
    }

    // Walk the merge tree to find the merge distance for every original
    // pair (i, j). Build cluster-membership lists: members[k] holds the
    // 0..N-1 indices in cluster k (0..N-1 are original; N..N+M-1 are merges).
    // The outer container needs uses-allocator construction; plain
    // std::pmr::vector<int> satisfies that (ScratchVec deletes copy ctor
    // and breaks the trait — use plain pmr::vector for nested cases).
    keel::Array<keel::Array<int>> members(scratch.allocator());
    members.reserve(N + M);
    for (keel::usize i = 0; i < N + M; ++i) members.emplace_back(scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) members[i].push_back((int)i);
    keel::Array<keel::f64> dcoph(Yn, 0.0, scratch.allocator());

    auto pdist_idx = [&](keel::usize i, keel::usize j) {
        if (i > j) std::swap(i, j);
        return i * N - (i * (i + 1)) / 2 + (j - i - 1);
    };

    for (keel::usize i = 0; i < M; ++i) {
        const int newid = (int)N + (int)i;
        const auto &lhs = members[a[i]];
        const auto &rhs = members[b[i]];
        for (int x : lhs)
            for (int y : rhs)
                dcoph[pdist_idx(x, y)] = d[i];
        members[newid].append(lhs);
        members[newid].append(rhs);
    }

    // Pearson correlation between Y and dcoph.
    keel::f64 mx = 0, my = 0;
    for (keel::usize i = 0; i < Yn; ++i) { mx += Y.elemAsDouble(i); my += dcoph[i]; }
    mx /= Yn; my /= Yn;
    keel::f64 sxy = 0, sx2 = 0, sy2 = 0;
    for (keel::usize i = 0; i < Yn; ++i) {
        const keel::f64 dx = Y.elemAsDouble(i) - mx;
        const keel::f64 dy = dcoph[i] - my;
        sxy += dx * dy; sx2 += dx * dx; sy2 += dy * dy;
    }
    const keel::f64 denom = std::sqrt(sx2) * std::sqrt(sy2);
    Value cv = Value::scalar(denom > 0.0 ? sxy / denom : 0.0, mr);

    // Build the 1×Yn cophenetic-distance row vector.
    Value dv = Value::matrix(1, Yn, ValueType::DOUBLE, mr);
    keel::f64 *dd = dv.doubleDataMut();
    for (keel::usize i = 0; i < Yn; ++i) dd[i] = dcoph[i];
    return {std::move(cv), std::move(dv)};
}

Value cophenet(const Value &Z, const Value &Y, keel::Allocator *mr)
{
    auto [c, _d] = cophenet_full(Z, Y, mr);
    return c;
}

// ════════════════════════════════════════════════════════════════════
// inconsistent — inconsistency coefficient
// ════════════════════════════════════════════════════════════════════

Value inconsistent(const Value &Z, int depth, keel::Allocator *mr) {
    inconsistent_argcheck(Z);
    const keel::usize M = Z.dims().rows();
    if (depth <= 0) depth = 2;
    const keel::usize N = M + 1;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<int>    a(M, scratch.allocator()), b(M, scratch.allocator());
    keel::Array<keel::f64> d(M, scratch.allocator());
    for (keel::usize i = 0; i < M; ++i) {
        a[i] = (int)Z.doubleData()[0 * M + i] - 1;
        b[i] = (int)Z.doubleData()[1 * M + i] - 1;
        d[i] = Z.doubleData()[2 * M + i];
    }

    auto child_dist = [&](int id) -> keel::f64 {
        if ((keel::usize)id < N) return -1.0;  // leaf
        return d[(keel::usize)id - N];
    };

    // For each merge node, walk down `depth` levels, collecting all link
    // distances; compute mean, std, count, inconsistency (= (d - mean)/std).
    auto collect = [&](auto &&self, int id, int rem, keel::Array<keel::f64> &acc) -> void {
        if ((keel::usize)id < N) return;
        if (rem == 0) return;
        acc.push_back(d[(keel::usize)id - N]);
        self(self, a[(keel::usize)id - N], rem - 1, acc);
        self(self, b[(keel::usize)id - N], rem - 1, acc);
    };

    Value out = Value::matrix(M, 4, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    for (keel::usize i = 0; i < M; ++i) {
        keel::Array<keel::f64> ds(scratch.allocator());
        collect(collect, (int)N + (int)i, depth, ds);
        const keel::f64 n = keel::f64(ds.size());
        keel::f64 mu = 0.0; for (keel::f64 v : ds) mu += v; mu /= n;
        keel::f64 var = 0.0; for (keel::f64 v : ds) var += (v - mu) * (v - mu);
        const keel::f64 sd = (n > 1.0) ? std::sqrt(var / (n - 1.0)) : 0.0;
        const keel::f64 inc = (sd > 0.0) ? ((d[i] - mu) / sd) : 0.0;
        od[0 * M + i] = mu;
        od[1 * M + i] = sd;
        od[2 * M + i] = n;
        od[3 * M + i] = inc;
    }
    (void)child_dist;
    return out;
}

} // namespace numkit::stats
