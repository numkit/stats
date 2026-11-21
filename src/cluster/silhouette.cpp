// toolboxes/stats/src/cluster/silhouette.cpp
//
// silhouette(X, clust [, metric, p]) — clustering quality metric.
// Built on top of stats::pdist2 for the distance matrix; the rest
// is a per-point average + per-cluster aggregation.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/hash_map.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/stats/cluster/silhouette.hpp>
#include <numkit/stats/cluster/distance.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::stats {

namespace {

void silhouette_argcheck(const Value &X, const Value &clust)
{
    if (X.dims().rows() > 0 && clust.numel() != X.dims().rows())
        throw Error("silhouette: clust length must equal size(X, 1)",
                    0, 0, "silhouette", "", "numkit:silhouette:size");
}

} // namespace

Value silhouette(const Value &X, const Value &clust, keel::StringRef metric, keel::f64 p, keel::Allocator *mr)
{
    silhouette_argcheck(X, clust);
    const auto &dx = X.dims();
    const keel::usize N = dx.rows();
    if (N == 0)
        return Value::matrix(0, 1, ValueType::DOUBLE, mr);

    // Pull labels into int vector. The reference engine allows numeric or char/string —
    // we accept anything that elemAsDouble can read (round to int).
    keel::Array<int> lbl(N);
    for (keel::usize i = 0; i < N; ++i)
        lbl[i] = keel::as<int>(std::lround(clust.elemAsDouble(i)));

    // Catalogue labels and their member indices.
    keel::HashMap<int, keel::Array<keel::usize>> groups;
    groups.reserve(N);
    for (keel::usize i = 0; i < N; ++i)
        groups.try_insert(lbl[i], keel::Array<keel::usize>{}).value->push_back(i);

    // N×N distance matrix from pdist2.
    Value D = pdist2(X, X, metric, p, mr);
    if (D.dims().rows() != N || D.dims().cols() != N)
        throw Error("silhouette: pdist2 returned unexpected shape",
                    0, 0, "silhouette", "", "numkit:silhouette:internal");
    const keel::f64 *dd = D.doubleData();

    Value out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    for (keel::usize i = 0; i < N; ++i) {
        const int k = lbl[i];
        const auto &mine = *groups.get(k);

        // a(i): mean distance to other points in same cluster.
        keel::f64 a = 0.0;
        if (mine.size() <= 1) {
            // Singleton: s(i) = 0 by MATLAB convention.
            od[i] = 0.0;
            continue;
        }
        for (keel::usize j : mine) if (j != i)
            a += dd[i + j * N];   // column-major: D(i, j) = dd[i + j*N]
        a /= keel::as<keel::f64>(mine.size() - 1);

        // b(i): min over other clusters of mean distance to that cluster.
        keel::f64 b = std::numeric_limits<keel::f64>::infinity();
        for (auto [gk, members] : groups) {
            if (gk == k) continue;
            keel::f64 m = 0.0;
            for (keel::usize j : members) m += dd[i + j * N];
            m /= keel::as<keel::f64>(members.size());
            if (m < b) b = m;
        }
        if (!std::isfinite(b)) {
            // Only one cluster in total — undefined; MATLAB returns NaN
            // (per behaviour with one group). We mirror that.
            od[i] = std::numeric_limits<keel::f64>::quiet_NaN();
            continue;
        }

        const keel::f64 denom = std::max(a, b);
        od[i] = (denom > 0.0) ? (b - a) / denom : 0.0;
    }
    return out;
}

} // namespace numkit::stats
