// toolboxes/stats/src/cluster/kmeans.cpp
//
// Lloyd's algorithm with k-means++ initialisation. Single-threaded
// portable implementation; SIMD / parallel variants planned later.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/stats/cluster/kmeans.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>

#include "kmeans_detail.hpp"

namespace numkit::stats {

namespace {

void kmeans_argcheck(const Value &X, int K)
{
    const keel::usize N = X.dims().rows();
    if (K < 1 || static_cast<keel::usize>(K) > N)
        throw Error("kmeans: K must be in 1..N", 0, 0, "kmeans", "",
                    "numkit:kmeans:badK");
}

} // namespace

// Full kmeans with optional 4th D output (N×K squared distances).

KmeansResult
kmeans_full(::numkit::ops::RngContext &rng, const Value &X, int K, int max_iter, int replicates, keel::Allocator *mr)
{
    kmeans_argcheck(X, K);
    if (max_iter <= 0)   max_iter = 100;
    if (replicates <= 0) replicates = 1;

    const keel::usize N = X.dims().rows();
    const keel::usize D = X.dims().cols();

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Read X into a flat row-major scratch buffer.
    keel::Array<double> Xv(N * D, scratch.allocator());
    for (keel::usize r = 0; r < N; ++r)
        for (keel::usize c = 0; c < D; ++c)
            Xv[r * D + c] = X.elemAsDouble(c * N + r);

    auto &gen = rng;

    LloydResult best(scratch.allocator());
    double best_total = std::numeric_limits<double>::infinity();

    for (int rep = 0; rep < replicates; ++rep) {
        keel::Array<double> C0(scratch.allocator());
        {
            C0 = kmeanspp_init(Xv, N, D, K, gen.engine(), scratch.allocator());
        }
        LloydResult res = lloyd(Xv, N, D, K, std::move(C0), max_iter, scratch.allocator());
        if (res.total < best_total) {
            best_total = res.total;
            best = std::move(res);
        }
    }

    KmeansResult out;

    out.idx = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    {
        double *ip = out.idx.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) ip[i] = double(best.idx[i] + 1);
    }

    out.C = Value::matrix(K, D, ValueType::DOUBLE, mr);
    {
        double *cp = out.C.doubleDataMut();
        for (int k = 0; k < K; ++k)
            for (keel::usize d = 0; d < D; ++d)
                cp[d * K + k] = best.C[(keel::usize)k * D + d];
    }

    out.sumd = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    {
        double *sp = out.sumd.doubleDataMut();
        for (int k = 0; k < K; ++k) sp[k] = best.sumd[k];
    }

    // 4th output D — N×K squared distance from each point to each centroid.
    out.D = Value::matrix(N, K, ValueType::DOUBLE, mr);
    {
        double *dp = out.D.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i)
            for (int k = 0; k < K; ++k)
                dp[k * N + i] = sq_dist(&Xv[i * D], &best.C[(keel::usize)k * D], D);
    }

    return out;
}

// Backward-compat 3-output wrapper.
std::tuple<Value, Value, Value>
kmeans(::numkit::ops::RngContext &rng, const Value &X, int K, int max_iter, int replicates, keel::Allocator *mr)
{
    auto R = kmeans_full(rng, X, K, max_iter, replicates, mr);
    return std::make_tuple(std::move(R.idx), std::move(R.C),
                           std::move(R.sumd));
}

} // namespace numkit::stats
