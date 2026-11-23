// toolboxes/stats/src/cluster/kmedoids.cpp
//
// Partitioning Around Medoids (kmedoids) and density clustering (dbscan),
// delegating numerical kernels to keel::stats.

#include <keel/arena.h>
#include <keel/tuple.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/random.h>
#include <keel/ref.h>
#include <keel/stats.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/ops/rng_context.hpp>
#include <numkit/stats/cluster/kmedoids.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


#include "kmedoids_detail.hpp"

namespace numkit::stats {

namespace {

bool parse_metric(keel::StringRef s, keel::stats::DistanceMetric &out_metric)
{
    keel::String sl;
    sl.reserve(s.size());
    for (char c : s) sl.push_back(keel::to_ascii_lower(c));
    if (sl == "precomputed") {
        out_metric = keel::stats::DistanceMetric::kEuclidean;
        return true;
    }
    if (sl == "squaredeuclidean" || sl == "sqeuclidean") out_metric = keel::stats::DistanceMetric::kSqEuclidean;
    else if (sl == "cityblock")  out_metric = keel::stats::DistanceMetric::kCityblock;
    else if (sl == "chebychev")  out_metric = keel::stats::DistanceMetric::kChebychev;
    else if (sl == "minkowski")  out_metric = keel::stats::DistanceMetric::kMinkowski;
    else if (sl == "cosine")     out_metric = keel::stats::DistanceMetric::kCosine;
    else if (sl == "hamming")    out_metric = keel::stats::DistanceMetric::kHamming;
    else if (sl == "jaccard")    out_metric = keel::stats::DistanceMetric::kJaccard;
    else                         out_metric = keel::stats::DistanceMetric::kEuclidean;
    return false;
}

void kmedoids_argcheck(const Value &X, keel::i32 K)
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

KmedoidsResult
kmedoids_full(::numkit::ops::RngContext &rng, const Value &X, keel::i32 K,
              keel::i32 max_iter, keel::i32 replicates, keel::StringRef metric_name,
              keel::Allocator *mr)
{
    kmedoids_argcheck(X, K);
    if (max_iter <= 0)   max_iter = 100;
    if (replicates <= 0) replicates = 1;
    keel::stats::DistanceMetric metric = keel::stats::DistanceMetric::kEuclidean;
    (void)parse_metric(metric_name, metric);

    const keel::usize N = X.dims().rows();
    const keel::usize D = X.dims().cols();
    const keel::usize Ku = keel::as<keel::usize>(K);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(N * D, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N * D; ++i) xv[i] = X.elemAsDouble(i);

    keel::Array<keel::i32> best_med(Ku, 0, scratch.allocator());
    keel::Array<keel::i32> best_idx(N, 0, scratch.allocator());
    keel::Array<keel::f64> best_sumd(Ku, 0.0, scratch.allocator());
    keel::f64 best_total = keel::Limits<keel::f64>::infinity();
    keel::i32 best_iters = 0;
    keel::i32 best_rep   = 0;

    keel::Array<keel::i32> all(N, 0, scratch.allocator());
    keel::Array<keel::i32> init_med(Ku, 0, scratch.allocator());
    keel::Array<keel::i32> med(Ku, 0, scratch.allocator());
    keel::Array<keel::i32> idx(N, 0, scratch.allocator());
    keel::Array<keel::f64> sumd(Ku, 0.0, scratch.allocator());

    for (keel::i32 rep = 0; rep < replicates; ++rep) {
        for (keel::usize i = 0; i < N; ++i) all[i] = keel::as<keel::i32>(i);
        keel::random::shuffle(rng, keel::MutArrayRef<keel::i32>(all.data(), all.size()));
        for (keel::usize k = 0; k < Ku; ++k) init_med[k] = all[k];

        keel::f64 total = 0.0;
        keel::i32 rep_iters = 0;
        (void)keel::stats::kmedoids_pam_step(
            xv.as_ref(), N, D, init_med.as_ref(), max_iter, metric, 2.0,
            idx.as_mut_ref(), med.as_mut_ref(), sumd.as_mut_ref(),
            total, rep_iters, scratch.allocator());

        if (total < best_total) {
            best_total = total;
            keel::copy_n(best_med.data(), med.data(), Ku);
            keel::copy_n(best_idx.data(), idx.data(), N);
            keel::copy_n(best_sumd.data(), sumd.data(), Ku);
            best_iters = rep_iters;
            best_rep = rep + 1;
        }
    }

    KmedoidsResult R{};
    R.idx  = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    R.C    = Value::matrix(Ku, D, ValueType::DOUBLE, mr);
    R.sumd = Value::matrix(Ku, 1, ValueType::DOUBLE, mr);
    R.D    = Value::matrix(N, Ku, ValueType::DOUBLE, mr);
    R.midx = Value::matrix(Ku, 1, ValueType::DOUBLE, mr);

    for (keel::usize i = 0; i < N; ++i) R.idx.doubleDataMut()[i] = keel::as<keel::f64>(best_idx[i] + 1);
    for (keel::usize k = 0; k < Ku; ++k) {
        R.sumd.doubleDataMut()[k] = best_sumd[k];
        R.midx.doubleDataMut()[k] = keel::as<keel::f64>(best_med[k] + 1);
        for (keel::usize d = 0; d < D; ++d)
            R.C.doubleDataMut()[d * Ku + k] = xv[d * N + keel::as<keel::usize>(best_med[k])];
    }

    (void)keel::stats::pdist2(
        xv.as_ref(), N,
        keel::ArrayRef<keel::f64>(R.C.doubleData(), Ku * D), Ku, D,
        metric, 2.0, {},
        keel::MutArrayRef<keel::f64>(R.D.doubleDataMut(), N * Ku),
        scratch.allocator());

    R.iters = best_iters;
    R.best_rep = best_rep;
    return R;
}

keel::Tuple<Value, Value, Value>
kmedoids(::numkit::ops::RngContext &rng, const Value &X, keel::i32 K,
         keel::i32 max_iter, keel::i32 replicates, keel::StringRef metric_name,
         keel::Allocator *mr)
{
    auto R = kmedoids_full(rng, X, K, max_iter, replicates, metric_name, mr);
    return keel::make_tuple(keel::move(R.idx), keel::move(R.C), keel::move(R.sumd));
}

keel::Tuple<Value, Value>
dbscan(const Value &X, keel::f64 eps, keel::i32 minpts,
       keel::StringRef metric_name, keel::f64 p, keel::Allocator *mr)
{
    dbscan_argcheck(eps);
    if (minpts <= 0) minpts = 1;
    keel::stats::DistanceMetric metric = keel::stats::DistanceMetric::kEuclidean;
    const bool precomputed = parse_metric(metric_name, metric);

    const keel::usize N = X.dims().rows();
    const keel::usize D = X.dims().cols();
    if (precomputed && D != N)
        throw Error("dbscan: precomputed distance matrix must be N×N",
                    0, 0, "dbscan", "", "numkit:dbscan:badprecomp");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(N * D, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N * D; ++i) xv[i] = X.elemAsDouble(i);

    keel::Array<keel::i32> labels(N, 0, scratch.allocator());
    Value core_v = Value::matrix(N, 1, ValueType::LOGICAL, mr);

    (void)keel::stats::dbscan(
        xv.as_ref(), N, D, eps, minpts, metric, p, precomputed,
        labels.as_mut_ref(),
        keel::MutArrayRef<keel::u8>(core_v.logicalDataMut(), N),
        scratch.allocator());

    Value idx = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *ip = idx.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) ip[i] = keel::as<keel::f64>(labels[i]);
    return keel::make_tuple(keel::move(idx), keel::move(core_v));
}

keel::Tuple<Value, Value>
dbscan(const Value &X, keel::f64 eps, keel::i32 minpts,
       keel::StringRef metric_name, keel::Allocator *mr)
{
    return dbscan(X, eps, minpts, metric_name, 2.0, mr);
}

} // namespace numkit::stats
