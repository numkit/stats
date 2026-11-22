// toolboxes/stats/src/cluster/silhouette.cpp
//
// Clustering silhouette quality metric, delegating to keel::stats::silhouette.

#include <keel/arena.h>
#include <keel/math.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/stats.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/stats/cluster/silhouette.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::stats {

namespace {

keel::stats::DistanceMetric parse_sil_metric(keel::StringRef s)
{
    keel::String m;
    m.reserve(s.size());
    for (char c : s) m.push_back(keel::to_ascii_lower(c));
    if (m.empty() || m == "sqeuclidean" || m == "squaredeuclidean")
        return keel::stats::DistanceMetric::kSqEuclidean;
    if (m == "euclidean")   return keel::stats::DistanceMetric::kEuclidean;
    if (m == "cityblock")   return keel::stats::DistanceMetric::kCityblock;
    if (m == "chebychev")   return keel::stats::DistanceMetric::kChebychev;
    if (m == "minkowski")   return keel::stats::DistanceMetric::kMinkowski;
    if (m == "cosine")      return keel::stats::DistanceMetric::kCosine;
    if (m == "correlation") return keel::stats::DistanceMetric::kCorrelation;
    if (m == "hamming")     return keel::stats::DistanceMetric::kHamming;
    if (m == "jaccard")     return keel::stats::DistanceMetric::kJaccard;
    if (m == "mahalanobis") return keel::stats::DistanceMetric::kMahalanobis;
    if (m == "seuclidean")  return keel::stats::DistanceMetric::kSeuclidean;
    if (m == "spearman")    return keel::stats::DistanceMetric::kSpearman;
    return keel::stats::DistanceMetric::kSqEuclidean;
}

void silhouette_argcheck(const Value &X, const Value &clust)
{
    if (X.dims().rows() > 0 && clust.numel() != X.dims().rows())
        throw Error("silhouette: clust length must equal size(X, 1)",
                    0, 0, "silhouette", "", "numkit:silhouette:size");
}

} // namespace

Value silhouette(const Value &X, const Value &clust, keel::StringRef metric,
                 keel::f64 p, keel::Allocator *mr)
{
    silhouette_argcheck(X, clust);
    const keel::usize N = X.dims().rows();
    const keel::usize D = X.dims().cols();
    if (N == 0)
        return Value::matrix(0, 1, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(N * D, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N * D; ++i) xv[i] = X.elemAsDouble(i);
    keel::Array<keel::i32> lbl(N, 0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        lbl[i] = keel::as<keel::i32>(keel::round_i64(clust.elemAsDouble(i)));

    Value out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    (void)keel::stats::silhouette(
        xv.as_ref(), N, D, lbl.as_ref(), parse_sil_metric(metric), p,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N),
        scratch.allocator());
    return out;
}

} // namespace numkit::stats
