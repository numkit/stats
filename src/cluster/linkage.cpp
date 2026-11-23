// toolboxes/stats/src/cluster/linkage.cpp
//
// Hierarchical clustering (linkage, cluster_from_linkage, clusterdata,
// cophenet, inconsistent) delegating numerical kernels to keel::stats.

#include <keel/arena.h>
#include <keel/tuple.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/stats.h>
#include <numkit/stats/cluster/distance.hpp>
#include <numkit/stats/cluster/linkage.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "linkage_detail.hpp"

namespace numkit::stats {

namespace {

keel::stats::LinkageMethod parseLink(keel::StringRef s) noexcept
{
    if (s == "single")   return keel::stats::LinkageMethod::kSingle;
    if (s == "complete") return keel::stats::LinkageMethod::kComplete;
    if (s == "average")  return keel::stats::LinkageMethod::kAverage;
    if (s == "weighted") return keel::stats::LinkageMethod::kWeighted;
    if (s == "centroid") return keel::stats::LinkageMethod::kCentroid;
    if (s == "median")   return keel::stats::LinkageMethod::kMedian;
    if (s == "ward")     return keel::stats::LinkageMethod::kWard;
    return keel::stats::LinkageMethod::kSingle;
}

keel::usize solveNFromPdist(keel::usize k) noexcept
{
    return keel::as<keel::usize>(keel::round(0.5 * (1.0 + keel::sqrt(1.0 + 8.0 * keel::as<keel::f64>(k)))));
}

keel::Array<keel::f64> readFlat(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, 0.0, alloc);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

} // anonymous

Value linkage(const Value &Y, keel::StringRef method, keel::StringRef metric, keel::f64 p, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize Yrows = Y.dims().rows();
    const keel::usize Ycols = Y.dims().cols();
    keel::usize N = 0;
    keel::Array<keel::f64> D(scratch.allocator());

    if (Yrows == 1 || Ycols == 1) {
        const keel::usize n = Y.numel();
        N = solveNFromPdist(n);
        if (N * (N - 1) / 2 != n)
            throw Error("linkage: pdist input has invalid length",
                        0, 0, "linkage", "", "numkit:linkage:size");
        D = readFlat(Y, scratch.allocator());
    } else {
        Value Yp = pdist(Y, metric, p, mr);
        const keel::usize n = Yp.numel();
        N = Yrows;
        if (N * (N - 1) / 2 != n)
            throw Error("linkage: bad pdist intermediate", 0, 0, "linkage", "", "numkit:linkage:internal");
        D = readFlat(Yp, scratch.allocator());
    }

    if (N < 2) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    const keel::usize rows = N - 1;
    Value Z = Value::matrix(rows, 3, ValueType::DOUBLE, mr);
    (void)keel::stats::linkage(D.as_ref(), N, parseLink(method),
                               keel::MutArrayRef<keel::f64>(Z.doubleDataMut(), rows * 3),
                               scratch.allocator());
    return Z;
}

Value linkage(const Value &Y, keel::StringRef method, keel::Allocator *mr)
{
    return linkage(Y, method, "euclidean", 2.0, mr);
}

Value cluster_from_linkage(const Value &Z, keel::i32 maxclust, keel::f64 cutoff,
                           keel::StringRef criterion, keel::i32 depth, keel::Allocator *mr)
{
    if (Z.dims().cols() != 3)
        throw Error("cluster: Z must be (N-1)×3", 0, 0, "cluster", "", "numkit:cluster:size");

    const keel::usize M = Z.dims().rows();
    const keel::usize N = M + 1;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto zv = readFlat(Z, scratch.allocator());
    keel::Array<keel::i32> labels(N, 0, scratch.allocator());

    if (maxclust > 0 && keel::as<keel::usize>(maxclust) <= N) {
        (void)keel::stats::cluster_maxclust(zv.as_ref(), M, maxclust, labels.as_mut_ref(),
                                            scratch.allocator());
    } else if (cutoff > 0.0) {
        (void)keel::stats::cluster_cutoff(zv.as_ref(), M, cutoff, criterion == "distance", depth,
                                          labels.as_mut_ref(), scratch.allocator());
    } else {
        for (keel::usize i = 0; i < N; ++i) labels[i] = keel::as<keel::i32>(i + 1);
    }

    Value out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = keel::as<keel::f64>(labels[i]);
    return out;
}

Value cluster_from_linkage(const Value &Z, keel::i32 maxclust, keel::f64 cutoff,
                           keel::StringRef criterion, keel::Allocator *mr)
{
    return cluster_from_linkage(Z, maxclust, cutoff, criterion, 2, mr);
}

Value clusterdata(const Value &X, keel::i32 maxclust, keel::f64 cutoff,
                  keel::StringRef linkage_method, keel::StringRef criterion, keel::i32 depth,
                  keel::StringRef distance_metric, keel::f64 p, keel::Allocator *mr)
{
    Value Y = pdist(X, distance_metric, p, mr);
    Value Z = linkage(Y, linkage_method, mr);
    return cluster_from_linkage(Z, maxclust, cutoff, criterion, depth, mr);
}

Value clusterdata(const Value &X, keel::i32 maxclust, keel::f64 cutoff,
                  keel::StringRef linkage_method, keel::StringRef criterion, keel::i32 depth,
                  keel::Allocator *mr)
{
    return clusterdata(X, maxclust, cutoff, linkage_method, criterion, depth, "euclidean", 2.0, mr);
}

Value clusterdata(const Value &X, keel::i32 maxclust, keel::f64 cutoff,
                  keel::StringRef linkage_method, keel::StringRef criterion, keel::Allocator *mr)
{
    return clusterdata(X, maxclust, cutoff, linkage_method, criterion, 2, mr);
}

keel::Tuple<Value, Value> cophenet_full(const Value &Z, const Value &Y, keel::Allocator *mr)
{
    const keel::usize M = Z.dims().rows();
    const keel::usize N = M + 1;
    if (Z.dims().cols() != 3)
        throw Error("cophenet: Z must be (N-1)×3", 0, 0, "cophenet", "", "numkit:cophenet:size");
    if (N * (N - 1) / 2 != Y.numel())
        throw Error("cophenet: Y / Z size mismatch", 0, 0, "cophenet", "", "numkit:cophenet:size");

    const keel::usize Yn = Y.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto zv = readFlat(Z, scratch.allocator());
    const auto yv = readFlat(Y, scratch.allocator());

    Value dv = Value::matrix(1, Yn, ValueType::DOUBLE, mr);
    keel::f64 corr = 0.0;
    (void)keel::stats::cophenet(zv.as_ref(), M, yv.as_ref(),
                                keel::MutArrayRef<keel::f64>(dv.doubleDataMut(), Yn), corr,
                                scratch.allocator());
    return {Value::scalar(corr, mr), keel::move(dv)};
}

Value cophenet(const Value &Z, const Value &Y, keel::Allocator *mr)
{
    auto [c, _d] = cophenet_full(Z, Y, mr);
    return c;
}

Value inconsistent(const Value &Z, keel::i32 depth, keel::Allocator *mr)
{
    if (Z.dims().cols() != 3)
        throw Error("inconsistent: Z must be (N-1)×3", 0, 0, "inconsistent", "", "numkit:inconsistent:size");

    const keel::usize M = Z.dims().rows();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto zv = readFlat(Z, scratch.allocator());
    Value out = Value::matrix(M, 4, ValueType::DOUBLE, mr);
    (void)keel::stats::inconsistent(zv.as_ref(), M, depth,
                                    keel::MutArrayRef<keel::f64>(out.doubleDataMut(), M * 4),
                                    scratch.allocator());
    return out;
}

} // namespace numkit::stats
