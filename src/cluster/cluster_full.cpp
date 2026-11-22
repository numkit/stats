// toolboxes/stats/src/cluster/cluster_full.cpp
//
// Full parse+validate+compute contracts for the clustering entries —
// pdist / pdist2 / mahal / linkage / cluster / inconsistent /
// silhouette (probed against the reference engine R2025b:
// build/probe77.m). The register adapters marshal only; the distance
// and dendrogram kernels live in their existing TUs.
//
// Metric/method/criterion name tables are validatestring-exact
// (case-insensitive, unique prefix) — the same semantics the
// reference's getParamVal applies. Rejection voices follow the probed
// classes: an unknown pdist metric is DistanceFunctionNotFound, an
// unknown pdist2 DISTANCE argument is UnrecognizedDistance, an unknown
// linkage METHOD / cluster Criterion is the value-list class.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/cluster/distance.hpp>
#include <numkit/stats/cluster/linkage.hpp>
#include <numkit/stats/cluster/silhouette.hpp>
#include "cluster/distance_detail.hpp"   // pdist2_topk

#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>

namespace numkit::stats {

namespace {

keel::String lowerCopy(keel::StringRef sv)
{
    keel::String s = keel::String::from(sv);
    s.to_ascii_lower_in_place();
    return s;
}

// The distance-metric name table shared by pdist / pdist2 /
// silhouette (validatestring prefix semantics via matchStringChoice).
constexpr keel::StringRef kMetrics[] = {
    "euclidean", "squaredeuclidean", "seuclidean", "cityblock",
    "chebychev", "minkowski", "cosine", "correlation", "hamming",
    "jaccard", "mahalanobis", "spearman",
};
constexpr keel::StringRef kLinkMethods[] = {
    "single", "complete", "average", "weighted", "centroid", "median",
    "ward",
};
constexpr keel::StringRef kCriteria[] = {
    "inconsistent", "distance", "threshold",
};

struct ClusterOpts {
    keel::String metric;
    keel::f64 p = 2.0;
    const Value *C = nullptr;   // Mahalanobis covariance
};

// Positional metric tail: a string names the metric, a scalar is the
// Minkowski exponent, a larger numeric array is the Mahalanobis
// covariance (the same shape the reference's positional forms take).
ClusterOpts pdist_argcheck(keel::ArrayRef<Value> tail, const char *fn,
                           const char *badMetricTag)
{
    ClusterOpts o{keel::String::from("euclidean"), 2.0, nullptr};
    for (keel::usize i = 0; i < tail.size(); ++i) {
        const Value &v = tail[i];
        if (v.isChar() || v.isString()) {
            const keel::String raw = v.toString();
            const keel::i32 hit = argcheck::matchStringChoice(
                v, keel::ArrayRef<keel::StringRef>(kMetrics));
            if (hit < 0)
                throw Error(keel::format("{}: the distance function '{}' was not found.", fn, raw),
                            0, 0, fn, "", badMetricTag);
            o.metric = keel::String::from(
                kMetrics[keel::as<keel::usize>(hit)]);
        } else if (v.numel() == 1) {
            o.p = v.elemAsDouble(0);
        } else {
            o.C = &v;
        }
    }
    return o;
}

} // namespace

// ── pdist / pdist2 / mahal ────────────────────────────────────────────

Value pdist_full(const Value &X, keel::ArrayRef<Value> tail,
                 keel::Allocator *mr)
{
    const ClusterOpts o = pdist_argcheck(tail, "pdist",
                                         "numkit:pdist:distanceFunctionNotFound");
    Value d = pdist(X, o.metric, o.p, o.C, mr);
    // Single inputs produce single distances (probe P07).
    if (X.type() == ValueType::SINGLE && d.type() == ValueType::DOUBLE) {
        const keel::usize n = keel::as<keel::usize>(d.numel());
        Value out = Value::matrix(
            keel::as<keel::usize>(d.dims().dim(0)),
            keel::as<keel::usize>(d.dims().dim(1)),
            ValueType::SINGLE, mr);
        for (keel::usize k = 0; k < n; ++k)
            out.singleDataMut()[k] =
                keel::as<keel::f32>(d.elemAsDouble(k));
        return out;
    }
    return d;
}

Pdist2Out pdist2_full(const Value &X, const Value &Y, keel::ArrayRef<Value> tail,
                      keel::Allocator *mr)
{
    // Smallest/Largest top-k selection follows the positional Distance
    // argument (tail index >= 1): pdist2(X, Y, Distance, 'Smallest', K).
    // Passing 'Smallest'/'Largest' in the Distance slot (tail[0]) falls
    // through to pdist_argcheck and raises unrecognizedDistance (probe P09).
    bool topk = false, largest = false;
    keel::usize k = 0;
    keel::Array<Value> rest;
    for (keel::usize i = 0; i < tail.size(); ++i) {
        if (i >= 1 && (tail[i].isChar() || tail[i].isString())
            && i + 1 < tail.size()) {
            const keel::String s = lowerCopy(tail[i].toString());
            if (s == "smallest" || s == "largest") {
                const keel::f64 kv = argcheck::expectedReal(
                    tail[i + 1], 4, "k", "pdist2");
                if (!(kv >= 1.0 && kv == std::floor(kv)))
                    throw Error(keel::format("pdist2: the 'Smallest'/'Largest' count "
                                "must be a positive integer scalar (got {}).", kv),
                                0, 0, "pdist2", "", "numkit:pdist2:badK");
                topk = true;
                largest = (s == "largest");
                k = keel::as<keel::usize>(kv);
                ++i;   // consume the count
                continue;
            }
        }
        rest.push_back(tail[i]);
    }
    const ClusterOpts o = pdist_argcheck(
        keel::ArrayRef<Value>(rest.data(), rest.size()), "pdist2",
        "numkit:pdist2:unrecognizedDistance");
    if (topk) {
        Pdist2Out r;
        pdist2_topk(X, Y, o.metric, o.p, o.C, k, largest, r.D, r.I, mr);
        return r;
    }
    Pdist2Out r;
    r.D = pdist2(X, Y, o.metric, o.p, o.C, mr);
    return r;
}

// ── linkage ───────────────────────────────────────────────────────────

Value linkage_full(const Value &Y, keel::ArrayRef<Value> tail,
                   keel::Allocator *mr)
{
    keel::String method = keel::String::from("single");
    keel::String metric = keel::String::from("euclidean");
    keel::f64 p = 2.0;
    // Method: a string 2nd argument (validatestring against the
    // linkage methods — the reference's getParamVal class).
    if (!tail.empty() && (tail[0].isChar() || tail[0].isString())) {
        const keel::i32 hit = argcheck::matchStringChoice(
            tail[0], keel::ArrayRef<keel::StringRef>(kLinkMethods));
        if (hit < 0)
            throw Error(keel::format("linkage: '{}' is not a valid value for the METHOD argument.",
                                     tail[0].toString()),
                        0, 0, "linkage", "", "numkit:linkage:badMethod");
        method = keel::String::from(
            kLinkMethods[keel::as<keel::usize>(hit)]);
    } else if (!tail.empty()) {
        throw Error(keel::format("linkage: METHOD must be one of: single, complete, "
                    "average, weighted, centroid, median, ward (got a {}).",
                    mtypeName(tail[0].type())),
                    0, 0, "linkage", "", "numkit:linkage:badMethod");
    }
    // Metric (implicit pdist) + Minkowski exponent.
    bool haveMetric = false;
    for (keel::usize i = 1; i < tail.size(); ++i) {
        const Value &v = tail[i];
        if (v.isChar() || v.isString()) {
            if (haveMetric)
                throw Error("linkage: unexpected extra argument.",
                            0, 0, "linkage", "", "numkit:linkage:nargin");
            const keel::i32 hit = argcheck::matchStringChoice(
                v, keel::ArrayRef<keel::StringRef>(kMetrics));
            if (hit < 0)
                throw Error(keel::format("linkage: the distance function '{}' was not found.",
                                         v.toString()),
                            0, 0, "linkage", "",
                            "numkit:linkage:distanceFunctionNotFound");
            metric = keel::String::from(
                kMetrics[keel::as<keel::usize>(hit)]);
            haveMetric = true;
        } else if (v.numel() == 1) {
            p = v.elemAsDouble(0);
        } else {
            throw Error("linkage: unexpected extra argument.",
                        0, 0, "linkage", "", "numkit:linkage:nargin");
        }
    }
    return linkage(Y, method, metric, p, mr);
}

// ── cluster ───────────────────────────────────────────────────────────

Value cluster_full(const Value &Z, keel::ArrayRef<Value> tail,
                   keel::Allocator *mr)
{
    keel::i32 maxclust = -1;
    keel::f64 cutoff = -1.0;
    keel::i32 depth = 2;
    keel::String criterion = keel::String::from("inconsistent");
    for (keel::usize i = 0; i < tail.size(); ++i) {
        if (!(tail[i].isChar() || tail[i].isString())) {
            if (i == 0) {
                // Legacy positional cutoff scalar.
                cutoff = tail[i].elemAsDouble(0);
                continue;
            }
            throw Error("cluster: expected a parameter name.",
                        0, 0, "cluster", "", "numkit:cluster:nvMismatch");
        }
        if (i + 1 >= tail.size())
            throw Error("cluster: each parameter name needs a value.",
                        0, 0, "cluster", "", "numkit:cluster:nvMismatch");
        const keel::String rawName = tail[i].toString();
        const keel::String name = lowerCopy(rawName);
        const Value &v = tail[i + 1];
        if      (name == "maxclust")  maxclust = keel::as<keel::i32>(v.elemAsDouble(0));
        else if (name == "cutoff")    cutoff = v.elemAsDouble(0);
        else if (name == "depth")     depth = keel::as<keel::i32>(v.elemAsDouble(0));
        else if (name == "criterion") {
            const keel::i32 hit = argcheck::matchStringChoice(
                v, keel::ArrayRef<keel::StringRef>(kCriteria));
            if (hit < 0)
                throw Error(keel::format("cluster: '{}' is not a "
                            "valid value for the Criterion argument. "
                            "Valid values are 'inconsistent', 'distance', "
                            "and 'threshold'.", v.toString()),
                            0, 0, "cluster", "",
                            "numkit:cluster:badCriterion");
            criterion = keel::String::from(
                kCriteria[keel::as<keel::usize>(hit)]);
        } else {
            throw Error(keel::format("cluster: unrecognized parameter name '{}'.", rawName),
                        0, 0, "cluster", "", "numkit:cluster:badParamName");
        }
        ++i;   // consume the value
    }
    if (maxclust < 0 && cutoff < 0.0)
        throw Error("cluster: one of 'MaxClust' or 'Cutoff' is required.",
                    0, 0, "cluster", "", "numkit:cluster:needOption");
    return cluster_from_linkage(Z, maxclust, cutoff, criterion, depth, mr);
}

// ── inconsistent ──────────────────────────────────────────────────────

Value inconsistent_full(const Value &Z, keel::ArrayRef<Value> tail,
                        keel::Allocator *mr)
{
    keel::i32 depth = 2;
    if (!tail.empty()) {
        const keel::f64 d = tail[0].elemAsDouble(0);
        if (!(d >= 1.0 && d == std::floor(d)))
            throw Error(keel::format("inconsistent: depth must be a positive integer "
                        "scalar (got {}).", d),
                        0, 0, "inconsistent", "", "numkit:inconsistent:badDepth");
        depth = keel::as<keel::i32>(d);
    }
    return inconsistent(Z, depth, mr);
}

// ── silhouette ────────────────────────────────────────────────────────

Value silhouette_full(const Value &X, const Value &clust,
                      keel::ArrayRef<Value> tail, keel::Allocator *mr)
{
    ClusterOpts o{keel::String::from("sqeuclidean"), 2.0, nullptr};   // silhouette default
    for (keel::usize i = 0; i < tail.size(); ++i) {
        const Value &v = tail[i];
        if (v.isChar() || v.isString()) {
            const keel::i32 hit = argcheck::matchStringChoice(
                v, keel::ArrayRef<keel::StringRef>(kMetrics));
            if (hit < 0)
                throw Error(keel::format("silhouette: the distance function '{}' was not found.",
                                         v.toString()),
                            0, 0, "silhouette", "",
                            "numkit:silhouette:distanceFunctionNotFound");
            o.metric = lowerCopy(v.toString());
        } else if (v.numel() == 1) {
            o.p = v.elemAsDouble(0);
        } else {
            o.C = &v;
        }
    }
    return silhouette(X, clust, o.metric, o.p, mr);
}

} // namespace numkit::stats
