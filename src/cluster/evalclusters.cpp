// toolboxes/stats/src/cluster/evalclusters.cpp
//
// Engine-free compute implementation for evalclusters (CalinskiHarabasz,
// DaviesBouldin, Silhouette) supporting 'kmeans', 'linkage', and precomputed
// cluster label matrices (Phase 2b compute/register split).

#include "cluster/kmeans_detail.hpp"
#include <numkit/stats/cluster/linkage.hpp>

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/string.h>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::stats {

namespace {

static keel::f64 evalCalinskiHarabasz(const Value &X, const Value &idxVal, keel::i32 k)
{
    const keel::usize n = X.dims().rows();
    const keel::usize d = X.dims().cols();
    if (k <= 1 || keel::as<keel::usize>(k) >= n)
        return keel::Limits<keel::f64>::quiet_nan();
    keel::Array<keel::f64> grand(d, 0.0);
    for (keel::usize j = 0; j < d; ++j) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < n; ++i) s += X.elemAsDouble(j * n + i);
        grand[j] = s / keel::as<keel::f64>(n);
    }
    const keel::usize ku = keel::as<keel::usize>(k);
    keel::Array<keel::usize> cnt(ku, 0);
    keel::Array<keel::f64> cent(ku * d, 0.0);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::i32 c = keel::as<keel::i32>(idxVal.elemAsDouble(i)) - 1;
        if (c < 0 || c >= k) continue;
        const keel::usize cu = keel::as<keel::usize>(c);
        cnt[cu] += 1;
        for (keel::usize j = 0; j < d; ++j)
            cent[cu * d + j] += X.elemAsDouble(j * n + i);
    }
    keel::usize nonEmpty = 0;
    for (keel::usize c = 0; c < ku; ++c) {
        if (cnt[c] > 0) {
            ++nonEmpty;
            for (keel::usize j = 0; j < d; ++j)
                cent[c * d + j] /= keel::as<keel::f64>(cnt[c]);
        }
    }
    if (nonEmpty <= 1 || nonEmpty >= n)
        return keel::Limits<keel::f64>::quiet_nan();

    keel::f64 ssb = 0.0;
    for (keel::usize c = 0; c < ku; ++c) {
        if (cnt[c] == 0) continue;
        keel::f64 dist2 = 0.0;
        for (keel::usize j = 0; j < d; ++j) {
            const keel::f64 diff = cent[c * d + j] - grand[j];
            dist2 += diff * diff;
        }
        ssb += keel::as<keel::f64>(cnt[c]) * dist2;
    }
    keel::f64 ssw = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::i32 c = keel::as<keel::i32>(idxVal.elemAsDouble(i)) - 1;
        if (c < 0 || c >= k) continue;
        const keel::usize cu = keel::as<keel::usize>(c);
        for (keel::usize j = 0; j < d; ++j) {
            const keel::f64 diff = X.elemAsDouble(j * n + i) - cent[cu * d + j];
            ssw += diff * diff;
        }
    }
    if (ssw <= 0.0) return keel::Limits<keel::f64>::infinity();
    return (ssb / keel::as<keel::f64>(nonEmpty - 1))
         / (ssw / keel::as<keel::f64>(n - nonEmpty));
}

static keel::f64 evalDaviesBouldin(const Value &X, const Value &idxVal, keel::i32 k)
{
    const keel::usize n = X.dims().rows();
    const keel::usize d = X.dims().cols();
    if (k <= 1 || keel::as<keel::usize>(k) >= n)
        return keel::Limits<keel::f64>::quiet_nan();
    const keel::usize ku = keel::as<keel::usize>(k);
    keel::Array<keel::usize> cnt(ku, 0);
    keel::Array<keel::f64> cent(ku * d, 0.0);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::i32 c = keel::as<keel::i32>(idxVal.elemAsDouble(i)) - 1;
        if (c < 0 || c >= k) continue;
        const keel::usize cu = keel::as<keel::usize>(c);
        cnt[cu] += 1;
        for (keel::usize j = 0; j < d; ++j)
            cent[cu * d + j] += X.elemAsDouble(j * n + i);
    }
    for (keel::usize c = 0; c < ku; ++c) {
        if (cnt[c] > 0) {
            for (keel::usize j = 0; j < d; ++j)
                cent[c * d + j] /= keel::as<keel::f64>(cnt[c]);
        }
    }
    keel::Array<keel::f64> disp(ku, 0.0);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::i32 c = keel::as<keel::i32>(idxVal.elemAsDouble(i)) - 1;
        if (c < 0 || c >= k) continue;
        const keel::usize cu = keel::as<keel::usize>(c);
        keel::f64 dist2 = 0.0;
        for (keel::usize j = 0; j < d; ++j) {
            const keel::f64 diff = X.elemAsDouble(j * n + i) - cent[cu * d + j];
            dist2 += diff * diff;
        }
        disp[cu] += keel::sqrt(dist2);
    }
    for (keel::usize c = 0; c < ku; ++c)
        if (cnt[c] > 0) disp[c] /= keel::as<keel::f64>(cnt[c]);

    keel::f64 sumMax = 0.0;
    keel::usize active = 0;
    for (keel::usize i = 0; i < ku; ++i) {
        if (cnt[i] == 0) continue;
        ++active;
        keel::f64 mx = 0.0;
        for (keel::usize j = 0; j < ku; ++j) {
            if (j == i || cnt[j] == 0) continue;
            keel::f64 d2 = 0.0;
            for (keel::usize c = 0; c < d; ++c) {
                const keel::f64 diff = cent[i * d + c] - cent[j * d + c];
                d2 += diff * diff;
            }
            const keel::f64 dij = keel::sqrt(d2);
            const keel::f64 rij = (dij > 0.0) ? (disp[i] + disp[j]) / dij
                                              : keel::Limits<keel::f64>::infinity();
            if (rij > mx) mx = rij;
        }
        sumMax += mx;
    }
    if (active <= 1) return keel::Limits<keel::f64>::quiet_nan();
    return sumMax / keel::as<keel::f64>(active);
}

static keel::f64 evalMeanSilhouette(const Value &X, const Value &idxVal, keel::i32 k)
{
    const keel::usize n = X.dims().rows();
    const keel::usize d = X.dims().cols();
    if (k <= 1 || keel::as<keel::usize>(k) >= n)
        return keel::Limits<keel::f64>::quiet_nan();
    const keel::usize ku = keel::as<keel::usize>(k);
    keel::Array<keel::i32> labels(n, 0);
    keel::Array<keel::usize> cnt(ku, 0);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::i32 c = keel::as<keel::i32>(idxVal.elemAsDouble(i)) - 1;
        labels[i] = c;
        if (c >= 0 && c < k) cnt[keel::as<keel::usize>(c)] += 1;
    }
    keel::f64 total = 0.0;
    keel::Array<keel::f64> dsum(ku, 0.0);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::i32 ci = labels[i];
        if (ci < 0 || ci >= k) continue;
        for (keel::usize c = 0; c < ku; ++c) dsum[c] = 0.0;
        for (keel::usize j = 0; j < n; ++j) {
            if (j == i) continue;
            const keel::i32 cj = labels[j];
            if (cj < 0 || cj >= k) continue;
            keel::f64 sq = 0.0;
            for (keel::usize col = 0; col < d; ++col) {
                const keel::f64 diff = X.elemAsDouble(col * n + i) - X.elemAsDouble(col * n + j);
                sq += diff * diff;
            }
            dsum[keel::as<keel::usize>(cj)] += keel::sqrt(sq);
        }
        const keel::usize ciu = keel::as<keel::usize>(ci);
        if (cnt[ciu] <= 1) continue;
        const keel::f64 a = dsum[ciu] / keel::as<keel::f64>(cnt[ciu] - 1);
        keel::f64 b = keel::Limits<keel::f64>::infinity();
        for (keel::usize c = 0; c < ku; ++c) {
            if (c == ciu || cnt[c] == 0) continue;
            const keel::f64 avg = dsum[c] / keel::as<keel::f64>(cnt[c]);
            if (avg < b) b = avg;
        }
        const keel::f64 denom = keel::max(a, b);
        if (keel::is_finite(b) && denom > 0.0) total += (b - a) / denom;
    }
    return total / keel::as<keel::f64>(n);
}

static keel::i32 inferClusterCount(const Value &colIdx)
{
    keel::i32 maxLabel = 0;
    for (keel::usize i = 0; i < colIdx.numel(); ++i) {
        const keel::f64 v = colIdx.elemAsDouble(i);
        if (keel::is_finite(v)) {
            const keel::i32 iv = keel::as<keel::i32>(v);
            if (iv > maxLabel) maxLabel = iv;
        }
    }
    return maxLabel;
}

} // namespace

Value evalclusters_compute(::numkit::ops::RngContext &rng,
                           const Value &X,
                           const Value &clustArg,
                           keel::StringRef criterionRaw,
                           keel::ArrayRef<keel::i32> klistIn,
                           keel::Allocator *mr)
{
    if (!X.isNumeric() || X.isEmpty())
        throw Error("evalclusters: X must be a nonempty numeric matrix",
                    0, 0, "evalclusters", "", "numkit:evalclusters:badX");

    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        s.to_ascii_lower_in_place();
        return s;
    };

    const keel::String crit = lower(criterionRaw);
    const bool isCH  = (crit == "calinskiharabasz");
    const bool isDB  = (crit == "daviesbouldin");
    const bool isSil = (crit == "silhouette");
    if (!isCH && !isDB && !isSil)
        throw Error(keel::format("evalclusters: unsupported criterion '{}'", criterionRaw),
                    0, 0, "evalclusters", "", "numkit:evalclusters:badCriterion");

    const keel::usize n = X.dims().rows();
    const bool precomputed = clustArg.isNumeric();
    bool useKmeans = false;
    bool useLinkage = false;
    keel::String clustName;

    if (precomputed) {
        if (clustArg.dims().rows() != n || clustArg.dims().cols() == 0)
            throw Error("evalclusters: precomputed cluster matrix must have the same number of rows as X",
                        0, 0, "evalclusters", "", "numkit:evalclusters:badClustShape");
        clustName = keel::String::from("custom");
    } else if (clustArg.isChar() || clustArg.isString()) {
        const keel::String cname = lower(clustArg.toString().as_ref());
        if (cname == "kmeans") {
            useKmeans = true;
            clustName = keel::String::from("kmeans");
        } else if (cname == "linkage") {
            useLinkage = true;
            clustName = keel::String::from("linkage");
        } else {
            throw Error(keel::format("evalclusters: unsupported clustering algorithm '{}' (expected 'kmeans', 'linkage', or cluster matrix)", clustArg.toString()),
                        0, 0, "evalclusters", "", "numkit:evalclusters:badClust");
        }
    } else {
        throw Error("evalclusters: 2nd argument must be 'kmeans', 'linkage', or a numeric cluster matrix",
                    0, 0, "evalclusters", "", "numkit:evalclusters:badClust");
    }

    keel::Array<keel::i32> klist;
    if (!klistIn.empty()) {
        for (keel::usize j = 0; j < klistIn.size(); ++j) {
            if (klistIn[j] < 1 || keel::as<keel::usize>(klistIn[j]) > n)
                throw Error("evalclusters: KList values must be integers in [1, N]",
                            0, 0, "evalclusters", "", "numkit:evalclusters:badKList");
            klist.push_back(klistIn[j]);
        }
    } else if (precomputed) {
        const keel::usize mCols = clustArg.dims().cols();
        for (keel::usize j = 0; j < mCols; ++j) {
            Value col = Value::matrix(n, 1, ValueType::DOUBLE, mr);
            keel::f64 *cd = col.doubleDataMut();
            for (keel::usize r = 0; r < n; ++r) cd[r] = clustArg.elemAsDouble(j * n + r);
            klist.push_back(inferClusterCount(col));
        }
    } else {
        throw Error("evalclusters: 'KList' must be specified when clust is a clustering algorithm name",
                    0, 0, "evalclusters", "", "numkit:evalclusters:missingKList");
    }

    Value linkageTree;
    if (useLinkage)
        linkageTree = linkage(X, "ward", "euclidean", 2.0, mr);

    const keel::usize m = klist.size();
    Value inspectedK = Value::matrix(1, m, ValueType::DOUBLE, mr);
    Value critVals   = Value::matrix(1, m, ValueType::DOUBLE, mr);
    keel::f64 *kd = inspectedK.doubleDataMut();
    keel::f64 *cv = critVals.doubleDataMut();

    keel::Array<Value> solutions;
    solutions.reserve(m);
    for (keel::usize j = 0; j < m; ++j) {
        const keel::i32 k = klist[j];
        kd[j] = keel::as<keel::f64>(k);
        Value idx;
        if (precomputed) {
            idx = Value::matrix(n, 1, ValueType::DOUBLE, mr);
            keel::f64 *id = idx.doubleDataMut();
            for (keel::usize r = 0; r < n; ++r)
                id[r] = clustArg.elemAsDouble(j * n + r);
        } else if (useKmeans) {
            auto R = kmeans_full(rng, X, k, 100, 1, mr);
            idx = keel::move(R.idx);
        } else {
            idx = cluster_from_linkage(linkageTree, k, 0.0, "inconsistent", mr);
        }
        if (isCH)       cv[j] = evalCalinskiHarabasz(X, idx, k);
        else if (isDB)  cv[j] = evalDaviesBouldin(X, idx, k);
        else            cv[j] = evalMeanSilhouette(X, idx, k);
        solutions.push_back(keel::move(idx));
    }

    keel::usize bestIdx = 0;
    bool found = false;
    for (keel::usize j = 0; j < m; ++j) {
        if (!keel::is_finite(cv[j])) continue;
        if (!found || (isDB ? (cv[j] < cv[bestIdx]) : (cv[j] > cv[bestIdx]))) {
            bestIdx = j;
            found = true;
        }
    }

    Value eva = Value::structure(mr);
    eva.field("NumObservations") = Value::scalar(keel::as<keel::f64>(n), mr);
    eva.field("InspectedK") = keel::move(inspectedK);
    eva.field("CriterionValues") = keel::move(critVals);
    eva.field("OptimalK") = Value::scalar(found ? keel::as<keel::f64>(klist[bestIdx])
                                                : keel::Limits<keel::f64>::quiet_nan(), mr);
    eva.field("OptimalIndex") = Value::scalar(found ? keel::as<keel::f64>(bestIdx + 1)
                                                    : keel::Limits<keel::f64>::quiet_nan(), mr);
    eva.field("OptimalY") = found ? solutions[bestIdx] : Value::matrix(0, 0, ValueType::DOUBLE, mr);
    eva.field("CriterionName") = Value::fromString(
        isCH ? "CalinskiHarabasz" : (isDB ? "DaviesBouldin" : "Silhouette"), mr);
    eva.field("ClusteringFunction") = Value::fromString(clustName, mr);
    return eva;
}

} // namespace numkit::stats
