// toolboxes/signal/src/cluster/kmeans_reg.cpp
//
// CallContext register half of cluster/kmeans.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/cluster/kmeans.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "cluster/kmeans_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>


namespace numkit::stats {

namespace detail {

void kmeans_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "kmeans");  // no max: name-value tail in the reference
    const keel::i32 K = (keel::i32)args[1].toScalar();
    keel::i32 max_iter  = 100;
    keel::i32 replicates = 1;
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        s.to_ascii_lower_in_place();
        return s;
    };
    for (keel::usize i = 2; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        const keel::String key = lower(args[i].toString());
        const Value &v = args[i + 1];
        if      (key == "maxiter")    max_iter   = (keel::i32)v.toScalar();
        else if (key == "replicates") replicates = (keel::i32)v.toScalar();
        else if (key == "distance") {
            const keel::String dn = lower(v.toString());
            if (dn != "sqeuclidean" && dn != "squaredeuclidean")
                throw Error(keel::format("kmeans: only Distance = 'sqeuclidean' is "
                            "implemented (got '{}')", dn),
                            0, 0, "kmeans", "", "numkit:kmeans:distance");
        }
        else if (key == "start") {
            const keel::String sn = lower(v.toString());
            if (sn != "plus")
                throw Error(keel::format("kmeans: only Start = 'plus' is implemented "
                            "(got '{}')", sn),
                            0, 0, "kmeans", "", "numkit:kmeans:start");
        }
        // 'Display' / 'EmptyAction' / 'OnlinePhase' / 'Options' silently
        // accepted (no-op).
    }
    auto R = kmeans_full(ctx.engine->rng(), args[0], K, max_iter, replicates, ctx.engine->allocator());
    outs[0] = keel::move(R.idx);
    if (nargout > 1) outs[1] = keel::move(R.C);
    if (nargout > 2) outs[2] = keel::move(R.sumd);
    if (nargout > 3) outs[3] = keel::move(R.D);
}

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

} // namespace

void evalclusters_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 15, "evalclusters");
    auto *mr = ctx.engine->allocator();
    const Value &X = args[0];
    if (!X.isNumeric() || X.isEmpty())
        throw Error("evalclusters: X must be a nonempty numeric matrix",
                    0, 0, "evalclusters", "", "numkit:evalclusters:badX");

    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        s.to_ascii_lower_in_place();
        return s;
    };

    const keel::String critRaw = args[2].toString();
    const keel::String crit = lower(critRaw.as_ref());
    const bool isCH  = (crit == "calinskiharabasz");
    const bool isDB  = (crit == "daviesbouldin");
    const bool isSil = (crit == "silhouette");
    if (!isCH && !isDB && !isSil)
        throw Error(keel::format("evalclusters: unsupported criterion '{}'", critRaw),
                    0, 0, "evalclusters", "", "numkit:evalclusters:badCriterion");

    keel::Array<keel::i32> klist;
    for (keel::usize i = 3; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        if (lower(args[i].toString().as_ref()) == "klist") {
            const Value &kv = args[i + 1];
            for (keel::usize j = 0; j < kv.numel(); ++j)
                klist.push_back(keel::as<keel::i32>(kv.elemAsDouble(j)));
        }
    }

    const bool precomputed = args[1].isNumeric();
    keel::String clustName = precomputed ? keel::String::from("custom") : args[1].toString();
    if (klist.empty()) {
        if (precomputed) {
            const keel::usize m = args[1].dims().cols();
            for (keel::usize j = 0; j < m; ++j)
                klist.push_back(keel::as<keel::i32>(j + 1));
        } else {
            throw Error("evalclusters: 'KList' must be specified when clust is a function name",
                        0, 0, "evalclusters", "", "numkit:evalclusters:missingKList");
        }
    }

    const keel::usize n = X.dims().rows();
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
                id[r] = args[1].elemAsDouble(j * n + r);
        } else {
            auto R = kmeans_full(ctx.engine->rng(), X, k, 100, 1, mr);
            idx = keel::move(R.idx);
        }
        if (isCH)       cv[j] = evalCalinskiHarabasz(X, idx, k);
        else if (isDB)  cv[j] = evalDaviesBouldin(X, idx, k);
        else            cv[j] = evalMeanSilhouette(X, idx, k);
        solutions.push_back(keel::move(idx));
    }

    // OptimalK: max for CalinskiHarabasz / Silhouette, min for DaviesBouldin.
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
    outs[0] = keel::move(eva);
}

} // namespace detail

} // namespace numkit::stats
