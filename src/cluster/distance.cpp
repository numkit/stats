// toolboxes/stats/src/cluster/distance.cpp
//
// Pairwise distances (pdist, pdist2, pdist2_topk, squareform, mahal)
// delegating pure numerical kernels to keel::stats.

#include <keel/arena.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/cluster/distance.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


#include "distance_detail.hpp"

namespace numkit::stats {

namespace {

keel::stats::DistanceMetric toKeelMetric(Metric m) noexcept
{
    switch (m) {
        case Metric::Euclidean:   return keel::stats::DistanceMetric::kEuclidean;
        case Metric::SqEuclidean: return keel::stats::DistanceMetric::kSqEuclidean;
        case Metric::Cityblock:   return keel::stats::DistanceMetric::kCityblock;
        case Metric::Chebychev:   return keel::stats::DistanceMetric::kChebychev;
        case Metric::Minkowski:   return keel::stats::DistanceMetric::kMinkowski;
        case Metric::Cosine:      return keel::stats::DistanceMetric::kCosine;
        case Metric::Correlation: return keel::stats::DistanceMetric::kCorrelation;
        case Metric::Hamming:     return keel::stats::DistanceMetric::kHamming;
        case Metric::Jaccard:     return keel::stats::DistanceMetric::kJaccard;
        case Metric::Mahalanobis: return keel::stats::DistanceMetric::kMahalanobis;
        case Metric::Seuclidean:  return keel::stats::DistanceMetric::kSeuclidean;
        case Metric::Spearman:    return keel::stats::DistanceMetric::kSpearman;
    }
    return keel::stats::DistanceMetric::kEuclidean;
}

keel::Array<keel::f64> readColMajor(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, 0.0, alloc);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

void validateExtraParam(Metric m, keel::usize D, const Value *C_opt, const char *fn,
                        keel::Array<keel::f64> &extra)
{
    if (!C_opt) return;
    if (m == Metric::Mahalanobis) {
        if (C_opt->dims().rows() != D || C_opt->dims().cols() != D)
            throw Error(keel::format("{}: mahalanobis covariance C must be D×D", fn),
                        0, 0, fn, "", keel::format("numkit:{}:mahal", fn));
        extra.resize(D * D);
        for (keel::usize i = 0; i < D * D; ++i) extra[i] = C_opt->elemAsDouble(i);
    } else if (m == Metric::Seuclidean) {
        if (C_opt->numel() != D)
            throw Error(keel::format("{}: seuclidean scale must have one element per column", fn),
                        0, 0, fn, "", keel::format("numkit:{}:seuclidean", fn));
        extra.resize(D);
        for (keel::usize i = 0; i < D; ++i) extra[i] = C_opt->elemAsDouble(i);
    }
}

} // namespace

Value pdist(const Value &X, keel::StringRef metric, keel::f64 p, const Value *C_opt, keel::Allocator *mr)
{
    const Metric m = parse_metric(metric);
    const keel::usize M = X.dims().rows();
    const keel::usize D = X.dims().cols();
    if (M < 2) return Value::matrix(1, 0, ValueType::DOUBLE, mr);

    const keel::usize Npairs = M * (M - 1) / 2;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> extra(scratch.allocator());
    validateExtraParam(m, D, C_opt, "pdist", extra);
    const auto xv = readColMajor(X, scratch.allocator());

    Value out = Value::matrix(1, Npairs, ValueType::DOUBLE, mr);
    const auto st = keel::stats::pdist(xv.as_ref(), M, D, toKeelMetric(m), p, extra.as_ref(),
                                       keel::MutArrayRef<keel::f64>(out.doubleDataMut(), Npairs),
                                       scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("pdist: covariance matrix is singular for Mahalanobis distance",
                    0, 0, "pdist", "", "numkit:pdist:singular");
    return out;
}

Value pdist(const Value &X, keel::StringRef metric, keel::f64 p, keel::Allocator *mr)
{
    return pdist(X, metric, p, nullptr, mr);
}

Value pdist2(const Value &X, const Value &Y, keel::StringRef metric, keel::f64 p, const Value *C_opt, keel::Allocator *mr)
{
    const Metric m = parse_metric(metric);
    if (X.dims().cols() != Y.dims().cols())
        throw Error("pdist2: column counts must match", 0, 0, "pdist2", "", "numkit:pdist2:size");
    const keel::usize Mx = X.dims().rows();
    const keel::usize Dx = X.dims().cols();
    const keel::usize My = Y.dims().rows();
    Value out = Value::matrix(Mx, My, ValueType::DOUBLE, mr);
    if (Mx == 0 || My == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> extra(scratch.allocator());
    validateExtraParam(m, Dx, C_opt, "pdist2", extra);
    const auto xv = readColMajor(X, scratch.allocator());
    const auto yv = readColMajor(Y, scratch.allocator());

    const auto st = keel::stats::pdist2(xv.as_ref(), Mx, yv.as_ref(), My, Dx, toKeelMetric(m), p,
                                        extra.as_ref(),
                                        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), Mx * My),
                                        scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("pdist: covariance matrix is singular for Mahalanobis distance",
                    0, 0, "pdist", "", "numkit:pdist:singular");
    return out;
}

Value pdist2(const Value &X, const Value &Y, keel::StringRef metric, keel::f64 p, keel::Allocator *mr)
{
    return pdist2(X, Y, metric, p, nullptr, mr);
}

void pdist2_topk(const Value &X, const Value &Y, keel::StringRef metric, keel::f64 p,
                 const Value *C_opt, keel::usize k, bool largest, Value &Dout, Value &Iout,
                 keel::Allocator *mr)
{
    Value full = pdist2(X, Y, metric, p, C_opt, mr);
    const keel::usize Mx = X.dims().rows();
    const keel::usize My = Y.dims().rows();
    const keel::usize kk = keel::min(k, Mx);
    Dout = Value::matrix(kk, My, ValueType::DOUBLE, mr);
    Iout = Value::matrix(kk, My, ValueType::DOUBLE, mr);
    keel::f64 *dd = Dout.doubleDataMut();
    keel::f64 *ii = Iout.doubleDataMut();
    const keel::f64 *fd = full.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::Pair<keel::f64, keel::usize>> col(Mx, scratch.allocator());
    for (keel::usize j = 0; j < My; ++j) {
        for (keel::usize i = 0; i < Mx; ++i) {
            col[i].first = fd[j * Mx + i];
            col[i].second = i;
        }
        if (largest) {
            keel::partial_sort(col.begin(), col.begin() + kk, col.end(),
                [](const auto &a, const auto &b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second < b.second;
                });
        } else {
            keel::partial_sort(col.begin(), col.begin() + kk, col.end(),
                [](const auto &a, const auto &b) {
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second;
                });
        }
        for (keel::usize r = 0; r < kk; ++r) {
            dd[j * kk + r] = col[r].first;
            ii[j * kk + r] = keel::as<keel::f64>(col[r].second + 1);
        }
    }
}

Value squareform(const Value &d, keel::Allocator *mr)
{
    const auto &dims = d.dims();
    const keel::usize r = dims.rows(), c = dims.cols();
    const keel::usize n = d.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    if (r == 1 || c == 1) {
        const keel::usize N = keel::as<keel::usize>(keel::round(0.5 * (1.0 + keel::sqrt(1.0 + 8.0 * keel::as<keel::f64>(n)))));
        if (N * (N - 1) / 2 != n)
            throw Error("squareform: vector length must be triangular number",
                        0, 0, "squareform", "", "numkit:squareform:size");
        const auto dv = readColMajor(d, scratch.allocator());
        Value out = Value::matrix(N, N, ValueType::DOUBLE, mr);
        (void)keel::stats::squareform_to_matrix(dv.as_ref(), N,
                                                keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N * N));
        return out;
    }
    if (r != c)
        throw Error("squareform: matrix must be square", 0, 0, "squareform", "", "numkit:squareform:size");
    const keel::usize N = r;
    const keel::usize npairs = N * (N - 1) / 2;
    const auto mv = readColMajor(d, scratch.allocator());
    Value out = Value::matrix(1, npairs, ValueType::DOUBLE, mr);
    (void)keel::stats::squareform_to_condensed(mv.as_ref(), N,
                                               keel::MutArrayRef<keel::f64>(out.doubleDataMut(), npairs));
    return out;
}

Value mahal(const Value &Y, const Value &X, keel::Allocator *mr)
{
    const keel::usize Mx = X.dims().rows();
    const keel::usize D  = X.dims().cols();
    const keel::usize My = Y.dims().rows();
    if (Y.dims().cols() != D)
        throw Error("mahal: Y and X must have same column count",
                    0, 0, "mahal", "", "numkit:mahal:size");
    if (Mx < 2)
        throw Error("mahal: X must have at least 2 rows for covariance",
                    0, 0, "mahal", "", "numkit:mahal:size");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto xv = readColMajor(X, scratch.allocator());
    const auto yv = readColMajor(Y, scratch.allocator());
    Value out = Value::matrix(My, 1, ValueType::DOUBLE, mr);
    (void)keel::stats::mahal_qr(yv.as_ref(), My, xv.as_ref(), Mx, D,
                                keel::MutArrayRef<keel::f64>(out.doubleDataMut(), My),
                                scratch.allocator());
    return out;
}

} // namespace numkit::stats
