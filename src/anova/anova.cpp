// toolboxes/stats/src/anova/anova.cpp

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/anova/anova.hpp>

#include <numkit/stats/distributions/fisher_f.hpp>
#include <numkit/stats/distributions/chi2.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "anova_detail.hpp"

namespace numkit::stats {


std::tuple<keel::f64, keel::f64, keel::f64, keel::f64, keel::f64, keel::f64>
anova1(const Value &y, const Value &group, keel::Allocator *mr)
{
    auto buckets = bucket(y, group);
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    if (buckets.size() < 2)
        return std::make_tuple(nan, nan, 0.0, 0.0, nan, nan);

    keel::usize N = 0;
    keel::f64 grandSum = 0.0;
    keel::Array<keel::f64> means(buckets.size());
    for (keel::usize k = 0; k < buckets.size(); ++k) {
        const auto &v = buckets[k].values;
        keel::f64 s = 0.0;
        for (keel::f64 x : v) s += x;
        means[k] = (v.empty() ? 0.0 : s / keel::f64(v.size()));
        N += v.size();
        grandSum += s;
    }
    const keel::f64 grandMean = grandSum / keel::f64(N);

    keel::f64 ssB = 0.0, ssW = 0.0;
    for (keel::usize k = 0; k < buckets.size(); ++k) {
        const auto &v = buckets[k].values;
        ssB += keel::f64(v.size()) * (means[k] - grandMean) * (means[k] - grandMean);
        for (keel::f64 x : v) {
            const keel::f64 d = x - means[k];
            ssW += d * d;
        }
    }
    const keel::f64 dfB = keel::f64(buckets.size() - 1);
    const keel::f64 dfW = keel::f64(N - buckets.size());
    if (dfW <= 0.0)
        return std::make_tuple(nan, nan, dfB, dfW, ssB, ssW);

    const keel::f64 msB = ssB / dfB;
    const keel::f64 msW = ssW / dfW;
    if (msW <= 0.0)
        return std::make_tuple(0.0, std::numeric_limits<keel::f64>::infinity(),
                               dfB, dfW, ssB, ssW);
    const keel::f64 F = msB / msW;

    Value Fv = Value::scalar(F, mr);
    const keel::f64 cdf = fcdf(Fv, dfB, dfW, mr).toScalar();
    const keel::f64 p = std::max(0.0, 1.0 - cdf);
    return std::make_tuple(p, F, dfB, dfW, ssB, ssW);
}

namespace {

void anova2_argcheck(const Value &Y)
{
    if (Y.dims().ndim() != 2)
        throw Error("anova2: input must be a 2D matrix",
                    0, 0, "anova2", "", "numkit:anova2:notMatrix");
    const keel::usize m = keel::as<keel::usize>(Y.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(Y.dims().dim(1));
    if (m < 2 || n < 2)
        throw Error("anova2: input must be at least 2x2",
                    0, 0, "anova2", "", "numkit:anova2:tooSmall");
}

void friedman_argcheck(keel::i32 reps)
{
    if (reps < 1)
        throw Error("friedman: reps must be a positive integer",
                    0, 0, "friedman", "", "numkit:friedman:reps");
}

} // namespace

// ── anova2 (two-way ANOVA without replication, reps=1) ───────────────
//
// Y is m × n: rows = levels of factor A, columns = levels of factor B.
// Returns (p_cols, p_rows, F_cols, F_rows, df_cols, df_rows, df_err,
//          ss_cols, ss_rows, ss_err).
// Reference engine convention: anova2 puts COLUMNS factor first (factor B),
// then ROWS factor (factor A).
std::tuple<keel::f64, keel::f64, keel::f64, keel::f64,
           keel::f64, keel::f64, keel::f64,
           keel::f64, keel::f64, keel::f64>
anova2(const Value &Y, keel::Allocator *mr)
{
    anova2_argcheck(Y);
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::usize m = keel::as<keel::usize>(Y.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(Y.dims().dim(1));

    const keel::f64 *yd = Y.doubleData();
    const keel::f64 N = keel::as<keel::f64>(m * n);

    // Row means (size m), column means (size n), grand mean.
    keel::Array<keel::f64> rowSum(m, 0.0);
    keel::Array<keel::f64> colSum(n, 0.0);
    keel::f64 grandSum = 0.0;
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < m; ++i) {
            const keel::f64 v = yd[i + j * m];
            rowSum[i] += v;
            colSum[j] += v;
            grandSum += v;
        }
    const keel::f64 grandMean = grandSum / N;
    keel::Array<keel::f64> rowMean(m), colMean(n);
    for (keel::usize i = 0; i < m; ++i) rowMean[i] = rowSum[i] / keel::as<keel::f64>(n);
    for (keel::usize j = 0; j < n; ++j) colMean[j] = colSum[j] / keel::as<keel::f64>(m);

    // Sum of squares.
    keel::f64 ssRows = 0.0, ssCols = 0.0, ssTotal = 0.0;
    for (keel::usize i = 0; i < m; ++i) {
        const keel::f64 d = rowMean[i] - grandMean;
        ssRows += keel::as<keel::f64>(n) * d * d;
    }
    for (keel::usize j = 0; j < n; ++j) {
        const keel::f64 d = colMean[j] - grandMean;
        ssCols += keel::as<keel::f64>(m) * d * d;
    }
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < m; ++i) {
            const keel::f64 d = yd[i + j * m] - grandMean;
            ssTotal += d * d;
        }
    const keel::f64 ssErr = ssTotal - ssRows - ssCols;

    const keel::f64 dfRows = keel::as<keel::f64>(m - 1);
    const keel::f64 dfCols = keel::as<keel::f64>(n - 1);
    const keel::f64 dfErr  = keel::as<keel::f64>((m - 1) * (n - 1));
    if (dfErr <= 0.0)
        return std::make_tuple(nan, nan, nan, nan, dfCols, dfRows, dfErr,
                               ssCols, ssRows, ssErr);

    const keel::f64 msErr = ssErr / dfErr;
    if (msErr <= 0.0) {
        const keel::f64 inf = std::numeric_limits<keel::f64>::infinity();
        return std::make_tuple(0.0, 0.0, inf, inf, dfCols, dfRows, dfErr,
                               ssCols, ssRows, ssErr);
    }

    const keel::f64 Fcols = (ssCols / dfCols) / msErr;
    const keel::f64 Frows = (ssRows / dfRows) / msErr;
    Value FcolsV = Value::scalar(Fcols, mr);
    Value FrowsV = Value::scalar(Frows, mr);
    const keel::f64 pCols = std::max(0.0, 1.0 - fcdf(FcolsV, dfCols, dfErr, mr).toScalar());
    const keel::f64 pRows = std::max(0.0, 1.0 - fcdf(FrowsV, dfRows, dfErr, mr).toScalar());

    return std::make_tuple(pCols, pRows, Fcols, Frows,
                           dfCols, dfRows, dfErr,
                           ssCols, ssRows, ssErr);
}

std::tuple<Value, Value, Value, Value>
kruskalwallis(const Value &y, const Value &group, keel::Allocator *mr)
{
    auto buckets = bucket(y, group);
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    if (buckets.size() < 2)
        return std::make_tuple(Value::scalar(nan, mr),
                               Value::scalar(nan, mr),
                               Value::scalar(0.0, mr),
                               Value::scalar(0.0, mr));

    // Flatten to single (value, group_idx) array for ranking.
    struct V { keel::f64 val; keel::usize g; };
    keel::Array<V> all;
    keel::usize N = 0;
    for (keel::usize k = 0; k < buckets.size(); ++k) {
        for (keel::f64 v : buckets[k].values) all.push_back({v, k});
        N += buckets[k].values.size();
    }
    if (N == 0) return std::make_tuple(Value::scalar(nan, mr),
                                       Value::scalar(nan, mr),
                                       Value::scalar(0.0, mr),
                                       Value::scalar(0.0, mr));

    keel::Array<keel::usize> ord(N);
    for (keel::usize i = 0; i < N; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(),
              [&](keel::usize a, keel::usize b) { return all[a].val < all[b].val; });

    // Assign mid-ranks for ties.
    keel::Array<keel::f64> ranks(N);
    keel::Array<keel::usize> tieGroupSizes;
    keel::usize i = 0;
    while (i < N) {
        keel::usize j = i + 1;
        while (j < N && all[ord[j]].val == all[ord[i]].val) ++j;
        const keel::f64 avg = keel::as<keel::f64>(i + j + 1) / 2.0;
        for (keel::usize k = i; k < j; ++k) ranks[ord[k]] = avg;
        if (j - i > 1) tieGroupSizes.push_back(j - i);
        i = j;
    }

    // Sum ranks by group.
    keel::Array<keel::f64> R(buckets.size(), 0.0);
    keel::Array<keel::usize> ng(buckets.size(), 0);
    for (keel::usize k = 0; k < N; ++k) {
        R[all[k].g] += ranks[k];
        ng[all[k].g]++;
    }

    keel::f64 sumR2_n = 0.0;
    for (keel::usize g = 0; g < buckets.size(); ++g)
        if (ng[g] > 0) sumR2_n += R[g] * R[g] / keel::f64(ng[g]);

    const keel::f64 Nd = keel::f64(N);
    keel::f64 H = (12.0 / (Nd * (Nd + 1.0))) * sumR2_n - 3.0 * (Nd + 1.0);

    // Tie correction.
    keel::f64 tieSum = 0.0;
    for (keel::usize t : tieGroupSizes) {
        const keel::f64 td = keel::f64(t);
        tieSum += td * td * td - td;
    }
    if (tieSum > 0.0 && Nd > 1.0)
        H /= (1.0 - tieSum / (Nd * Nd * Nd - Nd));

    const keel::f64 df = keel::f64(buckets.size() - 1);
    Value Hv = Value::scalar(H, mr);
    const keel::f64 cdf = chi2cdf(Hv, df, mr).toScalar();
    const keel::f64 p = std::max(0.0, 1.0 - cdf);

    return std::make_tuple(Value::scalar(p, mr),
                           Value::scalar(H, mr),
                           Value::scalar(df, mr),
                           Value::scalar(sumR2_n, mr));
}

std::tuple<Value, Value, Value>
friedman(const Value &x, keel::i32 reps, keel::Allocator *mr)
{
    friedman_argcheck(reps);
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::usize nrows = x.dims().rows();
    const keel::usize k     = x.dims().cols();
    const keel::usize R     = keel::as<keel::usize>(reps);   // replicates per cell
    if (k < 2 || nrows < 2 * R || R == 0 || (nrows % R) != 0)
        return std::make_tuple(Value::scalar(nan, mr), Value::scalar(nan, mr),
                               Value::scalar(0.0, mr));
    const keel::usize n = nrows / R;   // blocks
    const keel::usize M = R * k;       // observations per block

    keel::Array<keel::f64> Rj(k, 0.0);      // column rank sums
    keel::f64              tieTerm = 0.0;   // Σ (t³ − t) over every tie group in every block
    keel::Array<keel::f64> blkVals(M), rank(M);
    keel::Array<keel::usize> blkCols(M), ord(M);
    for (keel::usize b = 0; b < n; ++b) {
        keel::usize idx = 0;
        for (keel::usize c = 0; c < k; ++c) {
            for (keel::usize r = 0; r < R; ++r) {
                blkVals[idx] = x.elemAsDouble(c * nrows + (b * R + r));
                blkCols[idx] = c;
                ord[idx]     = idx;
                ++idx;
            }
        }
        std::sort(ord.begin(), ord.end(), [&](keel::usize a, keel::usize bb) { return blkVals[a] < blkVals[bb]; });
        keel::usize i = 0;                    // mid-ranks within the block (1 .. R*k)
        while (i < M) {
            keel::usize j = i + 1;
            while (j < M && blkVals[ord[j]] == blkVals[ord[i]]) ++j;
            const keel::f64 avg = keel::as<keel::f64>(i + j + 1) / 2.0;
            for (keel::usize t = i; t < j; ++t) rank[ord[t]] = avg;
            const keel::f64 tg = keel::as<keel::f64>(j - i);
            if (tg > 1.0) tieTerm += tg * tg * tg - tg;
            i = j;
        }
        for (keel::usize m = 0; m < M; ++m) Rj[blkCols[m]] += rank[m];
    }

    const keel::f64 nd = keel::as<keel::f64>(n);
    const keel::f64 kd = keel::as<keel::f64>(k);
    const keel::f64 Rd = keel::as<keel::f64>(R);
    const keel::f64 Md = Rd * kd;
    keel::f64 sumRj2 = 0.0;
    for (keel::f64 v : Rj) sumRj2 += v * v;
    keel::f64 Q = 12.0 / (nd * kd * Rd * Rd * (Md + 1.0)) * sumRj2 - 3.0 * nd * (Md + 1.0);
    const keel::f64 C = 1.0 - tieTerm / (nd * Md * (Md * Md - 1.0));   // tie correction
    if (C > 0.0) Q /= C;

    const keel::f64 df = kd - 1.0;
    Value        Qv = Value::scalar(Q, mr);
    const keel::f64 p  = std::max(0.0, 1.0 - chi2cdf(Qv, df, mr).toScalar());
    return std::make_tuple(Value::scalar(p, mr), Value::scalar(Q, mr), Value::scalar(df, mr));
}

Value dummyvar(const Value &group, keel::Allocator *mr)
{
    const keel::usize N = group.numel();
    if (N == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    // Find unique labels, sorted ascending (matches MATLAB).
    keel::Array<keel::f64> uniq;
    uniq.reserve(N);
    for (keel::usize i = 0; i < N; ++i) uniq.push_back(group.elemAsDouble(i));
    std::sort(uniq.begin(), uniq.end());
    uniq.resize(std::unique(uniq.begin(), uniq.end()) - uniq.begin());
    const keel::usize K = uniq.size();

    Value out = Value::matrix(N, K, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N * K; ++i) od[i] = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 li = group.elemAsDouble(i);
        // Linear search across at most K labels.
        for (keel::usize k = 0; k < K; ++k) {
            if (uniq[k] == li) { od[i + k * N] = 1.0; break; }
        }
    }
    return out;
}

} // namespace numkit::stats
