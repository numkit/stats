// toolboxes/.../distance_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by distance.cpp + distance_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)



namespace numkit::stats {

namespace {

// Distance metric IDs.
enum class Metric { Euclidean, SqEuclidean, Cityblock, Chebychev, Minkowski,
                    Cosine, Correlation, Hamming, Jaccard, Mahalanobis,
                    Seuclidean, Spearman };

[[maybe_unused]] Metric parse_metric(keel::StringRef raw) {
    // Case-insensitive + accept MATLAB-style aliases (e.g. 'sqEuclidean'
    // for squared-euclidean).
    keel::String s; s.reserve(raw.size());
    for (char c : raw) s.push_back(keel::to_ascii_lower(c));
    if (s == "euclidean")        return Metric::Euclidean;
    if (s == "squaredeuclidean" ||
        s == "sqeuclidean")      return Metric::SqEuclidean;
    if (s == "cityblock")        return Metric::Cityblock;
    if (s == "chebychev")        return Metric::Chebychev;
    if (s == "minkowski")        return Metric::Minkowski;
    if (s == "cosine")           return Metric::Cosine;
    if (s == "correlation")      return Metric::Correlation;
    if (s == "hamming")          return Metric::Hamming;
    if (s == "jaccard")          return Metric::Jaccard;
    if (s == "mahalanobis")      return Metric::Mahalanobis;
    if (s == "seuclidean")       return Metric::Seuclidean;
    if (s == "spearman")         return Metric::Spearman;
    throw Error(keel::format("pdist: unknown metric '{}'", raw), 0, 0, "pdist", "",
                "numkit:pdist:badmetric");
}

// Invert a small dense matrix in-place (D × D, row-major in `A`) using
// Gauss-Jordan with partial pivoting. Uses the supplied scratch arena
// for the 2D-wide augmented matrix. Throws on singular matrix.
inline void invert_small(keel::f64 *A, keel::usize D,
                         keel::Allocator *scratch_mr)
{
    keel::Array<keel::f64> aug(D * 2 * D, 0.0, scratch_mr);
    // Build augmented [A | I].
    for (keel::usize r = 0; r < D; ++r) {
        for (keel::usize c = 0; c < D; ++c) aug[r * 2 * D + c] = A[r * D + c];
        aug[r * 2 * D + D + r] = 1.0;
    }
    for (keel::usize k = 0; k < D; ++k) {
        // Partial pivot.
        keel::usize piv = k;
        keel::f64 pmax = keel::abs(aug[k * 2 * D + k]);
        for (keel::usize r = k + 1; r < D; ++r) {
            const keel::f64 v = keel::abs(aug[r * 2 * D + k]);
            if (v > pmax) { pmax = v; piv = r; }
        }
        if (pmax < 1e-14)
            throw Error("pdist: covariance matrix is singular for "
                        "Mahalanobis distance",
                        0, 0, "pdist", "", "numkit:pdist:singular");
        if (piv != k) {
            for (keel::usize c = 0; c < 2 * D; ++c)
                keel::swap(aug[k * 2 * D + c], aug[piv * 2 * D + c]);
        }
        const keel::f64 pivval = aug[k * 2 * D + k];
        const keel::f64 inv_pivval = 1.0 / pivval;
        for (keel::usize c = 0; c < 2 * D; ++c) aug[k * 2 * D + c] *= inv_pivval;
        for (keel::usize r = 0; r < D; ++r) {
            if (r == k) continue;
            const keel::f64 factor = aug[r * 2 * D + k];
            if (factor == 0.0) continue;
            for (keel::usize c = 0; c < 2 * D; ++c)
                aug[r * 2 * D + c] -= factor * aug[k * 2 * D + c];
        }
    }
    for (keel::usize r = 0; r < D; ++r)
        for (keel::usize c = 0; c < D; ++c)
            A[r * D + c] = aug[r * 2 * D + D + c];
}

// Compute the unbiased covariance matrix of the M×D data `X` (column-major
// in the Value) into the supplied D*D row-major buffer.
[[maybe_unused]] void data_cov(const Value &X, keel::f64 *cov_out,
              keel::Allocator *scratch_mr)
{
    const keel::usize M = X.dims().rows();
    const keel::usize D = X.dims().cols();
    keel::Array<keel::f64> mean(D, 0.0, scratch_mr);
    for (keel::usize c = 0; c < D; ++c) {
        for (keel::usize r = 0; r < M; ++r) mean[c] += X.elemAsDouble(c * M + r);
        mean[c] /= keel::as<keel::f64>(M);
    }
    keel::fill(cov_out, cov_out + D * D, 0.0);
    keel::Array<keel::f64> dev(D, scratch_mr);
    for (keel::usize r = 0; r < M; ++r) {
        for (keel::usize c = 0; c < D; ++c)
            dev[c] = X.elemAsDouble(c * M + r) - mean[c];
        for (keel::usize a = 0; a < D; ++a)
            for (keel::usize b = 0; b < D; ++b)
                cov_out[a * D + b] += dev[a] * dev[b];
    }
    const keel::f64 inv_dof = (M > 1) ? 1.0 / keel::as<keel::f64>(M - 1) : 1.0;
    for (keel::usize i = 0; i < D * D; ++i) cov_out[i] *= inv_dof;
}

// Mahalanobis distance: d = sqrt((u - v)ᵀ · C_inv · (u - v)). Uses the
// caller's `diff` scratch buffer to avoid per-call allocation.
inline keel::f64 mahal_distance(const keel::f64 *u, const keel::f64 *v, keel::usize D,
                              const keel::f64 *Cinv, keel::f64 *diff)
{
    for (keel::usize k = 0; k < D; ++k) diff[k] = u[k] - v[k];
    keel::f64 s = 0.0;
    for (keel::usize a = 0; a < D; ++a) {
        keel::f64 row = 0.0;
        for (keel::usize b = 0; b < D; ++b) row += Cinv[a * D + b] * diff[b];
        s += diff[a] * row;
    }
    return keel::sqrt(keel::max(0.0, s));
}

// Standardized-euclidean distance: √Σ((xₖ−yₖ)·inv_scaleₖ)². `inv_scale[k]`
// is 1/Sₖ where S defaults to the per-column sample std of the data set.
inline keel::f64 seuclidean_distance(const keel::f64 *x, const keel::f64 *y,
                                  const keel::f64 *inv_scale, keel::usize D)
{
    keel::f64 s = 0.0;
    for (keel::usize k = 0; k < D; ++k) {
        const keel::f64 d = (x[k] - y[k]) * inv_scale[k];
        s += d * d;
    }
    return keel::sqrt(s);
}

// Correlation distance 1 − corr(x, y). NaN when either vector is constant
// (zero variance) — MATLAB's behaviour. Shared by the 'correlation' metric
// and (applied to tied ranks) by 'spearman'.
inline keel::f64 correlation_distance(const keel::f64 *x, const keel::f64 *y, keel::usize D)
{
    keel::f64 mx = 0, my = 0;
    for (keel::usize k = 0; k < D; ++k) { mx += x[k]; my += y[k]; }
    mx /= keel::as<keel::f64>(D);
    my /= keel::as<keel::f64>(D);
    keel::f64 xy = 0, xx = 0, yy = 0;
    for (keel::usize k = 0; k < D; ++k) {
        const keel::f64 dx = x[k] - mx, dy = y[k] - my;
        xy += dx * dy; xx += dx * dx; yy += dy * dy;
    }
    const keel::f64 denom = keel::sqrt(xx) * keel::sqrt(yy);
    return (denom > 0.0) ? (1.0 - xy / denom)
                         : keel::Limits<keel::f64>::quiet_nan();
}

inline keel::f64 row_distance(const keel::f64 *x, const keel::f64 *y, keel::usize D,
                           Metric m, keel::f64 p)
{
    switch (m) {
        case Metric::Euclidean: {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < D; ++k) { keel::f64 d = x[k] - y[k]; s += d * d; }
            return keel::sqrt(s);
        }
        case Metric::SqEuclidean: {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < D; ++k) { keel::f64 d = x[k] - y[k]; s += d * d; }
            return s;
        }
        case Metric::Cityblock: {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < D; ++k) s += keel::abs(x[k] - y[k]);
            return s;
        }
        case Metric::Chebychev: {
            keel::f64 m_ = 0.0;
            for (keel::usize k = 0; k < D; ++k) {
                keel::f64 d = keel::abs(x[k] - y[k]);
                if (d > m_) m_ = d;
            }
            return m_;
        }
        case Metric::Minkowski: {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < D; ++k) {
                keel::f64 d = keel::abs(x[k] - y[k]);
                s += keel::pow(d, p);
            }
            return keel::pow(s, 1.0 / p);
        }
        case Metric::Cosine: {
            keel::f64 xy = 0, xx = 0, yy = 0;
            for (keel::usize k = 0; k < D; ++k) {
                xy += x[k] * y[k];
                xx += x[k] * x[k];
                yy += y[k] * y[k];
            }
            const keel::f64 denom = keel::sqrt(xx) * keel::sqrt(yy);
            // Zero-norm row → cosine undefined; MATLAB returns NaN (not 1).
            return (denom > 0.0) ? (1.0 - xy / denom)
                                 : keel::Limits<keel::f64>::quiet_nan();
        }
        case Metric::Correlation:
            return correlation_distance(x, y, D);
        case Metric::Hamming: {
            keel::i32 diff = 0;
            for (keel::usize k = 0; k < D; ++k) if (x[k] != y[k]) ++diff;
            return keel::f64(diff) / keel::f64(D);
        }
        case Metric::Jaccard: {
            keel::i32 diff = 0, considered = 0;
            for (keel::usize k = 0; k < D; ++k) {
                const bool xi = x[k] != 0.0, yi = y[k] != 0.0;
                if (xi || yi) {
                    ++considered;
                    if (xi != yi) ++diff;
                }
            }
            return considered > 0 ? keel::f64(diff) / keel::f64(considered) : 0.0;
        }
        case Metric::Mahalanobis:
        case Metric::Seuclidean:
        case Metric::Spearman:
            // These need data-set-wide pre-computation (covariance, per-column
            // scale, or tied ranks) and are handled by dedicated paths in
            // pdist/pdist2 — never via the pairwise row_distance switch.
            break;
    }
    return 0.0;
}

// Read row r of an M×D Value into a flat raw buffer. Column-major storage.
inline void read_row(const Value &X, keel::usize r, keel::f64 *out) {
    const keel::usize M = X.dims().rows();
    const keel::usize D = X.dims().cols();
    for (keel::usize k = 0; k < D; ++k) out[k] = X.elemAsDouble(k * M + r);
}

// Per-column sample standard deviation (divisor n−1) of an M×D Value into
// `scale_out` (length D). The default 'seuclidean' coordinate scale.
inline void col_std(const Value &X, keel::f64 *scale_out,
                    keel::Allocator * /*scratch_mr*/)
{
    const keel::usize M = X.dims().rows();
    const keel::usize D = X.dims().cols();
    for (keel::usize c = 0; c < D; ++c) {
        keel::f64 mean = 0.0;
        for (keel::usize r = 0; r < M; ++r) mean += X.elemAsDouble(c * M + r);
        mean /= keel::as<keel::f64>(M);
        keel::f64 ss = 0.0;
        for (keel::usize r = 0; r < M; ++r) {
            const keel::f64 d = X.elemAsDouble(c * M + r) - mean;
            ss += d * d;
        }
        scale_out[c] = (M > 1) ? keel::sqrt(ss / keel::as<keel::f64>(M - 1)) : 0.0;
    }
}

// Average-tie ranks (MATLAB `tiedrank`) of the D values in `v`, written to
// `r` in the same order. Ties share the mean of their 1-based ranks.
inline void tiedrank_row(const keel::f64 *v, keel::usize D, keel::f64 *r,
                         keel::Allocator *scratch_mr)
{
    keel::Array<keel::usize> idx(D, scratch_mr);
    for (keel::usize k = 0; k < D; ++k) idx[k] = k;
    keel::sort(idx.begin(), idx.end(),
              [&](keel::usize a, keel::usize b) { return v[a] < v[b]; });
    keel::usize i = 0;
    while (i < D) {
        keel::usize j = i;
        while (j + 1 < D && v[idx[j + 1]] == v[idx[i]]) ++j;
        const keel::f64 avg = (keel::as<keel::f64>(i + 1) +
                            keel::as<keel::f64>(j + 1)) / 2.0;
        for (keel::usize t = i; t <= j; ++t) r[idx[t]] = avg;
        i = j + 1;
    }
}

// Build the M×D row-major tied-rank transform of X (each row ranked across
// its D coordinates) into `R`. Backs the 'spearman' metric.
inline void tiedrank_rows(const Value &X, keel::f64 *R,
                          keel::Allocator *scratch_mr)
{
    const keel::usize M = X.dims().rows();
    const keel::usize D = X.dims().cols();
    keel::Array<keel::f64> row(D, scratch_mr);
    for (keel::usize r = 0; r < M; ++r) {
        read_row(X, r, row.data());
        tiedrank_row(row.data(), D, R + r * D, scratch_mr);
    }
}

} // anonymous

// Internal pdist/pdist2 overloads (with optional Mahalanobis covariance C_opt)
// + the top-k variant, used by the reg adapters. Defs in distance.cpp (external).
Value pdist(const Value &X, keel::StringRef metric, keel::f64 p,
            const Value *C_opt, keel::Allocator *mr);
Value pdist2(const Value &X, const Value &Y, keel::StringRef metric, keel::f64 p,
             const Value *C_opt, keel::Allocator *mr);
void pdist2_topk(const Value &X, const Value &Y, keel::StringRef metric,
                 keel::f64 p, const Value *C_opt, keel::usize k, bool largest,
                 Value &Dout, Value &Iout, keel::Allocator *mr);

} // namespace numkit::stats
