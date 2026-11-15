// toolboxes/stats/src/cluster/distance.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/math.h>
#include <numkit/stats/cluster/distance.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>

#include "distance_detail.hpp"

namespace numkit::stats {

namespace {

void pdist2_argcheck(const Value &X, const Value &Y)
{
    if (X.dims().cols() != Y.dims().cols())
        throw Error("pdist2: column counts must match", 0, 0, "pdist2", "",
                    "numkit:pdist2:size");
}

void mahal_argcheck(const Value &Y, const Value &X)
{
    if (Y.dims().cols() != X.dims().cols())
        throw Error("mahal: Y and X must have same column count",
                    0, 0, "mahal", "", "numkit:mahal:size");
    if (X.dims().rows() < 2)
        throw Error("mahal: X must have at least 2 rows for covariance",
                    0, 0, "mahal", "", "numkit:mahal:size");
}

} // namespace

Value pdist(const Value &X, const std::string &metric, double p, const Value *C_opt, keel::Allocator *mr)
{
    const Metric m = parse_metric(metric);
    const keel::usize M = X.dims().rows();
    const keel::usize D = X.dims().cols();
    if (M < 2) return Value::matrix(1, 0, ValueType::DOUBLE, mr);
    const keel::usize Npairs = M * (M - 1) / 2;
    Value out = Value::matrix(1, Npairs, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> xi(D, scratch.allocator()), xj(D, scratch.allocator());

    if (m == Metric::Mahalanobis) {
        // Build Cinv once.
        keel::Array<double> Cinv(D * D, 0.0, scratch.allocator());
        if (C_opt) {
            const Value &C = *C_opt;
            if (C.dims().rows() != D || C.dims().cols() != D)
                throw Error("pdist: mahalanobis covariance C must be D×D",
                            0, 0, "pdist", "", "numkit:pdist:mahal");
            for (keel::usize r = 0; r < D; ++r)
                for (keel::usize c = 0; c < D; ++c)
                    Cinv[r * D + c] = C.elemAsDouble(c * D + r);
        } else {
            data_cov(X, Cinv.data(), scratch.allocator());
        }
        invert_small(Cinv.data(), D, scratch.allocator());
        keel::Array<double> diff(D, scratch.allocator());
        keel::usize idx = 0;
        for (keel::usize i = 0; i < M; ++i) {
            read_row(X, i, xi.data());
            for (keel::usize j = i + 1; j < M; ++j) {
                read_row(X, j, xj.data());
                od[idx++] = mahal_distance(xi.data(), xj.data(), D,
                                           Cinv.data(), diff.data());
            }
        }
        return out;
    }

    if (m == Metric::Seuclidean) {
        // Per-coordinate scaling: divide each difference by Sₖ (default
        // Sₖ = per-column sample std of X; explicit scale vector via C_opt).
        keel::Array<double> inv_scale(D, scratch.allocator());
        if (C_opt) {
            const Value &S = *C_opt;
            if (S.numel() != D)
                throw Error("pdist: seuclidean scale must have one element "
                            "per column", 0, 0, "pdist", "",
                            "numkit:pdist:seuclidean");
            for (keel::usize k = 0; k < D; ++k) inv_scale[k] = 1.0 / S.elemAsDouble(k);
        } else {
            keel::Array<double> sd(D, scratch.allocator());
            col_std(X, sd.data(), scratch.allocator());
            for (keel::usize k = 0; k < D; ++k) inv_scale[k] = 1.0 / sd[k];
        }
        keel::usize idx = 0;
        for (keel::usize i = 0; i < M; ++i) {
            read_row(X, i, xi.data());
            for (keel::usize j = i + 1; j < M; ++j) {
                read_row(X, j, xj.data());
                od[idx++] = seuclidean_distance(xi.data(), xj.data(),
                                                inv_scale.data(), D);
            }
        }
        return out;
    }

    if (m == Metric::Spearman) {
        // Tied-rank each row, then correlation distance on the ranks.
        keel::Array<double> R(M * D, scratch.allocator());
        tiedrank_rows(X, R.data(), scratch.allocator());
        keel::usize idx = 0;
        for (keel::usize i = 0; i < M; ++i)
            for (keel::usize j = i + 1; j < M; ++j)
                od[idx++] = correlation_distance(R.data() + i * D,
                                                 R.data() + j * D, D);
        return out;
    }

    keel::usize idx = 0;
    for (keel::usize i = 0; i < M; ++i) {
        read_row(X, i, xi.data());
        for (keel::usize j = i + 1; j < M; ++j) {
            read_row(X, j, xj.data());
            od[idx++] = row_distance(xi.data(), xj.data(), D, m, p);
        }
    }
    return out;
}

// Backward-compat wrapper without C.
Value pdist(const Value &X, const std::string &metric, double p, keel::Allocator *mr)
{
    return pdist(X, metric, p, nullptr, mr);
}

Value pdist2(const Value &X, const Value &Y, const std::string &metric, double p, const Value *C_opt, keel::Allocator *mr)
{
    const Metric m = parse_metric(metric);
    pdist2_argcheck(X, Y);
    const keel::usize Mx = X.dims().rows();
    const keel::usize Dx = X.dims().cols();
    const keel::usize My = Y.dims().rows();
    const keel::usize Dy = Y.dims().cols();
    Value out = Value::matrix(Mx, My, ValueType::DOUBLE, mr);
    if (Mx == 0 || My == 0) return out;
    double *od = out.doubleDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> xi(Dx, scratch.allocator()), yj(Dy, scratch.allocator());

    if (m == Metric::Mahalanobis) {
        // For pdist2 with Mahalanobis, MATLAB uses cov(X) by default
        // (the *first* arg). Verified via R2025b probe.
        keel::Array<double> Cinv(Dx * Dx, 0.0, scratch.allocator());
        if (C_opt) {
            const Value &C = *C_opt;
            if (C.dims().rows() != Dx || C.dims().cols() != Dx)
                throw Error("pdist2: mahalanobis covariance C must be D×D",
                            0, 0, "pdist2", "", "numkit:pdist2:mahal");
            for (keel::usize r = 0; r < Dx; ++r)
                for (keel::usize c = 0; c < Dx; ++c)
                    Cinv[r * Dx + c] = C.elemAsDouble(c * Dx + r);
        } else {
            data_cov(X, Cinv.data(), scratch.allocator());
        }
        invert_small(Cinv.data(), Dx, scratch.allocator());
        keel::Array<double> diff(Dx, scratch.allocator());
        for (keel::usize j = 0; j < My; ++j) {
            read_row(Y, j, yj.data());
            for (keel::usize i = 0; i < Mx; ++i) {
                read_row(X, i, xi.data());
                od[j * Mx + i] = mahal_distance(xi.data(), yj.data(), Dx,
                                                Cinv.data(), diff.data());
            }
        }
        return out;
    }

    if (m == Metric::Seuclidean) {
        // Default scale = per-column sample std of X (the FIRST arg), matching
        // MATLAB; explicit scale vector via C_opt.
        keel::Array<double> inv_scale(Dx, scratch.allocator());
        if (C_opt) {
            const Value &S = *C_opt;
            if (S.numel() != Dx)
                throw Error("pdist2: seuclidean scale must have one element "
                            "per column", 0, 0, "pdist2", "",
                            "numkit:pdist2:seuclidean");
            for (keel::usize k = 0; k < Dx; ++k) inv_scale[k] = 1.0 / S.elemAsDouble(k);
        } else {
            keel::Array<double> sd(Dx, scratch.allocator());
            col_std(X, sd.data(), scratch.allocator());
            for (keel::usize k = 0; k < Dx; ++k) inv_scale[k] = 1.0 / sd[k];
        }
        for (keel::usize j = 0; j < My; ++j) {
            read_row(Y, j, yj.data());
            for (keel::usize i = 0; i < Mx; ++i) {
                read_row(X, i, xi.data());
                od[j * Mx + i] = seuclidean_distance(xi.data(), yj.data(),
                                                     inv_scale.data(), Dx);
            }
        }
        return out;
    }

    if (m == Metric::Spearman) {
        // Tied-rank rows of X and Y independently, then correlation distance.
        keel::Array<double> RX(Mx * Dx, scratch.allocator());
        keel::Array<double> RY(My * Dy, scratch.allocator());
        tiedrank_rows(X, RX.data(), scratch.allocator());
        tiedrank_rows(Y, RY.data(), scratch.allocator());
        for (keel::usize j = 0; j < My; ++j)
            for (keel::usize i = 0; i < Mx; ++i)
                od[j * Mx + i] = correlation_distance(RX.data() + i * Dx,
                                                      RY.data() + j * Dy, Dx);
        return out;
    }

    for (keel::usize j = 0; j < My; ++j) {
        read_row(Y, j, yj.data());
        for (keel::usize i = 0; i < Mx; ++i) {
            read_row(X, i, xi.data());
            od[j * Mx + i] = row_distance(xi.data(), yj.data(), Dx, m, p);
        }
    }
    return out;
}

Value pdist2(const Value &X, const Value &Y, const std::string &metric, double p, keel::Allocator *mr)
{
    return pdist2(X, Y, metric, p, nullptr, mr);
}

// pdist2 with per-column top-k selection ('Smallest' / 'Largest').
// Returns D (k × My, sorted asc/desc) and I (k × My, 1-based row indices
// into X). On exit, D and I are k × My or min(Mx,k) × My when Mx < k.
void pdist2_topk(const Value &X, const Value &Y, const std::string &metric, double p, const Value *C_opt, keel::usize k, bool largest, Value &Dout, Value &Iout, keel::Allocator *mr)
{
    Value full = pdist2(X, Y, metric, p, C_opt, mr);
    const keel::usize Mx = X.dims().rows();
    const keel::usize My = Y.dims().rows();
    const keel::usize kk = std::min(k, Mx);
    Dout = Value::matrix(kk, My, ValueType::DOUBLE, mr);
    Iout = Value::matrix(kk, My, ValueType::DOUBLE, mr);
    double *dd = Dout.doubleDataMut();
    double *ii = Iout.doubleDataMut();
    const double *fd = full.doubleData();

    // Per-column sort.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::pair<double, keel::usize>> col(Mx, scratch.allocator());
    for (keel::usize j = 0; j < My; ++j) {
        for (keel::usize i = 0; i < Mx; ++i) {
            col[i].first = fd[j * Mx + i];
            col[i].second = i;
        }
        if (largest) {
            std::partial_sort(col.begin(), col.begin() + kk, col.end(),
                [](const auto &a, const auto &b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second < b.second;
                });
        } else {
            std::partial_sort(col.begin(), col.begin() + kk, col.end(),
                [](const auto &a, const auto &b) {
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second;
                });
        }
        for (keel::usize r = 0; r < kk; ++r) {
            dd[j * kk + r] = col[r].first;
            ii[j * kk + r] = double(col[r].second + 1);
        }
    }
}

Value squareform(const Value &d, keel::Allocator *mr) {
    // Detect whether `d` is a row vector (pdist form) or a square matrix.
    const auto &dims = d.dims();
    const keel::usize r = dims.rows(), c = dims.cols();
    const keel::usize n = d.numel();

    if (r == 1 || c == 1) {
        // Row/col → square: solve N(N-1)/2 = n.
        const double Nf = (1.0 + std::sqrt(1.0 + 8.0 * double(n))) / 2.0;
        const keel::usize N = (keel::usize)std::round(Nf);
        if (N * (N - 1) / 2 != n)
            throw Error("squareform: vector length must be triangular number",
                        0, 0, "squareform", "", "numkit:squareform:size");
        Value out = Value::matrix(N, N, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        keel::usize k = 0;
        for (keel::usize i = 0; i < N; ++i)
            for (keel::usize j = i + 1; j < N; ++j) {
                const double v = d.elemAsDouble(k++);
                od[j * N + i] = v;  // (i, j)
                od[i * N + j] = v;  // (j, i)
            }
        return out;
    }
    // Square → vector.
    if (r != c)
        throw Error("squareform: matrix must be square", 0, 0, "squareform", "",
                    "numkit:squareform:size");
    const keel::usize N = r;
    Value out = Value::matrix(1, N * (N - 1) / 2, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    keel::usize k = 0;
    for (keel::usize i = 0; i < N; ++i)
        for (keel::usize j = i + 1; j < N; ++j)
            od[k++] = d.elemAsDouble(j * N + i);
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

    // Mean of X.
    keel::Array<double> mu(D, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < Mx; ++i)
        for (keel::usize k = 0; k < D; ++k) mu[k] += X.elemAsDouble(k * Mx + i);
    for (auto &m : mu) m /= double(Mx);

    // MATLAB's mahal formulation (mahal.m): QR of the CENTERED
    // reference, d² = (n−1)·|Rᵀ⁻¹(y−μ)|². Defined for rank-deficient X
    // (the old covariance+Cholesky path threw "not positive definite"
    // on collinear references where MATLAB returns a value —
    // bugs/closed/stats/mahal-singular).
    keel::Array<double> Xc(Mx * D, scratch.allocator());
    for (keel::usize i = 0; i < Mx; ++i)
        for (keel::usize k = 0; k < D; ++k)
            Xc[i + k * Mx] = X.elemAsDouble(k * Mx + i) - mu[k];

    // Householder QR (column-major work copy), R kept D×D upper.
    keel::Array<double> R(D * D, 0.0, scratch.allocator());
    {
        double xnorm = 0.0;
        for (keel::usize i = 0; i < Mx * D; ++i) xnorm += Xc[i] * Xc[i];
        (void)xnorm;
        for (keel::usize k = 0; k < D; ++k) {
            double tail_sq = 0.0;
            for (keel::usize i = k + 1; i < Mx; ++i)
                tail_sq += Xc[i + k * Mx] * Xc[i + k * Mx];
            if (tail_sq == 0.0) {
                R[k + k * D] = Xc[k + k * Mx];   // dlarfg tail rule
                continue;
            }
            double norm_sq = tail_sq + Xc[k + k * Mx] * Xc[k + k * Mx];
            const double norm = std::sqrt(norm_sq);
            const double xk = Xc[k + k * Mx];
            const double alpha = (xk >= 0.0) ? -norm : norm;
            keel::Array<double> v(Mx - k, scratch.allocator());
            v[0] = xk - alpha;
            for (keel::usize i = k + 1; i < Mx; ++i) v[i - k] = Xc[i + k * Mx];
            double v_sq = 0.0;
            for (double e : v) v_sq += e * e;
            const double tau = 2.0 / v_sq;
            for (keel::usize j = k; j < D; ++j) {
                double dot = 0.0;
                for (keel::usize i = k; i < Mx; ++i)
                    dot += v[i - k] * Xc[i + j * Mx];
                const double s = tau * dot;
                for (keel::usize i = k; i < Mx; ++i)
                    Xc[i + j * Mx] -= s * v[i - k];
            }
            R[k + k * D] = alpha;
            for (keel::usize j = k + 1; j < D; ++j)
                R[k + j * D] = Xc[k + j * Mx];
        }
    }

    // Solve Rᵀ z = (y−μ) by forward substitution; a (near-)zero diagonal
    // (rank-deficient direction) contributes z_i = 0, mirroring mldivide's
    // thresholded triangular solve.
    const double diagTol = 1e-12;
    Value out = Value::matrix(My, 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    keel::Array<double> dy(D, scratch.allocator());
    keel::Array<double> z(D, 0.0, scratch.allocator());
    for (keel::usize r = 0; r < My; ++r) {
        for (keel::usize k = 0; k < D; ++k) dy[k] = Y.elemAsDouble(k * My + r) - mu[k];
        std::fill(z.begin(), z.end(), 0.0);
        for (keel::usize i = 0; i < D; ++i) {
            const double Rii = R[i + i * D];
            if (keel::abs(Rii) < diagTol)
                continue;
            double s = dy[i];
            for (keel::usize k = 0; k < i; ++k) s -= R[k + i * D] * z[k];
            z[i] = s / Rii;
        }
        double m2 = 0.0;
        for (auto v : z) m2 += v * v;
        od[r] = m2 * double(Mx - 1);
    }
    return out;
}

} // namespace numkit::stats
