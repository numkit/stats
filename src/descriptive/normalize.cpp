// toolboxes/stats/src/descriptive/normalize.cpp
//
// normalize / rescale / zscore — basic data pre-processing.
// Operate column-wise on matrices, whole-vector on 1-D inputs (MATLAB
// convention). Output is always DOUBLE regardless of input class.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "normalize_detail.hpp"

namespace numkit::stats {

namespace {

// Iterate over columns of A and call `op(col_idx, ptr_to_double_buf, len)`
// for each. Vectors are treated as a single "column" of length numel.
template <typename Op>
void forEachColumn(const Value &A, Op op) {
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    const keel::usize N = A.numel();
    if (N == H * W && (H == 1 || W == 1)) {
        // 1-D: pull whole vector as a single column.
        keel::Array<double> buf(N);
        for (keel::usize i = 0; i < N; ++i) buf[i] = A.elemAsDouble(i);
        op(0, buf.data(), N);
        return;
    }
    // Matrix / N-D: walk columns. Column j spans linear indices
    // [j*H .. j*H + H - 1]; for N-D we still treat the FIRST dim as
    // "rows" (first non-singleton — same simplification as nk's
    // mean/var on 2-D path).
    if (N != H * W)
        throw Error("normalize/rescale/zscore: only 1-D or 2-D inputs supported",
                    0, 0, "normalize", "", "numkit:normalize:nd");
    keel::Array<double> buf(H);
    for (keel::usize j = 0; j < W; ++j) {
        for (keel::usize i = 0; i < H; ++i) buf[i] = A.elemAsDouble(j * H + i);
        op(j, buf.data(), H);
    }
}

void writeColumn(Value &out, keel::usize colIdx, const double *src, keel::usize len,
                 keel::usize H, keel::usize W)
{
    double *od = out.doubleDataMut();
    if (W == 1 || H == 1) {
        for (keel::usize i = 0; i < len; ++i) od[i] = src[i];
    } else {
        for (keel::usize i = 0; i < len; ++i) od[colIdx * H + i] = src[i];
    }
}

double colMean(const double *x, keel::usize n) {
    double s = 0.0;
    for (keel::usize i = 0; i < n; ++i) s += x[i];
    return (n > 0) ? s / keel::as<double>(n) : 0.0;
}

double colStdPop(const double *x, keel::usize n) {
    if (n == 0) return 0.0;
    const double m = colMean(x, n);
    double s = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const double d = x[i] - m;
        s += d * d;
    }
    return std::sqrt(s / keel::as<double>(n));
}

// Sample (N-1) standard deviation — MATLAB's default for std/zscore/normalize.
double colStdSample(const double *x, keel::usize n) {
    if (n < 2) return 0.0;
    const double m = colMean(x, n);
    double s = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const double d = x[i] - m;
        s += d * d;
    }
    return std::sqrt(s / keel::as<double>(n - 1));
}

double colMedian(keel::Array<double> v) {
    std::sort(v.begin(), v.end());
    const keel::usize n = v.size();
    if (n == 0) return 0.0;
    return (n & 1) ? v[n / 2]
                   : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double colIQR(keel::Array<double> v) {
    std::sort(v.begin(), v.end());
    const keel::usize n = v.size();
    if (n < 2) return 0.0;
    // MATLAB's iqr/normalize use the prctile convention: the i-th sorted
    // value (0-based) sits at cumulative fraction (i+0.5)/n; linear-interp
    // between bracketing samples, clamped to the extremes. (NOT the R-type-7
    // (n-1)*p rule, which gave a different IQR for small n — e.g. for
    // [1 2 4 8 16 32] prctile gives Q1=2,Q3=16,IQR=14, not 2.5/14/11.5.)
    auto q = [&](double pfrac) {
        const double pos = pfrac * double(n) - 0.5;   // index in [-0.5, n-0.5]
        if (pos <= 0.0)               return v[0];
        if (pos >= double(n - 1))     return v[n - 1];
        const keel::usize lo   = keel::as<keel::usize>(std::floor(pos));
        const double frac = pos - double(lo);
        return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
    };
    return q(0.75) - q(0.25);
}

bool methodMatches(keel::StringRef m, const char *want) {
    if (m.size() != keel::StringRef(want).size()) return false;
    for (keel::usize i = 0; i < m.size(); ++i) {
        char a = m[i], b = want[i];
        if (a >= 'A' && a <= 'Z') a = char(a + 32);
        if (b >= 'A' && b <= 'Z') b = char(b + 32);
        if (a != b) return false;
    }
    return true;
}

} // anonymous

NormalizeResult normalize(const Value &A, keel::StringRef method,
                          const Value &param, keel::Allocator *mr)
{
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    NormalizeResult R;
    R.n = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value &out = R.n;
    if (H == 0 || W == 0) {
        R.c = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        R.s = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return R;
    }
    // Centering (C) and scaling (S) outputs for MATLAB [N,C,S]=normalize:
    // one value per operating slice — 1×1 for a vector, 1×W for a matrix
    // (column-wise) — such that N == (A - C) ./ S. The reg adapter discards
    // c/s when nargout < 2/3.
    const keel::usize csW = (H == 1 || W == 1) ? 1 : W;
    R.c = Value::matrix(1, csW, ValueType::DOUBLE, mr);
    R.s = Value::matrix(1, csW, ValueType::DOUBLE, mr);
    double *cOutD = R.c.doubleDataMut();
    double *sOutD = R.s.doubleDataMut();

    const keel::StringRef m = method.empty() ? keel::StringRef("zscore") : method;

    // Decode the optional method parameter once. A string param (e.g.
    // 'first'/'median') vs a numeric param (range bounds / norm-p / divisor).
    const bool hasParam = !param.isEmpty();
    const bool paramIsStr = hasParam && (param.isChar() || param.isString());
    keel::String paramStr;
    if (paramIsStr) {
        paramStr = param.toString();
        for (char &c : paramStr) if (c >= 'A' && c <= 'Z') c = char(c + 32);
    }
    const double paramNum = (hasParam && !paramIsStr) ? param.elemAsDouble(0) : 0.0;

    forEachColumn(A, [&](keel::usize j, double *col, keel::usize n) {
        keel::Array<double> y(n);
        double cVal = 0.0, sVal = 1.0;  // centering / scaling -> [N,C,S]
        if (methodMatches(m, "zscore")) {
            // Default 'std': centre by mean, scale by sample (N-1) std.
            // 'robust': centre by median, scale by the (raw) median absolute
            // deviation MAD = median(|x - median(x)|) — MATLAB R2025b.
            if (paramIsStr && paramStr == "robust") {
                keel::Array<double> tmp(keel::ArrayRef<double>(col, n));
                const double med = colMedian(tmp.clone());
                for (double &v : tmp) v = std::fabs(v - med);
                const double mad = colMedian(std::move(tmp));
                const double inv = (mad != 0.0) ? 1.0 / mad : 0.0;
                cVal = med; sVal = mad;
                for (keel::usize i = 0; i < n; ++i) y[i] = (col[i] - med) * inv;
            } else {
                const double mu = colMean(col, n);
                const double sd = colStdSample(col, n);  // MATLAB default: N-1
                const double inv = (sd != 0.0) ? 1.0 / sd : 0.0;
                cVal = mu; sVal = sd;
                for (keel::usize i = 0; i < n; ++i) y[i] = (col[i] - mu) * inv;
            }
        } else if (methodMatches(m, "center")) {
            // default 'mean'; 'median' or a numeric centre also accepted.
            double c;
            if (hasParam && !paramIsStr)              c = paramNum;
            else if (paramIsStr && paramStr == "median") {
                keel::Array<double> tmp(keel::ArrayRef<double>(col, n)); c = colMedian(std::move(tmp));
            } else                                     c = colMean(col, n);
            cVal = c; sVal = 1.0;
            for (keel::usize i = 0; i < n; ++i) y[i] = col[i] - c;
        } else if (methodMatches(m, "scale")) {
            // default 'std'; 'first'/'iqr'/'mad' or a numeric divisor.
            double sc;
            if (hasParam && !paramIsStr)               sc = paramNum;
            else if (paramIsStr && paramStr == "first") sc = col[0];
            else if (paramIsStr && paramStr == "iqr") {
                keel::Array<double> tmp(keel::ArrayRef<double>(col, n));  sc = colIQR(std::move(tmp));
            } else if (paramIsStr && paramStr == "mad") {
                keel::Array<double> tmp(keel::ArrayRef<double>(col, n));
                const double med = colMedian(tmp.clone());
                for (double &v : tmp) v = std::fabs(v - med);
                sc = colMedian(std::move(tmp));
            } else                                      sc = colStdSample(col, n);
            const double inv = (sc != 0.0) ? 1.0 / sc : 0.0;
            cVal = 0.0; sVal = sc;
            for (keel::usize i = 0; i < n; ++i) y[i] = col[i] * inv;
        } else if (methodMatches(m, "range")) {
            // default [0 1]; custom [lo hi] supported.
            double rlo = 0.0, rhi = 1.0;
            if (hasParam && !paramIsStr && param.numel() >= 2) {
                rlo = param.elemAsDouble(0);
                rhi = param.elemAsDouble(1);
            }
            double lo = col[0], hi = col[0];
            for (keel::usize i = 1; i < n; ++i) {
                if (col[i] < lo) lo = col[i];
                if (col[i] > hi) hi = col[i];
            }
            const double r = hi - lo;
            cVal = lo; sVal = r;  // default [0 1]: C=min, S=max-min
            if (r != 0.0) {
                const double scl = (rhi - rlo) / r;
                for (keel::usize i = 0; i < n; ++i) y[i] = rlo + (col[i] - lo) * scl;
            } else {
                for (keel::usize i = 0; i < n; ++i) y[i] = rlo;
            }
        } else if (methodMatches(m, "norm")) {
            // default p=2; p=1 / p=Inf / general p supported.
            const double p = hasParam && !paramIsStr ? paramNum : 2.0;
            double L;
            if (std::isinf(p)) {
                L = 0.0;
                for (keel::usize i = 0; i < n; ++i) L = std::max(L, std::fabs(col[i]));
            } else if (p == 1.0) {
                L = 0.0;
                for (keel::usize i = 0; i < n; ++i) L += std::fabs(col[i]);
            } else if (p == 2.0) {
                double s = 0.0;
                for (keel::usize i = 0; i < n; ++i) s += col[i] * col[i];
                L = std::sqrt(s);
            } else {
                double s = 0.0;
                for (keel::usize i = 0; i < n; ++i) s += std::pow(std::fabs(col[i]), p);
                L = std::pow(s, 1.0 / p);
            }
            const double inv = (L != 0.0) ? 1.0 / L : 0.0;
            cVal = 0.0; sVal = L;
            for (keel::usize i = 0; i < n; ++i) y[i] = col[i] * inv;
        } else if (methodMatches(m, "medianiqr")) {
            keel::Array<double> tmp(keel::ArrayRef<double>(col, n));
            const double med = colMedian(tmp.clone());
            const double iqr = colIQR(std::move(tmp));
            const double inv = (iqr != 0.0) ? 1.0 / iqr : 0.0;
            cVal = med; sVal = iqr;
            for (keel::usize i = 0; i < n; ++i) y[i] = (col[i] - med) * inv;
        } else {
            throw Error(keel::format("normalize: unknown method '{}'", m),
                        0, 0, "normalize", "", "numkit:normalize:method");
        }
        if (cOutD) cOutD[j] = cVal;
        if (sOutD) sOutD[j] = sVal;
        writeColumn(out, j, y.data(), n, H, W);
    });
    return R;
}

namespace {

// rescale_argcheck: the input-type contract, crossed by every call
// path. R2025b rejects non-numeric/non-logical first arguments
// (probed: rescale('x') -> "First argument must be a real numeric or
// logical array"); nk's column walker assumed doubles.
void rescale_argcheck(const Value &A)
{
    if (!(A.isNumeric() || A.isLogical()) || A.isComplex())
        throw Error("rescale: first argument must be a real numeric or "
                    "logical array.",
                    0, 0, "rescale", "", "numkit:rescale:InvalidA");
}

} // namespace

Value rescale(const Value &A, double lo, double hi,
              double inputMin, double inputMax, keel::Allocator *mr)
{
    rescale_argcheck(A);
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0) return out;
    const bool haveImn = !std::isnan(inputMin);
    const bool haveImx = !std::isnan(inputMax);
    forEachColumn(A, [&](keel::usize j, double *col, keel::usize n) {
        // Input range: explicit 'InputMin'/'InputMax' override the data
        // min/max. When given, values are clamped to [mn, mx] (MATLAB).
        double mn = col[0], mx = col[0];
        for (keel::usize i = 1; i < n; ++i) {
            if (col[i] < mn) mn = col[i];
            if (col[i] > mx) mx = col[i];
        }
        if (haveImn) mn = inputMin;
        if (haveImx) mx = inputMax;
        const double r = mx - mn;
        keel::Array<double> y(n);
        if (r == 0.0) {
            std::fill(y.begin(), y.end(), lo);
        } else {
            const double scl = (hi - lo) / r;
            for (keel::usize i = 0; i < n; ++i) {
                double v = col[i];
                if (v < mn) v = mn;     // clamp to input range
                if (v > mx) v = mx;
                y[i] = lo + (v - mn) * scl;
            }
        }
        writeColumn(out, j, y.data(), n, H, W);
    });
    return out;
}

// Flag-aware, dim-aware z-score. flag 0 (default) -> sample std (N-1),
// flag 1 -> population std (N). dim<=0 auto-selects the first non-singleton
// dimension (MATLAB convention); dim 1 = columns, dim 2 = rows.
// When muOut / sigmaOut are non-null they receive the per-slice mean and
// standard deviation, matching MATLAB's [Z, MU, SIGMA] = zscore(X). They have
// the operating dimension collapsed to length 1 (row vector for dim 1,
// column vector for dim 2).
Value zscoreCore(const Value &A, int flag, int dim,
                 Value *muOut, Value *sigmaOut, keel::Allocator *mr)
{
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0) {
        if (muOut)    *muOut    = Value::matrix(H, W, ValueType::DOUBLE, mr);
        if (sigmaOut) *sigmaOut = Value::matrix(H, W, ValueType::DOUBLE, mr);
        return out;
    }
    double *od = out.doubleDataMut();
    auto sdf = [&](const double *x, keel::usize n) {
        return (flag == 1) ? colStdPop(x, n) : colStdSample(x, n);
    };
    int opDim = dim;
    if (opDim <= 0) opDim = (H == 1 && W > 1) ? 2 : 1;  // first non-singleton
    if (opDim == 1) {
        if (muOut)    *muOut    = Value::matrix(1, W, ValueType::DOUBLE, mr);
        if (sigmaOut) *sigmaOut = Value::matrix(1, W, ValueType::DOUBLE, mr);
        double *mud = muOut ? muOut->doubleDataMut() : nullptr;
        double *sgd = sigmaOut ? sigmaOut->doubleDataMut() : nullptr;
        keel::Array<double> col(H);
        for (keel::usize j = 0; j < W; ++j) {
            for (keel::usize i = 0; i < H; ++i) col[i] = A.elemAsDouble(j * H + i);
            const double mu = colMean(col.data(), H);
            const double sd = sdf(col.data(), H);
            const double inv = (sd != 0.0) ? 1.0 / sd : 0.0;
            for (keel::usize i = 0; i < H; ++i) od[j * H + i] = (col[i] - mu) * inv;
            if (mud) mud[j] = mu;
            if (sgd) sgd[j] = sd;
        }
    } else {  // opDim == 2, row-wise
        if (muOut)    *muOut    = Value::matrix(H, 1, ValueType::DOUBLE, mr);
        if (sigmaOut) *sigmaOut = Value::matrix(H, 1, ValueType::DOUBLE, mr);
        double *mud = muOut ? muOut->doubleDataMut() : nullptr;
        double *sgd = sigmaOut ? sigmaOut->doubleDataMut() : nullptr;
        keel::Array<double> row(W);
        for (keel::usize i = 0; i < H; ++i) {
            for (keel::usize j = 0; j < W; ++j) row[j] = A.elemAsDouble(j * H + i);
            const double mu = colMean(row.data(), W);
            const double sd = sdf(row.data(), W);
            const double inv = (sd != 0.0) ? 1.0 / sd : 0.0;
            for (keel::usize j = 0; j < W; ++j) od[j * H + i] = (row[j] - mu) * inv;
            if (mud) mud[i] = mu;
            if (sgd) sgd[i] = sd;
        }
    }
    return out;
}

Value zscore(const Value &A, keel::Allocator *mr)
{
    return zscoreCore(A, /*flag=*/0, /*dim=*/0, /*muOut=*/nullptr,
                      /*sigmaOut=*/nullptr, mr);
}

} // namespace numkit::stats
