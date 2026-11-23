// toolboxes/stats/src/descriptive/descriptive_extras.cpp
//
// Descriptive stats extras (B2): bounds, iqr, maxk, mink, rmse.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/stats/distributions/normal.hpp>   // norminv (via descriptive_detail.hpp)
#include <numkit/stats/distributions/students_t.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value_type.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>


#include "descriptive_extras_detail.hpp"
#include "descriptive_detail.hpp"   // narrowToSingle

namespace numkit::stats {


// ── bounds ────────────────────────────────────────────────────────────
keel::Tuple<Value, Value>
bounds(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = resolveDim(x, dim, "bounds");
    auto lo = applyAlongDim(x, d,
        [](keel::usize, const keel::f64 *s, keel::usize n) { return sliceMin(s, n); }, mr);
    auto hi = applyAlongDim(x, d,
        [](keel::usize, const keel::f64 *s, keel::usize n) { return sliceMax(s, n); }, mr);
    return keel::make_tuple(keel::move(lo), keel::move(hi));
}

// ── iqr ───────────────────────────────────────────────────────────────
Value iqr(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = resolveDim(x, dim, "iqr");
    return applyAlongDim(x, d,
        [](keel::usize, const keel::f64 *s, keel::usize n) -> keel::f64 {
            // MATLAB iqr ignores NaN values.
            keel::Array<keel::f64> buf;
            buf.reserve(n);
            for (keel::usize i = 0; i < n; ++i)
                if (!keel::is_nan(s[i])) buf.push_back(s[i]);
            const keel::usize k = buf.size();
            if (k == 0) return keel::Limits<keel::f64>::quiet_nan();
            keel::Array<keel::f64> buf2 = buf.clone();
            const keel::f64 q3 = sliceQuantile(buf.data(), k, 0.75);
            const keel::f64 q1 = sliceQuantile(buf2.data(), k, 0.25);
            return q3 - q1;
        }, mr);
}


// ── maxk / mink ───────────────────────────────────────────────────────
Value maxk(const Value &x, keel::i32 k, keel::i32 dim, keel::Allocator *mr)
{
    return topKAlongDim(x, dim, k, /*ascending=*/false, "maxk", mr);
}

Value mink(const Value &x, keel::i32 k, keel::i32 dim, keel::Allocator *mr)
{
    return topKAlongDim(x, dim, k, /*ascending=*/true, "mink", mr);
}

namespace {

void rmse_argcheck(const Value &f, const Value &a)
{
    if (f.dims() != a.dims() && !(f.isScalar() || a.isScalar()))
        throw Error("rmse: F and A must have compatible sizes",
                     0, 0, "rmse", "", "numkit:rmse:sizeMismatch");
}

void mape_argcheck(const Value &f, const Value &a)
{
    if (f.dims() != a.dims() && !(f.isScalar() || a.isScalar()))
        throw Error("mape: F and A must have compatible sizes",
                     0, 0, "mape", "", "numkit:mape:sizeMismatch");
}

void partialcorr_of_argcheck(const Value &X, const Value &Y, const Value &Z)
{
    if (X.dims().ndim() != 2 || Y.dims().ndim() != 2 || Z.dims().ndim() != 2)
        throw Error("partialcorr: X, Y, Z must be 2D matrices",
                    0, 0, "partialcorr", "", "numkit:partialcorr:notMatrix");
    const keel::usize m = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize mY = keel::as<keel::usize>(Y.dims().dim(0));
    const keel::usize mZ = keel::as<keel::usize>(Z.dims().dim(0));
    if (mY != m || mZ != m)
        throw Error("partialcorr: X, Y, Z must have the same number of rows",
                    0, 0, "partialcorr", "", "numkit:partialcorr:dimMismatch");
}

} // namespace

// ── rmse ──────────────────────────────────────────────────────────────
Value rmse(const Value &f, const Value &a, keel::i32 dim, keel::Allocator *mr)
{
    rmse_argcheck(f, a);
    // Build the squared-difference array, then reduce.
    const keel::usize n = keel::max(f.numel(), a.numel());
    auto diff = (f.numel() >= a.numel())
                  ? createLike(f, ValueType::DOUBLE, mr)
                  : createLike(a, ValueType::DOUBLE, mr);
    keel::f64 *dst = diff.doubleDataMut();
    if (f.isScalar()) {
        const keel::f64 fs = f.toScalar();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 d = a.elemAsDouble(i) - fs;
            dst[i] = d * d;
        }
    } else if (a.isScalar()) {
        const keel::f64 as = a.toScalar();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 d = f.elemAsDouble(i) - as;
            dst[i] = d * d;
        }
    } else {
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 d = f.elemAsDouble(i) - a.elemAsDouble(i);
            dst[i] = d * d;
        }
    }
    const keel::i32 dResolved = resolveDim(diff, dim, "rmse");
    auto v = applyAlongDim(diff, dResolved,
        [](keel::usize, const keel::f64 *s, keel::usize k) -> keel::f64 {
            if (k == 0) return keel::Limits<keel::f64>::quiet_nan();
            keel::f64 sum = 0.0;
            for (keel::usize i = 0; i < k; ++i) sum += s[i];
            return keel::sqrt(sum / keel::as<keel::f64>(k));
        }, mr);
    return v;
}

// ── mape ──────────────────────────────────────────────────────────────
Value mape(const Value &f, const Value &a, keel::i32 dim, keel::Allocator *mr)
{
    mape_argcheck(f, a);
    const keel::usize n = keel::max(f.numel(), a.numel());
    auto pct = (f.numel() >= a.numel())
                  ? createLike(f, ValueType::DOUBLE, mr)
                  : createLike(a, ValueType::DOUBLE, mr);
    keel::f64 *dst = pct.doubleDataMut();

    if (f.isScalar()) {
        const keel::f64 fs = f.toScalar();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 ai = a.elemAsDouble(i);
            dst[i] = keel::abs((ai - fs) / ai) * 100.0;
        }
    } else if (a.isScalar()) {
        const keel::f64 as = a.toScalar();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 fi = f.elemAsDouble(i);
            dst[i] = keel::abs((as - fi) / as) * 100.0;
        }
    } else {
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 fi = f.elemAsDouble(i);
            const keel::f64 ai = a.elemAsDouble(i);
            dst[i] = keel::abs((ai - fi) / ai) * 100.0;
        }
    }

    const keel::i32 dResolved = resolveDim(pct, dim, "mape");
    return applyAlongDim(pct, dResolved,
        [](keel::usize, const keel::f64 *s, keel::usize k) -> keel::f64 {
            if (k == 0) return keel::Limits<keel::f64>::quiet_nan();
            keel::f64 sum = 0.0;
            for (keel::usize i = 0; i < k; ++i) sum += s[i];
            return sum / keel::as<keel::f64>(k);
        }, mr);
}

// ── partialcorr (regress-out + correlate) ───────────────────────────

Value partialcorr_of(const Value &X, const Value &Y, const Value &Z, keel::Allocator *mr)
{
    partialcorr_of_argcheck(X, Y, Z);
    const keel::usize m = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize pX = keel::as<keel::usize>(X.dims().dim(1));
    const keel::usize pY = keel::as<keel::usize>(Y.dims().dim(1));
    const keel::usize pZ = keel::as<keel::usize>(Z.dims().dim(1));

    // Augment Z with intercept column (column of ones) for proper regression.
    // Z_full is m × (pZ + 1).
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize pZ1 = pZ + 1;
    keel::Array<keel::f64> Zf(m * pZ1, scratch.allocator());
    for (keel::usize i = 0; i < m; ++i) Zf[i + 0 * m] = 1.0;  // intercept
    const keel::f64 *Zd = Z.doubleData();
    for (keel::usize j = 0; j < pZ; ++j)
        for (keel::usize i = 0; i < m; ++i)
            Zf[i + (j + 1) * m] = Zd[i + j * m];

    // Compute Zf' * Zf and its LU once.
    // M = Zf' * Zf  (pZ1 × pZ1).
    keel::Array<keel::f64> M(pZ1 * pZ1, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < pZ1; ++i)
        for (keel::usize j = 0; j < pZ1; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < m; ++k)
                s += Zf[k + i * m] * Zf[k + j * m];
            M[i + j * pZ1] = s;
        }

    auto regressOut = [&](const Value &W) -> keel::Array<keel::f64> {
        const keel::usize p = keel::as<keel::usize>(W.dims().dim(1));
        const keel::f64 *Wd = W.doubleData();
        keel::Array<keel::f64> Res(m * p, 0.0, scratch.allocator());
        // For each column of W:
        for (keel::usize col = 0; col < p; ++col) {
            // b = Zf' * W(:, col)   (pZ1)
            keel::Array<keel::f64> b(pZ1, 0.0, scratch.allocator());
            for (keel::usize i = 0; i < pZ1; ++i)
                for (keel::usize k = 0; k < m; ++k)
                    b[i] += Zf[k + i * m] * Wd[k + col * m];
            // Solve M * coef = b via inline Gaussian elimination.
            keel::Array<keel::f64> Mcopy(keel::ArrayRef<keel::f64>(M.data(), M.size()), scratch.allocator());
            keel::Array<keel::f64> coef = b.clone(scratch.allocator());
            bool singular = false;
            for (keel::usize kc = 0; kc < pZ1 && !singular; ++kc) {
                // Pivot.
                keel::usize piv = kc;
                keel::f64 pmax = keel::abs(Mcopy[kc + kc * pZ1]);
                for (keel::usize r = kc + 1; r < pZ1; ++r) {
                    const keel::f64 v = keel::abs(Mcopy[r + kc * pZ1]);
                    if (v > pmax) { pmax = v; piv = r; }
                }
                if (pmax == 0.0) { singular = true; break; }
                if (piv != kc) {
                    for (keel::usize j = 0; j < pZ1; ++j)
                        keel::swap(Mcopy[kc + j * pZ1], Mcopy[piv + j * pZ1]);
                    keel::swap(coef[kc], coef[piv]);
                }
                const keel::f64 pivVal = Mcopy[kc + kc * pZ1];
                for (keel::usize r = kc + 1; r < pZ1; ++r) {
                    const keel::f64 f = Mcopy[r + kc * pZ1] / pivVal;
                    for (keel::usize j = kc; j < pZ1; ++j)
                        Mcopy[r + j * pZ1] -= f * Mcopy[kc + j * pZ1];
                    coef[r] -= f * coef[kc];
                }
            }
            if (singular) {
                keel::fill(coef.begin(), coef.end(), 0.0);
            } else {
                // Back-substitute.
                for (keel::usize kk = pZ1; kk-- > 0;) {
                    keel::f64 s = coef[kk];
                    for (keel::usize j = kk + 1; j < pZ1; ++j)
                        s -= Mcopy[kk + j * pZ1] * coef[j];
                    coef[kk] = s / Mcopy[kk + kk * pZ1];
                }
            }
            // Residual: W(:, col) - Zf * coef
            for (keel::usize i = 0; i < m; ++i) {
                keel::f64 pred = 0.0;
                for (keel::usize j = 0; j < pZ1; ++j)
                    pred += Zf[i + j * m] * coef[j];
                Res[i + col * m] = Wd[i + col * m] - pred;
            }
        }
        return Res;
    };

    // Build residual matrices.
    auto Xres_data = regressOut(X);
    auto Yres_data = regressOut(Y);

    // Compute correlation matrix between Xres (m×pX) and Yres (m×pY).
    // R[i, j] = corr(Xres(:,i), Yres(:,j)) = cov / (std_x * std_y).
    auto Rout = Value::matrix(pX, pY, ValueType::DOUBLE, mr);
    keel::f64 *R = Rout.doubleDataMut();

    auto colMean = [&](const keel::Array<keel::f64> &v, keel::usize col) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < m; ++i) s += v[i + col * m];
        return s / keel::as<keel::f64>(m);
    };
    auto colStd = [&](const keel::Array<keel::f64> &v, keel::usize col, keel::f64 mean) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < m; ++i) {
            const keel::f64 d = v[i + col * m] - mean;
            s += d * d;
        }
        return keel::sqrt(s / keel::as<keel::f64>(m - 1));
    };

    for (keel::usize i = 0; i < pX; ++i) {
        const keel::f64 mx = colMean(Xres_data, i);
        const keel::f64 sx = colStd(Xres_data, i, mx);
        for (keel::usize j = 0; j < pY; ++j) {
            const keel::f64 my = colMean(Yres_data, j);
            const keel::f64 sy = colStd(Yres_data, j, my);
            keel::f64 cov = 0.0;
            for (keel::usize k = 0; k < m; ++k)
                cov += (Xres_data[k + i * m] - mx) * (Yres_data[k + j * m] - my);
            cov /= keel::as<keel::f64>(m - 1);
            R[i + j * pX] = (sx > 0.0 && sy > 0.0) ? cov / (sx * sy) : keel::kNaN;
        }
    }
    return Rout;
}

// ── partialcorr 1-arg / 2-arg forms ─────────────────────────────────
//
// `partialcorr(X, Z)` is `partialcorr(X, X, Z)` with the diagonal forced
// to exactly 1 (FP cancellation drops self-correlation to 1 - O(ε), but
// MATLAB returns exactly 1).
//
// `partialcorr(X)` residualises on a per-pair-distinct control set:
// for pair (i, j), control = all X columns except i and j (plus
// intercept). Implementation matches the residualise-then-Pearson
// approach used by `partialcorr_of`, but rebuilds the design matrix
// per pair. For small p (typical use ≤ 20) this is cheap; cost is
// O(p² · (p³ + m·p²)).


Value partialcorr_xx(const Value &X, keel::Allocator *mr)
{
    if (X.dims().ndim() != 2)
        throw Error("partialcorr: X must be a 2D matrix",
                    0, 0, "partialcorr", "", "numkit:partialcorr:notMatrix");
    const keel::usize m = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize p = keel::as<keel::usize>(X.dims().dim(1));

    auto Rout = Value::matrix(p, p, ValueType::DOUBLE, mr);
    keel::f64 *R = Rout.doubleDataMut();
    for (keel::usize i = 0; i < p; ++i) R[i + i * p] = 1.0;
    if (p < 2 || m < 2) return Rout;

    const keel::f64 *Xd = X.doubleData();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    // Control matrix: intercept + (p - 2) "other" cols = (p - 1) cols.
    const keel::usize pC = (p >= 2) ? (p - 1) : 1;
    keel::Array<keel::f64> C(m * pC, 0.0, scratch.allocator());
    keel::Array<keel::f64> MM(pC * pC, 0.0, scratch.allocator());
    keel::Array<keel::f64> ri(m, 0.0, scratch.allocator());
    keel::Array<keel::f64> rj(m, 0.0, scratch.allocator());

    for (keel::usize i = 0; i < p; ++i) {
        for (keel::usize j = i + 1; j < p; ++j) {
            // Build C column-major: col 0 = intercept, then X cols ≠ i, j.
            keel::fill(C.begin(), C.end(), 0.0);
            for (keel::usize k = 0; k < m; ++k) C[k + 0 * m] = 1.0;
            keel::usize cidx = 1;
            for (keel::usize k = 0; k < p; ++k) {
                if (k == i || k == j) continue;
                for (keel::usize r = 0; r < m; ++r)
                    C[r + cidx * m] = Xd[r + k * m];
                ++cidx;
            }
            // MM = C' · C  (column-major pC × pC).
            keel::fill(MM.begin(), MM.end(), 0.0);
            for (keel::usize a = 0; a < pC; ++a)
                for (keel::usize b = 0; b < pC; ++b) {
                    keel::f64 s = 0.0;
                    for (keel::usize k = 0; k < m; ++k)
                        s += C[k + a * m] * C[k + b * m];
                    MM[a + b * pC] = s;
                }
            residualiseColumn_local(C.data(), m, pC, MM.data(),
                                     Xd + i * m, ri.data(), mr);
            residualiseColumn_local(C.data(), m, pC, MM.data(),
                                     Xd + j * m, rj.data(), mr);
            const keel::f64 rij = pearsonOnResiduals(ri.data(), rj.data(), m);
            R[i + j * p] = rij;
            R[j + i * p] = rij;
        }
    }
    return Rout;
}

Value partialcorr_xz(const Value &X, const Value &Z, keel::Allocator *mr)
{
    // Equivalent to partialcorr_of(X, X, Z) with diagonal forced to 1
    // (FP cancellation in self-correlation may drift to 1 - O(ε)).
    auto R = partialcorr_of(X, X, Z, mr);
    const keel::usize p = keel::as<keel::usize>(R.dims().dim(0));
    keel::f64 *Rd = R.doubleDataMut();
    for (keel::usize i = 0; i < p; ++i) Rd[i + i * p] = 1.0;
    // Symmetrise: off-diagonal pairs are mathematically equal but may
    // differ by O(ε) due to non-associative summation in cov.
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = i + 1; j < p; ++j) {
            const keel::f64 m = 0.5 * (Rd[i + j * p] + Rd[j + i * p]);
            Rd[i + j * p] = m;
            Rd[j + i * p] = m;
        }
    return R;
}

// ── corr (Pearson alias) ─────────────────────────────────────────────

Value corr_xx(const Value &X, keel::Allocator *mr)
{
    return corrcoef(X, mr);
}

Value corr_xy(const Value &X, const Value &Y, keel::Allocator *mr)
{
    // Pairwise Pearson correlation: X (n×p), Y (n×q) → p×q matrix where
    // out(i,j) = corr(X(:,i), Y(:,j)). Matches MATLAB corr(X, Y).
    // Vector × vector returns a 1×1 matrix (scalar-convertible) —
    // differs from corrcoef(x,y) which returns the 2×2 [x;y] matrix.
    const keel::usize n = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize p = (X.dims().ndim() >= 2)
                            ? keel::as<keel::usize>(X.dims().dim(1)) : 1;
    if (keel::as<keel::usize>(Y.dims().dim(0)) != n)
        throw Error("corr: X and Y must have the same number of rows",
                    0, 0, "corr", "", "numkit:corr:rows");
    const keel::usize q = (Y.dims().ndim() >= 2)
                            ? keel::as<keel::usize>(Y.dims().dim(1)) : 1;

    auto out = Value::matrix(p, q, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    // Pre-compute means + centered-norms per column.
    keel::Array<keel::f64> mx(p), my(q);
    keel::Array<keel::f64> nrmX(p), nrmY(q);
    for (keel::usize i = 0; i < p; ++i) {
        keel::f64 s = 0.0;
        for (keel::usize r = 0; r < n; ++r)
            s += X.elemAsDouble(r + i * n);
        mx[i] = s / keel::f64(n);
    }
    for (keel::usize j = 0; j < q; ++j) {
        keel::f64 s = 0.0;
        for (keel::usize r = 0; r < n; ++r)
            s += Y.elemAsDouble(r + j * n);
        my[j] = s / keel::f64(n);
    }
    for (keel::usize i = 0; i < p; ++i) {
        keel::f64 ss = 0.0;
        for (keel::usize r = 0; r < n; ++r) {
            const keel::f64 d = X.elemAsDouble(r + i * n) - mx[i];
            ss += d * d;
        }
        nrmX[i] = keel::sqrt(ss);
    }
    for (keel::usize j = 0; j < q; ++j) {
        keel::f64 ss = 0.0;
        for (keel::usize r = 0; r < n; ++r) {
            const keel::f64 d = Y.elemAsDouble(r + j * n) - my[j];
            ss += d * d;
        }
        nrmY[j] = keel::sqrt(ss);
    }
    for (keel::usize j = 0; j < q; ++j) {
        for (keel::usize i = 0; i < p; ++i) {
            keel::f64 s = 0.0;
            for (keel::usize r = 0; r < n; ++r) {
                const keel::f64 dx = X.elemAsDouble(r + i * n) - mx[i];
                const keel::f64 dy = Y.elemAsDouble(r + j * n) - my[j];
                s += dx * dy;
            }
            const keel::f64 denom = nrmX[i] * nrmY[j];
            od[i + j * p] = (denom > 0.0) ? (s / denom)
                                          : keel::Limits<keel::f64>::quiet_nan();
        }
    }
    return out;
}

// ── detrend (polynomial trend removal) ──────────────────────────────


Value detrend_of(const Value &x, keel::i32 order, keel::Allocator *mr)
{
    // Simple public form of the full detrend contract (cleaning.cpp):
    // detrend(x, order). Crosses the same float gate / NaN policy /
    // single + N-D handling as the interpreter path.
    const Value ord = Value::scalar(keel::as<keel::f64>(order), mr);
    return detrend_full(x, keel::ArrayRef<Value>(&ord, 1), mr);
}

// Continuous piecewise-linear detrend with breakpoints — MATLAB
// detrend(x, 1, bp). Replicates the classic detrend.m design matrix:
// breakpoints become bp = unique([0; bp; N-1]) (0-based sample offsets);
// the design has one ramp column per segment (rows off..N-1 hold
// (1:M)/M, M = N-off) plus a constant column, and each data column is
// detrended by subtracting its least-squares fit a·(a\col). Operates
// along dim 1 (per column) like the ordinary detrend.
Value detrendBP_of(const Value &x, keel::ArrayRef<keel::f64> bpUser,
                   keel::Allocator *mr)
{
    if (x.numel() == 0) return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    const keel::usize r = keel::as<keel::usize>(x.dims().dim(0));
    const keel::usize c = (x.dims().ndim() >= 2)
                            ? keel::as<keel::usize>(x.dims().dim(1)) : 1;
    const keel::usize N = (r == 1 || c == 1) ? r * c : r;
    if (N < 2) {                       // nothing to fit
        auto out = Value::matrix(r, c, ValueType::DOUBLE, mr);
        const keel::f64 *src0 = x.doubleData();
        keel::f64 *dst0 = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) dst0[i] = src0[i];
        return out;
    }

    // Breakpoint offsets: unique([0; round(bp); N-1]) within (0, N-1).
    keel::Array<keel::f64> bps;
    bps.push_back(0.0);
    for (keel::f64 v : bpUser) {
        const keel::f64 off = keel::floor(v + 0.5);
        if (off > 0.0 && off < keel::as<keel::f64>(N - 1)) bps.push_back(off);
    }
    bps.push_back(keel::as<keel::f64>(N - 1));
    keel::sort(bps.begin(), bps.end());
    bps.resize(keel::unique(bps.begin(), bps.end()) - bps.begin());
    const keel::usize lb = bps.size() - 1;
    const keel::usize k = lb + 1;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> a(N * k, 0.0, scratch.allocator());     // column-major N×k
    for (keel::usize row = 0; row < N; ++row) a[(k - 1) * N + row] = 1.0;
    for (keel::usize kb = 0; kb < lb; ++kb) {
        const keel::usize off = keel::as<keel::usize>(bps[kb]);
        const keel::usize M = N - off;
        for (keel::usize i = 0; i < M; ++i)
            a[kb * N + (off + i)] = keel::as<keel::f64>(i + 1) / keel::as<keel::f64>(M);
    }

    auto out = Value::matrix(r, c, ValueType::DOUBLE, mr);
    const keel::f64 *xd = x.doubleData();
    keel::f64 *od = out.doubleDataMut();
    if (r == 1 || c == 1) {
        detrendColumnBP(a.data(), N, k, xd, od, scratch.allocator());
    } else {
        for (keel::usize j = 0; j < c; ++j)
            detrendColumnBP(a.data(), N, k, xd + j * r, od + j * r, scratch.allocator());
    }
    return out;
}

// ── isoutlier / rmoutliers / fillmissing / rmmissing / standardizeMissing ──

// isoutlier(x) — boolean array marking outliers via median + MAD
// (default MATLAB method: more than 3 scaled MADs from median).
Value isoutlier_of(const Value &x, keel::Allocator *mr)
{
    // Default-method public form: median + 3 scaled MADs, per column
    // (the whole run for vectors), exactly the interpreter default.
    return isoutlier_full(x, keel::ArrayRef<Value>(), false, mr).tf;
}

// ── filloutliers — detect outliers then replace with fillmethod ─────
//
// Detection methods covered:
//   * "median" (default) — outliers > 3 * 1.4826 * MAD from median.
//   * "mean"             — outliers > 3 * std    from mean.
//   * "quartiles"        — outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR].
//
// Fill methods covered (vector-wise, single-column path):
//   * numeric scalar     — constant fill.
//   * "center"           — center value used by detection (median /
//                          mean / 0.5*(Q1+Q3)).
//   * "clip"             — clamp to [L, U] threshold.
//   * "previous"         — last non-outlier value (NaN if leading).
//   * "next"             — first non-outlier value (NaN if trailing).
//   * "nearest"          — closer of prev/next (ties → NEXT, MATLAB).
//   * "linear"           — linear interpolation between flanking
//                          non-outliers; extrapolates ends from the
//                          slope of the nearest interior pair.
//
// Deferred (require extra infrastructure or rare): "spline", "pchip",
// "makima", "movmedian"/"movmean" detection, "grubbs", "gesd",
// SamplePoints / OutlierLocations / MaxNumOutliers / ReplaceValues.
// ThresholdFactor is honoured for "median" and "mean" methods.

Value filloutliers_of(const Value &x,
                      const Value &fillArg,       // string OR numeric scalar
                      keel::StringRef detect,
                      keel::f64 thresholdFactor,
                      keel::f64 loP, keel::f64 hiP,     // used iff detect=="percentiles"
                      keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (n == 0) return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    const keel::f64 *xd = x.doubleData();
    const keel::usize r = keel::as<keel::usize>(x.dims().dim(0));
    const keel::usize c = (x.dims().ndim() >= 2)
                            ? keel::as<keel::usize>(x.dims().dim(1)) : 1;
    auto out = Value::matrix(r, c, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    keel::copy(xd, xd + n, od);

    // Parse fill arg: scalar numeric → constant fill; string → method.
    bool fill_is_constant = false;
    keel::f64 constVal = 0.0;
    keel::String fillMethod;
    if (fillArg.isChar() || fillArg.isString()) {
        fillMethod = fillArg.toString();
        const keel::StringRef fm = fillMethod.as_ref();
        if (fm != "center" && fm != "clip" &&
            fm != "previous" && fm != "next" &&
            fm != "nearest" && fm != "linear")
            throw Error("filloutliers: fillmethod must be a scalar, "
                        "'center', 'clip', 'previous', 'next', "
                        "'nearest', or 'linear'",
                        0, 0, "filloutliers", "",
                        "numkit:filloutliers:fillmethod");
    } else {
        fill_is_constant = true;
        constVal = fillArg.toScalar();
    }

    auto run_col = [&](keel::f64 *p, keel::usize len) {
        FoDetect d = (detect == "percentiles")
                       ? detect_percentile_column(p, len, loP, hiP)
                       : detect_one_column(p, len, detect, thresholdFactor);
        apply_fill(p, len, d.mask, d.center, d.lo, d.hi,
                   fillMethod.as_ref(), constVal, fill_is_constant);
    };
    if (r == 1) {
        run_col(od, c);
    } else {
        for (keel::usize col = 0; col < c; ++col)
            run_col(od + col * r, r);
    }
    return out;
}

// rmoutliers(x) — public simple form of the full rmoutliers contract.
Value rmoutliers_of(const Value &x, keel::Allocator *mr)
{
    return rmoutliers_full(x, keel::ArrayRef<Value>(), mr).B;
}

// fillmissing(x, method[, constant_value]) — replace NaN with method.
// MATLAB-canonical methods: 'constant' (needs value), 'previous',
// 'next', 'nearest', 'linear'. Internal 'mean'/'median' kept as a
// nk convenience (undocumented; use mean(x,'omitnan') +
// 'constant' for portability). Per-column processing matches
// MATLAB's default (each column filled independently). For vectors
// (rows or cols) the per-column scan reduces to a flat scan.

Value fillmissing_of(const Value &x, keel::StringRef method, keel::f64 constVal, keel::Allocator *mr)
{
    // Public simple form of the full fillmissing contract (cleaning.cpp).
    // Only 'constant' takes the numeric third argument; every other
    // method forwards the method string alone (a bare numeric would be
    // parsed as the dimension, exactly like the interpreter path).
    const Value m = Value::fromString(method, mr);
    if (method == "constant") {
        const Value cv = Value::scalar(constVal, mr);
        const Value tail[2] = {m, cv};
        return fillmissing_full(x, keel::ArrayRef<Value>(tail, 2), false, mr).B;
    }
    return fillmissing_full(x, keel::ArrayRef<Value>(&m, 1), false, mr).B;
}

// rmmissing(x) — public simple form of the full rmmissing contract
// (row removal for matrices, entry removal for vectors).
Value rmmissing_of(const Value &x, keel::Allocator *mr)
{
    return rmmissing_full(x, keel::ArrayRef<Value>(), mr).B;
}

// standardizeMissing(x, indicators) — public form of the full
// indicator-vector contract (cleaning.cpp).
Value standardizeMissing_of(const Value &x, const Value &indicators,
                            keel::Allocator *mr)
{
    return standardizeMissing_full(x, indicators, mr);
}

// ── error-metrics family: flags + Weights + the gated computes ──────

ErrorMetricOpts parseErrorMetricOpts(keel::ArrayRef<Value> tail,
                                     keel::usize start, bool forMape,
                                     const Value &F, const Value &A,
                                     keel::usize dimLen, const char *fn)
{
    ErrorMetricOpts o;
    static constexpr keel::StringRef kNanFlags[] = {
        "omitnan", "includenan", "omitmissing", "includemissing"};
    static constexpr keel::StringRef kZeroFlags[] = {"omitzero", "includezero"};
    auto matchFamily = [](const Value &v, keel::ArrayRef<keel::StringRef> list) {
        keel::String low;
        if (v.isChar() || v.isString()) {
            low = v.toString();
            low.to_ascii_lower_in_place();
        }
        keel::i32 hit = -1; keel::i32 cnt = 0;
        const keel::StringRef lref = low.as_ref();
        for (keel::usize i = 0; i < list.size(); ++i)
            if (lref.size() <= list[i].size() && list[i].substr(0, lref.size()) == lref) {
                ++cnt; hit = keel::as<keel::i32>(i);
            }
        return keel::Pair<keel::i32, keel::i32>(hit, cnt);   // (index, match count)
    };
    keel::usize i = start;
    // Up to two positional flags (mape only; rmse takes one). A string
    // that matches no flag starts the Weights NV region when more args
    // follow, and is the lone-string UnknownFlag error otherwise
    // (reference: mape(F,A,'bogus') errors, mape(F,A,'Weights',w) is NV).
    for (keel::i32 flagPos = 0; flagPos < (forMape ? 2 : 1); ++flagPos) {
        if (i >= tail.size() || !(tail[i].isChar() || tail[i].isString()))
            break;
        const auto nanM = matchFamily(tail[i],
            keel::ArrayRef<keel::StringRef>(kNanFlags));
        const auto zeroM = matchFamily(tail[i],
            keel::ArrayRef<keel::StringRef>(kZeroFlags));
        const keel::i32 total = nanM.second + zeroM.second;
        if (total == 0) {
            if (i + 1 >= tail.size()) {
                keel::String rawFlag = tail[i].toString();
                throw Error(keel::format("{}: unrecognized flag '{}' (expected omitnan, "
                                         "includenan{}; prefixes allowed)",
                                         fn, rawFlag.as_ref(),
                                         forMape ? ", omitzero, includezero" : ""),
                            0, 0, fn, "",
                            keel::format("numkit:{}{}", fn,
                                         forMape ? ":unknownFlagMape" : ":unknownFlag"));
            }
            break;   // NV region starts here
        }
        if (forMape && total > 1) {
            keel::String rawFlag = tail[i].toString();
            throw Error(keel::format("{}: ambiguous flag '{}' (matches several of "
                                     "omitnan, includenan, omitmissing, includemissing, "
                                     "omitzero, includezero)",
                                     fn, rawFlag.as_ref()),
                        0, 0, fn, "",
                        keel::format("numkit:{}:unknownFlagMape", fn));
        }
        if (nanM.second == 1)
            o.omitnan = o.omitnan || (nanM.first == 0 || nanM.first == 2);
        else if (zeroM.second == 1)
            o.omitzero = o.omitzero || (zeroM.first == 0);
        ++i;
    }
    // Weights NV tail.
    static constexpr keel::StringRef kWNames[] = {"weights"};
    keel::String badWId = keel::format("numkit:{}:badWeights", fn);
    argcheck::forEachNvPair(
        tail, i, keel::ArrayRef<keel::StringRef>(kWNames), fn,
        badWId.as_ref(),
        [&](keel::usize, const Value &w) {
            if (!w.isNumeric() || w.isComplex() ||
                (w.type() != ValueType::DOUBLE && w.type() != ValueType::SINGLE))
                throw Error(keel::format("{}: Weights must be a double "
                                         "or single array of real nonnegative values "
                                         "(got {})", fn, mtypeName(w.type())),
                            0, 0, fn, "",
                            keel::format("numkit:{}:invalidWeight", fn));
            for (keel::usize k = 0; k < w.numel(); ++k) {
                const keel::f64 wv = w.elemAsDouble(k);
                if (o.omitnan ? (wv < 0.0) : !(wv >= 0.0))
                    throw Error(keel::format("{}: Weights must be "
                                             "nonnegative (got {})", fn, wv),
                                0, 0, fn, "",
                                keel::format("numkit:{}:invalidWeight", fn));
            }
            const keel::usize dn = keel::max(F.numel(), A.numel());
            const bool okSize =
                (w.numel() == dn) ||
                (w.dims().isVector() && w.numel() == dimLen) ||
                (w.numel() == A.numel()) || (w.numel() == F.numel());
            if (!okSize)
                throw Error(keel::format("{}: Weights must have the same "
                                         "size as the inputs or be a vector of length {} "
                                         "along the operating dimension", fn, dimLen),
                            0, 0, fn, "",
                            keel::format("numkit:{}:invalidWeight", fn));
            o.weights = &w;
        });
    return o;
}

namespace {

// Shared reduction for the error metrics: X (already the per-element
// metric), flags, optional weights (indexable along the flat run), and
// the run layout. Returns the reduced value for one run; `madeNaN`
// reports elements that became NaN during the metric computation
// (0-division under omitzero, Inf-Inf) — under omitnan those poison
// the reduced slice per the reference's post-rule.
keel::f64 reduceMetricRun(const keel::f64 *X, const keel::f64 *w, keel::usize n,
                       bool omit, bool anyMadeNaN)
{
    keel::f64 sw = 0.0, sx = 0.0;
    keel::usize cnt = 0;
    for (keel::usize j = 0; j < n; ++j) {
        if (omit && keel::is_nan(X[j])) continue;
        const keel::f64 wj = w ? w[j] : 1.0;
        if (omit && keel::is_nan(wj)) continue;
        sw += wj;
        sx += wj * X[j];
        ++cnt;
    }
    if (cnt == 0 || sw == 0.0)
        return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 r = sx / sw;
    if (omit && anyMadeNaN && keel::is_nan(r))
        return keel::Limits<keel::f64>::quiet_nan();
    return r;
}

} // anonymous

// rmse with the full flag/weight contract (probe71 E-section).
Value rmse(const Value &f, const Value &a, keel::i32 dim, const ErrorMetricOpts &o,
           keel::Allocator *mr)
{
    const char *fn = "rmse";
    argcheck::requireClassSet(
        f, argcheck::classMaskOf(argcheck::ClsDbl) |
              argcheck::classMaskOf(argcheck::ClsSgl) |
              argcheck::classMaskOf(argcheck::ClsCpx),
        fn, "F");
    argcheck::requireClassSet(
        a, argcheck::classMaskOf(argcheck::ClsDbl) |
              argcheck::classMaskOf(argcheck::ClsSgl) |
              argcheck::classMaskOf(argcheck::ClsCpx),
        fn, "A");
    const bool single = (f.type() == ValueType::SINGLE ||
                         a.type() == ValueType::SINGLE);
    // Per-element metric on the (scalar-expanded) difference: |A-F|^2
    // (complex: real^2 + imag^2 — real result, probe71 E07).
    const keel::usize n = keel::max(f.numel(), a.numel());
    keel::ArenaTemp sc = keel::get_scratch(mr);
    keel::Array<keel::f64> X(n, 0.0, sc.allocator());
    const bool fCpx = f.type() == ValueType::COMPLEX;
    const bool aCpx = a.type() == ValueType::COMPLEX;
    for (keel::usize i = 0; i < n; ++i) {
        keel::f64 dr, di = 0.0;
        if (fCpx || aCpx) {
            const keel::c64 fz = fCpx ? f.complexData()[f.numel() == 1 ? 0 : i]
                                    : keel::c64(f.elemAsDouble(f.numel() == 1 ? 0 : i), 0.0);
            const keel::c64 az = aCpx ? a.complexData()[a.numel() == 1 ? 0 : i]
                                    : keel::c64(a.elemAsDouble(a.numel() == 1 ? 0 : i), 0.0);
            dr = az.real() - fz.real();
            di = az.imag() - fz.imag();
        } else {
            dr = a.elemAsDouble(a.numel() == 1 ? 0 : i) -
                 f.elemAsDouble(f.numel() == 1 ? 0 : i);
        }
        X[i] = dr * dr + di * di;
    }
    const bool omit = o.omitnan;
    // Reduce along the run layout.
    const keel::i32 d = (dim == 0) ? 1 : dim;
    const auto &fd = f.dims();
    const keel::usize r = keel::as<keel::usize>(fd.dim(0));
    const keel::usize c = (fd.ndim() >= 2)
        ? keel::as<keel::usize>(fd.numel() / keel::max<keel::usize>(r, 1)) : 1;
    // Treat the inputs in the (dim-1 columns) layout like the rest of
    // the family: flatten when the shapes are vectors (the old
    // behaviour) — the reg normalizes vecdim/'all' away.
    const bool vecInput = (f.dims().isVector() || f.isScalar()) &&
                          (a.dims().isVector() || a.isScalar());
    // Simplify: single run over the whole flattened metric for the
    // vector/scalar shapes (the only shapes the reference documents for
    // error metrics without a dim), and per-column runs for matrices.
    if (vecInput) {
        const keel::f64 *wd = o.weights ? o.weights->doubleData() : nullptr;
        const keel::f64 v = keel::sqrt(reduceMetricRun(X.data(), wd, n, omit, false));
        return single ? narrowToSingle(Value::scalar(v, mr), mr)
                      : Value::scalar(v, mr);
    }
    const bool along2 = (d == 2);
    const keel::usize runLen = along2 ? c : r;
    const keel::usize nRuns = along2 ? r : c;
    Value R = Value::matrix(along2 ? r : 1,
                            along2 ? 1 : c,
                            ValueType::DOUBLE, mr);
    keel::f64 *rd = R.doubleDataMut();
    keel::Array<keel::f64> run(runLen, sc.allocator());
    for (keel::usize k = 0; k < nRuns; ++k) {
        for (keel::usize j = 0; j < runLen; ++j) {
            const keel::usize lin = along2 ? k + j * r : k * runLen + j;
            run[j] = X[lin];
        }
        const keel::f64 *wd = nullptr;
        if (o.weights) {
            if (o.weights->numel() == n)
                wd = o.weights->doubleData() + (along2 ? 0 : k * runLen);
            else
                wd = o.weights->doubleData();   // dim-length vector
        }
        keel::f64 v = reduceMetricRun(run.data(), wd, runLen, omit, false);
        rd[k] = keel::sqrt(v);
    }
    if (single) R = narrowToSingle(keel::move(R), mr);
    return R;
}

// mape with the full flag/weight contract (probe71 E-section).
Value mape(const Value &f, const Value &a, keel::i32 dim, const ErrorMetricOpts &o,
           keel::Allocator *mr)
{
    const char *fn = "mape";
    argcheck::requireClassSet(
        f, argcheck::classMaskOf(argcheck::ClsDbl) |
              argcheck::classMaskOf(argcheck::ClsSgl) |
              argcheck::classMaskOf(argcheck::ClsCpx),
        fn, "F");
    argcheck::requireClassSet(
        a, argcheck::classMaskOf(argcheck::ClsDbl) |
              argcheck::classMaskOf(argcheck::ClsSgl) |
              argcheck::classMaskOf(argcheck::ClsCpx),
        fn, "A");
    const bool single = (f.type() == ValueType::SINGLE ||
                         a.type() == ValueType::SINGLE);
    const keel::usize n = keel::max(f.numel(), a.numel());
    keel::ArenaTemp sc = keel::get_scratch(mr);
    keel::Array<keel::f64> X(n, 0.0, sc.allocator());
    keel::Array<char> madeNaN(n, 0, sc.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 fi = f.elemAsDouble(f.numel() == 1 ? 0 : i);
        const keel::f64 ai = a.elemAsDouble(a.numel() == 1 ? 0 : i);
        // |A-F|/|A| — complex inputs: modulus ratio (real result).
        X[i] = keel::abs(ai - fi) / keel::abs(ai);
        if (o.omitzero && !keel::is_finite(X[i]) && keel::is_finite(ai - fi))
            madeNaN[i] = 1;
    }
    if (o.omitzero)
        for (keel::usize i = 0; i < n; ++i)
            if (madeNaN[i]) X[i] = keel::Limits<keel::f64>::quiet_nan();
    const bool omit = o.omitnan || o.omitzero;
    const bool vecInput = (f.dims().isVector() || f.isScalar()) &&
                          (a.dims().isVector() || a.isScalar());
    if (vecInput) {
        const keel::f64 *wd = o.weights ? o.weights->doubleData() : nullptr;
        keel::f64 v = reduceMetricRun(X.data(), wd, n, omit, false) * 100.0;
        return single ? narrowToSingle(Value::scalar(v, mr), mr)
                      : Value::scalar(v, mr);
    }
    const keel::i32 d = (dim == 0) ? 1 : dim;
    const auto &fd = f.dims();
    const keel::usize r = keel::as<keel::usize>(fd.dim(0));
    const keel::usize c = (fd.ndim() >= 2)
        ? keel::as<keel::usize>(fd.numel() / keel::max<keel::usize>(r, 1)) : 1;
    const bool along2 = (d == 2);
    const keel::usize runLen = along2 ? c : r;
    const keel::usize nRuns = along2 ? r : c;
    Value R = Value::matrix(along2 ? r : 1, along2 ? 1 : c,
                            ValueType::DOUBLE, mr);
    keel::f64 *rd = R.doubleDataMut();
    keel::Array<keel::f64> run(runLen, sc.allocator());
    for (keel::usize k = 0; k < nRuns; ++k) {
        bool anyMade = false;
        for (keel::usize j = 0; j < runLen; ++j) {
            const keel::usize lin = along2 ? k + j * r : k * runLen + j;
            run[j] = X[lin];
            anyMade = anyMade || madeNaN[lin];
        }
        const keel::f64 *wd = nullptr;
        if (o.weights) {
            if (o.weights->numel() == n)
                wd = o.weights->doubleData() + (along2 ? 0 : k * runLen);
            else
                wd = o.weights->doubleData();
        }
        // include-mode: any residual NaN (incl. 0/0 without omitzero)
        // poisons the slice; omit-mode: only metric-made NaNs do.
        keel::f64 v;
        if (!omit) {
            bool anyNaN = false;
            for (keel::f64 xj : run) anyNaN = anyNaN || keel::is_nan(xj);
            v = anyNaN ? keel::Limits<keel::f64>::quiet_nan()
                       : reduceMetricRun(run.data(), wd, runLen, false, false);
        } else {
            v = reduceMetricRun(run.data(), wd, runLen, true, anyMade);
        }
        rd[k] = v * 100.0;
    }
    if (single) R = narrowToSingle(keel::move(R), mr);
    return R;
}

// ── range / mad / geomean / harmmean / moment / trimmean ─────────────

// range(x) = max(x) - min(x) along dim.
Value range_of(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = resolveDim(x, dim, "range");
    return applyAlongDim(x, d,
        [](keel::usize, const keel::f64 *s, keel::usize n) -> keel::f64 {
            if (n == 0) return keel::Limits<keel::f64>::quiet_nan();
            keel::f64 lo = s[0], hi = s[0];
            for (keel::usize i = 1; i < n; ++i) {
                if (s[i] < lo) lo = s[i];
                if (s[i] > hi) hi = s[i];
            }
            return hi - lo;
        }, mr);
}

// mad(x) -- mean absolute deviation: mean(abs(x - mean(x))).
// mad(x, 1) -- median absolute deviation: median(abs(x - median(x))).
Value mad_of(const Value &x, keel::i32 flag, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = resolveDim(x, dim, "mad");
    if (flag == 0) {
        // Mean form. MATLAB mad ignores NaN values.
        return applyAlongDim(x, d,
            [](keel::usize, const keel::f64 *s, keel::usize n) -> keel::f64 {
                keel::Array<keel::f64> v;
                v.reserve(n);
                for (keel::usize i = 0; i < n; ++i)
                    if (!keel::is_nan(s[i])) v.push_back(s[i]);
                const keel::usize k = v.size();
                if (k == 0) return keel::Limits<keel::f64>::quiet_nan();
                keel::f64 mean = 0.0;
                for (keel::usize i = 0; i < k; ++i) mean += v[i];
                mean /= keel::as<keel::f64>(k);
                keel::f64 sum = 0.0;
                for (keel::usize i = 0; i < k; ++i) sum += keel::abs(v[i] - mean);
                return sum / keel::as<keel::f64>(k);
            }, mr);
    }
    // Median form. MATLAB mad ignores NaN values.
    return applyAlongDim(x, d,
        [](keel::usize, const keel::f64 *s, keel::usize n) -> keel::f64 {
            keel::Array<keel::f64> buf;
            buf.reserve(n);
            for (keel::usize i = 0; i < n; ++i)
                if (!keel::is_nan(s[i])) buf.push_back(s[i]);
            const keel::usize k = buf.size();
            if (k == 0) return keel::Limits<keel::f64>::quiet_nan();
            const keel::f64 med = sliceQuantile(buf.data(), k, 0.5);
            keel::Array<keel::f64> dev(k);
            for (keel::usize i = 0; i < k; ++i) dev[i] = keel::abs(buf[i] - med);
            return sliceQuantile(dev.data(), k, 0.5);
        }, mr);
}

// geomean(x) = (prod(x))^(1/n) = exp(mean(log(x))). x must be >= 0.
Value geomean_of(const Value &x, keel::i32 dim, bool omitnan, keel::Allocator *mr)
{
    const keel::i32 d = resolveDim(x, dim, "geomean");
    return applyAlongDim(x, d,
        [omitnan](keel::usize, const keel::f64 *s, keel::usize n) -> keel::f64 {
            keel::f64 sum = 0.0;
            keel::usize k = 0;
            for (keel::usize i = 0; i < n; ++i) {
                if (omitnan && keel::is_nan(s[i])) continue;  // 'omitnan'
                if (s[i] < 0.0)
                    return keel::Limits<keel::f64>::quiet_nan();
                if (s[i] == 0.0) return 0.0;
                sum += keel::log(s[i]);
                ++k;
            }
            if (k == 0) return keel::Limits<keel::f64>::quiet_nan();
            return keel::exp(sum / keel::as<keel::f64>(k));
        }, mr);
}

// harmmean(x) = n / sum(1./x). x must be > 0.
Value harmmean_of(const Value &x, keel::i32 dim, bool omitnan, keel::Allocator *mr)
{
    const keel::i32 d = resolveDim(x, dim, "harmmean");
    return applyAlongDim(x, d,
        [omitnan](keel::usize, const keel::f64 *s, keel::usize n) -> keel::f64 {
            keel::f64 sum = 0.0;
            keel::usize k = 0;
            for (keel::usize i = 0; i < n; ++i) {
                if (omitnan && keel::is_nan(s[i])) continue;  // 'omitnan'
                if (s[i] <= 0.0)
                    return keel::Limits<keel::f64>::quiet_nan();
                sum += 1.0 / s[i];
                ++k;
            }
            if (k == 0) return keel::Limits<keel::f64>::quiet_nan();
            return keel::as<keel::f64>(k) / sum;
        }, mr);
}

// moment(x, k) = mean((x - mean(x))^k). k >= 2.
Value moment_of(const Value &x, keel::i32 order, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = resolveDim(x, dim, "moment");
    const keel::i32 k = order;
    return applyAlongDim(x, d,
        [k](keel::usize, const keel::f64 *s, keel::usize n) -> keel::f64 {
            if (n == 0) return keel::Limits<keel::f64>::quiet_nan();
            if (k < 0) return keel::Limits<keel::f64>::quiet_nan();
            if (k == 0) return 1.0;          // m_0 = 1
            if (k == 1) return 0.0;          // central first moment
            keel::f64 mean = 0.0;
            for (keel::usize i = 0; i < n; ++i) mean += s[i];
            mean /= keel::as<keel::f64>(n);
            keel::f64 sum = 0.0;
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 d2 = s[i] - mean;
                sum += keel::pow(d2, k);
            }
            return sum / keel::as<keel::f64>(n);
        }, mr);
}

// trimmean(x, p) = mean of x after trimming p/2% from each end (p in [0, 100]).
// `useFloor` selects MATLAB's flag: false ('round', the default) rounds the
// per-end count n*p/200 to the nearest integer with ties going DOWN (so
// k = ceil(n*p/200 - 0.5)); true ('floor') takes the plain floor.
Value trimmean_of(const Value &x, keel::f64 pct, keel::i32 dim, bool useFloor,
                  keel::Allocator *mr)
{
    if (pct < 0.0 || pct >= 100.0)
        throw Error("trimmean: percent must be in [0, 100)",
                    0, 0, "trimmean", "", "numkit:trimmean:badPct");
    const keel::i32 d = resolveDim(x, dim, "trimmean");
    const keel::f64 p = pct;
    return applyAlongDim(x, d,
        [p, useFloor](keel::usize, const keel::f64 *s, keel::usize n) -> keel::f64 {
            if (n == 0) return keel::Limits<keel::f64>::quiet_nan();
            // Number of values to trim from EACH end.
            const keel::f64 kf = keel::as<keel::f64>(n) * p / 200.0;
            const keel::usize k = useFloor
                ? keel::as<keel::usize>(keel::floor(kf))
                : keel::as<keel::usize>(keel::max(0.0, keel::ceil(kf - 0.5)));
            if (2 * k >= n) return keel::Limits<keel::f64>::quiet_nan();
            keel::Array<keel::f64> buf(keel::ArrayRef<keel::f64>(s, n));
            keel::sort(buf.begin(), buf.end());
            keel::f64 sum = 0.0;
            for (keel::usize i = k; i < n - k; ++i) sum += buf[i];
            return sum / keel::as<keel::f64>(n - 2 * k);
        }, mr);
}

// ── ksdensity ─────────────────────────────────────────────────────────


// Result struct for the extended ksdensity API. Forward-declared also
// above the engine-adapter `ksdensity_reg` (in the outer namespace).

KsdensityFull
ksdensity_full(const Value &x, const Value &pts, keel::f64 bw_user, keel::StringRef kernel_name, keel::StringRef function_mode, keel::usize numpoints, const Value *weights, keel::Allocator *mr, keel::f64 supportLo, keel::f64 supportHi, keel::StringRef boundary_correction)
{
    const keel::usize N = x.numel();
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    if (N == 0)
        return {Value::matrix(0, 0, ValueType::DOUBLE, mr),
                Value::matrix(0, 0, ValueType::DOUBLE, mr),
                Value::scalar(nan, mr)};

    const KsKernel kernel = parse_ks_kernel(kernel_name);
    const bool hasLo = keel::is_finite(supportLo);
    const bool hasHi = keel::is_finite(supportHi);
    const bool hasBounds = hasLo || hasHi;
    const bool useReflection = hasBounds && boundary_correction.equals_ignore_ascii_case("reflection");
    const bool useLogTransform = hasBounds && !useReflection;

    // Transform helpers for 'log' boundary correction on bounded/semi-bounded support.
    auto fwdTransform = [&](keel::f64 v) -> keel::f64 {
        if (hasLo && hasHi) return keel::log((v - supportLo) / (supportHi - v));
        if (hasLo)          return keel::log(v - supportLo);
        return -keel::log(supportHi - v);
    };
    auto invTransform = [&](keel::f64 y) -> keel::f64 {
        if (hasLo && hasHi) {
            const keel::f64 ey = keel::exp(y);
            return (supportLo + supportHi * ey) / (1.0 + ey);
        }
        if (hasLo) return supportLo + keel::exp(y);
        return supportHi - keel::exp(-y);
    };
    auto jacTransform = [&](keel::f64 v) -> keel::f64 {
        if (hasLo && hasHi) return (supportHi - supportLo) / ((v - supportLo) * (supportHi - v));
        if (hasLo)          return 1.0 / (v - supportLo);
        return 1.0 / (supportHi - v);
    };

    keel::Array<keel::f64> xv(N);
    keel::Array<keel::f64> wv(N, 1.0);
    if (weights && weights->numel() == N)
        for (keel::usize i = 0; i < N; ++i) wv[i] = weights->elemAsDouble(i);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = x.elemAsDouble(i);
        if (hasLo && !(v > supportLo))
            throw Error("ksdensity: data values must be strictly greater than lower Support bound",
                        0, 0, "ksdensity", "", "numkit:ksdensity:support");
        if (hasHi && !(v < supportHi))
            throw Error("ksdensity: data values must be strictly less than upper Support bound",
                        0, 0, "ksdensity", "", "numkit:ksdensity:support");
        xv[i] = useLogTransform ? fwdTransform(v) : v;
    }
    // Normalize weights so Σw = 1.
    keel::f64 Wsum = 0.0;
    for (keel::f64 w : wv) Wsum += w;
    if (!(Wsum > 0.0)) Wsum = keel::f64(N);
    const keel::f64 Winv = 1.0 / Wsum;

    // Sort xv and wv jointly.
    keel::Array<keel::usize> idx(N);
    for (keel::usize i = 0; i < N; ++i) idx[i] = i;
    keel::sort(idx.begin(), idx.end(),
              [&](keel::usize a, keel::usize b) { return xv[a] < xv[b]; });
    keel::Array<keel::f64> xs(N), ws(N);
    for (keel::usize i = 0; i < N; ++i) { xs[i] = xv[idx[i]]; ws[i] = wv[idx[i]]; }

    // Default bandwidth for the normal kernel:
    //   sigma = mad(x, 1) / 0.6745 if positive, else iqr(x) / 1.349
    //   bw    = sigma · (4 / (3·n))^(1/5)
    keel::f64 bw = bw_user;
    if (!(bw > 0.0)) {
        // median absolute deviation
        keel::Array<keel::f64> tmp = xs.clone();          // already sorted
        const keel::f64 med = (N % 2 == 1) ? tmp[N / 2]
                                        : 0.5 * (tmp[N / 2 - 1] + tmp[N / 2]);
        keel::Array<keel::f64> dev(N);
        for (keel::usize i = 0; i < N; ++i) dev[i] = keel::abs(xs[i] - med);
        keel::sort(dev.begin(), dev.end());
        const keel::f64 mad = (N % 2 == 1) ? dev[N / 2]
                                        : 0.5 * (dev[N / 2 - 1] + dev[N / 2]);
        keel::f64 sigma = (mad > 0.0) ? mad / 0.6745 : 0.0;
        if (!(sigma > 0.0)) {
            // IQR fallback.
            auto pct = [&](keel::f64 p) {
                const keel::f64 pos = p * (keel::f64(N) - 1.0);
                const keel::usize lo = (keel::usize)keel::floor(pos);
                const keel::usize hi = (keel::usize)keel::ceil(pos);
                const keel::f64 t = pos - keel::f64(lo);
                return xs[lo] * (1.0 - t) + xs[hi] * t;
            };
            const keel::f64 iqr = pct(0.75) - pct(0.25);
            sigma = (iqr > 0.0) ? iqr / 1.349 : 1.0;
        }
        bw = sigma * keel::pow(4.0 / (3.0 * keel::f64(N)), 0.2);
        if (!(bw > 0.0)) bw = 1.0;
    }

    // Apply kernel-specific bandwidth scaling so h has consistent
    // standard-deviation semantics across kernels.
    const keel::f64 h_eff = bw * ks_h_factor(kernel);

    // Build evaluation grid (in original x-space).
    keel::Array<keel::f64> grid;
    keel::usize outRows = 1;
    keel::usize outCols = 0;
    if (pts.isEmpty()) {
        const keel::usize M = (numpoints > 0) ? numpoints : 100;
        outRows = 1;
        outCols = M;
        grid.resize(M);
        if (useLogTransform) {
            const keel::f64 ymin = xs.front() - 3.0 * h_eff;
            const keel::f64 ymax = xs.back()  + 3.0 * h_eff;
            if (M == 1) grid[0] = invTransform(ymin);
            else {
                const keel::f64 step = (ymax - ymin) / keel::f64(M - 1);
                for (keel::usize i = 0; i < M; ++i)
                    grid[i] = invTransform((i + 1 == M) ? ymax : ymin + step * keel::f64(i));
            }
        } else {
            keel::f64 xmin = xs.front() - 3.0 * h_eff;
            keel::f64 xmax = xs.back()  + 3.0 * h_eff;
            if (hasLo && xmin < supportLo) xmin = supportLo;
            if (hasHi && xmax > supportHi) xmax = supportHi;
            if (M == 1) grid[0] = xmin;
            else {
                const keel::f64 step = (xmax - xmin) / keel::f64(M - 1);
                for (keel::usize i = 0; i < M; ++i) grid[i] = xmin + step * keel::f64(i);
                grid[M - 1] = xmax;
            }
        }
    } else {
        const keel::usize M = pts.numel();
        outRows = pts.dims().rows();
        outCols = pts.dims().cols();
        grid.resize(M);
        for (keel::usize i = 0; i < M; ++i) grid[i] = pts.elemAsDouble(i);
    }

    const keel::usize M = grid.size();
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        s.to_ascii_lower_in_place();
        return s;
    };
    const keel::String mode = function_mode.empty() ? keel::String::from("pdf")
                                                    : lower(function_mode);
    const keel::StringRef modeRef = mode.as_ref();
    Value fv = Value::matrix(outRows, outCols, ValueType::DOUBLE, mr);
    keel::f64 *fd = fv.doubleDataMut();
    const keel::f64 inv_h = 1.0 / h_eff;

    auto evalWindowedPdf = [&](keel::f64 q, bool useWindow, keel::f64 cutoff) -> keel::f64 {
        keel::usize i0 = 0, i1 = N;
        if (useWindow && N > 0) {
            const keel::f64 lo = q - cutoff;
            const keel::f64 hi = q + cutoff;
            i0 = keel::as<keel::usize>(
                keel::lower_bound(xs.begin(), xs.end(), lo) - xs.begin());
            if (i0 >= N) i0 = N - 1;
            const keel::usize rIdx = keel::as<keel::usize>(
                keel::lower_bound(xs.begin(), xs.end(), hi) - xs.begin());
            i1 = keel::min(N, rIdx + 1);
        }
        keel::f64 sum = 0.0;
        for (keel::usize i = i0; i < i1; ++i)
            sum += ws[i] * ks_pdf((q - xs[i]) * inv_h, kernel);
        return sum * inv_h * Winv;
    };

    if (modeRef == "pdf") {
        const bool useWindow = (N * M > 30000);
        const keel::f64 cutoff = (kernel == KsKernel::Normal ? 4.0 : 1.0) * h_eff;
        for (keel::usize j = 0; j < M; ++j) {
            const keel::f64 xq = grid[j];
            if (hasBounds && ((hasLo && !(xq > supportLo)) || (hasHi && !(xq < supportHi)))) {
                fd[j] = 0.0;
                continue;
            }
            if (useLogTransform) {
                const keel::f64 yq = fwdTransform(xq);
                fd[j] = evalWindowedPdf(yq, useWindow, cutoff) * jacTransform(xq);
            } else if (useReflection) {
                keel::f64 val = evalWindowedPdf(xq, useWindow, cutoff);
                if (hasLo) val += evalWindowedPdf(2.0 * supportLo - xq, useWindow, cutoff);
                if (hasHi) val += evalWindowedPdf(2.0 * supportHi - xq, useWindow, cutoff);
                fd[j] = val;
            } else {
                fd[j] = evalWindowedPdf(xq, useWindow, cutoff);
            }
        }
    } else if (modeRef == "cdf" || modeRef == "survivor" || modeRef == "cumhazard"
               || modeRef == "cumulative hazard") {
        const keel::f64 cutoff = (kernel == KsKernel::Normal ? 4.0 : 1.0) * h_eff;
        keel::Array<keel::f64> w_prefix(N + 1, 0.0);
        for (keel::usize i = 0; i < N; ++i)
            w_prefix[i + 1] = w_prefix[i] + ws[i];
        auto evalRawCdf = [&](keel::f64 q) -> keel::f64 {
            keel::usize i0 = 0, i1 = N;
            if (N > 0) {
                const keel::f64 lo = q - cutoff;
                const keel::f64 hi = q + cutoff;
                i0 = keel::as<keel::usize>(
                    keel::lower_bound(xs.begin(), xs.end(), lo) - xs.begin());
                const keel::usize rIdx = keel::as<keel::usize>(
                    keel::lower_bound(xs.begin(), xs.end(), hi) - xs.begin());
                i1 = keel::min(N, rIdx + 1);
            }
            keel::f64 sum = w_prefix[i0];
            for (keel::usize i = i0; i < i1; ++i)
                sum += ws[i] * ks_cdf((q - xs[i]) * inv_h, kernel);
            return sum * Winv;
        };
        for (keel::usize j = 0; j < M; ++j) {
            const keel::f64 xq = grid[j];
            keel::f64 F = 0.0;
            if (hasLo && xq <= supportLo) {
                F = 0.0;
            } else if (hasHi && xq >= supportHi) {
                F = 1.0;
            } else if (useLogTransform) {
                F = evalRawCdf(fwdTransform(xq));
            } else if (useReflection) {
                F = evalRawCdf(xq);
                if (hasLo) F -= evalRawCdf(2.0 * supportLo - xq);
                if (hasHi) F += (1.0 - evalRawCdf(2.0 * supportHi - xq));
                F = keel::clamp(F, 0.0, 1.0);
            } else {
                F = evalRawCdf(xq);
            }
            if      (modeRef == "cdf")      fd[j] = F;
            else if (modeRef == "survivor") fd[j] = 1.0 - F;
            else                            fd[j] = -keel::log(keel::max(1.0 - F, 1e-300));
        }
    } else if (modeRef == "icdf") {
        // Inverse CDF: the `grid` values are PROBABILITIES p in [0, 1].
        // Solve F(x) = p via Newton's method on the smoothed cdf/pdf, seeded
        // by linear inverse-interpolation of a 100-point grid cdf over the
        // data range.
        const keel::f64 infv = keel::Limits<keel::f64>::infinity();
        auto cdf_at = [&](keel::f64 xq) {
            keel::f64 s = 0.0;
            for (keel::usize i = 0; i < N; ++i)
                s += ws[i] * ks_cdf((xq - xs[i]) * inv_h, kernel);
            return s * Winv;
        };
        auto pdf_at = [&](keel::f64 xq) {
            keel::f64 s = 0.0;
            for (keel::usize i = 0; i < N; ++i)
                s += ws[i] * ks_pdf((xq - xs[i]) * inv_h, kernel);
            return s * inv_h * Winv;
        };
        // 100-point grid cdf over [min(x), max(x)] for the initial guess.
        const keel::usize G = 100;
        keel::Array<keel::f64> gx(G), gF(G);
        {
            const keel::f64 gmin = xs.front();
            const keel::f64 gmax = xs.back();
            const keel::f64 step = (G > 1) ? (gmax - gmin) / keel::f64(G - 1) : 0.0;
            for (keel::usize g = 0; g < G; ++g) {
                gx[g] = (g + 1 == G) ? gmax : gmin + step * keel::f64(g);
                gF[g] = cdf_at(gx[g]);
            }
        }
        const keel::f64 min_dF0 = keel::sqrt(keel::Limits<keel::f64>::kEpsilon);
        for (keel::usize j = 0; j < M; ++j) {
            const keel::f64 p = grid[j];
            if (p < 0.0 || p > 1.0) { fd[j] = nan;   continue; }
            if (p == 0.0)           { fd[j] = hasLo ? supportLo : -infv; continue; }
            if (p == 1.0)           { fd[j] = hasHi ? supportHi :  infv; continue; }
            // Initial guess via linear inverse interpolation on (gF, gx).
            keel::f64 x0;
            if      (p <= gF.front()) x0 = gx.front();
            else if (p >= gF.back())  x0 = gx.back();
            else {
                keel::usize lo = 0;
                while (lo + 1 < G && gF[lo + 1] < p) ++lo;
                const keel::f64 denom = gF[lo + 1] - gF[lo];
                const keel::f64 t = (denom != 0.0) ? (p - gF[lo]) / denom : 0.0;
                x0 = gx[lo] + t * (gx[lo + 1] - gx[lo]);
            }
            // Newton refinement on the smoothed CDF.
            for (keel::i32 iter = 0; iter < 100; ++iter) {
                const keel::f64 F0 = cdf_at(x0);
                keel::f64 dF0 = pdf_at(x0);
                if (dF0 < min_dF0) dF0 = min_dF0;
                const keel::f64 dp = p - F0;
                const keel::f64 dx = dp / dF0;
                x0 += dx;
                if (keel::abs(dx) <= 1e-6 * keel::abs(x0) || keel::abs(dp) <= 1e-8)
                    break;
            }
            fd[j] = useLogTransform ? invTransform(x0) : x0;
        }
    } else {
        throw Error(keel::format("ksdensity: unknown Function '{}'", modeRef),
                    0, 0, "ksdensity", "", "numkit:ksdensity:badfn");
    }

    Value xiV = Value::matrix(outRows, outCols, ValueType::DOUBLE, mr);
    keel::f64 *xd = xiV.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) xd[i] = grid[i];
    return {keel::move(fv), keel::move(xiV), Value::scalar(bw, mr)};
}

// Backward-compat 4-arg form: pdf with normal kernel, default
// numpoints=100, no weights.
keel::Tuple<Value, Value, Value>
ksdensity(const Value &x, const Value &pts, keel::f64 bw_user, keel::Allocator *mr)
{
    auto R = ksdensity_full(x, pts, bw_user, "normal", "pdf", 100, nullptr, mr);
    return keel::make_tuple(keel::move(R.f), keel::move(R.xi), keel::move(R.bw));
}

// ── prepareCurveData / prepareSurfaceData ─────────────────────────────


keel::Tuple<Value, Value, Value>
prepareCurveData(const Value &x, const Value &y, const Value &w, keel::Allocator *mr)
{
    const keel::usize Nx = x.numel();
    const keel::usize Ny = y.numel();
    const bool   hasW = !w.isEmpty();
    const keel::usize Nw = hasW ? w.numel() : Nx;
    if (Nx != Ny || (hasW && Nw != Nx))
        throw Error(keel::format("prepareCurveData: x, y{} must be same length",
                                 hasW ? ", w" : ""),
                    0, 0, "prepareCurveData", "", "numkit:prepCD:size");

    keel::Array<keel::f64> xv(Nx), yv(Nx), wv(hasW ? Nx : 0);
    keel::Array<keel::u8> keep(Nx, 1);
    for (keel::usize i = 0; i < Nx; ++i) {
        xv[i] = x.elemAsDouble(i);
        yv[i] = y.elemAsDouble(i);
        if (hasW) wv[i] = w.elemAsDouble(i);
        if (!finite_double(xv[i]) || !finite_double(yv[i]) ||
            (hasW && !finite_double(wv[i])))
            keep[i] = 0;
    }

    Value xo = pack_filtered(xv, keep, mr);
    Value yo = pack_filtered(yv, keep, mr);
    Value wo = hasW ? pack_filtered(wv, keep, mr)
                    : Value::matrix(0, 1, ValueType::DOUBLE, mr);
    return {keel::move(xo), keel::move(yo), keel::move(wo)};
}

keel::Tuple<Value, Value, Value>
prepareSurfaceData(const Value &x, const Value &y, const Value &z, keel::Allocator *mr)
{
    // For surface fits MATLAB lets x and y be either vectors of length
    // numel(z), or matrices the same shape as z (meshgrid). Normalise
    // by linearising in column-major order; numel must match.
    const keel::usize Nz = z.numel();
    const keel::usize Nx = x.numel();
    const keel::usize Ny = y.numel();

    auto broadcastTo = [&](const Value &v, keel::usize target) -> keel::Array<keel::f64> {
        keel::Array<keel::f64> out(target);
        if (v.numel() == target) {
            for (keel::usize i = 0; i < target; ++i) out[i] = v.elemAsDouble(i);
        } else {
            // Allow x = row of length cols, y = col of length rows for
            // implicit meshgrid (matches MATLAB behaviour).
            const auto &dz = z.dims();
            const keel::usize rows = dz.rows();
            const keel::usize cols = dz.cols();
            if (v.numel() == cols) {
                // treat as x-coords per column
                for (keel::usize c = 0; c < cols; ++c)
                    for (keel::usize r = 0; r < rows; ++r)
                        out[r + c * rows] = v.elemAsDouble(c);
            } else if (v.numel() == rows) {
                for (keel::usize c = 0; c < cols; ++c)
                    for (keel::usize r = 0; r < rows; ++r)
                        out[r + c * rows] = v.elemAsDouble(r);
            } else {
                throw Error("prepareSurfaceData: x, y, z size mismatch",
                            0, 0, "prepareSurfaceData", "", "numkit:prepSD:size");
            }
        }
        return out;
    };

    keel::Array<keel::f64> xv = broadcastTo(x, Nz);
    keel::Array<keel::f64> yv = broadcastTo(y, Nz);
    keel::Array<keel::f64> zv(Nz);
    for (keel::usize i = 0; i < Nz; ++i) zv[i] = z.elemAsDouble(i);

    keel::Array<keel::u8> keep(Nz, 1);
    for (keel::usize i = 0; i < Nz; ++i) {
        if (!finite_double(xv[i]) || !finite_double(yv[i]) ||
            !finite_double(zv[i]))
            keep[i] = 0;
    }

    Value xo = pack_filtered(xv, keep, mr);
    Value yo = pack_filtered(yv, keep, mr);
    Value zo = pack_filtered(zv, keep, mr);
    (void)Nx; (void)Ny;
    return {keel::move(xo), keel::move(yo), keel::move(zo)};
}

// ── datastats ─────────────────────────────────────────────────────────

keel::Tuple<Value, Value, Value, Value, Value, Value, Value>
datastats(const Value &x, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    keel::Array<keel::f64> v;
    v.reserve(N);
    for (keel::usize i = 0; i < N; ++i) v.push_back(x.elemAsDouble(i));

    const auto nan = keel::Limits<keel::f64>::quiet_nan();
    if (N == 0) {
        return keel::make_tuple(Value::scalar(0.0, mr),
                               Value::scalar(nan, mr),
                               Value::scalar(nan, mr),
                               Value::scalar(nan, mr),
                               Value::scalar(nan, mr),
                               Value::scalar(nan, mr),
                               Value::scalar(nan, mr));
    }

    keel::f64 mn = v[0], mx = v[0], sum = 0.0;
    for (keel::f64 vi : v) {
        if (vi < mn) mn = vi;
        if (vi > mx) mx = vi;
        sum += vi;
    }
    const keel::f64 mean = sum / keel::as<keel::f64>(N);

    keel::Array<keel::f64> sorted = v.clone();
    keel::sort(sorted.begin(), sorted.end());
    keel::f64 median;
    if (N % 2 == 1) median = sorted[N / 2];
    else            median = 0.5 * (sorted[N / 2 - 1] + sorted[N / 2]);

    keel::f64 sd = 0.0;
    if (N > 1) {
        keel::f64 sq = 0.0;
        for (keel::f64 vi : v) { const keel::f64 d = vi - mean; sq += d * d; }
        sd = keel::sqrt(sq / keel::as<keel::f64>(N - 1));
    }
    const keel::f64 range = mx - mn;

    return keel::make_tuple(Value::scalar(keel::as<keel::f64>(N), mr),
                           Value::scalar(mx,     mr),
                           Value::scalar(mn,     mr),
                           Value::scalar(mean,   mr),
                           Value::scalar(median, mr),
                           Value::scalar(range,  mr),
                           Value::scalar(sd,     mr));
}

// Forward declarations for ecdf (defined at the end of this TU).

// Forward declaration for ksdensity_full. KsdensityFull is defined above.

// ── filloutliers (public typed entry; see descriptive.hpp) ────────────
// Typed front over filloutliers_of covering the median / mean / quartiles
// find-methods. 'percentiles' (needs a [lo hi] pair) and the deferred
// grubbs/gesd/movmedian/movmean methods stay script-only via the adapter.
Value filloutliers(const Value &A, const Value &fillMethod,
                   keel::StringRef findMethod, keel::f64 thresholdFactor,
                   keel::Allocator *mr)
{
    keel::String detect = keel::String::from(findMethod);
    detect.to_ascii_lower_in_place();
    const keel::StringRef detRef = detect.as_ref();
    if (detRef != "median" && detRef != "mean" && detRef != "quartiles")
        throw Error("filloutliers: findMethod must be 'median', 'mean', or "
                    "'quartiles' for the C++ API ('percentiles' and "
                    "grubbs/gesd/movmedian/movmean are script-only / deferred)",
                    0, 0, "filloutliers", "", "numkit:filloutliers:findmethod");
    // Per-method default threshold when thresholdFactor is NaN (matches the
    // adapter): median/mean default 3; quartiles internal 3.0 (= MATLAB k=1.5).
    // A user-supplied factor for quartiles is the IQR multiplier k; the
    // internal formula uses 0.5·tf·IQR, so scale by 2.
    keel::f64 tf;
    if (keel::is_nan(thresholdFactor))
        tf = 3.0;
    else
        tf = (detRef == "quartiles") ? 2.0 * thresholdFactor : thresholdFactor;
    return filloutliers_of(A, fillMethod, detRef, tf, 0.0, 0.0, mr);
}

// ── Engine adapters ───────────────────────────────────────────────────

// ── ecdfhist ─────────────────────────────────────────────────────────
// Convert (f, x) from ecdf into a probability-density histogram.
// Algorithm: probs[k] = f[k+1] - f[k]; vals[k] = x[k+1] (the unique
// data values). Bin into m equal-width bins over [min(x), max(x)],
// each height = sum(probs in bin) / bin_width.
keel::Tuple<Value, Value>
ecdfhist(const Value &f, const Value &x, keel::i32 m, keel::Allocator *mr)
{
    if (m < 1)
        throw Error("ecdfhist: number of bins must be >= 1",
                    0, 0, "ecdfhist", "", "numkit:ecdfhist:nbins");
    const keel::usize Lf = f.numel();
    const keel::usize Lx = x.numel();
    if (Lf != Lx)
        throw Error("ecdfhist: f and x must have the same length",
                    0, 0, "ecdfhist", "", "numkit:ecdfhist:size");
    if (Lf < 2) {
        Value n_empty = Value::matrix(1, keel::as<keel::usize>(m), ValueType::DOUBLE, mr);
        Value c_empty = Value::matrix(1, keel::as<keel::usize>(m), ValueType::DOUBLE, mr);
        return {keel::move(n_empty), keel::move(c_empty)};
    }

    // Build (vals, probs) from the ecdf step structure.
    const keel::usize K = Lf - 1;
    keel::Array<keel::f64> vals(K), probs(K);
    for (keel::usize k = 0; k < K; ++k) {
        vals[k]  = x.elemAsDouble(k + 1);
        probs[k] = f.elemAsDouble(k + 1) - f.elemAsDouble(k);
    }

    const keel::f64 xmin = vals.front();
    const keel::f64 xmax = vals.back();
    const keel::f64 width = (xmax > xmin)
        ? (xmax - xmin) / keel::as<keel::f64>(m)
        : 1.0;  // degenerate single-value case: arbitrary width=1

    Value n_out = Value::matrix(1, keel::as<keel::usize>(m), ValueType::DOUBLE, mr);
    Value c_out = Value::matrix(1, keel::as<keel::usize>(m), ValueType::DOUBLE, mr);
    keel::f64 *nd = n_out.doubleDataMut();
    keel::f64 *cd = c_out.doubleDataMut();

    for (keel::i32 k = 0; k < m; ++k) {
        cd[k] = xmin + (k + 0.5) * width;
        nd[k] = 0.0;
    }
    if (xmax <= xmin) {
        // All-equal data: drop full mass into the centre bin.
        nd[m / 2] = 1.0 / width;
        return {keel::move(n_out), keel::move(c_out)};
    }

    // MATLAB convention: a value v at an edge belongs to the LOWER bin,
    // i.e. bin k contains (edge[k-1], edge[k]] in 1-based indexing.
    // Equivalent: `bin = ceil((v - xmin) / width) - 1` in 0-based.
    // Bug fix 2026-05-08: previous impl used `floor`, which sent
    // boundary values to the upper bin (off-by-one shift in counts).
    for (keel::usize k = 0; k < K; ++k) {
        const keel::f64 v = vals[k];
        keel::i32 bin = keel::as<keel::i32>(keel::ceil((v - xmin) / width)) - 1;
        if (bin < 0)  bin = 0;
        if (bin >= m) bin = m - 1;
        nd[bin] += probs[k];
    }
    for (keel::i32 k = 0; k < m; ++k) nd[k] /= width;
    return {keel::move(n_out), keel::move(c_out)};
}

// ── ecdf ─────────────────────────────────────────────────────────────
// Empirical CDF / survivor / cumulative-hazard. Optional Frequency
// weights; optional 95% confidence bounds.
//
// Output shapes match MATLAB R2025b: f and x are column vectors of
// length K+1 (K = number of distinct sample values). For cdf/survivor:
//   f[0] = 0 (cdf) or 1 (survivor); x[0] = min sample value.
// For cumulative hazard: same shape but f is the Nelson-Aalen
// estimator, NOT -log(1-cdf).

// `EcdfFull` forward-declared above. f/x/flo/fup are K+1 column vectors.
EcdfFull ecdf_full(const Value &y, const Value *freq, keel::StringRef function_mode, keel::f64 alpha, bool want_bounds, keel::Allocator *mr)
{
    const keel::usize n = y.numel();
    const bool has_freq = (freq && freq->numel() == n);
    if (freq && freq->numel() != 0 && freq->numel() != n)
        throw Error("ecdf: Frequency length must match data length",
                    0, 0, "ecdf", "", "numkit:ecdf:freqsize");

    // Collect (value, weight) pairs, dropping NaNs. Sort by value.
    keel::Array<keel::Pair<keel::f64, keel::f64>> vw;
    vw.reserve(n);
    keel::f64 Wtotal = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 s = y.elemAsDouble(i);
        if (keel::is_nan(s)) continue;
        const keel::f64 w = has_freq ? freq->elemAsDouble(i) : 1.0;
        if (w == 0.0) continue;
        vw.push_back({s, w});
        Wtotal += w;
    }
    EcdfFull R{};
    if (vw.empty() || Wtotal <= 0.0) {
        R.f   = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        R.x   = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        R.flo = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        R.fup = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        return R;
    }
    keel::sort(vw.begin(), vw.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    const keel::f64 N = Wtotal;

    // Walk through sorted (v, w) and emit one row per distinct value.
    keel::Array<keel::f64> fs, xs;
    keel::Array<keel::f64> at_risk;   // n_i (for Nelson-Aalen and Greenwood)
    keel::Array<keel::f64> events;    // d_i (events at this distinct value)
    fs.push_back(0.0);
    xs.push_back(vw.front().first);  // F = 0 at x = min(y)
    at_risk.push_back(N);
    events.push_back(0.0);

    keel::f64 cum_w = 0.0;
    keel::usize i = 0;
    while (i < vw.size()) {
        keel::usize j = i + 1;
        keel::f64 w_block = vw[i].second;
        while (j < vw.size() && vw[j].first == vw[i].first) {
            w_block += vw[j].second;
            ++j;
        }
        const keel::f64 n_i = N - cum_w;       // at-risk just before this event
        cum_w += w_block;
        fs.push_back(cum_w / N);
        xs.push_back(vw[i].first);
        at_risk.push_back(n_i);
        events.push_back(w_block);
        i = j;
    }

    const keel::usize L = fs.size();

    // Apply Function mode.
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        s.to_ascii_lower_in_place();
        return s;
    };
    const keel::String mode = lower(function_mode);
    const keel::StringRef modeRef = mode.as_ref();
    keel::Array<keel::f64> ff(L);
    if (modeRef == "cdf" || modeRef.empty()) {
        for (keel::usize k = 0; k < L; ++k) ff[k] = fs[k];
    } else if (modeRef == "survivor") {
        for (keel::usize k = 0; k < L; ++k) ff[k] = 1.0 - fs[k];
    } else if (modeRef == "cumulative hazard" || modeRef == "cumhazard") {
        // Nelson-Aalen estimator: H(x) = sum over t_i ≤ x of d_i / n_i.
        ff[0] = 0.0;
        keel::f64 H = 0.0;
        for (keel::usize k = 1; k < L; ++k) {
            if (at_risk[k] > 0.0) H += events[k] / at_risk[k];
            ff[k] = H;
        }
    } else {
        throw Error(keel::format("ecdf: unknown Function mode '{}'", modeRef),
                    0, 0, "ecdf", "", "numkit:ecdf:badmode");
    }

    R.f = Value::matrix(L, 1, ValueType::DOUBLE, mr);
    R.x = Value::matrix(L, 1, ValueType::DOUBLE, mr);
    {
        keel::f64 *fd = R.f.doubleDataMut();
        keel::f64 *xd = R.x.doubleDataMut();
        for (keel::usize k = 0; k < L; ++k) { fd[k] = ff[k]; xd[k] = xs[k]; }
    }

    if (!want_bounds) {
        R.flo = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        R.fup = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        return R;
    }

    // Greenwood-style binomial Wald CI for cdf / survivor; analogous
    // log-transform for cumulative hazard. Match MATLAB R2025b: first
    // and last rows return NaN bounds.
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    R.flo = Value::matrix(L, 1, ValueType::DOUBLE, mr);
    R.fup = Value::matrix(L, 1, ValueType::DOUBLE, mr);
    keel::f64 *lo = R.flo.doubleDataMut();
    keel::f64 *hi = R.fup.doubleDataMut();

    // z = -norminv(α/2). For α=0.05 → ~1.959964.
    const keel::f64 z = keel::sqrt(2.0) * [&]{
        keel::f64 e = 1.0 - alpha;
        for (keel::i32 it = 0; it < 50; ++it) {
            const keel::f64 f = keel::erf(e) - (1.0 - alpha);
            const keel::f64 fp = (2.0 / keel::sqrt(keel::kPi))
                              * keel::exp(-e * e);
            e -= f / fp;
        }
        return e;
    }();
    for (keel::usize k = 0; k < L; ++k) {
        if (k == 0 || k == L - 1) { lo[k] = nan; hi[k] = nan; continue; }
        const keel::f64 F = ff[k];
        const keel::f64 se = keel::sqrt(F * (1.0 - F) / N);
        keel::f64 l = F - z * se;
        keel::f64 h = F + z * se;
        if (l < 0.0) l = 0.0;
        if (h > 1.0) h = 1.0;
        lo[k] = l;
        hi[k] = h;
    }
    return R;
}

// Backward-compat 1-arg form.
keel::Tuple<Value, Value>
ecdf(const Value &y, keel::Allocator *mr)
{
    auto R = ecdf_full(y, nullptr, "cdf", 0.05, false, mr);
    return {keel::move(R.f), keel::move(R.x)};
}

} // namespace numkit::stats
