// toolboxes/stats/src/mvdist/mvdist.cpp
//
// mvnpdf / mvtpdf / mnpdf — closed-form multivariate PDFs/PMFs, owning
// their full parse+validate+compute contracts (probed against the
// reference engine R2025b: build/probe73.m P/T/M batteries).
//
// Shared contract pieces (all three read column vectors provisionally
// as data and reinterpret them as rows when a matrix-shaped parameter
// demands it — exactly the reference's rule):
//   * column X + no matrix parameter → one d = n observation row;
//   * 1×k parameter rows are "single row" forms; n×k full vectors;
//   * a single data row is replicated to a parameter's page/row count;
//   * single inputs produce single outputs (compute runs in double,
//     cast down — the reference runs single arithmetic, O(1e-7) tail).

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <numkit/stats/mvdist/mvdist.hpp>

#include <numkit/linalg/eig.hpp>             // eig_symmetric
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::stats {

namespace {

constexpr keel::f64 kLog2Pi = 1.8378770664093454835606594728112352;
constexpr keel::f64 kNaN = std::numeric_limits<keel::f64>::quiet_NaN();

bool isMatrixValue(const Value &V)
{
    return V.dims().ndim() <= 2 && V.numel() > 0;
}

// Symmetric eigendecomposition of a d×d column-major matrix read via
// elemAsDouble. Eigenvalues ascend in `eval`; eigenvectors are the
// columns of `evec` (column-major d×d).
void eigSymmetric(keel::ArrayRef<keel::f64> M, keel::usize d,
                  keel::Array<keel::f64> &eval, keel::Array<keel::f64> &evec,
                  keel::Allocator *mr)
{
    Value Mv = Value::matrix(d, d, ValueType::DOUBLE, mr);
    std::copy(M.begin(), M.end(), Mv.doubleDataMut());
    auto [V, D] = ::numkit::linalg::eig_symmetric(Mv, mr);
    evec.resize(d * d);
    std::copy(V.doubleData(), V.doubleData() + d * d, evec.begin());
    eval.resize(d);
    for (keel::usize i = 0; i < d; ++i) eval[i] = D.doubleData()[i + i * d];
}

// Quadratic-form machinery shared by the normal and t densities:
// factor sigma = V·diag(λ)·V' with negative eigenvalues clamped to 0
// (positive SEMI-definite, the reference's cholcov semantics). Returns
// false when sigma is not even semidefinite (λ < -tol). `quad` fills
// per-row quadratic forms q = Σ_{λi>0} (v_i'·dx)²/λi and logHalfDet
// fills Σ log(sqrt(λi)) over ALL eigenvalues — a zero eigenvalue gives
// -Inf, which drives a degenerate density to 0 (probe P15).
bool psdQuadForms(const Value &X0, keel::usize n, keel::usize d,
                  keel::ArrayRef<keel::f64> Sig,
                  keel::Array<keel::f64> &quad, keel::f64 &logHalfDet,
                  keel::Allocator *mr, bool *degenerate = nullptr)
{
    keel::Array<keel::f64> eval, evec;
    eigSymmetric(Sig, d, eval, evec, mr);
    keel::f64 maxAbs = 0.0;
    for (keel::f64 e : eval) maxAbs = std::max(maxAbs, keel::abs(e));
    const keel::f64 tol = keel::as<keel::f64>(d) * std::numeric_limits<keel::f64>::epsilon()
                     * std::max(maxAbs, 1.0);
    for (keel::f64 e : eval)
        if (e < -tol) return false;
    logHalfDet = 0.0;
    if (degenerate) *degenerate = false;
    for (keel::f64 e : eval) {
        if (e <= tol && degenerate) *degenerate = true;
        logHalfDet += 0.5 * std::log(std::max(e, 0.0));   // 0 → -Inf
    }
    quad = keel::Array<keel::f64>(n, 0.0);
    for (keel::usize i = 0; i < d; ++i) {
        if (eval[i] <= tol) continue;                     // null direction
        for (keel::usize r = 0; r < n; ++r) {
            keel::f64 s = 0.0;
            for (keel::usize j = 0; j < d; ++j)
                s += evec[j + i * d] * X0.elemAsDouble(r + j * n);
            quad[r] += s * s / eval[i];
        }
    }
    return true;
}

// Diagonal-sigma quadratic forms + log half determinant.
void diagQuadForms(const Value &X0, keel::usize n, keel::usize d,
                   keel::ArrayRef<keel::f64> sd,
                   keel::Array<keel::f64> &quad, keel::f64 &logHalfDet)
{
    logHalfDet = 0.0;
    for (keel::usize j = 0; j < d; ++j) logHalfDet += std::log(sd[j]);
    quad = keel::Array<keel::f64>(n, 0.0);
    for (keel::usize r = 0; r < n; ++r) {
        keel::f64 q = 0.0;
        for (keel::usize j = 0; j < d; ++j) {
            const keel::f64 z = X0.elemAsDouble(r + j * n) / sd[j];
            q += z * z;
        }
        quad[r] = q;
    }
}

Value castDown(const Value &X, bool single, keel::Allocator *mr)
{
    if (!single || X.type() == ValueType::SINGLE) return X;
    const keel::usize r = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize c = keel::as<keel::usize>(X.dims().dim(1));
    Value out = Value::matrix(r, c, ValueType::SINGLE, mr);
    const keel::usize cnt = r * c;
    for (keel::usize k = 0; k < cnt; ++k)
        out.singleDataMut()[k] = keel::as<keel::f32>(X.elemAsDouble(k));
    return out;
}

} // namespace

namespace {
void mvnpdf_argcheck(const Value &X)
{
    if (!isMatrixValue(X))
        throw Error(keel::format("mvnpdf: X must be a non-empty matrix (got a {} array).",
                                 mtypeName(X.type())),
                    0, 0, "mvnpdf", "", "numkit:mvnpdf:invalidData");
    const keel::usize d = keel::as<keel::usize>(X.dims().dim(1));
    if (d < 1)
        throw Error("mvnpdf: X must have at least one column.",
                    0, 0, "mvnpdf", "", "numkit:mvnpdf:tooFewDimensions");
}
} // namespace

Value mvnpdf(const Value &X, const Value &mu, const Value &Sigma,
             keel::Allocator *mr)
{
    // ── X gate ──────────────────────────────────────────────────────
    mvnpdf_argcheck(X);
    keel::usize n = keel::as<keel::usize>(X.dims().dim(0));
    keel::usize d = keel::as<keel::usize>(X.dims().dim(1));
    bool single = (X.type() == ValueType::SINGLE);

    // ── mu: empty | scalar | matrix (row, full, or rep-when-single) ──
    Value X0;
    if (mu.isEmpty()) {
        X0 = X;
    } else if (mu.isScalar()) {
        const keel::f64 m = mu.elemAsDouble(0);
        X0 = Value::matrix(n, d, ValueType::DOUBLE, mr);
        for (keel::usize k = 0; k < n * d; ++k)
            X0.doubleDataMut()[k] = X.elemAsDouble(k) - m;
    } else if (isMatrixValue(mu)) {
        const keel::usize n2 = keel::as<keel::usize>(mu.dims().dim(0));
        const keel::usize d2 = keel::as<keel::usize>(mu.dims().dim(1));
        if (d2 != d)
            throw Error(keel::format("mvnpdf: X and MU must have the same number of "
                                     "columns (X has {}, MU has {}).", d, d2),
                        0, 0, "mvnpdf", "", "numkit:mvnpdf:colSizeMismatch");
        if (n2 == 1 || n2 == n) {
            X0 = Value::matrix(n, d, ValueType::DOUBLE, mr);
            for (keel::usize j = 0; j < d; ++j)
                for (keel::usize r = 0; r < n; ++r)
                    X0.doubleDataMut()[r + j * n] =
                        X.elemAsDouble(r + j * n)
                        - mu.elemAsDouble((n2 == 1 ? 0 : r) + j * n2);
        } else if (n == 1) {
            n = n2;
            X0 = Value::matrix(n, d, ValueType::DOUBLE, mr);
            for (keel::usize j = 0; j < d; ++j)
                for (keel::usize r = 0; r < n; ++r)
                    X0.doubleDataMut()[r + j * n] =
                        X.elemAsDouble(j * 1) - mu.elemAsDouble(r + j * n);
        } else {
            throw Error(keel::format("mvnpdf: MU must be a row vector or have one row "
                                     "per X row (X has {} rows, MU has {}).", n, n2),
                        0, 0, "mvnpdf", "", "numkit:mvnpdf:rowSizeMismatch");
        }
    } else {
        throw Error("mvnpdf: MU must be a scalar or a matrix.",
                    0, 0, "mvnpdf", "", "numkit:mvnpdf:badMu");
    }
    single = single || (mu.type() == ValueType::SINGLE);

    // ── Sigma: empty | 1×k diag row | d×d matrix | d×d×n stack ──────
    keel::usize pages = 1;
    bool sigmaIsDiag = false;
    keel::Array<keel::f64> sd;                        // diag form: sqrt entries
    keel::Array<keel::f64> Sig;                       // matrix form, column-major
    keel::Array<keel::Array<keel::f64>> pageSig;      // stack form
    keel::usize sz1 = 0, sz2 = 0;
    if (Sigma.isEmpty()) {
        // No Sigma: a column X is one d = n observation row.
        if (d == 1 && X.numel() > 1) {
            Value Xt = Value::matrix(1, n, ValueType::DOUBLE, mr);
            for (keel::usize k = 0; k < n; ++k)
                Xt.doubleDataMut()[k] = X0.elemAsDouble(k);
            X0 = Xt;
            d = n;
            n = 1;
        }
        sigmaIsDiag = true;
        sd = keel::Array<keel::f64>(d, 1.0);
    } else {
        sz1 = keel::as<keel::usize>(Sigma.dims().dim(0));
        sz2 = keel::as<keel::usize>(Sigma.dims().dim(1));
        if (Sigma.dims().ndim() <= 2 && sz1 == 1 && sz2 > 1) {
            sigmaIsDiag = true;
            sz1 = sz2;
        }
        // Column X + matrix Sigma whose rows match → reinterpret as row.
        if (!sigmaIsDiag && d == 1 && X.numel() > 1 && sz1 == n) {
            Value Xt = Value::matrix(1, n, ValueType::DOUBLE, mr);
            for (keel::usize k = 0; k < n; ++k)
                Xt.doubleDataMut()[k] = X0.elemAsDouble(k);
            X0 = Xt;
            d = n;
            n = 1;
        }
        single = single || (Sigma.type() == ValueType::SINGLE);
        pages = Sigma.dims().ndim() >= 3
                  ? keel::as<keel::usize>(Sigma.dims().dim(2)) : 1;
        if (pages == 1) {
            if (sz1 != sz2)
                throw Error("mvnpdf: SIGMA must be a square covariance "
                            "matrix, the diagonal of one as a row vector, "
                            "or a stack of matrices.",
                            0, 0, "mvnpdf", "", "numkit:mvnpdf:badCovariance");
            if (sz1 != d)
                throw Error(keel::format("mvnpdf: SIGMA must be {}-by-{} to match X "
                                         "(got {}-by-{}).", d, d, sz1, sz2),
                            0, 0, "mvnpdf", "", "numkit:mvnpdf:covSizeMismatch");
            if (sigmaIsDiag) {
                sd = keel::Array<keel::f64>(d, 0.0);
                for (keel::usize j = 0; j < d; ++j) {
                    const keel::f64 v = Sigma.elemAsDouble(j);
                    if (!(v > 0.0))
                        throw Error(keel::format("mvnpdf: all elements of the diagonal "
                                                 "SIGMA must be positive (got {} at position {}).",
                                                 v, j + 1),
                                    0, 0, "mvnpdf", "",
                                    "numkit:mvnpdf:badDiagSigma");
                    sd[j] = std::sqrt(v);
                }
            } else {
                Sig = keel::Array<keel::f64>(d * d, 0.0);
                for (keel::usize k = 0; k < d * d; ++k)
                    Sig[k] = Sigma.elemAsDouble(k);
            }
        } else {
            // Stack: a single data row replicates across the pages.
            if (n == 1) {
                n = pages;
                Value Xr = Value::matrix(n, d, ValueType::DOUBLE, mr);
                for (keel::usize j = 0; j < d; ++j)
                    for (keel::usize r = 0; r < n; ++r)
                        Xr.doubleDataMut()[r + j * n] = X0.elemAsDouble(j);
                X0 = Xr;
            }
            if (sz1 != sz2)
                throw Error("mvnpdf: SIGMA must be a square covariance "
                            "matrix, the diagonal of one as a row vector, "
                            "or a stack of matrices.",
                            0, 0, "mvnpdf", "",
                            "numkit:mvnpdf:badCovarianceMultiple");
            if (sz1 != d)
                throw Error(keel::format("mvnpdf: each SIGMA page must be {}-by-{} to match X.",
                                         d, d),
                            0, 0, "mvnpdf", "",
                            "numkit:mvnpdf:covSizeMismatchMultiple");
            if (pages != n)
                throw Error(keel::format("mvnpdf: SIGMA must have one page per X row "
                                         "(X has {} rows, SIGMA has {} pages).", n, pages),
                            0, 0, "mvnpdf", "",
                            "numkit:mvnpdf:covSizeMismatchPages");
            if (sigmaIsDiag) {
                // Row diag [1×d×n] → per-page sqrt rows.
                for (keel::usize j = 0; j < d; ++j) {
                    for (keel::usize p = 0; p < pages; ++p) {
                        const keel::f64 v = Sigma.elemAsDouble(j + p * d);
                        if (!(v > 0.0))
                            throw Error("mvnpdf: all elements of the "
                                        "diagonal SIGMA must be positive.",
                                        0, 0, "mvnpdf", "",
                                        "numkit:mvnpdf:badDiagSigma");
                    }
                }
            } else {
                pageSig.reserve(pages);
                for (keel::usize p = 0; p < pages; ++p) {
                    keel::Array<keel::f64> mat(d * d, 0.0);
                    for (keel::usize k = 0; k < d * d; ++k)
                        mat[k] = Sigma.elemAsDouble(k + p * d * d);
                    pageSig.push_back(keel::move(mat));
                }
            }
        }
    }

    // ── density ─────────────────────────────────────────────────────
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 base = -0.5 * keel::as<keel::f64>(d) * kLog2Pi;
    auto fillRows = [&](keel::ArrayRef<keel::f64> quad, keel::f64 logHalfDet) {
        for (keel::usize r = 0; r < n; ++r)
            od[r] = std::exp(base - logHalfDet - 0.5 * quad[r]);
    };
    if (pages == 1) {
        keel::Array<keel::f64> quad;
        keel::f64 logHalfDet = 0.0;
        bool degenerate = false;
        if (sigmaIsDiag) {
            diagQuadForms(X0, n, d, sd.as_ref(), quad, logHalfDet);
        } else if (!psdQuadForms(X0, n, d, Sig.as_ref(), quad, logHalfDet, mr,
                                 &degenerate)) {
            throw Error("mvnpdf: SIGMA must be a square, symmetric, "
                        "positive semi-definite matrix.",
                        0, 0, "mvnpdf", "", "numkit:mvnpdf:badMatrixSigma");
        }
        if (degenerate) {
            for (keel::usize r = 0; r < n; ++r) od[r] = 0.0;
        } else {
            fillRows(quad.as_ref(), logHalfDet);
        }
    } else if (sigmaIsDiag) {
        // Per-page diagonal densities.
        for (keel::usize r = 0; r < n; ++r) {
            keel::f64 q = 0.0, lhd = 0.0;
            for (keel::usize j = 0; j < d; ++j) {
                const keel::f64 s = std::sqrt(Sigma.elemAsDouble(j + r * d));
                const keel::f64 z = X0.elemAsDouble(r + j * n) / s;
                q += z * z;
                lhd += std::log(s);
            }
            od[r] = std::exp(base - lhd - 0.5 * q);
        }
    } else {
        for (keel::usize r = 0; r < n; ++r) {
            // One-page slice through the shared machinery.
            Value Xrow = Value::matrix(1, d, ValueType::DOUBLE, mr);
            for (keel::usize j = 0; j < d; ++j)
                Xrow.doubleDataMut()[j] = X0.elemAsDouble(r + j * n);
            keel::Array<keel::f64> quad;
            keel::f64 logHalfDet = 0.0;
            bool degenerate = false;
            if (!psdQuadForms(Xrow, 1, d, pageSig[r].as_ref(), quad, logHalfDet, mr,
                              &degenerate))
                throw Error("mvnpdf: SIGMA must be a square, symmetric, "
                            "positive semi-definite matrix.",
                            0, 0, "mvnpdf", "",
                            "numkit:mvnpdf:badMatrixSigmaMultiple");
            od[r] = degenerate ? 0.0
                               : std::exp(base - logHalfDet - 0.5 * quad[0]);
        }
    }
    return castDown(out, single, mr);
}

Value mvtpdfCompute(const Value &X, const Value &C, const Value &df,
                    bool single, keel::Allocator *mr);

Value mvtpdf(const Value &X, const Value &C, const Value &df,
             keel::Allocator *mr)
{
    if (X.dims().ndim() != 2)
        throw Error("mvtpdf: X must be a matrix.",
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:invalidData");
    keel::usize n = keel::as<keel::usize>(X.dims().dim(0));
    keel::usize d = keel::as<keel::usize>(X.dims().dim(1));
    if (d < 1)
        throw Error("mvtpdf: X must have at least one column.",
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:tooFewDimensions");
    bool single = (X.type() == ValueType::SINGLE);

    // Row reinterpretation: vector X whose length matches rows(C).
    const keel::usize cr = C.dims().dim(0);
    if (X.dims().isVector() && cr == n) {
        Value Xt = Value::matrix(1, n, ValueType::DOUBLE, mr);
        for (keel::usize k = 0; k < n; ++k)
            Xt.doubleDataMut()[k] = X.elemAsDouble(k);
        return mvtpdfCompute(Xt, C, df, single, mr);
    }
    return mvtpdfCompute(X, C, df, single, mr);
}

Value mvtpdfCompute(const Value &X, const Value &C, const Value &df,
                    bool single, keel::Allocator *mr)
{
    keel::usize n = keel::as<keel::usize>(X.dims().dim(0));
    keel::usize d = keel::as<keel::usize>(X.dims().dim(1));

    const keel::usize cr = keel::as<keel::usize>(C.dims().dim(0));
    const keel::usize cc = keel::as<keel::usize>(C.dims().dim(1));
    if (cr != cc)
        throw Error(keel::format("mvtpdf: C must be a square matrix (got {}-by-{}).", cr, cc),
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:badCorrelationNotSquare");
    if (cr != d)
        throw Error(keel::format("mvtpdf: C must be a square matrix with size equal to "
                                 "the number of columns of X ({}-by-{}, got {}-by-{}).",
                                 d, d, cr, cc),
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:inputSizeMismatchC");
    single = single || (C.type() == ValueType::SINGLE);

    // df: scalar, or one per row.
    keel::Array<keel::f64> dfv(n);
    if (df.isScalar()) {
        std::fill(dfv.begin(), dfv.end(), df.elemAsDouble(0));
    } else if (df.dims().isVector()
               && df.numel() == n) {
        for (keel::usize r = 0; r < n; ++r) dfv[r] = df.elemAsDouble(r);
    } else {
        throw Error("mvtpdf: DF must be a scalar or a vector with one "
                    "element for each row of X.",
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:inputSizeMismatchDF");
    }
    for (keel::f64 v : dfv)
        if (!(v > 0.0))
            throw Error(keel::format("mvtpdf: DF must be positive (got {}).", v),
                        0, 0, "mvtpdf", "", "numkit:mvtpdf:invalidDF");
    single = single || (df.type() == ValueType::SINGLE);

    // Standardize C to a correlation matrix (the reference does this
    // silently; probe T06/T10).
    keel::Array<keel::f64> Cm(d * d, 0.0);
    for (keel::usize k = 0; k < d * d; ++k) Cm[k] = C.elemAsDouble(k);
    keel::Array<keel::f64> sq(d);
    bool unitDiag = true;
    for (keel::usize i = 0; i < d; ++i) {
        sq[i] = std::sqrt(Cm[i + i * d]);
        if (Cm[i + i * d] != 1.0) unitDiag = false;
    }
    if (!unitDiag)
        for (keel::usize j = 0; j < d; ++j)
            for (keel::usize i = 0; i < d; ++i)
                Cm[i + j * d] /= sq[i] * sq[j];

    keel::Array<keel::f64> quad;
    keel::f64 logHalfDet = 0.0;
    if (!psdQuadForms(X, n, d, Cm.as_ref(), quad, logHalfDet, mr))
        throw Error("mvtpdf: C must be a symmetric positive "
                    "(semi-)definite matrix.",
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:badCorrelationSymPos");

    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize r = 0; r < n; ++r) {
        const keel::f64 nu = dfv[r];
        const keel::f64 logpdf =
            std::lgamma((nu + keel::as<keel::f64>(d)) / 2.0)
            - std::lgamma(nu / 2.0)
            - 0.5 * keel::as<keel::f64>(d) * std::log(nu * M_PI)
            - logHalfDet
            - ((nu + keel::as<keel::f64>(d)) / 2.0) * std::log1p(quad[r] / nu);
        od[r] = std::exp(logpdf);
    }
    return castDown(out, single, mr);
}

Value mnpdf(const Value &X, const Value &P, keel::Allocator *mr)
{
    Value x = X, p = P;
    bool single = (X.type() == ValueType::SINGLE)
                  || (P.type() == ValueType::SINGLE);
    const keel::f64 epsD = std::numeric_limits<keel::f64>::epsilon();

    // A column p that sums to one is a single probability vector —
    // transpose it (and a column x alongside it).
    if (p.dims().dim(1) == 1 && p.dims().dim(0) > 1) {
        keel::f64 s = 0.0;
        const keel::usize len = keel::as<keel::usize>(p.numel());
        for (keel::usize k = 0; k < len; ++k) s += p.elemAsDouble(k);
        if (keel::abs(s - 1.0) <= keel::as<keel::f64>(len) * epsD) {
            Value pt = Value::matrix(1, len, ValueType::DOUBLE, mr);
            for (keel::usize k = 0; k < len; ++k)
                pt.doubleDataMut()[k] = p.elemAsDouble(k);
            p = pt;
            if (x.dims().dim(1) == 1 && x.dims().dim(0) > 1) {
                Value xt = Value::matrix(1, x.numel(), ValueType::DOUBLE, mr);
                for (keel::usize k = 0; k < keel::as<keel::usize>(x.numel()); ++k)
                    xt.doubleDataMut()[k] = x.elemAsDouble(k);
                x = xt;
            }
        }
    }

    keel::usize m = keel::as<keel::usize>(x.dims().dim(0));
    const keel::usize k = keel::as<keel::usize>(x.dims().dim(1));
    if (k < 1)
        throw Error("mnpdf: X must have at least one column.",
                    0, 0, "mnpdf", "", "numkit:mnpdf:noCategories");

    const keel::usize mm = keel::as<keel::usize>(p.dims().dim(0));
    const keel::usize kk = keel::as<keel::usize>(p.dims().dim(1));
    if (kk != k)
        throw Error(keel::format("mnpdf: P and X must have the same number of columns "
                                 "(X has {}, P has {}).", k, kk),
                    0, 0, "mnpdf", "", "numkit:mnpdf:colSizeMismatch");

    Value out = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    if (mm == 1 || m == 1 || mm == m) {
        if (m == 1 && mm > 1) {
            // Single data row replicates across p's rows.
            Value xt = Value::matrix(mm, k, ValueType::DOUBLE, mr);
            for (keel::usize j = 0; j < k; ++j)
                for (keel::usize r = 0; r < mm; ++r)
                    xt.doubleDataMut()[r + j * mm] = x.elemAsDouble(j);
            x = xt;
            m = mm;
            out = Value::matrix(m, 1, ValueType::DOUBLE, mr);
            od = out.doubleDataMut();
        }
        for (keel::usize r = 0; r < m; ++r) {
            const keel::usize pr = (mm == 1) ? 0 : r;
            // Row totals and the bad-value classes.
            keel::f64 n = 0.0;
            for (keel::usize j = 0; j < k; ++j)
                n += x.elemAsDouble(r + j * m);
            bool xBad = false;
            for (keel::usize j = 0; j < k; ++j) {
                const keel::f64 v = x.elemAsDouble(r + j * m);
                if (v < 0.0 || v != std::round(v)) xBad = true;
            }
            keel::f64 pSum = 0.0;
            bool pBad = false;
            for (keel::usize j = 0; j < k; ++j) {
                const keel::f64 v = p.elemAsDouble(pr + j * mm);
                if (v < 0.0 || 1.0 < v) pBad = true;
                pSum += v;
            }
            if (keel::abs(pSum - 1.0)
                    > keel::as<keel::f64>(k) * epsD)
                pBad = true;
            const bool nBad = (n < 0.0) || (std::round(n) != n);
            if (pBad || nBad) {
                od[r] = kNaN;
                continue;
            }
            if (xBad) {
                od[r] = 0.0;
                continue;
            }
            keel::f64 lp = std::lgamma(n + 1.0);
            for (keel::usize j = 0; j < k; ++j) {
                const keel::f64 xv = x.elemAsDouble(r + j * m);
                lp -= std::lgamma(xv + 1.0);
                if (xv > 0.0)
                    lp += xv * std::log(p.elemAsDouble(pr + j * mm));
                // xv == 0 with any p contributes 0.
            }
            od[r] = std::exp(lp);
        }
    } else {
        throw Error(keel::format("mnpdf: P and X must have the same number of rows "
                                 "(X has {}, P has {}).", m, mm),
                    0, 0, "mnpdf", "", "numkit:mnpdf:rowSizeMismatch");
    }
    return castDown(out, single, mr);
}

} // namespace numkit::stats
