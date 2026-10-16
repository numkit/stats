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

#include <numkit/stats/mvdist/mvdist.hpp>

#include <numkit/linalg/eig.hpp>             // eig_symmetric
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::stats {

namespace {

constexpr double kLog2Pi = 1.8378770664093454835606594728112352;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool isMatrixValue(const Value &V)
{
    return V.dims().ndim() <= 2 && V.numel() > 0;
}

// Symmetric eigendecomposition of a d×d column-major matrix read via
// elemAsDouble. Eigenvalues ascend in `eval`; eigenvectors are the
// columns of `evec` (column-major d×d).
void eigSymmetric(const std::vector<double> &M, std::size_t d,
                  std::vector<double> &eval, std::vector<double> &evec,
                  std::pmr::memory_resource *mr)
{
    Value Mv = Value::matrix(d, d, ValueType::DOUBLE, mr);
    std::copy(M.begin(), M.end(), Mv.doubleDataMut());
    auto [V, D] = ::numkit::linalg::eig_symmetric(Mv, mr);
    evec.resize(d * d);
    std::copy(V.doubleData(), V.doubleData() + d * d, evec.begin());
    eval.resize(d);
    for (std::size_t i = 0; i < d; ++i) eval[i] = D.doubleData()[i + i * d];
}

// Quadratic-form machinery shared by the normal and t densities:
// factor sigma = V·diag(λ)·V' with negative eigenvalues clamped to 0
// (positive SEMI-definite, the reference's cholcov semantics). Returns
// false when sigma is not even semidefinite (λ < -tol). `quad` fills
// per-row quadratic forms q = Σ_{λi>0} (v_i'·dx)²/λi and logHalfDet
// fills Σ log(sqrt(λi)) over ALL eigenvalues — a zero eigenvalue gives
// -Inf, which drives a degenerate density to 0 (probe P15).
bool psdQuadForms(const Value &X0, std::size_t n, std::size_t d,
                  const std::vector<double> &Sig,
                  std::vector<double> &quad, double &logHalfDet,
                  std::pmr::memory_resource *mr, bool *degenerate = nullptr)
{
    std::vector<double> eval, evec;
    eigSymmetric(Sig, d, eval, evec, mr);
    double maxAbs = 0.0;
    for (double e : eval) maxAbs = std::max(maxAbs, std::abs(e));
    const double tol = static_cast<double>(d) * std::numeric_limits<double>::epsilon()
                     * std::max(maxAbs, 1.0);
    for (double e : eval)
        if (e < -tol) return false;
    logHalfDet = 0.0;
    if (degenerate) *degenerate = false;
    for (double e : eval) {
        if (e <= tol && degenerate) *degenerate = true;
        logHalfDet += 0.5 * std::log(std::max(e, 0.0));   // 0 → -Inf
    }
    quad.assign(n, 0.0);
    std::vector<double> proj(d);
    for (std::size_t i = 0; i < d; ++i) {
        if (eval[i] <= tol) continue;                     // null direction
        for (std::size_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (std::size_t j = 0; j < d; ++j)
                s += evec[j + i * d] * X0.elemAsDouble(r + j * n);
            quad[r] += s * s / eval[i];
        }
    }
    return true;
}

// Diagonal-sigma quadratic forms + log half determinant.
void diagQuadForms(const Value &X0, std::size_t n, std::size_t d,
                   const std::vector<double> &sd,
                   std::vector<double> &quad, double &logHalfDet)
{
    logHalfDet = 0.0;
    for (std::size_t j = 0; j < d; ++j) logHalfDet += std::log(sd[j]);
    quad.assign(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        double q = 0.0;
        for (std::size_t j = 0; j < d; ++j) {
            const double z = X0.elemAsDouble(r + j * n) / sd[j];
            q += z * z;
        }
        quad[r] = q;
    }
}

Value castDown(const Value &X, bool single, std::pmr::memory_resource *mr)
{
    if (!single || X.type() == ValueType::SINGLE) return X;
    const std::size_t r = static_cast<std::size_t>(X.dims().dim(0));
    const std::size_t c = static_cast<std::size_t>(X.dims().dim(1));
    Value out = Value::matrix(r, c, ValueType::SINGLE, mr);
    const std::size_t cnt = r * c;
    for (std::size_t k = 0; k < cnt; ++k)
        out.singleDataMut()[k] = static_cast<float>(X.elemAsDouble(k));
    return out;
}

} // namespace

Value mvnpdf(const Value &X, const Value &mu, const Value &Sigma,
             std::pmr::memory_resource *mr)
{
    // ── X gate ──────────────────────────────────────────────────────
    if (!isMatrixValue(X))
        throw Error("mvnpdf: X must be a non-empty matrix (got a "
                    + std::string(mtypeName(X.type())) + " array).",
                    0, 0, "mvnpdf", "", "nk:mvnpdf:invalidData");
    std::size_t n = static_cast<std::size_t>(X.dims().dim(0));
    std::size_t d = static_cast<std::size_t>(X.dims().dim(1));
    if (d < 1)
        throw Error("mvnpdf: X must have at least one column.",
                    0, 0, "mvnpdf", "", "nk:mvnpdf:tooFewDimensions");
    bool single = (X.type() == ValueType::SINGLE);

    // ── mu: empty | scalar | matrix (row, full, or rep-when-single) ──
    Value X0;
    if (mu.isEmpty()) {
        X0 = X;
    } else if (mu.isScalar()) {
        const double m = mu.elemAsDouble(0);
        X0 = Value::matrix(n, d, ValueType::DOUBLE, mr);
        for (std::size_t k = 0; k < n * d; ++k)
            X0.doubleDataMut()[k] = X.elemAsDouble(k) - m;
    } else if (isMatrixValue(mu)) {
        const std::size_t n2 = static_cast<std::size_t>(mu.dims().dim(0));
        const std::size_t d2 = static_cast<std::size_t>(mu.dims().dim(1));
        if (d2 != d)
            throw Error("mvnpdf: X and MU must have the same number of "
                        "columns (X has " + std::to_string(d) + ", MU has "
                        + std::to_string(d2) + ").",
                        0, 0, "mvnpdf", "", "nk:mvnpdf:colSizeMismatch");
        if (n2 == 1 || n2 == n) {
            X0 = Value::matrix(n, d, ValueType::DOUBLE, mr);
            for (std::size_t j = 0; j < d; ++j)
                for (std::size_t r = 0; r < n; ++r)
                    X0.doubleDataMut()[r + j * n] =
                        X.elemAsDouble(r + j * n)
                        - mu.elemAsDouble((n2 == 1 ? 0 : r) + j * n2);
        } else if (n == 1) {
            n = n2;
            X0 = Value::matrix(n, d, ValueType::DOUBLE, mr);
            for (std::size_t j = 0; j < d; ++j)
                for (std::size_t r = 0; r < n; ++r)
                    X0.doubleDataMut()[r + j * n] =
                        X.elemAsDouble(j * 1) - mu.elemAsDouble(r + j * n);
        } else {
            throw Error("mvnpdf: MU must be a row vector or have one row "
                        "per X row (X has " + std::to_string(n)
                        + " rows, MU has " + std::to_string(n2) + ").",
                        0, 0, "mvnpdf", "", "nk:mvnpdf:rowSizeMismatch");
        }
    } else {
        throw Error("mvnpdf: MU must be a scalar or a matrix.",
                    0, 0, "mvnpdf", "", "nk:mvnpdf:badMu");
    }
    single = single || (mu.type() == ValueType::SINGLE);

    // ── Sigma: empty | 1×k diag row | d×d matrix | d×d×n stack ──────
    std::size_t pages = 1;
    bool sigmaIsDiag = false;
    std::vector<double> sd;                    // diag form: sqrt entries
    std::vector<double> Sig;                   // matrix form, column-major
    std::vector<std::vector<double>> pageSig;  // stack form
    std::size_t sz1 = 0, sz2 = 0;
    if (Sigma.isEmpty()) {
        // No Sigma: a column X is one d = n observation row.
        if (d == 1 && X.numel() > 1) {
            Value Xt = Value::matrix(1, n, ValueType::DOUBLE, mr);
            for (std::size_t k = 0; k < n; ++k)
                Xt.doubleDataMut()[k] = X0.elemAsDouble(k);
            X0 = Xt;
            d = n;
            n = 1;
        }
        sigmaIsDiag = true;
        sd.assign(d, 1.0);
    } else {
        sz1 = static_cast<std::size_t>(Sigma.dims().dim(0));
        sz2 = static_cast<std::size_t>(Sigma.dims().dim(1));
        if (Sigma.dims().ndim() <= 2 && sz1 == 1 && sz2 > 1) {
            sigmaIsDiag = true;
            sz1 = sz2;
        }
        // Column X + matrix Sigma whose rows match → reinterpret as row.
        if (!sigmaIsDiag && d == 1 && X.numel() > 1 && sz1 == n) {
            Value Xt = Value::matrix(1, n, ValueType::DOUBLE, mr);
            for (std::size_t k = 0; k < n; ++k)
                Xt.doubleDataMut()[k] = X0.elemAsDouble(k);
            X0 = Xt;
            d = n;
            n = 1;
        }
        single = single || (Sigma.type() == ValueType::SINGLE);
        pages = Sigma.dims().ndim() >= 3
                  ? static_cast<std::size_t>(Sigma.dims().dim(2)) : 1;
        if (pages == 1) {
            if (sz1 != sz2)
                throw Error("mvnpdf: SIGMA must be a square covariance "
                            "matrix, the diagonal of one as a row vector, "
                            "or a stack of matrices.",
                            0, 0, "mvnpdf", "", "nk:mvnpdf:badCovariance");
            if (sz1 != d)
                throw Error("mvnpdf: SIGMA must be " + std::to_string(d)
                            + "-by-" + std::to_string(d) + " to match X "
                            "(got " + std::to_string(sz1) + "-by-"
                            + std::to_string(sz2) + ").",
                            0, 0, "mvnpdf", "", "nk:mvnpdf:covSizeMismatch");
            if (sigmaIsDiag) {
                sd.assign(d, 0.0);
                for (std::size_t j = 0; j < d; ++j) {
                    const double v = Sigma.elemAsDouble(j);
                    if (!(v > 0.0))
                        throw Error("mvnpdf: all elements of the diagonal "
                                    "SIGMA must be positive (got "
                                    + std::to_string(v) + " at position "
                                    + std::to_string(j + 1) + ").",
                                    0, 0, "mvnpdf", "",
                                    "nk:mvnpdf:badDiagSigma");
                    sd[j] = std::sqrt(v);
                }
            } else {
                Sig.assign(d * d, 0.0);
                for (std::size_t k = 0; k < d * d; ++k)
                    Sig[k] = Sigma.elemAsDouble(k);
            }
        } else {
            // Stack: a single data row replicates across the pages.
            if (n == 1) {
                n = pages;
                Value Xr = Value::matrix(n, d, ValueType::DOUBLE, mr);
                for (std::size_t j = 0; j < d; ++j)
                    for (std::size_t r = 0; r < n; ++r)
                        Xr.doubleDataMut()[r + j * n] = X0.elemAsDouble(j);
                X0 = Xr;
            }
            if (sz1 != sz2)
                throw Error("mvnpdf: SIGMA must be a square covariance "
                            "matrix, the diagonal of one as a row vector, "
                            "or a stack of matrices.",
                            0, 0, "mvnpdf", "",
                            "nk:mvnpdf:badCovarianceMultiple");
            if (sz1 != d)
                throw Error("mvnpdf: each SIGMA page must be "
                            + std::to_string(d) + "-by-" + std::to_string(d)
                            + " to match X.",
                            0, 0, "mvnpdf", "",
                            "nk:mvnpdf:covSizeMismatchMultiple");
            if (pages != n)
                throw Error("mvnpdf: SIGMA must have one page per X row "
                            "(X has " + std::to_string(n) + " rows, SIGMA "
                            "has " + std::to_string(pages) + " pages).",
                            0, 0, "mvnpdf", "",
                            "nk:mvnpdf:covSizeMismatchPages");
            if (sigmaIsDiag) {
                // Row diag [1×d×n] → per-page sqrt rows.
                for (std::size_t j = 0; j < d; ++j) {
                    for (std::size_t p = 0; p < pages; ++p) {
                        const double v = Sigma.elemAsDouble(j + p * d);
                        if (!(v > 0.0))
                            throw Error("mvnpdf: all elements of the "
                                        "diagonal SIGMA must be positive.",
                                        0, 0, "mvnpdf", "",
                                        "nk:mvnpdf:badDiagSigma");
                    }
                }
            } else {
                pageSig.resize(pages);
                for (std::size_t p = 0; p < pages; ++p) {
                    pageSig[p].assign(d * d, 0.0);
                    for (std::size_t k = 0; k < d * d; ++k)
                        pageSig[p][k] = Sigma.elemAsDouble(k + p * d * d);
                }
            }
        }
    }

    // ── density ─────────────────────────────────────────────────────
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    const double base = -0.5 * static_cast<double>(d) * kLog2Pi;
    auto fillRows = [&](const std::vector<double> &quad, double logHalfDet) {
        for (std::size_t r = 0; r < n; ++r)
            od[r] = std::exp(base - logHalfDet - 0.5 * quad[r]);
    };
    if (pages == 1) {
        std::vector<double> quad;
        double logHalfDet = 0.0;
        bool degenerate = false;
        if (sigmaIsDiag) {
            diagQuadForms(X0, n, d, sd, quad, logHalfDet);
        } else if (!psdQuadForms(X0, n, d, Sig, quad, logHalfDet, mr,
                                 &degenerate)) {
            throw Error("mvnpdf: SIGMA must be a square, symmetric, "
                        "positive semi-definite matrix.",
                        0, 0, "mvnpdf", "", "nk:mvnpdf:badMatrixSigma");
        }
        if (degenerate) {
            for (std::size_t r = 0; r < n; ++r) od[r] = 0.0;
        } else {
            fillRows(quad, logHalfDet);
        }
    } else if (sigmaIsDiag) {
        // Per-page diagonal densities.
        for (std::size_t r = 0; r < n; ++r) {
            double q = 0.0, lhd = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                const double s = std::sqrt(Sigma.elemAsDouble(j + r * d));
                const double z = X0.elemAsDouble(r + j * n) / s;
                q += z * z;
                lhd += std::log(s);
            }
            od[r] = std::exp(base - lhd - 0.5 * q);
        }
    } else {
        for (std::size_t r = 0; r < n; ++r) {
            // One-page slice through the shared machinery.
            Value Xrow = Value::matrix(1, d, ValueType::DOUBLE, mr);
            for (std::size_t j = 0; j < d; ++j)
                Xrow.doubleDataMut()[j] = X0.elemAsDouble(r + j * n);
            std::vector<double> quad;
            double logHalfDet = 0.0;
            bool degenerate = false;
            if (!psdQuadForms(Xrow, 1, d, pageSig[r], quad, logHalfDet, mr,
                              &degenerate))
                throw Error("mvnpdf: SIGMA must be a square, symmetric, "
                            "positive semi-definite matrix.",
                            0, 0, "mvnpdf", "",
                            "nk:mvnpdf:badMatrixSigmaMultiple");
            od[r] = degenerate ? 0.0
                               : std::exp(base - logHalfDet - 0.5 * quad[0]);
        }
    }
    return castDown(out, single, mr);
}

Value mvtpdfCompute(const Value &X, const Value &C, const Value &df,
                    bool single, std::pmr::memory_resource *mr);

Value mvtpdf(const Value &X, const Value &C, const Value &df,
             std::pmr::memory_resource *mr)
{
    if (X.dims().ndim() != 2)
        throw Error("mvtpdf: X must be a matrix.",
                    0, 0, "mvtpdf", "", "nk:mvtpdf:invalidData");
    std::size_t n = static_cast<std::size_t>(X.dims().dim(0));
    std::size_t d = static_cast<std::size_t>(X.dims().dim(1));
    if (d < 1)
        throw Error("mvtpdf: X must have at least one column.",
                    0, 0, "mvtpdf", "", "nk:mvtpdf:tooFewDimensions");
    bool single = (X.type() == ValueType::SINGLE);

    // Row reinterpretation: vector X whose length matches rows(C).
    const std::size_t cr = static_cast<std::size_t>(C.dims().dim(0));
    const std::size_t cc = static_cast<std::size_t>(C.dims().dim(1));
    if (X.dims().isVector() && cr == n) {
        Value Xt = Value::matrix(1, n, ValueType::DOUBLE, mr);
        for (std::size_t k = 0; k < n; ++k)
            Xt.doubleDataMut()[k] = X.elemAsDouble(k);
        return mvtpdfCompute(Xt, C, df, single, mr);
    }
    return mvtpdfCompute(X, C, df, single, mr);
}

Value mvtpdfCompute(const Value &X, const Value &C, const Value &df,
                    bool single, std::pmr::memory_resource *mr)
{
    std::size_t n = static_cast<std::size_t>(X.dims().dim(0));
    std::size_t d = static_cast<std::size_t>(X.dims().dim(1));

    const std::size_t cr = static_cast<std::size_t>(C.dims().dim(0));
    const std::size_t cc = static_cast<std::size_t>(C.dims().dim(1));
    if (cr != cc)
        throw Error("mvtpdf: C must be a square matrix (got "
                    + std::to_string(cr) + "-by-" + std::to_string(cc)
                    + ").",
                    0, 0, "mvtpdf", "", "nk:mvtpdf:badCorrelationNotSquare");
    if (cr != d)
        throw Error("mvtpdf: C must be a square matrix with size equal to "
                    "the number of columns of X (" + std::to_string(d)
                    + "-by-" + std::to_string(d) + ", got "
                    + std::to_string(cr) + "-by-" + std::to_string(cc)
                    + ").",
                    0, 0, "mvtpdf", "", "nk:mvtpdf:inputSizeMismatchC");
    single = single || (C.type() == ValueType::SINGLE);

    // df: scalar, or one per row.
    std::vector<double> dfv(n);
    if (df.isScalar()) {
        dfv.assign(n, df.elemAsDouble(0));
    } else if (df.dims().isVector()
               && df.numel() == static_cast<long long>(n)) {
        for (std::size_t r = 0; r < n; ++r) dfv[r] = df.elemAsDouble(r);
    } else {
        throw Error("mvtpdf: DF must be a scalar or a vector with one "
                    "element for each row of X.",
                    0, 0, "mvtpdf", "", "nk:mvtpdf:inputSizeMismatchDF");
    }
    for (double v : dfv)
        if (!(v > 0.0))
            throw Error("mvtpdf: DF must be positive (got "
                        + std::to_string(v) + ").",
                        0, 0, "mvtpdf", "", "nk:mvtpdf:invalidDF");
    single = single || (df.type() == ValueType::SINGLE);

    // Standardize C to a correlation matrix (the reference does this
    // silently; probe T06/T10).
    std::vector<double> Cm(d * d, 0.0);
    for (std::size_t k = 0; k < d * d; ++k) Cm[k] = C.elemAsDouble(k);
    std::vector<double> sq(d);
    bool unitDiag = true;
    for (std::size_t i = 0; i < d; ++i) {
        sq[i] = std::sqrt(Cm[i + i * d]);
        if (Cm[i + i * d] != 1.0) unitDiag = false;
    }
    if (!unitDiag)
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t i = 0; i < d; ++i)
                Cm[i + j * d] /= sq[i] * sq[j];

    std::vector<double> quad;
    double logHalfDet = 0.0;
    if (!psdQuadForms(X, n, d, Cm, quad, logHalfDet, mr))
        throw Error("mvtpdf: C must be a symmetric positive "
                    "(semi-)definite matrix.",
                    0, 0, "mvtpdf", "", "nk:mvtpdf:badCorrelationSymPos");

    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t r = 0; r < n; ++r) {
        const double nu = dfv[r];
        const double logpdf =
            std::lgamma((nu + static_cast<double>(d)) / 2.0)
            - std::lgamma(nu / 2.0)
            - 0.5 * static_cast<double>(d) * std::log(nu * M_PI)
            - logHalfDet
            - ((nu + static_cast<double>(d)) / 2.0) * std::log1p(quad[r] / nu);
        od[r] = std::exp(logpdf);
    }
    return castDown(out, single, mr);
}

Value mnpdf(const Value &X, const Value &P, std::pmr::memory_resource *mr)
{
    Value x = X, p = P;
    bool single = (X.type() == ValueType::SINGLE)
                  || (P.type() == ValueType::SINGLE);
    const double epsD = std::numeric_limits<double>::epsilon();

    // A column p that sums to one is a single probability vector —
    // transpose it (and a column x alongside it).
    if (p.dims().dim(1) == 1 && p.dims().dim(0) > 1) {
        double s = 0.0;
        const std::size_t len = static_cast<std::size_t>(p.numel());
        for (std::size_t k = 0; k < len; ++k) s += p.elemAsDouble(k);
        if (std::abs(s - 1.0) <= static_cast<double>(len) * epsD) {
            Value pt = Value::matrix(1, len, ValueType::DOUBLE, mr);
            for (std::size_t k = 0; k < len; ++k)
                pt.doubleDataMut()[k] = p.elemAsDouble(k);
            p = pt;
            if (x.dims().dim(1) == 1 && x.dims().dim(0) > 1) {
                Value xt = Value::matrix(1, x.numel(), ValueType::DOUBLE, mr);
                for (std::size_t k = 0; k < static_cast<std::size_t>(x.numel()); ++k)
                    xt.doubleDataMut()[k] = x.elemAsDouble(k);
                x = xt;
            }
        }
    }

    std::size_t m = static_cast<std::size_t>(x.dims().dim(0));
    const std::size_t k = static_cast<std::size_t>(x.dims().dim(1));
    if (k < 1)
        throw Error("mnpdf: X must have at least one column.",
                    0, 0, "mnpdf", "", "nk:mnpdf:noCategories");

    const std::size_t mm = static_cast<std::size_t>(p.dims().dim(0));
    const std::size_t kk = static_cast<std::size_t>(p.dims().dim(1));
    if (kk != k)
        throw Error("mnpdf: P and X must have the same number of columns "
                    "(X has " + std::to_string(k) + ", P has "
                    + std::to_string(kk) + ").",
                    0, 0, "mnpdf", "", "nk:mnpdf:colSizeMismatch");

    Value out = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (mm == 1 || m == 1 || mm == m) {
        if (m == 1 && mm > 1) {
            // Single data row replicates across p's rows.
            Value xt = Value::matrix(mm, k, ValueType::DOUBLE, mr);
            for (std::size_t j = 0; j < k; ++j)
                for (std::size_t r = 0; r < mm; ++r)
                    xt.doubleDataMut()[r + j * mm] = x.elemAsDouble(j);
            x = xt;
            m = mm;
            out = Value::matrix(m, 1, ValueType::DOUBLE, mr);
            od = out.doubleDataMut();
        }
        for (std::size_t r = 0; r < m; ++r) {
            const std::size_t pr = (mm == 1) ? 0 : r;
            // Row totals and the bad-value classes.
            double n = 0.0;
            for (std::size_t j = 0; j < k; ++j)
                n += x.elemAsDouble(r + j * m);
            bool xBad = false;
            for (std::size_t j = 0; j < k; ++j) {
                const double v = x.elemAsDouble(r + j * m);
                if (v < 0.0 || v != std::round(v)) xBad = true;
            }
            double pSum = 0.0;
            bool pBad = false;
            for (std::size_t j = 0; j < k; ++j) {
                const double v = p.elemAsDouble(pr + j * mm);
                if (v < 0.0 || 1.0 < v) pBad = true;
                pSum += v;
            }
            if (std::abs(pSum - 1.0)
                    > static_cast<double>(k) * epsD)
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
            double lp = std::lgamma(n + 1.0);
            for (std::size_t j = 0; j < k; ++j) {
                const double xv = x.elemAsDouble(r + j * m);
                lp -= std::lgamma(xv + 1.0);
                if (xv > 0.0)
                    lp += xv * std::log(p.elemAsDouble(pr + j * mm));
                // xv == 0 with any p contributes 0.
            }
            od[r] = std::exp(lp);
        }
    } else {
        throw Error("mnpdf: P and X must have the same number of rows "
                    "(X has " + std::to_string(m) + ", P has "
                    + std::to_string(mm) + ").",
                    0, 0, "mnpdf", "", "nk:mnpdf:rowSizeMismatch");
    }
    return castDown(out, single, mr);
}

} // namespace numkit::stats
