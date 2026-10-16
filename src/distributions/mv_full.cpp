// toolboxes/stats/src/distributions/mv_full.cpp
//
// Full parse+validate+compute contracts for the multivariate random /
// CDF entries — mvnrnd / mvncdf / mvtrnd (probed against the reference
// engine R2025b: build/probe73.m R/C/T batteries). The register
// adapters marshal only; the simple public overloads delegate here.
//
// Shared machinery: an eigenvalue factorization that accepts positive
// SEMI-definite matrices (the reference's cholcov semantics — perfect
// correlation is legal) and a symmetry pre-check so nonsymmetric
// matrices get the Sym rejection class rather than the Pos one.

#include <numkit/stats/distributions/multivariate.hpp>

#include <numkit/builtin/datafun.hpp>          // RngContext
#include <numkit/linalg/eig.hpp>               // eig_symmetric
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "multivariate_detail.hpp"         // phiCdf / bivariateCdf / monteCarloCdf

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace numkit::stats {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

bool isMatrixValue(const Value &V)
{
    return V.dims().ndim() <= 2 && V.numel() > 0;
}

bool isSymmetric(const std::vector<double> &M, std::size_t d)
{
    for (std::size_t j = 0; j < d; ++j)
        for (std::size_t i = j + 1; i < d; ++i)
            if (M[i + j * d] != M[j + i * d]) return false;
    return true;
}

// Factor Sigma = T·T' with T d×d (rows beyond the rank are zero):
// T = V·diag(sqrt(λ clamped at 0)). Returns false when Sigma is not
// even semidefinite. `eval`/`evec` out-param the eigendecomposition
// (ascending eigenvalues; evec column-major).
bool psdFactor(const std::vector<double> &Sig, std::size_t d,
               std::vector<double> &T, std::vector<double> &eval,
               std::vector<double> &evec, std::pmr::memory_resource *mr)
{
    Value Mv = Value::matrix(d, d, ValueType::DOUBLE, mr);
    std::copy(Sig.begin(), Sig.end(), Mv.doubleDataMut());
    auto [V, D] = ::numkit::linalg::eig_symmetric(Mv, mr);
    evec.assign(V.doubleData(), V.doubleData() + d * d);
    eval.assign(d, 0.0);
    double maxAbs = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        eval[i] = D.doubleData()[i + i * d];
        maxAbs = std::max(maxAbs, std::abs(eval[i]));
    }
    const double tol = static_cast<double>(d)
                     * std::numeric_limits<double>::epsilon()
                     * std::max(maxAbs, 1.0);
    for (double e : eval)
        if (e < -tol) return false;
    // Strictly positive definite → upper Cholesky (matches the
    // reference's cholcov factor exactly, probe R09). Semidefinite →
    // the eigendecomposition factor (null-space rows are zero).
    bool pd = true;
    for (double e : eval)
        if (e <= tol) pd = false;
    if (pd) {
        T.assign(d * d, 0.0);
        for (std::size_t j = 0; j < d; ++j) {
            for (std::size_t i = j; i < d; ++i) {
                double v = Sig[i + j * d];   // column-major source
                for (std::size_t k = 0; k < j; ++k)
                    v -= T[i + k * d] * T[j + k * d];
                T[i + j * d] = (i == j) ? std::sqrt(v)
                                        : v / T[j + j * d];
                if (i == j && !(v > 0.0)) { pd = false; break; }
            }
            if (!pd) break;
        }
    }
    if (!pd) {
        T.assign(d * d, 0.0);
        for (std::size_t j = 0; j < d; ++j) {
            const double s = std::sqrt(std::max(eval[j], 0.0));
            for (std::size_t i = 0; i < d; ++i)
                T[i + j * d] = evec[i + j * d] * s;
        }
    }
    return true;
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

// mu as a flat row (length d) or per-case rows; X keeps the cases×d
// resolved data. Returns false for a shape error (caller throws).
double muAt(const Value &mu, std::size_t row, std::size_t col,
            std::size_t muN)
{
    return muN == 1 ? mu.elemAsDouble(col)
                    : mu.elemAsDouble(col * muN + row);
}

} // namespace

// ── mvnrnd ────────────────────────────────────────────────────────────

MvnrndOut mvnrnd_reinterpreted(::numkit::ops::RngContext &rng, const Value &mu,
                               const Value &Sigma, const Value &cases,
                               bool wantT, std::pmr::memory_resource *mr);

MvnrndOut mvnrnd_full(::numkit::ops::RngContext &rng, const Value &mu,
                      const Value &Sigma, const Value &cases, bool wantT,
                      std::pmr::memory_resource *mr)
{
    if (mu.isEmpty() || Sigma.isEmpty())
        throw Error("mvnrnd: requires the input arguments MU and SIGMA.",
                    0, 0, "mvnrnd", "", "numkit:mvnrnd:tooFewInputs");
    if (!isMatrixValue(mu))
        throw Error("mvnrnd: MU must be a matrix.",
                    0, 0, "mvnrnd", "", "numkit:mvnrnd:badMu");
    if (Sigma.dims().ndim() > 3)
        throw Error("mvnrnd: SIGMA must be a matrix or a 3-D array.",
                    0, 0, "mvnrnd", "", "numkit:mvnrnd:badSigma");

    std::size_t n = static_cast<std::size_t>(mu.dims().dim(0));
    std::size_t d = static_cast<std::size_t>(mu.dims().dim(1));
    std::size_t sz1 = static_cast<std::size_t>(Sigma.dims().dim(0));
    const std::size_t sz2 = static_cast<std::size_t>(Sigma.dims().dim(1));
    bool sigmaIsDiag = false;
    if (Sigma.dims().ndim() <= 2 && sz1 == 1 && sz2 > 1) {
        sigmaIsDiag = true;
        sz1 = sz2;
    }
    // Column mu + matrix Sigma whose rows match → reinterpret as a row.
    if (d == 1 && sz1 == n && !sigmaIsDiag) {
        Value mut = Value::matrix(1, n, ValueType::DOUBLE, mr);
        for (std::size_t k = 0; k < n; ++k)
            mut.doubleDataMut()[k] = mu.elemAsDouble(k);
        return mvnrnd_reinterpreted(rng, mut, Sigma, cases, wantT, mr);
    }
    return mvnrnd_reinterpreted(rng, mu, Sigma, cases, wantT, mr);
}

// Body with the mu orientation already resolved (always 1×d or n×d).
MvnrndOut mvnrnd_reinterpreted(::numkit::ops::RngContext &rng, const Value &mu,
                               const Value &Sigma, const Value &cases,
                               bool wantT, std::pmr::memory_resource *mr)
{
    std::size_t n = static_cast<std::size_t>(mu.dims().dim(0));
    std::size_t d = static_cast<std::size_t>(mu.dims().dim(1));
    std::size_t sz1 = static_cast<std::size_t>(Sigma.dims().dim(0));
    std::size_t sz2 = static_cast<std::size_t>(Sigma.dims().dim(1));
    bool sigmaIsDiag = false;
    if (Sigma.dims().ndim() <= 2 && sz1 == 1 && sz2 > 1) {
        sigmaIsDiag = true;
        sz1 = sz2;
    }
    const bool single = (mu.type() == ValueType::SINGLE)
                        || (Sigma.type() == ValueType::SINGLE);

    // cases contract: absent/empty → derive; must equal rows(mu) unless
    // mu is a single row (replicated); cases <= 0 → 0×d empty output.
    bool nocases = cases.isEmpty();
    long long casesN = -1;
    if (!nocases) {
        casesN = static_cast<long long>(cases.elemAsDouble(0));
        if (static_cast<long long>(n) == casesN) {
            // mu already has cases rows.
        } else if (n == 1) {
            n = static_cast<std::size_t>(std::max(casesN, 0LL));
        } else {
            throw Error("mvnrnd: MU must be a row vector, or must have "
                        "CASES rows (CASES is " + std::to_string(casesN)
                        + ", MU has " + std::to_string(n) + " rows).",
                        0, 0, "mvnrnd", "", "numkit:mvnrnd:inputSizeMismatchMu");
        }
    }
    if (nocases && Sigma.dims().ndim() >= 3 && n == 1) {
        // A single mu row with a Sigma stack: one case per page.
        n = static_cast<std::size_t>(Sigma.dims().dim(2));
    }
    const std::size_t muN = (static_cast<std::size_t>(
        mu.dims().dim(0)) == 1) ? 1 : n;

    MvnrndOut out;
    if (n == 0) {
        out.r = Value::matrix(0, d, ValueType::DOUBLE, mr);
        return out;
    }

    auto &gen = rng;
    std::normal_distribution<double> Nz(0.0, 1.0);
    out.r = Value::matrix(n, d, ValueType::DOUBLE, mr);
    double *od = out.r.doubleDataMut();

    const std::size_t pages = Sigma.dims().ndim() >= 3
                  ? static_cast<std::size_t>(Sigma.dims().dim(2)) : 1;
    if (pages == 1) {
        if (sz1 != sz2)
            throw Error("mvnrnd: SIGMA must be a square covariance matrix "
                        "(got " + std::to_string(sz1) + "-by-"
                        + std::to_string(sz2) + ").",
                        0, 0, "mvnrnd", "", "numkit:mvnrnd:badCovariance2DSize");
        if (sz1 != d)
            throw Error("mvnrnd: SIGMA must be " + std::to_string(d)
                        + "-by-" + std::to_string(d) + " to match MU.",
                        0, 0, "mvnrnd", "",
                        "numkit:mvnrnd:inputSizeMismatch2DSigma");
        if (sigmaIsDiag) {
            std::vector<double> sd(d);
            for (std::size_t j = 0; j < d; ++j) {
                const double v = Sigma.elemAsDouble(j);
                if (v < 0.0)
                    throw Error("mvnrnd: all elements of the diagonal "
                                "SIGMA must be non-negative (got "
                                + std::to_string(v) + ").",
                                0, 0, "mvnrnd", "",
                                "numkit:mvnrnd:badDiagSigma");
                sd[j] = std::sqrt(v);
            }
            if (wantT) {
                out.T = Value::matrix(d, d, ValueType::DOUBLE, mr);
                double *td = out.T.doubleDataMut();
                for (std::size_t j = 0; j < d; ++j) td[j + j * d] = sd[j];
            }
            std::vector<double> Z(n * d);
            for (std::size_t j = 0; j < d; ++j)
                for (std::size_t r = 0; r < n; ++r)
                    Z[r + j * n] = Nz(gen);
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t j = 0; j < d; ++j)
                    od[j * n + r] = sd[j] > 0.0
                        ? Z[r + j * n] * sd[j] + muAt(mu, r, j, muN)
                        : muAt(mu, r, j, muN);   // exact mean at zero var
        } else {
            std::vector<double> Sig(d * d), T, eval, evec;
            for (std::size_t k = 0; k < d * d; ++k)
                Sig[k] = Sigma.elemAsDouble(k);
            if (!isSymmetric(Sig, d))
                throw Error("mvnrnd: SIGMA must be symmetric.",
                            0, 0, "mvnrnd", "",
                            "numkit:mvnrnd:badCovariance2DSym");
            if (!psdFactor(Sig, d, T, eval, evec, mr))
                throw Error("mvnrnd: SIGMA must be positive "
                            "semi-definite.",
                            0, 0, "mvnrnd", "",
                            "numkit:mvnrnd:badCovariance2DPos");
            if (wantT) {
                // Upper factor (Sigma = T'·T), matching the reference's
                // cholcov output — the transpose of the sampling L.
                out.T = Value::matrix(d, d, ValueType::DOUBLE, mr);
                double *td = out.T.doubleDataMut();
                for (std::size_t i = 0; i < d; ++i)
                    for (std::size_t j = 0; j < d; ++j)
                        td[i + j * d] = T[j + i * d];
            }
            std::vector<double> Z(n * d);
            for (std::size_t k = 0; k < d; ++k)
                for (std::size_t r = 0; r < n; ++r)
                    Z[r + k * n] = Nz(gen);
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t j = 0; j < d; ++j) {
                    double acc = 0.0;
                    for (std::size_t k = 0; k < d; ++k)
                        acc += T[j + k * d] * Z[r + k * n];
                    od[j * n + r] = acc + muAt(mu, r, j, muN);
                }
            // Force the exact mean in zero-variance directions.
            for (std::size_t j = 0; j < d; ++j)
                if (Sigma.elemAsDouble(j + j * d) == 0.0)
                    for (std::size_t r = 0; r < n; ++r)
                        od[j * n + r] = muAt(mu, r, j, muN);
        }
    } else {
        // 3-D Sigma stack: one covariance per case.
        if (sz1 != sz2)
            throw Error("mvnrnd: each SIGMA page must be a square "
                        "covariance matrix.",
                        0, 0, "mvnrnd", "",
                        "numkit:mvnrnd:badCovariance3DSize");
        if (sz1 != d)
            throw Error("mvnrnd: each SIGMA page must be "
                        + std::to_string(d) + "-by-" + std::to_string(d)
                        + " to match MU.",
                        0, 0, "mvnrnd", "",
                        "numkit:mvnrnd:inputSizeMismatch3DSigma");
        if (pages != n)
            throw Error("mvnrnd: SIGMA must have CASES pages (CASES is "
                        + std::to_string(n) + ", SIGMA has "
                        + std::to_string(pages) + " pages).",
                        0, 0, "mvnrnd", "",
                        "numkit:mvnrnd:inputSizeMismatchSigmaDimension");
        if (wantT)
            out.T = Value::matrix3d(d, d, pages, ValueType::DOUBLE, mr);
        std::vector<double> Sig(d * d), T, eval, evec, z(d);
        for (std::size_t p = 0; p < pages; ++p) {
            if (sigmaIsDiag) {
                // 1×d×n diagonal stack: per-page standard deviations.
                for (std::size_t j = 0; j < d; ++j) {
                    const double v = Sigma.elemAsDouble(j + p * d);
                    if (v < 0.0)
                        throw Error("mvnrnd: all elements of the diagonal "
                                    "SIGMA must be non-negative.",
                                    0, 0, "mvnrnd", "",
                                    "numkit:mvnrnd:badDiagSigma");
                }
                for (std::size_t j = 0; j < d; ++j) {
                    const double s = std::sqrt(Sigma.elemAsDouble(j + p * d));
                    z[j] = s > 0.0 ? Nz(gen) * s
                                   : muAt(mu, p, j, muN);
                    od[j * n + p] = s > 0.0 ? z[j] + muAt(mu, p, j, muN)
                                            : muAt(mu, p, j, muN);
                    if (wantT)
                        out.T.doubleDataMut()[(j + j * d) + p * d * d] = s;
                }
                continue;
            }
            for (std::size_t k = 0; k < d * d; ++k)
                Sig[k] = Sigma.elemAsDouble(k + p * d * d);
            if (!isSymmetric(Sig, d))
                throw Error("mvnrnd: SIGMA must be symmetric.",
                            0, 0, "mvnrnd", "",
                            "numkit:mvnrnd:badCovariance3DSym");
            if (!psdFactor(Sig, d, T, eval, evec, mr))
                throw Error("mvnrnd: SIGMA must be positive "
                            "semi-definite.",
                            0, 0, "mvnrnd", "",
                            "numkit:mvnrnd:badCovariance3DPos");
            if (wantT) {
                double *td = out.T.doubleDataMut() + p * d * d;
                for (std::size_t i = 0; i < d; ++i)
                    for (std::size_t j = 0; j < d; ++j)
                        td[i + j * d] = T[j + i * d];
            }
            for (std::size_t k = 0; k < d; ++k) z[k] = Nz(gen);
            for (std::size_t j = 0; j < d; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < d; ++k)
                    acc += T[j + k * d] * z[k];
                od[j * n + p] = acc + muAt(mu, p, j, muN);
                if (Sigma.elemAsDouble(j + j * d + p * d * d) == 0.0)
                    od[j * n + p] = muAt(mu, p, j, muN);
            }
        }
    }
    out.r = castDown(out.r, single, mr);
    if (wantT && !out.T.isEmpty())
        out.T = castDown(out.T, single, mr);
    return out;
}

// ── mvtrnd ────────────────────────────────────────────────────────────

Value mvtrnd_full(::numkit::ops::RngContext &rng, const Value &C,
                  const Value &df, const Value &cases,
                  std::pmr::memory_resource *mr)
{
    std::size_t n = 1;
    if (!cases.isEmpty())
        n = static_cast<std::size_t>(std::max(
            static_cast<long long>(cases.elemAsDouble(0)), 0LL));

    const std::size_t m1 = static_cast<std::size_t>(C.dims().dim(0));
    const std::size_t m2 = static_cast<std::size_t>(C.dims().dim(1));
    if (m1 != m2)
        throw Error("mvtrnd: the correlation matrix C must be square (got "
                    + std::to_string(m1) + "-by-" + std::to_string(m2)
                    + ").",
                    0, 0, "mvtrnd", "", "numkit:mvtrnd:badCorrelationNotSquare");

    // df: scalar, or one value per case.
    std::vector<double> dfv(n);
    if (df.isScalar()) {
        dfv.assign(n, df.elemAsDouble(0));
    } else if (df.dims().isVector()
               && df.numel() == static_cast<long long>(n)) {
        for (std::size_t r = 0; r < n; ++r) dfv[r] = df.elemAsDouble(r);
    } else {
        throw Error("mvtrnd: DF must be a scalar or a vector with one "
                    "element for each row of the output matrix.",
                    0, 0, "mvtrnd", "", "numkit:mvtrnd:inputSizeMismatch");
    }
    for (double v : dfv)
        if (!(v > 0.0))
            throw Error("mvtrnd: DF must be positive (got "
                        + std::to_string(v) + ").",
                        0, 0, "mvtrnd", "", "numkit:mvtrnd:badDF");

    // Standardize C to a correlation matrix (silent, probe T06).
    const std::size_t d = m1;
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
    if (!isSymmetric(Cm, d))
        throw Error("mvtrnd: C must be a symmetric positive "
                    "semi-definite matrix.",
                    0, 0, "mvtrnd", "", "numkit:mvtrnd:badCorrelationSymPos");
    std::vector<double> T, eval, evec;
    if (!psdFactor(Cm, d, T, eval, evec, mr))
        throw Error("mvtrnd: C must be a symmetric positive "
                    "semi-definite matrix.",
                    0, 0, "mvtrnd", "", "numkit:mvtrnd:badCorrelationSymPos");

    // r = (randn·T) ./ sqrt(χ²(df)/df), one χ² per case.
    Value out = Value::matrix(n, d, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    auto &gen = rng;
    std::normal_distribution<double> Nz(0.0, 1.0);
    std::chi_squared_distribution<double> Chi;
    std::vector<double> Z(n * d);
    for (std::size_t k = 0; k < d; ++k)
        for (std::size_t r = 0; r < n; ++r)
            Z[r + k * n] = Nz(gen);
    std::vector<double> scale(n);
    for (std::size_t r = 0; r < n; ++r) {
        Chi.param(std::chi_squared_distribution<double>::param_type(dfv[r]));
        scale[r] = std::sqrt(dfv[r] / Chi(gen));
    }
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t j = 0; j < d; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < d; ++k)
                acc += T[j + k * d] * Z[r + k * n];
            od[j * n + r] = acc * scale[r];
        }
    return castDown(out, C.type() == ValueType::SINGLE
                          || df.type() == ValueType::SINGLE, mr);
}

// ── mvncdf ────────────────────────────────────────────────────────────

Value mvncdf_full(Span<const Value> args, std::pmr::memory_resource *mr)
{
    // A trailing options struct is stripped before parsing (the
    // reference's statset options; the integrator options are inert
    // for the closed-form d ≤ 2 paths).
    std::size_t nin = args.size();
    if (nin > 0 && args[nin - 1].isObject())
        --nin;
    if (nin == 0)
        throw Error("mvncdf: requires at least one input argument.",
                    0, 0, "mvncdf", "", "numkit:mvncdf:tooFewInputs");

    Value XU, XL;
    Value mu, Sigma;
    bool box = false;
    if (nin < 4) {
        XU = args[0];
        if (!isMatrixValue(XU))
            throw Error("mvncdf: X must be a non-empty matrix.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:invalidDataMatrix");
        XL = Value();   // -Inf lower limits, filled per-d below
        if (nin > 1) mu = args[1];
        if (nin > 2) Sigma = args[2];
    } else {
        box = true;
        XL = args[0];
        XU = args[1];
        mu = args[2];
        Sigma = args[3];
        if (!isMatrixValue(XU) || XL.dims().dim(0) != XU.dims().dim(0)
            || XL.dims().dim(1) != XU.dims().dim(1))
            throw Error("mvncdf: XL and XU must be matrices of the same "
                        "size.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:invalidDataMatrices");
        const std::size_t cnt = static_cast<std::size_t>(XU.numel());
        for (std::size_t k = 0; k < cnt; ++k)
            if (XL.elemAsDouble(k) > XU.elemAsDouble(k))
                throw Error("mvncdf: the lower limits XL must be less "
                            "than or equal to the upper limits XU.",
                            0, 0, "mvncdf", "",
                            "numkit:mvncdf:nonincreasingLimits");
    }

    std::size_t n = static_cast<std::size_t>(XU.dims().dim(0));
    std::size_t d = static_cast<std::size_t>(XU.dims().dim(1));
    if (d < 1)
        throw Error("mvncdf: X must have at least one column.",
                    0, 0, "mvncdf", "", "numkit:mvncdf:tooFewDimensions");
    bool single = (XU.type() == ValueType::SINGLE);

    // Center by mu: empty | scalar | row/full vector (column mu with a
    // matching matrix Sigma reinterprets X as rows — handled below with
    // the Sigma shape, mirroring the reference's order).
    auto subtractMu = [&](Value &V) {
        if (mu.isEmpty() || mu.isScalar()) {
            const double m = mu.isEmpty() ? 0.0 : mu.elemAsDouble(0);
            if (m != 0.0) {
                Value out = Value::matrix(V.dims().dim(0), V.dims().dim(1),
                                          ValueType::DOUBLE, mr);
                const std::size_t cnt = static_cast<std::size_t>(V.numel());
                for (std::size_t k = 0; k < cnt; ++k)
                    out.doubleDataMut()[k] = V.elemAsDouble(k) - m;
                V = out;
            }
            return;
        }
        if (!isMatrixValue(mu))
            throw Error("mvncdf: MU must be a scalar or a matrix.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:invalidMu");
        const std::size_t n2 = static_cast<std::size_t>(mu.dims().dim(0));
        const std::size_t d2 = static_cast<std::size_t>(mu.dims().dim(1));
        if (d2 != d || !(n2 == 1 || n2 == n))
            throw Error("mvncdf: MU must be a row vector with the same "
                        "number of columns as X, or one row per X row.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:inputSizeMismatch");
        Value out = Value::matrix(V.dims().dim(0), V.dims().dim(1),
                                  ValueType::DOUBLE, mr);
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t r = 0; r < n; ++r)
                out.doubleDataMut()[r + j * n] =
                    V.elemAsDouble(r + j * n)
                    - mu.elemAsDouble((n2 == 1 ? 0 : r) + j * n2);
        V = out;
    };
    single = single || (mu.type() == ValueType::SINGLE);

    // Sigma: empty | 1×k diag row | d×d PSD matrix.
    bool sigmaIsDiag = false;
    std::size_t sz1 = 0;
    if (Sigma.isEmpty()) {
        // No Sigma: a column X is one d = n observation row (the
        // reference's provisional-interpretation flip).
        if (d == 1 && XU.numel() > 1) {
            auto transpose = [&](Value &V) {
                Value t = Value::matrix(1, n, ValueType::DOUBLE, mr);
                for (std::size_t k = 0; k < n; ++k)
                    t.doubleDataMut()[k] = V.elemAsDouble(k);
                V = t;
            };
            transpose(XU);
            if (box) transpose(XL);
            d = n;
            n = 1;
        }
    } else {
        sz1 = static_cast<std::size_t>(Sigma.dims().dim(0));
        const std::size_t sz2 = static_cast<std::size_t>(Sigma.dims().dim(1));
        if (Sigma.dims().ndim() <= 2 && sz1 == 1 && sz2 > 1) {
            sigmaIsDiag = true;
            sz1 = sz2;
        }
        // Column X + matrix Sigma whose rows match → reinterpret as row.
        if (!sigmaIsDiag && d == 1 && sz1 == n) {
            auto transpose = [&](Value &V) {
                Value t = Value::matrix(1, n, ValueType::DOUBLE, mr);
                for (std::size_t k = 0; k < n; ++k)
                    t.doubleDataMut()[k] = V.elemAsDouble(k);
                V = t;
            };
            transpose(XU);
            if (box) transpose(XL);
            d = n;
            n = 1;
        } else if (d == 1 && !mu.isScalar() && !mu.isEmpty()) {
            throw Error("mvncdf: MU must be a row vector with the same "
                        "number of columns as X, or one row per X row.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:inputSizeMismatch");
        }
        single = single || (Sigma.type() == ValueType::SINGLE);
        if (sz1 != static_cast<std::size_t>(Sigma.dims().dim(1)))
            throw Error("mvncdf: SIGMA must be a square covariance matrix.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:badCovariance");
        if (sz1 != d)
            throw Error("mvncdf: SIGMA must be " + std::to_string(d)
                        + "-by-" + std::to_string(d) + " to match X.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:covSizeMismatch");
    }

    // Standard deviations (diag path) or correlation + factor (matrix).
    std::vector<double> sd(d, 1.0);
    std::vector<double> L;   // lower Cholesky row-major of correlation
    if (Sigma.isEmpty() || sigmaIsDiag) {
        if (!Sigma.isEmpty()) {
            for (std::size_t j = 0; j < d; ++j) {
                const double v = Sigma.elemAsDouble(j);
                if (!(v > 0.0))
                    throw Error("mvncdf: all elements of the diagonal "
                                "SIGMA must be positive.",
                                0, 0, "mvncdf", "",
                                "numkit:mvncdf:badDiagSigma");
                sd[j] = std::sqrt(v);
            }
        }
        L.assign(d * d, 0.0);
        for (std::size_t j = 0; j < d; ++j) L[j * d + j] = 1.0;
    } else {
        std::vector<double> Sig(d * d), T, eval, evec;
        for (std::size_t k = 0; k < d * d; ++k)
            Sig[k] = Sigma.elemAsDouble(k);
        if (!isSymmetric(Sig, d)
            || !psdFactor(Sig, d, T, eval, evec, mr))
            throw Error("mvncdf: SIGMA must be a square, symmetric, "
                        "positive semi-definite matrix.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:badMatrixSigma");
        // Standardize to a correlation and Cholesky it.
        std::vector<double> rho(d * d, 0.0);
        for (std::size_t j = 0; j < d; ++j) sd[j] = std::sqrt(Sig[j + j * d]);
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t i = 0; i < d; ++i)
                rho[i + j * d] = Sig[i + j * d] / (sd[i] * sd[j]);
        L.assign(d * d, 0.0);
        for (std::size_t j = 0; j < d; ++j) {
            double s = 1.0;
            for (std::size_t k = 0; k < j; ++k)
                s -= L[j * d + k] * L[j * d + k];
            if (s <= 0.0)
                throw Error("mvncdf: SIGMA must be a square, symmetric, "
                            "positive semi-definite matrix.",
                            0, 0, "mvncdf", "",
                            "numkit:mvncdf:badMatrixSigma");
            L[j * d + j] = std::sqrt(s);
            for (std::size_t i = j + 1; i < d; ++i) {
                double t = rho[i + j * d];
                for (std::size_t k = 0; k < j; ++k)
                    t -= L[i * d + k] * L[j * d + k];
                L[i * d + j] = t / L[j * d + j];
            }
        }
    }

    // Standardize the limits.
    subtractMu(XU);
    if (box) subtractMu(XL);
    auto scaleRow = [&](Value &V, std::size_t r, std::size_t j) {
        return V.elemAsDouble(r + j * n) / sd[j];
    };

    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();

    if (sigmaIsDiag || Sigma.isEmpty()) {
        // Independent coordinates: products of univariate probabilities.
        for (std::size_t r = 0; r < n; ++r) {
            double p = 1.0;
            for (std::size_t j = 0; j < d; ++j) {
                const double u = scaleRow(XU, r, j);
                if (box) {
                    const double l = scaleRow(XL, r, j);
                    p *= std::max(0.0, phiCdf(u) - phiCdf(l));
                } else {
                    p *= phiCdf(u);
                }
            }
            od[r] = p;
        }
    } else if (d == 1) {
        const double rho = 1.0;   // single coordinate: plain univariate
        (void)rho;
        for (std::size_t r = 0; r < n; ++r) {
            const double u = scaleRow(XU, r, 0);
            od[r] = box
                ? std::max(0.0, phiCdf(u) - phiCdf(scaleRow(XL, r, 0)))
                : phiCdf(u);
        }
    } else if (d == 2) {
        const double rhoV = L[1 * 2 + 0];   // correlation
        for (std::size_t r = 0; r < n; ++r) {
            const double hu = scaleRow(XU, r, 0);
            const double ku = scaleRow(XU, r, 1);
            if (!box) {
                od[r] = bivariateCdf(hu, ku, rhoV);
            } else {
                const double hl = scaleRow(XL, r, 0);
                const double kl = scaleRow(XL, r, 1);
                od[r] = std::max(0.0,
                    bivariateCdf(hu, ku, rhoV) - bivariateCdf(hl, ku, rhoV)
                    - bivariateCdf(hu, kl, rhoV) + bivariateCdf(hl, kl, rhoV));
            }
        }
    } else {
        // d ≥ 3: Monte Carlo over the standardized limits; the box form
        // uses corner inclusion-exclusion over the 2^d orthant corners.
        std::mt19937 gen(12345);
        std::vector<double> q(d);
        const std::vector<double> zeros(d, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            if (!box) {
                for (std::size_t j = 0; j < d; ++j)
                    q[j] = scaleRow(XU, r, j);
                od[r] = monteCarloCdf(L.data(), d, zeros.data(), q.data(),
                                      20000, gen);
            } else {
                // P(l < X < u) via corner sums (d ≤ ~10 practical).
                double acc = 0.0;
                const std::size_t corners = 1u << d;
                for (std::size_t mask = 0; mask < corners; ++mask) {
                    // bit set → upper corner for that coordinate.
                    std::size_t bits = 0;
                    for (std::size_t j = 0; j < d; ++j)
                        if (mask & (1u << j)) ++bits;
                    const double sign = ((d - bits) % 2 == 0) ? 1.0 : -1.0;
                    for (std::size_t j = 0; j < d; ++j)
                        q[j] = (mask & (1u << j)) ? scaleRow(XU, r, j)
                                                  : scaleRow(XL, r, j);
                    acc += sign * monteCarloCdf(L.data(), d, zeros.data(),
                                                q.data(), 20000, gen);
                }
                od[r] = std::min(1.0, std::max(0.0, acc));
            }
        }
    }
    return castDown(out, single, mr);
}

} // namespace numkit::stats
