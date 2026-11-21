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

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/math.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/distributions/multivariate.hpp>

#include <numkit/builtin/datafun.hpp>          // RngContext
#include <numkit/linalg/eig.hpp>               // eig_symmetric
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "multivariate_detail.hpp"         // phiCdf / bivariateCdf / monteCarloCdf

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::stats {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

bool isMatrixValue(const Value &V)
{
    return V.dims().ndim() <= 2 && V.numel() > 0;
}

bool isSymmetric(const keel::Array<double> &M, keel::usize d)
{
    for (keel::usize j = 0; j < d; ++j)
        for (keel::usize i = j + 1; i < d; ++i)
            if (M[i + j * d] != M[j + i * d]) return false;
    return true;
}

// Factor Sigma = L·L' with L d×r (r = rank of Sigma):
// Strictly PD → r = d and L is lower Cholesky.
// Rank-deficient PSD → r < d and L has r columns V(:,j)*sqrt(λ_j) for λ_j > tol.
// Returns false when Sigma has any eigenvalue < -tol.
bool psdFactor(const keel::Array<double> &Sig, keel::usize d,
               keel::Array<double> &T, keel::Array<double> &eval,
               keel::Array<double> &evec, keel::Allocator *mr,
               keel::usize *outRank = nullptr)
{
    Value Mv = Value::matrix(d, d, ValueType::DOUBLE, mr);
    std::copy(Sig.begin(), Sig.end(), Mv.doubleDataMut());
    auto [V, D] = ::numkit::linalg::eig_symmetric(Mv, mr);
    evec.assign(keel::ArrayRef<double>(V.doubleData(), d * d));
    eval.assign(d, 0.0);
    double maxAbs = 0.0;
    for (keel::usize i = 0; i < d; ++i) {
        eval[i] = D.doubleData()[i + i * d];
        maxAbs = std::max(maxAbs, keel::abs(eval[i]));
    }
    const double tol = keel::as<double>(d)
                     * std::numeric_limits<double>::epsilon()
                     * std::max(maxAbs, 1.0);
    for (double e : eval)
        if (e < -tol) return false;
    // Strictly positive definite → lower Cholesky L (so T = L' is upper Cholesky,
    // matching cholcov output, probe R09). Semidefinite → compact d×r factor.
    bool pd = true;
    for (double e : eval)
        if (e <= tol) pd = false;
    if (pd) {
        T.assign(d * d, 0.0);
        for (keel::usize j = 0; j < d; ++j) {
            for (keel::usize i = j; i < d; ++i) {
                double v = Sig[i + j * d];   // column-major source
                for (keel::usize k = 0; k < j; ++k)
                    v -= T[i + k * d] * T[j + k * d];
                T[i + j * d] = (i == j) ? std::sqrt(v)
                                        : v / T[j + j * d];
                if (i == j && !(v > 0.0)) { pd = false; break; }
            }
            if (!pd) break;
        }
        if (pd) {
            if (outRank) *outRank = d;
            return true;
        }
    }
    // Count positive eigenvalues and build d×r factor (descending eigenvalue order,
    // with cholcov sign convention: largest-magnitude element in each column positive).
    keel::Array<keel::usize> posCols;
    for (keel::usize j = 0; j < d; ++j) {
        if (eval[j] > tol) posCols.push_back(j);
    }
    const keel::usize r = posCols.size();
    if (outRank) *outRank = r;
    T.assign(d * r, 0.0);
    for (keel::usize col = 0; col < r; ++col) {
        const keel::usize j = posCols[col];
        const double s = std::sqrt(std::max(eval[j], 0.0));
        keel::usize maxIdx = 0;
        double maxVal = 0.0;
        for (keel::usize i = 0; i < d; ++i) {
            const double a = keel::abs(evec[i + j * d]);
            if (a > maxVal) { maxVal = a; maxIdx = i; }
        }
        const double sign = (evec[maxIdx + j * d] < 0.0) ? -1.0 : 1.0;
        for (keel::usize i = 0; i < d; ++i)
            T[i + col * d] = sign * evec[i + j * d] * s;
    }
    return true;
}

Value castDown(const Value &X, bool single, keel::Allocator *mr)
{
    if (!single || X.type() == ValueType::SINGLE) return X;
    const keel::usize r = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize c = keel::as<keel::usize>(X.dims().dim(1));
    Value out = Value::matrix(r, c, ValueType::SINGLE, mr);
    const keel::usize cnt = r * c;
    for (keel::usize k = 0; k < cnt; ++k)
        out.singleDataMut()[k] = keel::as<float>(X.elemAsDouble(k));
    return out;
}

// mu as a flat row (length d) or per-case rows; X keeps the cases×d
// resolved data. Returns false for a shape error (caller throws).
double muAt(const Value &mu, keel::usize row, keel::usize col,
            keel::usize muN)
{
    return muN == 1 ? mu.elemAsDouble(col)
                    : mu.elemAsDouble(col * muN + row);
}

} // namespace

// ── mvnrnd ────────────────────────────────────────────────────────────

MvnrndOut mvnrnd_reinterpreted(::numkit::ops::RngContext &rng, const Value &mu,
                               const Value &Sigma, const Value &cases,
                               bool wantT, keel::Allocator *mr);

namespace {
void mvnrnd_full_argcheck(const Value &mu, const Value &Sigma)
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
}
} // namespace

MvnrndOut mvnrnd_full(::numkit::ops::RngContext &rng, const Value &mu,
                      const Value &Sigma, const Value &cases, bool wantT,
                      keel::Allocator *mr)
{
    mvnrnd_full_argcheck(mu, Sigma);

    keel::usize n = keel::as<keel::usize>(mu.dims().dim(0));
    keel::usize d = keel::as<keel::usize>(mu.dims().dim(1));
    keel::usize sz1 = keel::as<keel::usize>(Sigma.dims().dim(0));
    const keel::usize sz2 = keel::as<keel::usize>(Sigma.dims().dim(1));
    bool sigmaIsDiag = false;
    if (Sigma.dims().ndim() <= 2 && sz1 == 1 && sz2 > 1) {
        sigmaIsDiag = true;
        sz1 = sz2;
    }
    // Column mu + matrix Sigma whose rows match → reinterpret as a row.
    if (d == 1 && sz1 == n && !sigmaIsDiag) {
        Value mut = Value::matrix(1, n, ValueType::DOUBLE, mr);
        for (keel::usize k = 0; k < n; ++k)
            mut.doubleDataMut()[k] = mu.elemAsDouble(k);
        return mvnrnd_reinterpreted(rng, mut, Sigma, cases, wantT, mr);
    }
    return mvnrnd_reinterpreted(rng, mu, Sigma, cases, wantT, mr);
}

// Body with the mu orientation already resolved (always 1×d or n×d).
MvnrndOut mvnrnd_reinterpreted(::numkit::ops::RngContext &rng, const Value &mu,
                               const Value &Sigma, const Value &cases,
                               bool wantT, keel::Allocator *mr)
{
    keel::usize n = keel::as<keel::usize>(mu.dims().dim(0));
    keel::usize d = keel::as<keel::usize>(mu.dims().dim(1));
    keel::usize sz1 = keel::as<keel::usize>(Sigma.dims().dim(0));
    keel::usize sz2 = keel::as<keel::usize>(Sigma.dims().dim(1));
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
        casesN = keel::as<long long>(cases.elemAsDouble(0));
        if (keel::as<long long>(n) == casesN) {
            // mu already has cases rows.
        } else if (n == 1) {
            n = keel::as<keel::usize>(std::max(casesN, 0LL));
        } else {
            throw Error(keel::format("mvnrnd: MU must be a row vector, or must have "
                        "CASES rows (CASES is {}, MU has {} rows).", casesN, n),
                        0, 0, "mvnrnd", "", "numkit:mvnrnd:inputSizeMismatchMu");
        }
    }
    if (nocases && Sigma.dims().ndim() >= 3 && n == 1) {
        // A single mu row with a Sigma stack: one case per page.
        n = keel::as<keel::usize>(Sigma.dims().dim(2));
    }
    const keel::usize muN = (keel::as<keel::usize>(
        mu.dims().dim(0)) == 1) ? 1 : n;

    MvnrndOut out;
    if (n == 0) {
        out.r = Value::matrix(0, d, ValueType::DOUBLE, mr);
        return out;
    }

    auto &gen = rng;
    auto Nz = [](::numkit::ops::RngContext &g) { return g.nextNormal(); };
    out.r = Value::matrix(n, d, ValueType::DOUBLE, mr);
    double *od = out.r.doubleDataMut();

    const keel::usize pages = Sigma.dims().ndim() >= 3
                  ? keel::as<keel::usize>(Sigma.dims().dim(2)) : 1;
    if (pages == 1) {
        if (sz1 != sz2)
            throw Error(keel::format("mvnrnd: SIGMA must be a square covariance matrix "
                        "(got {}-by-{}).", sz1, sz2),
                        0, 0, "mvnrnd", "", "numkit:mvnrnd:badCovariance2DSize");
        if (sz1 != d)
            throw Error(keel::format("mvnrnd: SIGMA must be {}-by-{} to match MU.", d, d),
                        0, 0, "mvnrnd", "",
                        "numkit:mvnrnd:inputSizeMismatch2DSigma");
        if (sigmaIsDiag) {
            keel::Array<double> sd(d);
            for (keel::usize j = 0; j < d; ++j) {
                const double v = Sigma.elemAsDouble(j);
                if (v < 0.0)
                    throw Error(keel::format("mvnrnd: all elements of the diagonal "
                                "SIGMA must be non-negative (got {}).", v),
                                0, 0, "mvnrnd", "",
                                "numkit:mvnrnd:badDiagSigma");
                sd[j] = std::sqrt(v);
            }
            if (wantT) {
                out.T = Value::matrix(d, d, ValueType::DOUBLE, mr);
                double *td = out.T.doubleDataMut();
                for (keel::usize j = 0; j < d; ++j) td[j + j * d] = sd[j];
            }
            keel::Array<double> Z(n * d);
            for (keel::usize j = 0; j < d; ++j)
                for (keel::usize r = 0; r < n; ++r)
                    Z[r + j * n] = Nz(gen);
            for (keel::usize r = 0; r < n; ++r)
                for (keel::usize j = 0; j < d; ++j)
                    od[j * n + r] = sd[j] > 0.0
                        ? Z[r + j * n] * sd[j] + muAt(mu, r, j, muN)
                        : muAt(mu, r, j, muN);   // exact mean at zero var
        } else {
            keel::Array<double> Sig(d * d), T, eval, evec;
            for (keel::usize k = 0; k < d * d; ++k)
                Sig[k] = Sigma.elemAsDouble(k);
            if (!isSymmetric(Sig, d))
                throw Error("mvnrnd: SIGMA must be symmetric.",
                            0, 0, "mvnrnd", "",
                            "numkit:mvnrnd:badCovariance2DSym");
            keel::usize rank = d;
            if (!psdFactor(Sig, d, T, eval, evec, mr, &rank))
                throw Error("mvnrnd: SIGMA must be positive "
                            "semi-definite.",
                            0, 0, "mvnrnd", "",
                            "numkit:mvnrnd:badCovariance2DPos");
            if (wantT) {
                // Upper factor (Sigma = T'·T) of shape rank×d, matching
                // cholcov output — the transpose of the d×rank sampling L.
                out.T = Value::matrix(rank, d, ValueType::DOUBLE, mr);
                double *td = out.T.doubleDataMut();
                for (keel::usize i = 0; i < rank; ++i)
                    for (keel::usize j = 0; j < d; ++j)
                        td[i + j * rank] = T[j + i * d];
            }
            keel::Array<double> Z(n * rank);
            for (keel::usize k = 0; k < rank; ++k)
                for (keel::usize r = 0; r < n; ++r)
                    Z[r + k * n] = Nz(gen);
            for (keel::usize r = 0; r < n; ++r)
                for (keel::usize j = 0; j < d; ++j) {
                    double acc = 0.0;
                    for (keel::usize k = 0; k < rank; ++k)
                        acc += T[j + k * d] * Z[r + k * n];
                    od[j * n + r] = acc + muAt(mu, r, j, muN);
                }
            // Force the exact mean in zero-variance directions.
            for (keel::usize j = 0; j < d; ++j)
                if (Sigma.elemAsDouble(j + j * d) == 0.0)
                    for (keel::usize r = 0; r < n; ++r)
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
            throw Error(keel::format("mvnrnd: each SIGMA page must be {}-by-{} to match MU.", d, d),
                        0, 0, "mvnrnd", "",
                        "numkit:mvnrnd:inputSizeMismatch3DSigma");
        if (pages != n)
            throw Error(keel::format("mvnrnd: SIGMA must have CASES pages (CASES is {}, SIGMA has {} pages).", n, pages),
                        0, 0, "mvnrnd", "",
                        "numkit:mvnrnd:inputSizeMismatchSigmaDimension");
        if (wantT)
            out.T = Value::matrix3d(d, d, pages, ValueType::DOUBLE, mr);
        keel::Array<double> Sig(d * d), T, eval, evec, z(d);
        for (keel::usize p = 0; p < pages; ++p) {
            if (sigmaIsDiag) {
                // 1×d×n diagonal stack: per-page standard deviations.
                for (keel::usize j = 0; j < d; ++j) {
                    const double v = Sigma.elemAsDouble(j + p * d);
                    if (v < 0.0)
                        throw Error("mvnrnd: all elements of the diagonal "
                                    "SIGMA must be non-negative.",
                                    0, 0, "mvnrnd", "",
                                    "numkit:mvnrnd:badDiagSigma");
                }
                for (keel::usize j = 0; j < d; ++j) {
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
            for (keel::usize k = 0; k < d * d; ++k)
                Sig[k] = Sigma.elemAsDouble(k + p * d * d);
            if (!isSymmetric(Sig, d))
                throw Error("mvnrnd: SIGMA must be symmetric.",
                            0, 0, "mvnrnd", "",
                            "numkit:mvnrnd:badCovariance3DSym");
            keel::usize rank = d;
            if (!psdFactor(Sig, d, T, eval, evec, mr, &rank))
                throw Error("mvnrnd: SIGMA must be positive "
                            "semi-definite.",
                            0, 0, "mvnrnd", "",
                            "numkit:mvnrnd:badCovariance3DPos");
            if (wantT) {
                double *td = out.T.doubleDataMut() + p * d * d;
                for (keel::usize i = 0; i < rank; ++i)
                    for (keel::usize j = 0; j < d; ++j)
                        td[i + j * d] = T[j + i * d];
            }
            for (keel::usize k = 0; k < rank; ++k) z[k] = Nz(gen);
            for (keel::usize j = 0; j < d; ++j) {
                double acc = 0.0;
                for (keel::usize k = 0; k < rank; ++k)
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
                  keel::Allocator *mr)
{
    keel::usize n = 1;
    if (!cases.isEmpty())
        n = keel::as<keel::usize>(std::max(
            keel::as<long long>(cases.elemAsDouble(0)), 0LL));

    const keel::usize m1 = keel::as<keel::usize>(C.dims().dim(0));
    const keel::usize m2 = keel::as<keel::usize>(C.dims().dim(1));
    if (m1 != m2)
        throw Error(keel::format("mvtrnd: the correlation matrix C must be square (got {}-by-{}).", m1, m2),
                    0, 0, "mvtrnd", "", "numkit:mvtrnd:badCorrelationNotSquare");

    // df: scalar, or one value per case.
    keel::Array<double> dfv(n);
    if (df.isScalar()) {
        dfv.assign(n, df.elemAsDouble(0));
    } else if (df.dims().isVector()
               && df.numel() == n) {
        for (keel::usize r = 0; r < n; ++r) dfv[r] = df.elemAsDouble(r);
    } else {
        throw Error("mvtrnd: DF must be a scalar or a vector with one "
                    "element for each row of the output matrix.",
                    0, 0, "mvtrnd", "", "numkit:mvtrnd:inputSizeMismatch");
    }
    for (double v : dfv)
        if (!(v > 0.0))
            throw Error(keel::format("mvtrnd: DF must be positive (got {}).", v),
                        0, 0, "mvtrnd", "", "numkit:mvtrnd:badDF");

    // Standardize C to a correlation matrix (silent, probe T06).
    const keel::usize d = m1;
    keel::Array<double> Cm(d * d, 0.0);
    for (keel::usize k = 0; k < d * d; ++k) Cm[k] = C.elemAsDouble(k);
    keel::Array<double> sq(d);
    bool unitDiag = true;
    for (keel::usize i = 0; i < d; ++i) {
        sq[i] = std::sqrt(Cm[i + i * d]);
        if (Cm[i + i * d] != 1.0) unitDiag = false;
    }
    if (!unitDiag)
        for (keel::usize j = 0; j < d; ++j)
            for (keel::usize i = 0; i < d; ++i)
                Cm[i + j * d] /= sq[i] * sq[j];
    if (!isSymmetric(Cm, d))
        throw Error("mvtrnd: C must be a symmetric positive "
                    "semi-definite matrix.",
                    0, 0, "mvtrnd", "", "numkit:mvtrnd:badCorrelationSymPos");
    keel::Array<double> T, eval, evec;
    keel::usize rank = d;
    if (!psdFactor(Cm, d, T, eval, evec, mr, &rank))
        throw Error("mvtrnd: C must be a symmetric positive "
                    "semi-definite matrix.",
                    0, 0, "mvtrnd", "", "numkit:mvtrnd:badCorrelationSymPos");

    // r = (randn·T) ./ sqrt(χ²(df)/df), one χ² per case via deterministic
    // Marsaglia-Tsang Gamma(df/2, 2) on RngContext (platform-independent).
    auto sampleChi2 = [](::numkit::ops::RngContext &g, double df_r) -> double {
        const double a = 0.5 * df_r;
        const double d_g = (a < 1.0 ? (a + 1.0) : a) - 1.0 / 3.0;
        const double c_g = 1.0 / std::sqrt(9.0 * d_g);
        double u = g.nextUniform();
        const double u_boost = (a < 1.0) ? g.nextUniform() : 1.0;
        double v = 0.0;
        for (;;) {
            const double z = g.nextNormal();
            v = 1.0 + c_g * z;
            if (v > 0.0) {
                v = v * v * v;
                const double z2 = z * z;
                if (u < 1.0 - 0.0331 * z2 * z2 ||
                    std::log(u) < 0.5 * z2 + d_g * (1.0 - v + std::log(v))) {
                    break;
                }
            }
            u = g.nextUniform();
        }
        const double gam = (a < 1.0) ? (d_g * v * std::pow(u_boost, 1.0 / a)) : (d_g * v);
        return 2.0 * gam;
    };

    Value out = Value::matrix(n, d, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    auto &gen = rng;
    keel::Array<double> Z(n * rank);
    for (keel::usize k = 0; k < rank; ++k)
        for (keel::usize r = 0; r < n; ++r)
            Z[r + k * n] = gen.nextNormal();
    keel::Array<double> scale(n);
    for (keel::usize r = 0; r < n; ++r) {
        scale[r] = std::sqrt(dfv[r] / sampleChi2(gen, dfv[r]));
    }
    for (keel::usize r = 0; r < n; ++r)
        for (keel::usize j = 0; j < d; ++j) {
            double acc = 0.0;
            for (keel::usize k = 0; k < rank; ++k)
                acc += T[j + k * d] * Z[r + k * n];
            od[j * n + r] = acc * scale[r];
        }
    return castDown(out, C.type() == ValueType::SINGLE
                          || df.type() == ValueType::SINGLE, mr);
}

// ── mvncdf ────────────────────────────────────────────────────────────

Value mvncdf_full(keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    // A trailing options struct (e.g. statset('TolFun', 1e-5, 'MaxFunEvals', 20000))
    // is parsed for TolFun / MaxFunEvals and stripped from positional args.
    keel::usize nin = args.size();
    keel::usize maxFunEvals = 10000;
    double tolFun = 1e-5;
    if (nin > 0 && (args[nin - 1].isObject() || args[nin - 1].isStruct())) {
        const Value &opts = args[nin - 1];
        if (opts.isStruct()) {
            const auto &fields = opts.structFields();
            auto getFieldCI = [&](const char *key) -> const Value * {
                for (auto entry : fields) {
                    keel::String k = keel::String::from(entry.key.as_ref());
                    for (char &ch : k) ch = keel::to_ascii_lower(ch);
                    if (k == key && !entry.value.isEmpty() && entry.value.isNumeric())
                        return fields.get(entry.key);
                }
                return nullptr;
            };
            if (const Value *tf = getFieldCI("tolfun")) {
                const double v = tf->elemAsDouble(0);
                if (v > 0.0 && std::isfinite(v)) tolFun = v;
            }
            if (const Value *mf = getFieldCI("maxfunevals")) {
                const double v = mf->elemAsDouble(0);
                if (v >= 100.0 && std::isfinite(v)) maxFunEvals = keel::as<keel::usize>(v);
            }
        }
        --nin;
    }
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
        const keel::usize cnt = keel::as<keel::usize>(XU.numel());
        for (keel::usize k = 0; k < cnt; ++k)
            if (XL.elemAsDouble(k) > XU.elemAsDouble(k))
                throw Error("mvncdf: the lower limits XL must be less "
                            "than or equal to the upper limits XU.",
                            0, 0, "mvncdf", "",
                            "numkit:mvncdf:nonincreasingLimits");
    }

    keel::usize n = keel::as<keel::usize>(XU.dims().dim(0));
    keel::usize d = keel::as<keel::usize>(XU.dims().dim(1));
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
                const keel::usize cnt = keel::as<keel::usize>(V.numel());
                for (keel::usize k = 0; k < cnt; ++k)
                    out.doubleDataMut()[k] = V.elemAsDouble(k) - m;
                V = out;
            }
            return;
        }
        if (!isMatrixValue(mu))
            throw Error("mvncdf: MU must be a scalar or a matrix.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:invalidMu");
        const keel::usize n2 = keel::as<keel::usize>(mu.dims().dim(0));
        const keel::usize d2 = keel::as<keel::usize>(mu.dims().dim(1));
        if (d2 != d || !(n2 == 1 || n2 == n))
            throw Error("mvncdf: MU must be a row vector with the same "
                        "number of columns as X, or one row per X row.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:inputSizeMismatch");
        Value out = Value::matrix(V.dims().dim(0), V.dims().dim(1),
                                  ValueType::DOUBLE, mr);
        for (keel::usize j = 0; j < d; ++j)
            for (keel::usize r = 0; r < n; ++r)
                out.doubleDataMut()[r + j * n] =
                    V.elemAsDouble(r + j * n)
                    - mu.elemAsDouble((n2 == 1 ? 0 : r) + j * n2);
        V = out;
    };
    single = single || (mu.type() == ValueType::SINGLE);

    // Sigma: empty | 1×k diag row | d×d PSD matrix.
    bool sigmaIsDiag = false;
    keel::usize sz1 = 0;
    if (Sigma.isEmpty()) {
        // No Sigma: a column X is one d = n observation row (the
        // reference's provisional-interpretation flip).
        if (d == 1 && XU.numel() > 1) {
            auto transpose = [&](Value &V) {
                Value t = Value::matrix(1, n, ValueType::DOUBLE, mr);
                for (keel::usize k = 0; k < n; ++k)
                    t.doubleDataMut()[k] = V.elemAsDouble(k);
                V = t;
            };
            transpose(XU);
            if (box) transpose(XL);
            d = n;
            n = 1;
        }
    } else {
        sz1 = keel::as<keel::usize>(Sigma.dims().dim(0));
        const keel::usize sz2 = keel::as<keel::usize>(Sigma.dims().dim(1));
        if (Sigma.dims().ndim() <= 2 && sz1 == 1 && sz2 > 1) {
            sigmaIsDiag = true;
            sz1 = sz2;
        }
        // Column X + matrix Sigma whose rows match → reinterpret as row.
        if (!sigmaIsDiag && d == 1 && sz1 == n) {
            auto transpose = [&](Value &V) {
                Value t = Value::matrix(1, n, ValueType::DOUBLE, mr);
                for (keel::usize k = 0; k < n; ++k)
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
        if (sz1 != keel::as<keel::usize>(Sigma.dims().dim(1)))
            throw Error("mvncdf: SIGMA must be a square covariance matrix.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:badCovariance");
        if (sz1 != d)
            throw Error(keel::format("mvncdf: SIGMA must be {}-by-{} to match X.", d, d),
                        0, 0, "mvncdf", "", "numkit:mvncdf:covSizeMismatch");
    }

    // Standard deviations (diag path) or correlation + factor (matrix).
    keel::Array<double> sd(d, 1.0);
    keel::Array<double> L;   // lower Cholesky row-major of correlation
    if (Sigma.isEmpty() || sigmaIsDiag) {
        if (!Sigma.isEmpty()) {
            for (keel::usize j = 0; j < d; ++j) {
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
        for (keel::usize j = 0; j < d; ++j) L[j * d + j] = 1.0;
    } else {
        keel::Array<double> Sig(d * d), T, eval, evec;
        for (keel::usize k = 0; k < d * d; ++k)
            Sig[k] = Sigma.elemAsDouble(k);
        if (!isSymmetric(Sig, d)
            || !psdFactor(Sig, d, T, eval, evec, mr))
            throw Error("mvncdf: SIGMA must be a square, symmetric, "
                        "positive semi-definite matrix.",
                        0, 0, "mvncdf", "", "numkit:mvncdf:badMatrixSigma");
        // Standardize to a correlation and Cholesky it.
        keel::Array<double> rho(d * d, 0.0);
        for (keel::usize j = 0; j < d; ++j) sd[j] = std::sqrt(Sig[j + j * d]);
        for (keel::usize j = 0; j < d; ++j)
            for (keel::usize i = 0; i < d; ++i)
                rho[i + j * d] = Sig[i + j * d] / (sd[i] * sd[j]);
        L.assign(d * d, 0.0);
        for (keel::usize j = 0; j < d; ++j) {
            double s = 1.0;
            for (keel::usize k = 0; k < j; ++k)
                s -= L[j * d + k] * L[j * d + k];
            if (s <= 0.0)
                throw Error("mvncdf: SIGMA must be a square, symmetric, "
                            "positive semi-definite matrix.",
                            0, 0, "mvncdf", "",
                            "numkit:mvncdf:badMatrixSigma");
            L[j * d + j] = std::sqrt(s);
            for (keel::usize i = j + 1; i < d; ++i) {
                double t = rho[i + j * d];
                for (keel::usize k = 0; k < j; ++k)
                    t -= L[i * d + k] * L[j * d + k];
                L[i * d + j] = t / L[j * d + j];
            }
        }
    }

    // Standardize the limits.
    subtractMu(XU);
    if (box) subtractMu(XL);
    auto scaleRow = [&](Value &V, keel::usize r, keel::usize j) {
        return V.elemAsDouble(r + j * n) / sd[j];
    };

    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();

    if (sigmaIsDiag || Sigma.isEmpty()) {
        // Independent coordinates: products of univariate probabilities.
        for (keel::usize r = 0; r < n; ++r) {
            double p = 1.0;
            for (keel::usize j = 0; j < d; ++j) {
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
        for (keel::usize r = 0; r < n; ++r) {
            const double u = scaleRow(XU, r, 0);
            od[r] = box
                ? std::max(0.0, phiCdf(u) - phiCdf(scaleRow(XL, r, 0)))
                : phiCdf(u);
        }
    } else if (d == 2) {
        const double rhoV = L[1 * 2 + 0];   // correlation
        for (keel::usize r = 0; r < n; ++r) {
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
        // d ≥ 3: Genz (1992) Separation-of-Variables (SOV) quasi-Monte Carlo
        // over standardized lower/upper limits (handles both one-sided and box
        // forms directly without 2^d corner expansion).
        keel::Array<double> ql(d, -kInf), qu(d);
        for (keel::usize r = 0; r < n; ++r) {
            for (keel::usize j = 0; j < d; ++j) {
                qu[j] = scaleRow(XU, r, j);
                ql[j] = box ? scaleRow(XL, r, j) : -kInf;
            }
            od[r] = genzSovCdf(L.data(), d, ql.data(), qu.data(),
                               maxFunEvals, tolFun);
        }
    }
    return castDown(out, single, mr);
}

} // namespace numkit::stats
