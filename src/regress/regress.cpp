// toolboxes/stats/src/regress/regress.cpp

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/regress/regress.hpp>

#include <numkit/stats/distributions/students_t.hpp>
#include <numkit/stats/distributions/fisher_f.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <cmath>
#include <limits>

#include "regress_detail.hpp"

namespace numkit::stats {

namespace {

void regress_full_argcheck(const Value &y, const Value &X)
{
    const keel::usize N = y.numel();
    const keel::usize p = X.dims().cols();
    if (X.dims().rows() != N || N == 0 || p == 0)
        throw Error("regress: X must be N×p with same N as y",
                    0, 0, "regress", "", "numkit:regress:size");
}

void lscov_argcheck(const Value &A, const Value &b, const Value &w)
{
    const keel::usize N = b.numel();
    const keel::usize p = A.dims().cols();
    if (A.dims().rows() != N || N == 0 || p == 0)
        throw Error("lscov: A must be N×p with same N as b",
                    0, 0, "lscov", "", "numkit:lscov:size");
    if (!w.isEmpty()) {
        const bool isVecN = (w.numel() == N);
        const bool isMatNN = (w.dims().rows() == N && w.dims().cols() == N);
        if (!isVecN && !isMatNN)
            throw Error("lscov: w must be a length-N vector or N×N covariance matrix",
                        0, 0, "lscov", "", "numkit:lscov:w");
    }
}

void ridge_argcheck(const Value &y, const Value &X, const Value &kVec)
{
    const keel::usize N = y.numel();
    const keel::usize p = X.dims().cols();
    if (X.dims().rows() != N || N == 0 || p == 0)
        throw Error("ridge: X must be N×p with same N as y",
                    0, 0, "ridge", "", "numkit:ridge:size");
    if (kVec.numel() == 0)
        throw Error("ridge: k must be non-empty",
                    0, 0, "ridge", "", "numkit:ridge:k");
}

} // namespace

// 5-output form: [b, bint, r, rint, stats]. rint = residual CI for
// outlier detection — added 2026-05-08.
std::tuple<Value, Value, Value, Value, Value>
regress_full(const Value &y, const Value &X, keel::f64 alpha, keel::Allocator *mr)
{
    regress_full_argcheck(y, X);
    const keel::usize N = y.numel();
    const keel::usize p = X.dims().cols();

    // Build XtX (p×p) and Xty (p×1) using column-major X.
    keel::Array<keel::f64> XtX(p * p, 0.0);
    keel::Array<keel::f64> Xty(p, 0.0);
    for (keel::usize j = 0; j < p; ++j) {
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 xij = X.elemAsDouble(i + j * N);
            Xty[j] += xij * y.elemAsDouble(i);
            for (keel::usize k = j; k < p; ++k) {
                const keel::f64 xik = X.elemAsDouble(i + k * N);
                XtX[k + j * p] += xij * xik;
            }
        }
    }
    // Mirror upper triangle.
    for (keel::usize j = 0; j < p; ++j)
        for (keel::usize k = 0; k < j; ++k)
            XtX[k + j * p] = XtX[j + k * p];

    // Cholesky factor.
    keel::Array<keel::f64> L(p * p, 0.0);
    if (!cholesky(XtX.data(), L.data(), p))
        throw Error("regress: design matrix is rank-deficient",
                    0, 0, "regress", "", "numkit:regress:rank");

    // Solve XtX β = Xty:  L z = Xty,  L^T β = z.
    keel::Array<keel::f64> beta(p), z(p);
    fwd_solve(L.data(), z.data(), Xty.data(), p);
    back_solve(L.data(), beta.data(), z.data(), p);

    // Residuals r = y - X β.
    keel::Array<keel::f64> r(N);
    keel::f64 SSR = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        keel::f64 pred = 0.0;
        for (keel::usize j = 0; j < p; ++j) pred += X.elemAsDouble(i + j * N) * beta[j];
        const keel::f64 ri = y.elemAsDouble(i) - pred;
        r[i] = ri;
        SSR += ri * ri;
    }
    const keel::f64 dfErr = keel::f64(N) - keel::f64(p);
    const keel::f64 sigma2 = (dfErr > 0) ? SSR / dfErr
                                      : std::numeric_limits<keel::f64>::quiet_NaN();

    // Compute the FULL (XtX)^{-1} (p×p, column-major) so we can compute
    // both the diagonal (for bint SE) and the leverage h_ii = X·M·X'
    // for rint.
    keel::Array<keel::f64> invMat(p * p, 0.0);
    keel::Array<keel::f64> invDiag(p, 0.0);
    {
        keel::Array<keel::f64> ej(p), zj(p), mj(p);
        for (keel::usize j = 0; j < p; ++j) {
            std::fill(ej.begin(), ej.end(), 0.0);
            ej[j] = 1.0;
            fwd_solve(L.data(), zj.data(), ej.data(), p);
            back_solve(L.data(), mj.data(), zj.data(), p);
            for (keel::usize i = 0; i < p; ++i) invMat[i + j * p] = mj[i];
            invDiag[j] = mj[j];
        }
    }

    // Confidence intervals for β.
    Value bV    = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value bintV = Value::matrix(p, 2, ValueType::DOUBLE, mr);
    Value rV    = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *bd = bV.doubleDataMut();
    keel::f64 *bid = bintV.doubleDataMut();
    keel::f64 *rd = rV.doubleDataMut();
    keel::f64 tcrit = 0.0;
    if (dfErr > 0) {
        Value pV = Value::scalar(1.0 - alpha / 2.0, mr);
        tcrit = tinv(pV, dfErr, mr).toScalar();
    }
    for (keel::usize j = 0; j < p; ++j) {
        bd[j] = beta[j];
        const keel::f64 se = (sigma2 == sigma2 && invDiag[j] > 0)
                          ? std::sqrt(sigma2 * invDiag[j])
                          : std::numeric_limits<keel::f64>::quiet_NaN();
        bid[j]     = beta[j] - tcrit * se;
        bid[j + p] = beta[j] + tcrit * se;
    }
    for (keel::usize i = 0; i < N; ++i) rd[i] = r[i];

    // stats = [R², F, p_F, sigma²]
    keel::f64 yMean = 0.0;
    for (keel::usize i = 0; i < N; ++i) yMean += y.elemAsDouble(i);
    yMean /= keel::f64(N);
    keel::f64 SStot = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 d = y.elemAsDouble(i) - yMean;
        SStot += d * d;
    }
    const keel::f64 R2 = (SStot > 0.0) ? 1.0 - SSR / SStot
                                    : std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::f64 dfModel = keel::f64(p) - 1.0;
    const keel::f64 F = (dfModel > 0.0 && dfErr > 0.0 && (1.0 - R2) > 0.0)
                     ? (R2 / dfModel) / ((1.0 - R2) / dfErr)
                     : std::numeric_limits<keel::f64>::quiet_NaN();
    keel::f64 pF = std::numeric_limits<keel::f64>::quiet_NaN();
    if (!std::isnan(F)) {
        Value fv = Value::scalar(F, mr);
        const keel::f64 cdf = fcdf(fv, dfModel, dfErr, mr).toScalar();
        pF = std::max(0.0, 1.0 - cdf);
    }
    Value statsV = Value::matrix(1, 4, ValueType::DOUBLE, mr);
    keel::f64 *sd = statsV.doubleDataMut();
    sd[0] = R2; sd[1] = F; sd[2] = pF; sd[3] = sigma2;

    // rint: 100·(1-α)% CI for residuals, used to diagnose outliers
    // (Chatterjee & Hadi 1986 eq.14; Belsley, Kuh & Welsch 1980 eq.2.26).
    // The reference engine uses the LEAVE-ONE-OUT variance estimate σ_i and (nu-1) dof,
    // NOT a flat σ·sqrt(1-h) with nu dof:
    //   h_ii  = X[i,:]·(XtX)^{-1}·X[i,:]'                  (leverage)
    //   σ_i   = sqrt(max(0, nu·s²/(nu-1) - r_i²/((nu-1)(1-h_ii))))
    //   ser   = sqrt(1-h_ii)·σ_i,   t = tinv(1-α/2, nu-1)
    //   rint  = r_i ± t·ser
    // When h_ii ≥ 1 (perfect leverage) the CI degenerates to [r_i, r_i].
    Value rintV = Value::matrix(N, 2, ValueType::DOUBLE, mr);
    keel::f64 *rid = rintV.doubleDataMut();
    const keel::f64 nu   = dfErr;
    const keel::f64 s2   = sigma2;
    const keel::f64 rmse = (s2 == s2 && s2 > 0.0) ? std::sqrt(s2) : 0.0;
    keel::f64 tcritR = tcrit;                       // nu<=1 fallback: nu dof
    if (nu > 1.0) {
        Value pV = Value::scalar(1.0 - alpha / 2.0, mr);
        tcritR = tinv(pV, nu - 1.0, mr).toScalar();
    }
    const keel::f64 epsThresh = 1.4901161193847656e-08;   // sqrt(eps)
    for (keel::usize i = 0; i < N; ++i) {
        // h_ii = sum_{j,k} X[i,j] · invMat[j,k] · X[i,k].
        keel::f64 hii = 0.0;
        for (keel::usize j = 0; j < p; ++j) {
            keel::f64 row = 0.0;
            for (keel::usize k = 0; k < p; ++k)
                row += invMat[j + k * p] * X.elemAsDouble(i + k * N);
            hii += X.elemAsDouble(i + j * N) * row;
        }
        const keel::f64 oneMinusH = 1.0 - hii;
        keel::f64 ser = 0.0;
        if (oneMinusH > epsThresh && std::isfinite(oneMinusH)) {
            if (nu > 1.0) {
                const keel::f64 denom  = (nu - 1.0) * oneMinusH;
                const keel::f64 sigmai = std::sqrt(std::max(0.0,
                    (nu * s2 / (nu - 1.0)) - (r[i] * r[i]) / denom));
                ser = std::sqrt(oneMinusH) * sigmai;
            } else {
                ser = std::sqrt(oneMinusH) * rmse;
            }
        }
        rid[i]     = r[i] - tcritR * ser;
        rid[i + N] = r[i] + tcritR * ser;
    }

    return {std::move(bV), std::move(bintV), std::move(rV),
            std::move(rintV), std::move(statsV)};
}

// Backward-compat 4-tuple wrapper (for legacy callers; rint dropped).
std::tuple<Value, Value, Value, Value>
regress(const Value &y, const Value &X, keel::f64 alpha, keel::Allocator *mr)
{
    auto [b, bint, r, rint, stats] = regress_full(y, X, alpha, mr);
    (void)rint;
    return {std::move(b), std::move(bint), std::move(r), std::move(stats)};
}

std::tuple<Value, Value, Value, Value>
lscov(const Value &A, const Value &b, const Value &w, keel::Allocator *mr)
{
    lscov_argcheck(A, b, w);
    const keel::usize N = b.numel();
    const keel::usize p = A.dims().cols();

    const bool weighted = !w.isEmpty();
    const bool fullCov = weighted && (w.dims().rows() == N && w.dims().cols() == N && N > 1);

    keel::Array<keel::f64> XtWX(p * p, 0.0);
    keel::Array<keel::f64> XtWy(p, 0.0);
    keel::Array<keel::f64> Atilde;
    keel::Array<keel::f64> btilde;

    if (fullCov) {
        keel::Array<keel::f64> V(N * N, 0.0);
        for (keel::usize j = 0; j < N; ++j) {
            for (keel::usize i = 0; i < N; ++i) {
                const keel::f64 vij = w.elemAsDouble(i + j * N);
                const keel::f64 vji = w.elemAsDouble(j + i * N);
                if (!std::isfinite(vij) || std::fabs(vij - vji) > 1e-10 * (1.0 + std::fabs(vij)))
                    throw Error("lscov: covariance matrix V must be symmetric positive definite",
                                0, 0, "lscov", "", "numkit:lscov:covNotSPD");
                V[i + j * N] = 0.5 * (vij + vji);
            }
        }
        keel::Array<keel::f64> LV(N * N, 0.0);
        if (!cholesky(V.data(), LV.data(), N))
            throw Error("lscov: covariance matrix V must be symmetric positive definite",
                        0, 0, "lscov", "", "numkit:lscov:covNotSPD");

        // Whiten A and b: solve LV * Atilde(:,j) = A(:,j), LV * btilde = b.
        Atilde = keel::Array<keel::f64>(N * p, 0.0);
        btilde = keel::Array<keel::f64>(N, 0.0);
        keel::Array<keel::f64> colIn(N);
        for (keel::usize j = 0; j < p; ++j) {
            for (keel::usize i = 0; i < N; ++i) colIn[i] = A.elemAsDouble(i + j * N);
            fwd_solve(LV.data(), Atilde.data() + j * N, colIn.data(), N);
        }
        for (keel::usize i = 0; i < N; ++i) colIn[i] = b.elemAsDouble(i);
        fwd_solve(LV.data(), btilde.data(), colIn.data(), N);

        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 bi = btilde[i];
            for (keel::usize j = 0; j < p; ++j) {
                const keel::f64 xij = Atilde[i + j * N];
                XtWy[j] += xij * bi;
                for (keel::usize k = j; k < p; ++k) {
                    const keel::f64 xik = Atilde[i + k * N];
                    XtWX[k + j * p] += xij * xik;
                }
            }
        }
    } else {
        auto wi = [&](keel::usize i) {
            return weighted ? w.elemAsDouble(i) : 1.0;
        };

        // Build XtWX and XtWy with weights baked in (W = diag(w)).
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 wii = wi(i);
            const keel::f64 bi  = b.elemAsDouble(i);
            for (keel::usize j = 0; j < p; ++j) {
                const keel::f64 xij = A.elemAsDouble(i + j * N);
                XtWy[j] += wii * xij * bi;
                for (keel::usize k = j; k < p; ++k) {
                    const keel::f64 xik = A.elemAsDouble(i + k * N);
                    XtWX[k + j * p] += wii * xij * xik;
                }
            }
        }
    }
    for (keel::usize j = 0; j < p; ++j)
        for (keel::usize k = 0; k < j; ++k)
            XtWX[k + j * p] = XtWX[j + k * p];

    keel::Array<keel::f64> L(p * p, 0.0);
    if (!cholesky(XtWX.data(), L.data(), p))
        throw Error("lscov: design matrix is rank-deficient",
                    0, 0, "lscov", "", "numkit:lscov:rank");

    keel::Array<keel::f64> beta(p), z(p);
    fwd_solve(L.data(), z.data(), XtWy.data(), p);
    back_solve(L.data(), beta.data(), z.data(), p);

    // weighted residuals + SSR
    keel::f64 SSRw = 0.0;
    if (fullCov) {
        for (keel::usize i = 0; i < N; ++i) {
            keel::f64 pred = 0.0;
            for (keel::usize j = 0; j < p; ++j) pred += Atilde[i + j * N] * beta[j];
            const keel::f64 ri = btilde[i] - pred;
            SSRw += ri * ri;
        }
    } else {
        for (keel::usize i = 0; i < N; ++i) {
            keel::f64 pred = 0.0;
            for (keel::usize j = 0; j < p; ++j) pred += A.elemAsDouble(i + j * N) * beta[j];
            const keel::f64 ri = b.elemAsDouble(i) - pred;
            const keel::f64 wii = weighted ? w.elemAsDouble(i) : 1.0;
            SSRw += wii * ri * ri;
        }
    }
    const keel::f64 dfErr = keel::f64(N) - keel::f64(p);
    const keel::f64 mse = (dfErr > 0.0) ? SSRw / dfErr
                                     : std::numeric_limits<keel::f64>::quiet_NaN();

    // (XtWX)⁻¹ then S = mse · (XtWX)⁻¹
    keel::Array<keel::f64> Sm(p * p, 0.0);
    {
        keel::Array<keel::f64> ej(p), zj(p), mj(p);
        for (keel::usize j = 0; j < p; ++j) {
            std::fill(ej.begin(), ej.end(), 0.0);
            ej[j] = 1.0;
            fwd_solve(L.data(), zj.data(), ej.data(), p);
            back_solve(L.data(), mj.data(), zj.data(), p);
            for (keel::usize i = 0; i < p; ++i)
                Sm[i + j * p] = mse * mj[i];
        }
    }

    Value xV    = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value stdxV = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value mseV  = Value::scalar(mse, mr);
    Value SV    = Value::matrix(p, p, ValueType::DOUBLE, mr);
    keel::f64 *xd  = xV.doubleDataMut();
    keel::f64 *sxd = stdxV.doubleDataMut();
    keel::f64 *Sd  = SV.doubleDataMut();
    for (keel::usize j = 0; j < p; ++j) {
        xd[j]  = beta[j];
        sxd[j] = std::sqrt(std::max(Sm[j + j * p], 0.0));
    }
    for (keel::usize i = 0; i < p * p; ++i) Sd[i] = Sm[i];

    return {std::move(xV), std::move(stdxV), std::move(mseV), std::move(SV)};
}

Value ridge(const Value &y, const Value &X, const Value &kVec, bool scaled, keel::Allocator *mr)
{
    ridge_argcheck(y, X, kVec);
    const keel::usize N = y.numel();
    const keel::usize p = X.dims().cols();
    const keel::usize Nk = kVec.numel();

    // Center y, X by column mean.
    keel::f64 yMean = 0.0;
    for (keel::usize i = 0; i < N; ++i) yMean += y.elemAsDouble(i);
    yMean /= keel::f64(N);
    keel::Array<keel::f64> yc(N);
    for (keel::usize i = 0; i < N; ++i) yc[i] = y.elemAsDouble(i) - yMean;

    keel::Array<keel::f64> xMean(p, 0.0), xStd(p, 0.0);
    for (keel::usize j = 0; j < p; ++j) {
        for (keel::usize i = 0; i < N; ++i) xMean[j] += X.elemAsDouble(i + j * N);
        xMean[j] /= keel::f64(N);
    }
    keel::Array<keel::f64> Xs(N * p);
    for (keel::usize j = 0; j < p; ++j) {
        keel::f64 sq = 0.0;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 d = X.elemAsDouble(i + j * N) - xMean[j];
            Xs[i + j * N] = d;
            sq += d * d;
        }
        // Sample std with N-1 normalisation (matches the reference engine ridge).
        const keel::f64 s = (N > 1) ? std::sqrt(sq / keel::f64(N - 1)) : 1.0;
        xStd[j] = s;
        if (s > 0.0)
            for (keel::usize i = 0; i < N; ++i) Xs[i + j * N] /= s;
    }

    // X_s' X_s and X_s' y_c.
    keel::Array<keel::f64> XtX(p * p, 0.0);
    keel::Array<keel::f64> Xty(p, 0.0);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 yi = yc[i];
        for (keel::usize j = 0; j < p; ++j) {
            const keel::f64 xij = Xs[i + j * N];
            Xty[j] += xij * yi;
            for (keel::usize kk = j; kk < p; ++kk) {
                const keel::f64 xik = Xs[i + kk * N];
                XtX[kk + j * p] += xij * xik;
            }
        }
    }
    for (keel::usize j = 0; j < p; ++j)
        for (keel::usize kk = 0; kk < j; ++kk)
            XtX[kk + j * p] = XtX[j + kk * p];

    // Allocate output
    const keel::usize outRows = scaled ? p : (p + 1);
    Value B = Value::matrix(outRows, Nk, ValueType::DOUBLE, mr);
    keel::f64 *Bd = B.doubleDataMut();

    keel::Array<keel::f64> M(p * p), L(p * p), beta(p), z(p);
    for (keel::usize kIdx = 0; kIdx < Nk; ++kIdx) {
        const keel::f64 k = kVec.elemAsDouble(kIdx);
        // M = XtX + k*I
        for (keel::usize j = 0; j < p * p; ++j) M[j] = XtX[j];
        for (keel::usize j = 0; j < p; ++j) M[j + j * p] += k;
        if (!cholesky(M.data(), L.data(), p))
            throw Error("ridge: regularised normal equations not PD",
                        0, 0, "ridge", "", "numkit:ridge:psd");
        fwd_solve(L.data(), z.data(), Xty.data(), p);
        back_solve(L.data(), beta.data(), z.data(), p);
        if (scaled) {
            for (keel::usize j = 0; j < p; ++j) Bd[j + kIdx * outRows] = beta[j];
        } else {
            // Rescale to original units; prepend intercept = mean(y) - Σ β_j·mean(X_j).
            keel::f64 intercept = yMean;
            for (keel::usize j = 0; j < p; ++j) {
                const keel::f64 bj = (xStd[j] > 0.0) ? beta[j] / xStd[j] : 0.0;
                Bd[(j + 1) + kIdx * outRows] = bj;
                intercept -= bj * xMean[j];
            }
            Bd[0 + kIdx * outRows] = intercept;
        }
    }
    return B;
}

} // namespace numkit::stats
