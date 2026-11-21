// toolboxes/stats/src/regress/lasso.cpp
//
// LASSO / elastic-net (lasso) + LASSO-regularised GLM (lassoglm) via
// cyclic coordinate descent on standardised predictors.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/regress/regress.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "lasso_detail.hpp"

namespace numkit::stats {

namespace {
void lasso_argcheck(const Value &X, const Value &y, const Value &lambdas, keel::f64 alpha)
{
    if (alpha < 0.0 || alpha > 1.0)
        throw Error("lasso: alpha must be in [0, 1]",
                    0, 0, "lasso", "", "numkit:lasso:badAlpha");
    const keel::usize n = X.dims().rows();
    if (y.numel() != n)
        throw Error("lasso: length(y) must equal rows(X)",
                    0, 0, "lasso", "", "numkit:lasso:shapeMismatch");
    if (lambdas.numel() == 0)
        throw Error("lasso: lambdas must be a non-empty vector",
                    0, 0, "lasso", "", "numkit:lasso:emptyLambda");
}
} // namespace

LassoResult lasso(const Value &X, const Value &y, const Value &lambdas,
                   keel::f64 alpha, keel::Allocator *mr)
{
    lasso_argcheck(X, y, lambdas, alpha);

    const keel::usize n = X.dims().rows();
    const keel::usize p = X.dims().cols();
    const keel::usize nL = lambdas.numel();

    // Copy X (n×p, col-major) and standardise in place.
    keel::Array<keel::f64> Xs(n * p);
    for (keel::usize i = 0; i < n * p; ++i) Xs[i] = X.elemAsDouble(i);
    keel::Array<keel::f64> mu_X, sd_X;
    standardise(Xs.data(), n, p, mu_X, sd_X);

    // y centred.
    keel::Array<keel::f64> yc(n);
    keel::f64 yMean = 0.0;
    for (keel::usize i = 0; i < n; ++i) yMean += y.elemAsDouble(i);
    yMean /= keel::as<keel::f64>(n);
    for (keel::usize i = 0; i < n; ++i) yc[i] = y.elemAsDouble(i) - yMean;

    // Pre-compute ||X_j||² for each (standardised) column = n.
    keel::Array<keel::f64> XjNormSq(p);
    for (keel::usize j = 0; j < p; ++j) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < n; ++i)
            s += Xs[j * n + i] * Xs[j * n + i];
        XjNormSq[j] = s;
    }

    // Output storage: coefficients in ORIGINAL units (transformed back
    // from standardised), intercepts per λ.
    auto Bout = Value::matrix(p, nL, ValueType::DOUBLE, mr);
    keel::f64 *Bd = Bout.doubleDataMut();
    auto intOut = Value::matrix(1, nL, ValueType::DOUBLE, mr);
    keel::f64 *intD = intOut.doubleDataMut();
    auto lambdaOut = Value::matrix(1, nL, ValueType::DOUBLE, mr);
    keel::f64 *lamD = lambdaOut.doubleDataMut();

    // Warm-start: walk λ values largest→smallest (typical lasso path).
    keel::Array<keel::usize> idx(nL);
    for (keel::usize i = 0; i < nL; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](keel::usize a, keel::usize b) {
        return lambdas.elemAsDouble(a) > lambdas.elemAsDouble(b);
    });

    keel::Array<keel::f64> beta(p, 0.0);
    keel::Array<keel::f64> r = yc.clone();         // residual on standardised X
    const int maxIter = 1000;
    const keel::f64 tol = 1e-7;

    for (keel::usize k : idx) {
        const keel::f64 lambda = lambdas.elemAsDouble(k);
        const keel::f64 lambda_eff = alpha * lambda;
        const keel::f64 lambda_l2  = (1.0 - alpha) * lambda;
        for (int it = 0; it < maxIter; ++it) {
            if (coordDescentSweep(Xs.data(), n, p, yc.data(),
                                   r, beta, XjNormSq.as_ref(),
                                   lambda_eff, lambda_l2, tol)) {
                break;
            }
        }
        // Untransform: β_orig = β_std / sd_X. Intercept = yMean - Σ β_orig · mu_X.
        keel::f64 interc = yMean;
        for (keel::usize j = 0; j < p; ++j) {
            const keel::f64 bOrig = beta[j] / sd_X[j];
            Bd[k * p + j] = bOrig;
            interc -= bOrig * mu_X[j];
        }
        intD[k] = interc;
        lamD[k] = lambda;
    }

    return { std::move(Bout), std::move(intOut), std::move(lambdaOut) };
}

// ── lassoglm — IRLS + lasso inner loop ─────────────────────────────

LassoResult lassoglm(const Value &X, const Value &y,
                      GlmDistribution distr,
                      const Value &lambdas, keel::f64 alpha,
                      keel::Allocator *mr)
{
    if (alpha < 0.0 || alpha > 1.0)
        throw Error("lassoglm: alpha must be in [0, 1]",
                    0, 0, "lassoglm", "", "numkit:lassoglm:badAlpha");

    const keel::usize n = X.dims().rows();
    const keel::usize p = X.dims().cols();
    if (y.numel() != n)
        throw Error("lassoglm: length(y) must equal rows(X)",
                    0, 0, "lassoglm", "", "numkit:lassoglm:shapeMismatch");
    const keel::usize nL = lambdas.numel();
    if (nL == 0)
        throw Error("lassoglm: lambdas must be a non-empty vector",
                    0, 0, "lassoglm", "", "numkit:lassoglm:emptyLambda");

    const LinkPair link = pickLink(distr);

    keel::Array<keel::f64> Xs(n * p);
    for (keel::usize i = 0; i < n * p; ++i) Xs[i] = X.elemAsDouble(i);
    keel::Array<keel::f64> mu_X, sd_X;
    standardise(Xs.data(), n, p, mu_X, sd_X);

    keel::Array<keel::f64> yv(n);
    for (keel::usize i = 0; i < n; ++i) yv[i] = y.elemAsDouble(i);

    auto Bout = Value::matrix(p, nL, ValueType::DOUBLE, mr);
    keel::f64 *Bd = Bout.doubleDataMut();
    auto intOut = Value::matrix(1, nL, ValueType::DOUBLE, mr);
    keel::f64 *intD = intOut.doubleDataMut();
    auto lambdaOut = Value::matrix(1, nL, ValueType::DOUBLE, mr);
    keel::f64 *lamD = lambdaOut.doubleDataMut();

    // Lambda path largest → smallest.
    keel::Array<keel::usize> idx(nL);
    for (keel::usize i = 0; i < nL; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](keel::usize a, keel::usize b) {
        return lambdas.elemAsDouble(a) > lambdas.elemAsDouble(b);
    });

    keel::Array<keel::f64> beta(p, 0.0);
    keel::f64 b0 = 0.0;
    // Initialise eta = b0 + Xs · β = b0; mu = invLink(eta).
    keel::Array<keel::f64> mu(n), eta(n, 0.0);
    for (keel::usize i = 0; i < n; ++i) mu[i] = initMu(yv[i], distr);
    // Bootstrap eta from mu via family-appropriate transform.
    auto resetEta = [&]() {
        for (keel::usize i = 0; i < n; ++i) {
            switch (distr) {
            case GlmDistribution::Normal:   eta[i] = mu[i]; break;
            case GlmDistribution::Binomial: {
                const keel::f64 m = std::min(std::max(mu[i], 1e-12), 1.0 - 1e-12);
                eta[i] = std::log(m / (1.0 - m)); break;
            }
            case GlmDistribution::Poisson:
                eta[i] = std::log(std::max(mu[i], 1e-12)); break;
            default: eta[i] = mu[i];
            }
        }
    };
    resetEta();

    for (keel::usize k : idx) {
        const keel::f64 lambda = lambdas.elemAsDouble(k);
        const keel::f64 lambda_eff = alpha * lambda;
        const keel::f64 lambda_l2  = (1.0 - alpha) * lambda;
        constexpr int outerIter = 25;
        for (int outer = 0; outer < outerIter; ++outer) {
            // Working response and weights.
            keel::Array<keel::f64> z(n), w(n);
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 dE = link.dEta(mu[i]);
                const keel::f64 V = varG(mu[i], distr);
                w[i] = 1.0 / (V * dE * dE + 1e-30);
                z[i] = eta[i] + (yv[i] - mu[i]) * dE;
            }
            // Centre z by weighted mean (so intercept = weighted-mean(z)).
            keel::f64 wsum = 0.0, wz = 0.0;
            for (keel::usize i = 0; i < n; ++i) { wsum += w[i]; wz += w[i] * z[i]; }
            const keel::f64 zMean = wz / std::max(wsum, 1e-30);
            keel::Array<keel::f64> zc(n);
            for (keel::usize i = 0; i < n; ++i) zc[i] = z[i] - zMean;

            // Weighted lasso inner coord descent. We absorb weights
            // into the variables: X_jw = sqrt(w) · X_j, z_w = sqrt(w) · z.
            keel::Array<keel::f64> Xw(n * p), zw(n);
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 sw = std::sqrt(w[i]);
                zw[i] = sw * zc[i];
                for (keel::usize j = 0; j < p; ++j)
                    Xw[j * n + i] = sw * Xs[j * n + i];
            }
            keel::Array<keel::f64> XjNorm(p);
            for (keel::usize j = 0; j < p; ++j) {
                keel::f64 s = 0.0;
                for (keel::usize i = 0; i < n; ++i)
                    s += Xw[j * n + i] * Xw[j * n + i];
                XjNorm[j] = s;
            }
            keel::Array<keel::f64> r = zw.clone();
            for (keel::usize i = 0; i < n; ++i)
                for (keel::usize j = 0; j < p; ++j)
                    r[i] -= Xw[j * n + i] * beta[j];

            constexpr int innerIter = 500;
            const keel::f64 tol = 1e-6;
            for (int inner = 0; inner < innerIter; ++inner) {
                if (coordDescentSweep(Xw.data(), n, p, zw.data(),
                                       r, beta, XjNorm.as_ref(),
                                       lambda_eff, lambda_l2, tol)) {
                    break;
                }
            }

            // Update eta from new β (intercept = zMean - Σ β · mu_X / sd_X).
            b0 = zMean;
            for (keel::usize i = 0; i < n; ++i) {
                keel::f64 e = b0;
                for (keel::usize j = 0; j < p; ++j)
                    e += beta[j] * Xs[j * n + i];
                eta[i] = e;
                mu[i] = link.inv(e);
            }
        }
        // Untransform.
        keel::f64 interc = b0;
        for (keel::usize j = 0; j < p; ++j) {
            const keel::f64 bOrig = beta[j] / sd_X[j];
            Bd[k * p + j] = bOrig;
            interc -= bOrig * mu_X[j];
        }
        intD[k] = interc;
        lamD[k] = lambda;
    }

    return { std::move(Bout), std::move(intOut), std::move(lambdaOut) };
}

} // namespace numkit::stats
