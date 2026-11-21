// toolboxes/stats/src/regress/robust.cpp
//
// Robust regression + covariance:
//   robustfit — IRLS regression with bisquare/Huber weighting
//   robustcov — concentration-step (FAST-MCD-lite) robust covariance

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/regress/regress.hpp>

#include <numkit/stats/distributions/chi2.hpp>      // chi2inv

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "robust_detail.hpp"

namespace numkit::stats {

namespace {
void robustfit_argcheck(const Value &X, const Value &y, keel::usize n, keel::usize p)
{
    (void)X;
    if (y.numel() != n)
        throw Error("robustfit: length(y) must equal rows(X)",
                    0, 0, "robustfit", "", "numkit:robustfit:shapeMismatch");
    if (p == 0 || n <= p)
        throw Error("robustfit: need rows(X) > cols(X)",
                    0, 0, "robustfit", "", "numkit:robustfit:noDOF");
}
} // namespace

RobustfitResult robustfit(const Value &X, const Value &y,
                           RobustWeight weight, keel::f64 tune,
                           keel::Allocator *mr)
{
    const keel::usize n = X.dims().rows();
    const keel::usize p = X.dims().cols();
    robustfit_argcheck(X, y, n, p);

    if (std::isnan(tune)) tune = defaultTune(weight);

    keel::Array<keel::f64> Xv(n * p);
    for (keel::usize i = 0; i < n * p; ++i) Xv[i] = X.elemAsDouble(i);
    keel::Array<keel::f64> yv(n);
    for (keel::usize i = 0; i < n; ++i) yv[i] = y.elemAsDouble(i);

    // Leverage adjustment (DuMouchel & O'Brien): radj = r / sqrt(1 - h),
    // h = hat-matrix diagonal. MATLAB statrobustfit.m.
    const keel::Array<keel::f64> h = leverage(Xv.data(), n, p);
    keel::Array<keel::f64> adj(n);
    for (keel::usize i = 0; i < n; ++i)
        adj[i] = 1.0 / std::sqrt(1.0 - h[i]);

    // Floor on the scale estimate so a (near-)perfect fit can't drive s to 0.
    keel::f64 tiny_s = 1e-6 * sampleStd(yv.as_ref());
    if (!(tiny_s > 0.0)) tiny_s = 1.0;

    // Initial OLS (uniform weights).
    keel::Array<keel::f64> w(n, 1.0);
    keel::Array<keel::f64> beta = weightedLS(Xv.data(), n, p, yv.data(), w.data());

    keel::Array<keel::f64> r(n), radj(n), beta_prev(p, 0.0);
    const int maxIter = 50;
    const keel::f64 D = 1.4901161193847656e-08;   // sqrt(eps)
    keel::f64 s = 1.0;
    for (int it = 0; it < maxIter; ++it) {
        // Residuals from the current fit, then the leverage-adjusted scale.
        for (keel::usize i = 0; i < n; ++i) {
            keel::f64 pred = 0.0;
            for (keel::usize j = 0; j < p; ++j)
                pred += Xv[j * n + i] * beta[j];
            r[i] = yv[i] - pred;
            radj[i] = r[i] * adj[i];
        }
        s = madsigma(radj.as_ref(), p);
        const keel::f64 denom = std::max(s, tiny_s) * tune;

        // New weights from the standardised, leverage-adjusted residuals.
        for (keel::usize i = 0; i < n; ++i)
            w[i] = robustWeight(weight, radj[i] / denom);

        beta_prev = keel::move(beta);
        beta = weightedLS(Xv.data(), n, p, yv.data(), w.data());

        // MATLAB: stop when no |b-b0| exceeds D*max(|b|,|b0|).
        bool changed = false;
        for (keel::usize j = 0; j < p; ++j)
            if (std::fabs(beta[j] - beta_prev[j])
                > D * std::max(std::fabs(beta[j]), std::fabs(beta_prev[j])))
                changed = true;
        if (!changed) break;
    }

    // Pack outputs.
    auto bv = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    std::memcpy(bv.doubleDataMut(), beta.data(), p * sizeof(keel::f64));
    return { std::move(bv), Value::scalar(s, mr) };
}


RobustcovResult robustcov(const Value &X, keel::Allocator *mr)
{
    const keel::usize n = X.dims().rows();
    const keel::usize d = X.dims().cols();
    if (n <= d + 1)
        throw Error("robustcov: need n > d + 1 observations",
                    0, 0, "robustcov", "", "numkit:robustcov:noDOF");

    keel::Array<keel::f64> Xv(n * d);
    for (keel::usize i = 0; i < n * d; ++i) Xv[i] = X.elemAsDouble(i);

    // Start from classical mean / cov of all rows.
    keel::Array<keel::usize> keep(n);
    for (keel::usize i = 0; i < n; ++i) keep[i] = i;
    keel::Array<keel::f64> mu, C, invC;
    meanCovSubset(Xv.data(), n, d, keep.as_ref(), mu, C);

    const keel::usize h = keel::as<keel::usize>(
        std::ceil(0.75 * keel::as<keel::f64>(n)));

    keel::Array<keel::usize> prevKeep;
    for (int it = 0; it < 30; ++it) {
        if (!invertSPD(C.as_ref(), d, invC)) break;
        auto dist = mahalanobisDist(Xv.data(), n, d, mu.data(), invC.data());
        // Pick h rows with smallest distance.
        keel::Array<std::pair<keel::f64, keel::usize>> pairs(n);
        for (keel::usize i = 0; i < n; ++i) pairs[i] = { dist[i], i };
        std::nth_element(pairs.begin(), pairs.begin() + h, pairs.end());
        keel::Array<keel::usize> newKeep(h);
        for (keel::usize i = 0; i < h; ++i) newKeep[i] = pairs[i].second;
        std::sort(newKeep.begin(), newKeep.end());
        if (newKeep.size() == prevKeep.size()
            && std::equal(newKeep.begin(), newKeep.end(), prevKeep.begin()))
            break;
        prevKeep = keel::move(newKeep);
        meanCovSubset(Xv.data(), n, d, prevKeep.as_ref(), mu, C);
        keep = prevKeep.clone();
    }

    // Consistency correction (Pison-Van Aelst-Willems 2002):
    //   c = (h/n) / F_{d+2}(F_d^{-1}(h/n))
    // where F_d is the chi-squared CDF with d DOF. For h/n = 0.75
    // and d = 2 this gives c ≈ 1.86, restoring the variance of a
    // trimmed multivariate-normal sample back to the population value.
    const keel::f64 hRatio = keel::as<keel::f64>(h) / keel::as<keel::f64>(n);
    Value pVal = Value::scalar(hRatio, mr);
    const keel::f64 q = chi2inv(pVal, keel::as<keel::f64>(d), mr).toScalar();
    Value qVal = Value::scalar(q, mr);
    const keel::f64 F_dp2 = chi2cdf(qVal, keel::as<keel::f64>(d + 2), mr).toScalar();
    const keel::f64 scale = (F_dp2 > 0.0) ? hRatio / F_dp2 : 1.0;
    for (auto &c : C) c *= scale;

    // Pack outputs.
    auto sigma = Value::matrix(d, d, ValueType::DOUBLE, mr);
    // C is row-major; transpose to col-major for Value layout.
    keel::f64 *sd = sigma.doubleDataMut();
    for (keel::usize i = 0; i < d; ++i)
        for (keel::usize j = 0; j < d; ++j)
            sd[j * d + i] = C[i * d + j];
    auto muV = Value::matrix(1, d, ValueType::DOUBLE, mr);
    std::memcpy(muV.doubleDataMut(), mu.data(), d * sizeof(keel::f64));
    return { std::move(sigma), std::move(muV) };
}

} // namespace numkit::stats
