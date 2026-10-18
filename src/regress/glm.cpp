// toolboxes/stats/src/regress/glm.cpp
//
// Generalized linear model fit + prediction:
//   glmfit — IRLS GLM fitting (normal / binomial / poisson / gamma)
//   glmval — predict from fitted coefficients

#include <numkit/stats/regress/regress.hpp>

#include <numkit/stats/distributions/normal.hpp>     // for probit via norminv/normcdf

#include <numkit/value/value.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "glm_detail.hpp"

namespace numkit::stats {


GlmfitResult glmfit(const Value &X, const Value &y,
                     GlmDistribution distr, GlmLink link,
                     std::pmr::memory_resource *mr)
{
    const std::size_t n = y.numel();
    if (X.dims().rows() != n)
        throw Error("glmfit: rows(X) must equal length(y)",
                    0, 0, "glmfit", "", "numkit:glmfit:shapeMismatch");
    const std::size_t p_in = X.dims().cols();
    const std::size_t p = p_in + 1;   // +1 for auto intercept
    if (n <= p)
        throw Error("glmfit: need rows(X) > cols(X) + 1",
                    0, 0, "glmfit", "", "numkit:glmfit:noDOF");

    // If user passed Identity AND it's not Normal, treat that as
    // "default" and switch to canonical link.
    if (link == GlmLink::Identity && distr != GlmDistribution::Normal)
        link = canonicalLink(distr);

    const LinkFn fn = makeLink(link);

    // Build design matrix Xb = [1, X] column-major.
    std::vector<double> Xb(n * p, 0.0);
    for (std::size_t i = 0; i < n; ++i) Xb[0 * n + i] = 1.0;
    for (std::size_t j = 0; j < p_in; ++j)
        for (std::size_t i = 0; i < n; ++i)
            Xb[(j + 1) * n + i] = X.elemAsDouble(j * n + i);

    std::vector<double> yv(n);
    for (std::size_t i = 0; i < n; ++i) yv[i] = y.elemAsDouble(i);

    // Initial mu and eta.
    std::vector<double> mu(n), eta(n);
    for (std::size_t i = 0; i < n; ++i) {
        mu[i] = initialMu(yv[i], distr);
        // eta = g(mu) — compute via numerical inverse:
        // We don't have direct g(mu) symbolic, but we can use the
        // identity g'(mu) = dEta/dMu and iterate. Simpler: bootstrap
        // eta from a transform consistent with the link.
        switch (link) {
        case GlmLink::Identity:   eta[i] = mu[i]; break;
        case GlmLink::Logit:      eta[i] = std::log(mu[i] / (1.0 - mu[i])); break;
        case GlmLink::Log:        eta[i] = std::log(std::max(mu[i], 1e-12)); break;
        case GlmLink::Reciprocal: eta[i] = 1.0 / std::max(mu[i], 1e-12); break;
        case GlmLink::Probit: {
            // bootstrap via logit-ish guess
            const double m = std::min(std::max(mu[i], 1e-12), 1.0 - 1e-12);
            eta[i] = std::log(m / (1.0 - m)) / 1.6;   // rough scaling
            break;
        }
        }
    }

    std::vector<double> beta(p, 0.0), beta_prev(p, 0.0);
    const int maxIter = 100;
    const double tol = 1e-8;
    double dev = 0.0;
    for (int it = 0; it < maxIter; ++it) {
        // Working response z = eta + (y - mu) * dEta/dMu, weight = 1 / (V(mu) * (dEta/dMu)²).
        std::vector<double> z(n), w(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double dE = fn.deta_dmu(mu[i]);
            const double V = varianceFn(mu[i], distr);
            w[i] = 1.0 / (V * dE * dE + 1e-30);
            z[i] = eta[i] + (yv[i] - mu[i]) * dE;
        }

        // Solve weighted least squares: (X' W X) β = X' W z.
        std::vector<double> M(p * p, 0.0), b(p, 0.0);
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k < n; ++k)
                    s += Xb[i * n + k] * w[k] * Xb[j * n + k];
                M[i * p + j] = s;
            }
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k)
                s += Xb[i * n + k] * w[k] * z[k];
            b[i] = s;
        }
        if (!gaussSolve(M.data(), b.data(), p)) {
            // Singular — bail with current beta.
            break;
        }

        beta_prev = beta;
        beta = b;

        // Update eta = X·β, then mu.
        for (std::size_t i = 0; i < n; ++i) {
            double e = 0.0;
            for (std::size_t j = 0; j < p; ++j)
                e += Xb[j * n + i] * beta[j];
            eta[i] = e;
            mu[i] = fn.invLink(e);
        }

        // Convergence on coefficient change.
        double maxStep = 0.0, maxBeta = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            maxStep = std::max(maxStep, std::fabs(beta[j] - beta_prev[j]));
            maxBeta = std::max(maxBeta, std::fabs(beta[j]));
        }
        if (it > 0 && maxStep < tol * std::max(maxBeta, 1.0)) break;
    }

    // Deviance and final working weights / covariance.
    dev = 0.0;
    double pearsonChi2 = 0.0;
    std::vector<double> wFinal(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        dev += devTerm(yv[i], mu[i], distr);
        const double dE = fn.deta_dmu(mu[i]);
        const double V = std::max(varianceFn(mu[i], distr), 1e-30);
        wFinal[i] = 1.0 / (V * dE * dE + 1e-30);
        const double r = yv[i] - mu[i];
        pearsonChi2 += (r * r) / V;
    }

    const double dfe = static_cast<double>(n - p);
    const double sfit = std::sqrt(dev / dfe);
    const bool estdisp = (distr == GlmDistribution::Normal
                          || distr == GlmDistribution::Gamma
                          || distr == GlmDistribution::InverseGaussian);
    const double sDisp = !estdisp ? 1.0
                       : (distr == GlmDistribution::Normal
                              ? sfit
                              : std::sqrt(pearsonChi2 / dfe));

    // Invert J = Xb' * W * Xb to obtain covb = s^2 * J^{-1}.
    std::vector<double> J(p * p, 0.0);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < n; ++k)
                acc += Xb[i * n + k] * wFinal[k] * Xb[j * n + k];
            J[i * p + j] = acc;
        }
    Value covb = Value::matrix(p, p, ValueType::DOUBLE, mr);
    double *covbD = covb.doubleDataMut();
    for (std::size_t col = 0; col < p; ++col) {
        std::vector<double> M = J;
        std::vector<double> rhs(p, 0.0);
        rhs[col] = 1.0;
        if (gaussSolve(M.data(), rhs.data(), p)) {
            for (std::size_t row = 0; row < p; ++row)
                covbD[row + col * p] = (sDisp * sDisp) * rhs[row];
        }
    }
    Value se = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value tStat = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    for (std::size_t j = 0; j < p; ++j) {
        const double sej = std::sqrt(std::max(0.0, covbD[j + j * p]));
        se.doubleDataMut()[j] = sej;
        tStat.doubleDataMut()[j] = (sej > 0.0) ? (beta[j] / sej) : 0.0;
    }
    Value coeffcorr = Value::matrix(p, p, ValueType::DOUBLE, mr);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < p; ++k) {
            const double sej = se.doubleData()[j];
            const double sek = se.doubleData()[k];
            coeffcorr.doubleDataMut()[j + k * p] =
                (sej > 0.0 && sek > 0.0)
                    ? covbD[j + k * p] / (sej * sek)
                    : (j == k ? 1.0 : 0.0);
        }
    Value resid = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    Value wts = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    for (std::size_t i = 0; i < n; ++i) {
        resid.doubleDataMut()[i] = yv[i] - mu[i];
        wts.doubleDataMut()[i] = wFinal[i];
    }

    auto bv = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    std::memcpy(bv.doubleDataMut(), beta.data(), p * sizeof(double));

    Value stats = Value::structure(mr);
    stats.field("beta") = bv;
    stats.field("dfe") = Value::scalar(dfe, mr);
    stats.field("sfit") = Value::scalar(sfit, mr);
    stats.field("s") = Value::scalar(sDisp, mr);
    stats.field("estdisp") = Value::scalar(estdisp ? 1.0 : 0.0, mr);
    stats.field("covb") = std::move(covb);
    stats.field("se") = std::move(se);
    stats.field("coeffcorr") = std::move(coeffcorr);
    stats.field("t") = std::move(tStat);
    stats.field("resid") = std::move(resid);
    stats.field("wts") = std::move(wts);

    return { std::move(bv), Value::scalar(dev, mr), std::move(stats) };
}

Value glmval(const Value &b, const Value &X, GlmLink link,
              std::pmr::memory_resource *mr)
{
    const std::size_t p = b.numel();
    const std::size_t m = X.dims().rows();
    const std::size_t p_in = X.dims().cols();
    if (p != p_in + 1)
        throw Error("glmval: length(b) must equal cols(X) + 1",
                    0, 0, "glmval", "", "numkit:glmval:shapeMismatch");

    const LinkFn fn = makeLink(link);
    std::vector<double> bv(p);
    for (std::size_t i = 0; i < p; ++i) bv[i] = b.elemAsDouble(i);

    auto out = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t i = 0; i < m; ++i) {
        double eta = bv[0];      // intercept
        for (std::size_t j = 0; j < p_in; ++j)
            eta += X.elemAsDouble(j * m + i) * bv[j + 1];
        od[i] = fn.invLink(eta);
    }
    return out;
}

} // namespace numkit::stats
