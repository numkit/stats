// toolboxes/stats/src/sampling/lhs.cpp
//
// Latin Hypercube sampling (lhsdesign / lhsnorm).

#include <numkit/stats/sampling/lhs.hpp>
#include <numkit/stats/distributions/multivariate.hpp>
#include <numkit/stats/distributions/normal.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/ops/rng_context.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

namespace numkit::stats {

namespace {

// Build a single LHS design into `od` (col-major n × p). Caller holds
// the RNG mutex. Template'd over RNG type to accept the project's
// MATLAB-compatible MT19937 stream.
template <typename Rng>
void buildOneLhs(double *od, std::size_t n, std::size_t p, bool smooth,
                 Rng &gen)
{
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<std::size_t> perm(n);
    const double inv_n = 1.0 / static_cast<double>(n);
    for (std::size_t j = 0; j < p; ++j) {
        std::iota(perm.begin(), perm.end(), std::size_t(1));
        for (std::size_t i = n - 1; i > 0; --i) {
            std::uniform_int_distribution<std::size_t> dist(0, i);
            std::swap(perm[i], perm[dist(gen)]);
        }
        if (smooth) {
            for (std::size_t i = 0; i < n; ++i)
                od[j * n + i] = (static_cast<double>(perm[i]) - U(gen)) * inv_n;
        } else {
            for (std::size_t i = 0; i < n; ++i)
                od[j * n + i] = (static_cast<double>(perm[i]) - 0.5) * inv_n;
        }
    }
}

// Min pairwise squared Euclidean distance (rows of X, col-major n × p).
double minPairwiseDistSquared(const double *X, std::size_t n, std::size_t p)
{
    if (n < 2) return 0.0;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = i + 1; k < n; ++k) {
            double s = 0.0;
            for (std::size_t j = 0; j < p; ++j) {
                const double d = X[j * n + i] - X[j * n + k];
                s += d * d;
            }
            if (s < best) best = s;
        }
    }
    return best;
}

// Max absolute off-diagonal Pearson correlation between columns of X.
double maxAbsColCorrelation(const double *X, std::size_t n, std::size_t p)
{
    if (p < 2 || n < 2) return 0.0;
    // Compute column means and SDs.
    std::vector<double> mu(p), sd(p);
    for (std::size_t j = 0; j < p; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += X[j * n + i];
        mu[j] = s / static_cast<double>(n);
    }
    for (std::size_t j = 0; j < p; ++j) {
        double v = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = X[j * n + i] - mu[j];
            v += d * d;
        }
        sd[j] = std::sqrt(v / static_cast<double>(n - 1));
        if (sd[j] < 1e-300) sd[j] = 1e-300;
    }
    double best = 0.0;
    for (std::size_t a = 0; a < p; ++a) {
        for (std::size_t b = a + 1; b < p; ++b) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                s += (X[a * n + i] - mu[a]) * (X[b * n + i] - mu[b]);
            const double r = s / (static_cast<double>(n - 1) * sd[a] * sd[b]);
            const double abr = std::fabs(r);
            if (abr > best) best = abr;
        }
    }
    return best;
}

} // anonymous

Value lhsdesign(::numkit::ops::RngContext &rng, std::size_t n, std::size_t p,
                bool smooth, LhsCriterion criterion, std::size_t iterations,
                std::pmr::memory_resource *mr)
{
    auto out = Value::matrix(n, p, ValueType::DOUBLE, mr);
    if (n == 0 || p == 0) return out;
    double *od = out.doubleDataMut();
    if (iterations < 1) iterations = 1;
    if (criterion == LhsCriterion::None) iterations = 1;

    auto &gen = rng;

    // First trial → direct write into `od`. Subsequent trials go to a
    // scratch buffer; replace `od` when the score improves.
    buildOneLhs(od, n, p, smooth, gen);

    if (iterations == 1 || criterion == LhsCriterion::None) return out;

    double best_score;
    auto score_of = [&](const double *X) {
        if (criterion == LhsCriterion::Maximin)
            return -minPairwiseDistSquared(X, n, p); // minimise neg → max dist
        return maxAbsColCorrelation(X, n, p);
    };
    best_score = score_of(od);

    std::vector<double> trial(n * p);
    for (std::size_t it = 1; it < iterations; ++it) {
        buildOneLhs(trial.data(), n, p, smooth, gen);
        const double s = score_of(trial.data());
        if (s < best_score) {
            best_score = s;
            std::copy(trial.begin(), trial.end(), od);
        }
    }
    return out;
}

Value lhsdesign(::numkit::ops::RngContext &rng, std::size_t n, std::size_t p,
                std::pmr::memory_resource *mr)
{
    // MATLAB defaults: smooth=on, criterion=maximin, iterations=5.
    return lhsdesign(rng, n, p, /*smooth=*/true, LhsCriterion::Maximin, 5, mr);
}

Value lhsnorm(::numkit::ops::RngContext &rng, const Value &mu, const Value &Sigma, std::size_t n,
              bool smooth, std::pmr::memory_resource *mr)
{
    // Resolve d from mu's length, validate against Sigma's shape.
    const std::size_t d = mu.numel();
    if (d == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (Sigma.dims().rows() != d || Sigma.dims().cols() != d)
        throw Error("lhsnorm: Sigma must be d × d where d = length(mu)",
                    0, 0, "lhsnorm", "", "numkit:lhsnorm:shape");

    auto out = Value::matrix(n, d, ValueType::DOUBLE, mr);
    if (n == 0) return out;
    double *od = out.doubleDataMut();

    // Stein (1987) rank-based Latin hypercube with covariance Sigma:
    // 1. Draw Z = mvnrnd(mu, Sigma, n) (validates PSD Sigma via psdFactor).
    // 2. For each column j = 1..d, compute the 1-based column ranks r_{i,j}
    //    of Z(:,j) and map them to stratified normal quantiles:
    //      u_{i,j} = (r_{i,j} - xi_{i,j}) / n,
    //      X_{i,j} = mu_j + sqrt(Sigma_{j,j}) * norminv(u_{i,j}).
    //    This preserves both exact 1-D Latin hypercube stratification in
    //    every marginal and the rank correlation structure of Sigma.
    Value muRow = Value::matrix(1, d, ValueType::DOUBLE, mr);
    for (std::size_t j = 0; j < d; ++j)
        muRow.doubleDataMut()[j] = mu.elemAsDouble(j);
    Value casesVal = Value::scalar(static_cast<double>(n), mr);
    MvnrndOut mv = mvnrnd_full(rng, muRow, Sigma, casesVal, /*wantT=*/false, mr);

    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double invN = 1.0 / static_cast<double>(n);
    std::vector<std::size_t> ord(n), rank(n);
    Value uCol = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *ud = uCol.doubleDataMut();

    for (std::size_t j = 0; j < d; ++j) {
        std::iota(ord.begin(), ord.end(), std::size_t(0));
        std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) {
            return mv.r.elemAsDouble(a + j * n) < mv.r.elemAsDouble(b + j * n);
        });
        for (std::size_t r = 0; r < n; ++r)
            rank[ord[r]] = r + 1;
        for (std::size_t i = 0; i < n; ++i) {
            const double xi = smooth ? U(rng) : 0.5;
            ud[i] = (static_cast<double>(rank[i]) - xi) * invN;
        }
        const double muj = mu.elemAsDouble(j);
        const double sdj = std::sqrt(std::max(0.0, Sigma.elemAsDouble(j + j * d)));
        if (sdj > 0.0) {
            Value zCol = norminv(uCol, muj, sdj, mr);
            for (std::size_t i = 0; i < n; ++i)
                od[j * n + i] = zCol.elemAsDouble(i);
        } else {
            for (std::size_t i = 0; i < n; ++i)
                od[j * n + i] = muj;
        }
    }
    return out;
}

} // namespace numkit::stats
