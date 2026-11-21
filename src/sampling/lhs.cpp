// toolboxes/stats/src/sampling/lhs.cpp
//
// Latin Hypercube sampling (lhsdesign / lhsnorm).

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/random.h>
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

namespace numkit::stats {

namespace {

// Build a single LHS design into `od` (col-major n × p). Caller holds
// the RNG mutex. Template'd over RNG type to accept the project's
// MATLAB-compatible MT19937 stream.
template <typename Rng>
void buildOneLhs(double *od, keel::usize n, keel::usize p, bool smooth,
                 Rng &gen)
{
    keel::Array<keel::usize> perm(n);
    const double inv_n = 1.0 / static_cast<double>(n);
    for (keel::usize j = 0; j < p; ++j) {
        std::iota(perm.begin(), perm.end(), keel::usize(1));
        for (keel::usize i = n - 1; i > 0; --i) {
            const keel::usize k = keel::random::uniform_usize(gen, 0, i);
            std::swap(perm[i], perm[k]);
        }
        if (smooth) {
            for (keel::usize i = 0; i < n; ++i)
                od[j * n + i] = (static_cast<double>(perm[i]) - keel::random::uniform_f64(gen)) * inv_n;
        } else {
            for (keel::usize i = 0; i < n; ++i)
                od[j * n + i] = (static_cast<double>(perm[i]) - 0.5) * inv_n;
        }
    }
}

// Min pairwise squared Euclidean distance (rows of X, col-major n × p).
double minPairwiseDistSquared(const double *X, keel::usize n, keel::usize p)
{
    if (n < 2) return 0.0;
    double best = std::numeric_limits<double>::infinity();
    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize k = i + 1; k < n; ++k) {
            double s = 0.0;
            for (keel::usize j = 0; j < p; ++j) {
                const double d = X[j * n + i] - X[j * n + k];
                s += d * d;
            }
            if (s < best) best = s;
        }
    }
    return best;
}

// Max absolute off-diagonal Pearson correlation between columns of X.
double maxAbsColCorrelation(const double *X, keel::usize n, keel::usize p)
{
    if (p < 2 || n < 2) return 0.0;
    // Compute column means and SDs.
    keel::Array<double> mu(p), sd(p);
    for (keel::usize j = 0; j < p; ++j) {
        double s = 0.0;
        for (keel::usize i = 0; i < n; ++i) s += X[j * n + i];
        mu[j] = s / static_cast<double>(n);
    }
    for (keel::usize j = 0; j < p; ++j) {
        double v = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            const double d = X[j * n + i] - mu[j];
            v += d * d;
        }
        sd[j] = std::sqrt(v / static_cast<double>(n - 1));
        if (sd[j] < 1e-300) sd[j] = 1e-300;
    }
    double best = 0.0;
    for (keel::usize a = 0; a < p; ++a) {
        for (keel::usize b = a + 1; b < p; ++b) {
            double s = 0.0;
            for (keel::usize i = 0; i < n; ++i)
                s += (X[a * n + i] - mu[a]) * (X[b * n + i] - mu[b]);
            const double r = s / (static_cast<double>(n - 1) * sd[a] * sd[b]);
            const double abr = std::fabs(r);
            if (abr > best) best = abr;
        }
    }
    return best;
}

} // anonymous

Value lhsdesign(::numkit::ops::RngContext &rng, keel::usize n, keel::usize p,
                bool smooth, LhsCriterion criterion, keel::usize iterations,
                keel::Allocator *mr)
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

    keel::Array<double> trial(n * p);
    for (keel::usize it = 1; it < iterations; ++it) {
        buildOneLhs(trial.data(), n, p, smooth, gen);
        const double s = score_of(trial.data());
        if (s < best_score) {
            best_score = s;
            std::copy(trial.begin(), trial.end(), od);
        }
    }
    return out;
}

Value lhsdesign(::numkit::ops::RngContext &rng, keel::usize n, keel::usize p,
                keel::Allocator *mr)
{
    // MATLAB defaults: smooth=on, criterion=maximin, iterations=5.
    return lhsdesign(rng, n, p, /*smooth=*/true, LhsCriterion::Maximin, 5, mr);
}

namespace {
void lhsnorm_argcheck(const Value &Sigma, keel::usize d)
{
    if (Sigma.dims().rows() != d || Sigma.dims().cols() != d)
        throw Error("lhsnorm: Sigma must be d × d where d = length(mu)",
                    0, 0, "lhsnorm", "", "numkit:lhsnorm:shape");
}
} // namespace

Value lhsnorm(::numkit::ops::RngContext &rng, const Value &mu, const Value &Sigma, keel::usize n,
              bool smooth, keel::Allocator *mr)
{
    // Resolve d from mu's length, validate against Sigma's shape.
    const keel::usize d = mu.numel();
    if (d == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    lhsnorm_argcheck(Sigma, d);

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
    for (keel::usize j = 0; j < d; ++j)
        muRow.doubleDataMut()[j] = mu.elemAsDouble(j);
    Value casesVal = Value::scalar(static_cast<double>(n), mr);
    MvnrndOut mv = mvnrnd_full(rng, muRow, Sigma, casesVal, /*wantT=*/false, mr);

    const double invN = 1.0 / static_cast<double>(n);
    keel::Array<keel::usize> ord(n), rank(n);
    Value uCol = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *ud = uCol.doubleDataMut();

    for (keel::usize j = 0; j < d; ++j) {
        std::iota(ord.begin(), ord.end(), keel::usize(0));
        std::sort(ord.begin(), ord.end(), [&](keel::usize a, keel::usize b) {
            return mv.r.elemAsDouble(a + j * n) < mv.r.elemAsDouble(b + j * n);
        });
        for (keel::usize r = 0; r < n; ++r)
            rank[ord[r]] = r + 1;
        for (keel::usize i = 0; i < n; ++i) {
            const double xi = smooth ? keel::random::uniform_f64(rng) : 0.5;
            ud[i] = (static_cast<double>(rank[i]) - xi) * invN;
        }
        const double muj = mu.elemAsDouble(j);
        const double sdj = std::sqrt(std::max(0.0, Sigma.elemAsDouble(j + j * d)));
        if (sdj > 0.0) {
            Value zCol = norminv(uCol, muj, sdj, mr);
            for (keel::usize i = 0; i < n; ++i)
                od[j * n + i] = zCol.elemAsDouble(i);
        } else {
            for (keel::usize i = 0; i < n; ++i)
                od[j * n + i] = muj;
        }
    }
    return out;
}

} // namespace numkit::stats
