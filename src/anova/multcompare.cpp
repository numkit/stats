// toolboxes/stats/src/anova/multcompare.cpp
//
// Multiple-comparison post-hoc tests for one-way ANOVA results:
// Tukey-Kramer HSD (default), Bonferroni-corrected pairwise t-tests,
// and Fisher LSD.

#include <numkit/stats/anova/anova.hpp>

#include <numkit/stats/distributions/students_t.hpp>   // tinv, tcdf

#include <numkit/value/value.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace numkit::stats {

namespace {

void multcompare_argcheck(const Value &stats, double alpha)
{
    if (!stats.isStruct())
        throw Error("multcompare: stats must be a struct (from anova1)",
                    0, 0, "multcompare", "", "numkit:multcompare:notStruct");
    if (!stats.hasField("means") || !stats.hasField("n")
        || !stats.hasField("s") || !stats.hasField("df"))
        throw Error("multcompare: stats must have fields {means, n, s, df}",
                    0, 0, "multcompare", "", "numkit:multcompare:missingField");
    if (alpha <= 0.0 || alpha >= 1.0 || std::isnan(alpha))
        throw Error("multcompare: alpha must be in (0, 1)",
                    0, 0, "multcompare", "", "numkit:multcompare:badAlpha");
    if (stats.field("means").numel() < 2)
        throw Error("multcompare: need at least 2 groups",
                    0, 0, "multcompare", "", "numkit:multcompare:tooFewGroups");
}

// Two-sided t p-value: 2 * (1 - tcdf(|t|, df)).
double tTwoSidedP(double tStat, double df, std::pmr::memory_resource *mr)
{
    Value tV = Value::scalar(std::fabs(tStat), mr);
    const double F = tcdf(tV, df, mr).toScalar();
    return std::min(1.0, std::max(0.0, 2.0 * (1.0 - F)));
}

// 16-point Gauss-Legendre positive nodes and weights on [-1, 1].
constexpr double kGL16X[8] = {
    0.0950125098376374401853,
    0.2816035507792589132305,
    0.4580167776572273863424,
    0.6178762444026437484467,
    0.7554044083550030338951,
    0.8656312023878317438805,
    0.9445750230732325760780,
    0.9894009349916499325961
};
constexpr double kGL16W[8] = {
    0.1894506104550684962854,
    0.1826034150449235888668,
    0.1691565193950025381893,
    0.1495959888165767320815,
    0.1246289712555338720525,
    0.0951585116824927848099,
    0.0622535239386478928628,
    0.0271524594117540948518
};

inline double normPhi(double z)
{
    constexpr double kInvSqrt2Pi = 0.3989422804014326779399;
    return kInvSqrt2Pi * std::exp(-0.5 * z * z);
}

inline double normCdfScalar(double z)
{
    constexpr double kInvSqrt2 = 0.7071067811865475244008;
    return 0.5 * std::erfc(-z * kInvSqrt2);
}

// Normal range CDF: P(max_{1..k} Z_i - min_{1..k} Z_i <= w)
// = k * \int_{-\infty}^{\infty} \phi(z) [\Phi(z + w) - \Phi(z)]^{k - 1} dz.
double normalRangeCDF(double w, std::size_t k)
{
    if (w <= 0.0) return 0.0;
    const double kd = static_cast<double>(k);
    const double km1 = static_cast<double>(k - 1);
    auto integrand = [&](double z) -> double {
        const double diff = normCdfScalar(z + w) - normCdfScalar(z);
        if (diff <= 0.0) return 0.0;
        return normPhi(z) * std::pow(diff, km1);
    };
    auto quad16 = [&](double a, double b) -> double {
        const double mid = 0.5 * (a + b);
        const double half = 0.5 * (b - a);
        double s = 0.0;
        for (int i = 0; i < 8; ++i) {
            const double dx = half * kGL16X[i];
            s += kGL16W[i] * (integrand(mid - dx) + integrand(mid + dx));
        }
        return s * half;
    };
    const double sum = quad16(-8.5, -1.5) + quad16(-1.5, 5.5);
    return std::min(1.0, std::max(0.0, kd * sum));
}

// Studentized range CDF F_Q(q; k, df).
double studentizedRangeCDF(double q, std::size_t k, double df)
{
    if (q <= 0.0) return 0.0;
    if (!std::isfinite(df) || df > 1e5) return normalRangeCDF(q, k);

    const double halfDf = 0.5 * df;
    const double logConst = 0.6931471805599453094172
                          + halfDf * std::log(halfDf)
                          - std::lgamma(halfDf);
    auto scalePdf = [&](double u) -> double {
        if (u <= 0.0) return 0.0;
        const double lp = logConst + (df - 1.0) * std::log(u) - halfDf * u * u;
        return std::exp(lp);
    };
    auto integrand = [&](double u) -> double {
        const double fu = scalePdf(u);
        if (fu <= 1e-30) return 0.0;
        return fu * normalRangeCDF(q * u, k);
    };
    auto quad16 = [&](double a, double b) -> double {
        if (b <= a) return 0.0;
        const double mid = 0.5 * (a + b);
        const double half = 0.5 * (b - a);
        double s = 0.0;
        for (int i = 0; i < 8; ++i) {
            const double dx = half * kGL16X[i];
            s += kGL16W[i] * (integrand(mid - dx) + integrand(mid + dx));
        }
        return s * half;
    };
    const double spread = 8.0 / std::sqrt(std::max(0.5, df));
    const double uMin = std::max(1e-6, 1.0 - spread);
    const double uMax = 1.0 + spread;
    const double res = quad16(uMin, 1.0) + quad16(1.0, uMax);
    return std::min(1.0, std::max(0.0, res));
}

// Inverse studentized range quantile q_{p; k, df} via bracketed bisection.
double studentizedRangeInv(double p, std::size_t k, double df)
{
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return std::numeric_limits<double>::infinity();
    double lo = 0.0;
    double hi = 4.0;
    while (studentizedRangeCDF(hi, k, df) < p && hi < 1e6) {
        lo = hi;
        hi *= 2.0;
    }
    for (int iter = 0; iter < 60; ++iter) {
        const double mid = 0.5 * (lo + hi);
        if (studentizedRangeCDF(mid, k, df) < p) lo = mid;
        else                                     hi = mid;
        if (hi - lo <= 1e-12 * std::max(1.0, mid)) break;
    }
    return 0.5 * (lo + hi);
}

} // namespace

Value multcompare(const Value &stats, double alpha, McCorrection ctype,
                   std::pmr::memory_resource *mr)
{
    multcompare_argcheck(stats, alpha);

    const Value &means = stats.field("means");
    const Value &nv    = stats.field("n");
    const double s     = stats.field("s").toScalar();
    const double df    = stats.field("df").toScalar();
    const std::size_t K = means.numel();

    std::vector<double> m(K), n(K);
    for (std::size_t i = 0; i < K; ++i) {
        m[i] = means.elemAsDouble(i);
        n[i] = nv.elemAsDouble(i);
    }

    // Number of pairs K · (K - 1) / 2.
    const std::size_t nPairs = K * (K - 1) / 2;

    double tCrit = 0.0;
    if (ctype == McCorrection::TukeyKramer) {
        const double qCrit = studentizedRangeInv(1.0 - alpha, K, df);
        tCrit = qCrit / std::sqrt(2.0);
    } else {
        const double alphaEffective = (ctype == McCorrection::Bonferroni)
            ? (alpha / static_cast<double>(nPairs))
            : alpha;
        Value pTail = Value::scalar(1.0 - alphaEffective / 2.0, mr);
        tCrit = tinv(pTail, df, mr).toScalar();
    }

    auto out = Value::matrix(nPairs, 6, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();

    std::size_t row = 0;
    for (std::size_t i = 0; i < K; ++i) {
        for (std::size_t j = i + 1; j < K; ++j) {
            const double diff = m[i] - m[j];
            const double se   = s * std::sqrt(1.0 / n[i] + 1.0 / n[j]);
            const double halfCI = tCrit * se;
            const double tStat  = (se > 0.0) ? diff / se : 0.0;
            double pVal = 1.0;
            if (ctype == McCorrection::TukeyKramer) {
                const double qStat = std::fabs(tStat) * std::sqrt(2.0);
                pVal = std::min(1.0, std::max(0.0, 1.0 - studentizedRangeCDF(qStat, K, df)));
            } else {
                pVal = tTwoSidedP(tStat, df, mr);
                if (ctype == McCorrection::Bonferroni)
                    pVal = std::min(1.0, pVal * static_cast<double>(nPairs));
            }

            // Column-major: out(row, c) = od[c * nPairs + row].
            od[0 * nPairs + row] = static_cast<double>(i + 1);
            od[1 * nPairs + row] = static_cast<double>(j + 1);
            od[2 * nPairs + row] = diff - halfCI;
            od[3 * nPairs + row] = diff;
            od[4 * nPairs + row] = diff + halfCI;
            od[5 * nPairs + row] = pVal;
            ++row;
        }
    }
    return out;
}

} // namespace numkit::stats
