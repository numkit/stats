// toolboxes/stats/src/anova/multcompare.cpp
//
// Multiple-comparison post-hoc tests for one-way ANOVA results:
// Tukey-Kramer HSD (default), Bonferroni-corrected pairwise t-tests,
// and Fisher LSD.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <numkit/stats/anova/anova.hpp>

#include <numkit/stats/distributions/students_t.hpp>   // tinv, tcdf

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::stats {

namespace {

void multcompare_argcheck(const Value &stats, keel::f64 alpha)
{
    if (!stats.isStruct())
        throw Error("multcompare: stats must be a struct (from anova1)",
                    0, 0, "multcompare", "", "numkit:multcompare:notStruct");
    if (!stats.hasField("means") || !stats.hasField("n")
        || !stats.hasField("s") || !stats.hasField("df"))
        throw Error("multcompare: stats must have fields {means, n, s, df}",
                    0, 0, "multcompare", "", "numkit:multcompare:missingField");
    if (alpha <= 0.0 || alpha >= 1.0 || keel::is_nan(alpha))
        throw Error("multcompare: alpha must be in (0, 1)",
                    0, 0, "multcompare", "", "numkit:multcompare:badAlpha");
    if (stats.field("means").numel() < 2)
        throw Error("multcompare: need at least 2 groups",
                    0, 0, "multcompare", "", "numkit:multcompare:tooFewGroups");
}

// Two-sided t p-value: 2 * (1 - tcdf(|t|, df)).
keel::f64 tTwoSidedP(keel::f64 tStat, keel::f64 df, keel::Allocator *mr)
{
    Value tV = Value::scalar(keel::abs(tStat), mr);
    const keel::f64 F = tcdf(tV, df, mr).toScalar();
    return std::min(1.0, std::max(0.0, 2.0 * (1.0 - F)));
}

// 16-point Gauss-Legendre positive nodes and weights on [-1, 1].
constexpr keel::f64 kGL16X[8] = {
    0.0950125098376374401853,
    0.2816035507792589132305,
    0.4580167776572273863424,
    0.6178762444026437484467,
    0.7554044083550030338951,
    0.8656312023878317438805,
    0.9445750230732325760780,
    0.9894009349916499325961
};
constexpr keel::f64 kGL16W[8] = {
    0.1894506104550684962854,
    0.1826034150449235888668,
    0.1691565193950025381893,
    0.1495959888165767320815,
    0.1246289712555338720525,
    0.0951585116824927848099,
    0.0622535239386478928628,
    0.0271524594117540948518
};

inline keel::f64 normPhi(keel::f64 z)
{
    constexpr keel::f64 kInvSqrt2Pi = keel::kInvSqrt2Pi;
    return kInvSqrt2Pi * keel::exp(-0.5 * z * z);
}

inline keel::f64 normCdfScalar(keel::f64 z)
{
    constexpr keel::f64 kInvSqrt2 = 0.7071067811865475244008;
    return 0.5 * keel::erfc(-z * kInvSqrt2);
}

// Normal range CDF: P(max_{1..k} Z_i - min_{1..k} Z_i <= w)
// = k * \int_{-\infty}^{\infty} \phi(z) [\Phi(z + w) - \Phi(z)]^{k - 1} dz.
keel::f64 normalRangeCDF(keel::f64 w, keel::usize k)
{
    if (w <= 0.0) return 0.0;
    const keel::f64 kd = keel::as<keel::f64>(k);
    const keel::f64 km1 = keel::as<keel::f64>(k - 1);
    auto integrand = [&](keel::f64 z) -> keel::f64 {
        const keel::f64 diff = normCdfScalar(z + w) - normCdfScalar(z);
        if (diff <= 0.0) return 0.0;
        return normPhi(z) * keel::pow(diff, km1);
    };
    auto quad16 = [&](keel::f64 a, keel::f64 b) -> keel::f64 {
        const keel::f64 mid = 0.5 * (a + b);
        const keel::f64 half = 0.5 * (b - a);
        keel::f64 s = 0.0;
        for (keel::i32 i = 0; i < 8; ++i) {
            const keel::f64 dx = half * kGL16X[i];
            s += kGL16W[i] * (integrand(mid - dx) + integrand(mid + dx));
        }
        return s * half;
    };
    const keel::f64 sum = quad16(-8.5, -1.5) + quad16(-1.5, 5.5);
    return std::min(1.0, std::max(0.0, kd * sum));
}

// Studentized range CDF F_Q(q; k, df).
keel::f64 studentizedRangeCDF(keel::f64 q, keel::usize k, keel::f64 df)
{
    if (q <= 0.0) return 0.0;
    if (!keel::is_finite(df) || df > 1e5) return normalRangeCDF(q, k);

    const keel::f64 halfDf = 0.5 * df;
    const keel::f64 logConst = 0.6931471805599453094172
                          + halfDf * keel::log(halfDf)
                          - keel::lgamma(halfDf);
    auto scalePdf = [&](keel::f64 u) -> keel::f64 {
        if (u <= 0.0) return 0.0;
        const keel::f64 lp = logConst + (df - 1.0) * keel::log(u) - halfDf * u * u;
        return keel::exp(lp);
    };
    auto integrand = [&](keel::f64 u) -> keel::f64 {
        const keel::f64 fu = scalePdf(u);
        if (fu <= 1e-30) return 0.0;
        return fu * normalRangeCDF(q * u, k);
    };
    auto quad16 = [&](keel::f64 a, keel::f64 b) -> keel::f64 {
        if (b <= a) return 0.0;
        const keel::f64 mid = 0.5 * (a + b);
        const keel::f64 half = 0.5 * (b - a);
        keel::f64 s = 0.0;
        for (keel::i32 i = 0; i < 8; ++i) {
            const keel::f64 dx = half * kGL16X[i];
            s += kGL16W[i] * (integrand(mid - dx) + integrand(mid + dx));
        }
        return s * half;
    };
    const keel::f64 spread = 8.0 / keel::sqrt(std::max(0.5, df));
    const keel::f64 uMin = std::max(1e-6, 1.0 - spread);
    const keel::f64 uMax = 1.0 + spread;
    const keel::f64 res = quad16(uMin, 1.0) + quad16(1.0, uMax);
    return std::min(1.0, std::max(0.0, res));
}

// Inverse studentized range quantile q_{p; k, df} via bracketed bisection.
keel::f64 studentizedRangeInv(keel::f64 p, keel::usize k, keel::f64 df)
{
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return keel::Limits<keel::f64>::infinity();
    keel::f64 lo = 0.0;
    keel::f64 hi = 4.0;
    while (studentizedRangeCDF(hi, k, df) < p && hi < 1e6) {
        lo = hi;
        hi *= 2.0;
    }
    for (keel::i32 iter = 0; iter < 60; ++iter) {
        const keel::f64 mid = 0.5 * (lo + hi);
        if (studentizedRangeCDF(mid, k, df) < p) lo = mid;
        else                                     hi = mid;
        if (hi - lo <= 1e-12 * std::max(1.0, mid)) break;
    }
    return 0.5 * (lo + hi);
}

} // namespace

Value multcompare(const Value &stats, keel::f64 alpha, McCorrection ctype,
                   keel::Allocator *mr)
{
    multcompare_argcheck(stats, alpha);

    const Value &means = stats.field("means");
    const Value &nv    = stats.field("n");
    const keel::f64 s     = stats.field("s").toScalar();
    const keel::f64 df    = stats.field("df").toScalar();
    const keel::usize K = means.numel();

    keel::Array<keel::f64> m(K), n(K);
    for (keel::usize i = 0; i < K; ++i) {
        m[i] = means.elemAsDouble(i);
        n[i] = nv.elemAsDouble(i);
    }

    // Number of pairs K · (K - 1) / 2.
    const keel::usize nPairs = K * (K - 1) / 2;

    keel::f64 tCrit = 0.0;
    if (ctype == McCorrection::TukeyKramer) {
        const keel::f64 qCrit = studentizedRangeInv(1.0 - alpha, K, df);
        tCrit = qCrit / keel::sqrt(2.0);
    } else {
        const keel::f64 alphaEffective = (ctype == McCorrection::Bonferroni)
            ? (alpha / keel::as<keel::f64>(nPairs))
            : alpha;
        Value pTail = Value::scalar(1.0 - alphaEffective / 2.0, mr);
        tCrit = tinv(pTail, df, mr).toScalar();
    }

    auto out = Value::matrix(nPairs, 6, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    keel::usize row = 0;
    for (keel::usize i = 0; i < K; ++i) {
        for (keel::usize j = i + 1; j < K; ++j) {
            const keel::f64 diff = m[i] - m[j];
            const keel::f64 se   = s * keel::sqrt(1.0 / n[i] + 1.0 / n[j]);
            const keel::f64 halfCI = tCrit * se;
            const keel::f64 tStat  = (se > 0.0) ? diff / se : 0.0;
            keel::f64 pVal = 1.0;
            if (ctype == McCorrection::TukeyKramer) {
                const keel::f64 qStat = keel::abs(tStat) * keel::sqrt(2.0);
                pVal = std::min(1.0, std::max(0.0, 1.0 - studentizedRangeCDF(qStat, K, df)));
            } else {
                pVal = tTwoSidedP(tStat, df, mr);
                if (ctype == McCorrection::Bonferroni)
                    pVal = std::min(1.0, pVal * keel::as<keel::f64>(nPairs));
            }

            // Column-major: out(row, c) = od[c * nPairs + row].
            od[0 * nPairs + row] = keel::as<keel::f64>(i + 1);
            od[1 * nPairs + row] = keel::as<keel::f64>(j + 1);
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
