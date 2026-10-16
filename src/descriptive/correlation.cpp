// toolboxes/stats/src/descriptive/correlation.cpp
//
// corr / partialcorr full entries — parse+validate+compute contracts,
// probed against the reference engine (build/probe71b.out P01-P15 for
// partialcorr, C01-C15 for corr). The register adapters marshal only.
//
// corr contract:
//   * positional form corr(X) / corr(X, Y): a char/string 2nd argument
//     starts the name-value tail (the reference's ischar dispatch);
//   * NV names: Type / Rows / Tail / Weights. NAME errors surface at
//     parse time; the Type/Tail VALUES are validated only after the
//     n<2 early-return, matching the reference's parseArgs/validate
//     ordering (a 1-row input short-circuits to NaN without ever
//     looking at the Type value);
//   * Weights: double/single N-by-1 nonnegative (NaN fails). Zero
//     weights drop their rows in every Rows mode (the exact drop rule
//     differs per mode: 'all' drops w==0, 'complete' turns w==0 into
//     NaN and drops them with the NaN rows, 'pairwise' keeps w>0).
//     Weighted p-values are NaN by definition (the reference does the
//     same). Rank types + Weights are a voiced stub (filed);
//   * Tail: 'left'/'both'/'right' (prefix) with the 'lt'/'ne'/'gt'
//     aliases — one-sided p-values for every type;
//   * complex data: complex rho for Pearson (probe C10), error when
//     p-values are requested (C11) or a rank type is chosen; the
//     complex + pairwise combination is a voiced stub (filed);
//   * n<2 (after the row prep): NaN rho and NaN p in the dominant
//     float class (probes C06/C07);
//   * single inputs produce single outputs (C13).
//
// partialcorr contract:
//   * positional forms partialcorr(X) / (X, Z) / (X, Y, Z): a numeric
//     2nd argument is Z (or Y when a numeric 3rd follows); anything
//     else starts the NV tail;
//   * gates: every positional must be a nonempty numeric 2-D matrix
//     (logical/char fail isnumeric → one rejection class), and at
//     least one positional must be float (the reference's superiorfloat
//     step errors when all inputs are integer);
//   * row-match: partialcorr(X, Y=..., ...) blames Y's row count
//     BEFORE Y's matrix-ness (reference order), Z after;
//   * NV: Rows {all|complete|pairwise}, Type {pearson|kendall|spearman}
//     with prefix matching — kendall is a parity ERROR (the reference
//     refuses it too), spearman ranks the data BEFORE residualising;
//   * pairwise: combinedXZ treats pairwise == complete (all columns
//     are controls for every pair — the reference says exactly this);
//     the (X, Z) / (X, Y, Z) pairwise loop stays a voiced stub (filed);
//   * Tail parses and is inert — the p-value output itself is a filed
//     missing output (see the family bug file);
//   * nonfinite guards: combinedXZ → all elements must be finite
//     after the row prep; (X, Z) forms → Z must be finite, NaNs in X
//     poison their columns through the regression (all = reference);
//   * single inputs produce single outputs (P14).

#include <numkit/stats/descriptive/descriptive.hpp>
#include "descriptive/descriptive_extras_detail.hpp"

#include <numkit/stats/distributions/students_t.hpp>   // tcdf
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace numkit::stats {

namespace {

using argcheck::ClsDbl;
using argcheck::ClsSgl;
using argcheck::classMaskOf;

inline constexpr argcheck::ClassMask kFloatReal =
    classMaskOf(ClsDbl) | classMaskOf(ClsSgl);

enum class CorrType { Pearson, Kendall, Spearman };
enum class CorrRows { All, Complete, Pairwise };
enum class CorrTail { Left, Both, Right };

const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

// tail = left|both|right by prefix, then the lt|ne|gt alias table
// (strncmpi semantics: the given string is a prefix of the choice).
CorrTail parseTailChoice(const Value &v, std::string_view fn)
{
    static constexpr std::string_view kTail[] = {"left", "both", "right"};
    static constexpr std::string_view kAlias[] = {"lt", "ne", "gt"};
    int m = argcheck::matchStringChoice(v, Span<const std::string_view>(kTail));
    if (m >= 0) return static_cast<CorrTail>(m);
    m = argcheck::matchStringChoice(v, Span<const std::string_view>(kAlias));
    if (m >= 0) return static_cast<CorrTail>(m);
    throw Error(std::string(fn) + ": Tail must be 'left', 'both', or 'right' "
                "(aliases 'lt', 'ne', 'gt'), got '" + v.toString() + "'.",
                0, 0, std::string(fn), "", "numkit:" + std::string(fn)
                    + ":unknownTail");
}

// ── rank / Kendall kernels (moved from the register adapter) ──────────

// Replace each column of X (n×p) with its tied (average) ranks. Pearson of
// the ranks is exactly the Spearman correlation.
Value rankColumns(const Value &X, std::pmr::memory_resource *mr)
{
    const std::size_t n = static_cast<std::size_t>(X.dims().dim(0));
    const std::size_t p = (X.dims().ndim() >= 2)
                            ? static_cast<std::size_t>(X.dims().dim(1)) : 1;
    Value R = Value::matrix(n, p, ValueType::DOUBLE, mr);
    double *rd = R.doubleDataMut();
    ScratchArena scratch(mr);
    for (std::size_t c = 0; c < p; ++c) {
        ScratchVec<std::size_t> idx(n, static_cast<std::size_t>(0), &scratch);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            return X.elemAsDouble(a + c * n) < X.elemAsDouble(b + c * n);
        });
        std::size_t i = 0;
        while (i < n) {
            std::size_t j = i;
            const double v = X.elemAsDouble(idx[i] + c * n);
            while (j + 1 < n && X.elemAsDouble(idx[j + 1] + c * n) == v) ++j;
            const double avg =
                (static_cast<double>(i + 1) + static_cast<double>(j + 1)) / 2.0;
            for (std::size_t k = i; k <= j; ++k) rd[idx[k] + c * n] = avg;
            i = j + 1;
        }
    }
    return R;
}

// Kendall tau-b between column ci of X and column cj of Y (length n).
double kendallTauB(const Value &X, std::size_t ci,
                   const Value &Y, std::size_t cj, std::size_t n)
{
    long nc = 0, nd = 0, n1 = 0, n2 = 0; // concordant, discordant, ties in x, ties in y
    for (std::size_t i = 0; i < n; ++i) {
        const double ai = X.elemAsDouble(i + ci * n);
        const double bi = Y.elemAsDouble(i + cj * n);
        for (std::size_t j = i + 1; j < n; ++j) {
            const double da = X.elemAsDouble(j + ci * n) - ai;
            const double db = Y.elemAsDouble(j + cj * n) - bi;
            const bool tiea = (da == 0.0);
            const bool tieb = (db == 0.0);
            if (tiea) ++n1;
            if (tieb) ++n2;
            if (!tiea && !tieb) {
                if ((da > 0.0) == (db > 0.0)) ++nc; else ++nd;
            }
        }
    }
    const double n0 = static_cast<double>(n) * (static_cast<double>(n) - 1.0) / 2.0;
    const double denom = std::sqrt((n0 - static_cast<double>(n1)) *
                                   (n0 - static_cast<double>(n2)));
    if (denom <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return (static_cast<double>(nc) - static_cast<double>(nd)) / denom;
}

// p×q matrix of Kendall tau-b for every column pair of X (n×p), Y (n×q).
Value kendallMatrix(const Value &X, const Value &Y, std::pmr::memory_resource *mr)
{
    const std::size_t n = static_cast<std::size_t>(X.dims().dim(0));
    const std::size_t p = (X.dims().ndim() >= 2)
                            ? static_cast<std::size_t>(X.dims().dim(1)) : 1;
    if (static_cast<std::size_t>(Y.dims().dim(0)) != n)
        throw Error("corr: X and Y must have the same number of rows",
                    0, 0, "corr", "", "numkit:corr:rows");
    const std::size_t q = (Y.dims().ndim() >= 2)
                            ? static_cast<std::size_t>(Y.dims().dim(1)) : 1;
    Value out = Value::matrix(p, q, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t j = 0; j < q; ++j)
        for (std::size_t i = 0; i < p; ++i)
            od[i + j * p] = kendallTauB(X, i, Y, j, n);
    return out;
}

// ── row prep ──────────────────────────────────────────────────────────

std::size_t corrRowsOf(const Value &X)
{
    return static_cast<std::size_t>(X.dims().dim(0));
}
std::size_t corrColsOf(const Value &X)
{
    return (X.dims().ndim() >= 2) ? static_cast<std::size_t>(X.dims().dim(1)) : 1;
}

// Gather X (and Y, sharing the row index) keeping only rows where keep
// is true; weights filtered alongside. keep has n entries.
void gatherRows(const Value &X, const Value *Y, const Value *wts,
                const std::vector<uint8_t> &keep, std::pmr::memory_resource *mr,
                Value &Xo, Value *Yo, Value &wo)
{
    const std::size_t n = corrRowsOf(X);
    const std::size_t pX = corrColsOf(X);
    const std::size_t pY = Y ? corrColsOf(*Y) : 0;
    const std::size_t m = static_cast<std::size_t>(
        std::count(keep.begin(), keep.end(), 1));
    Xo = Value::matrix(m, pX, ValueType::DOUBLE, mr);
    double *xo = Xo.doubleDataMut();
    for (std::size_t c = 0; c < pX; ++c)
        for (std::size_t r = 0, w = 0; r < n; ++r)
            if (keep[r]) xo[w++ + c * m] = X.elemAsDouble(r + c * n);
    if (Y && Yo) {
        *Yo = Value::matrix(m, pY, ValueType::DOUBLE, mr);
        double *yo = Yo->doubleDataMut();
        for (std::size_t c = 0; c < pY; ++c)
            for (std::size_t r = 0, w = 0; r < n; ++r)
                if (keep[r]) yo[w++ + c * m] = Y->elemAsDouble(r + c * n);
    }
    if (wts) {
        wo = Value::matrix(m, 1, ValueType::DOUBLE, mr);
        double *wd = wo.doubleDataMut();
        for (std::size_t r = 0, w = 0; r < n; ++r)
            if (keep[r]) wd[w++] = wts->elemAsDouble(r);
    }
}

// Listwise (complete-rows) deletion: keep rows with no NaN in any column
// of X (and Y, and no NaN weight — 'complete' turns zero weights into NaN
// so they drop here as well, exactly like the reference).
void dropCompleteRows(const Value &X, const Value *Y, const Value *wts,
                      std::pmr::memory_resource *mr,
                      Value &Xo, Value *Yo, Value &wo)
{
    const std::size_t n = corrRowsOf(X);
    const std::size_t pX = corrColsOf(X);
    const std::size_t pY = Y ? corrColsOf(*Y) : 0;
    std::vector<uint8_t> keep(n, 1);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < pX && keep[r]; ++c)
            if (std::isnan(X.elemAsDouble(r + c * n))) keep[r] = 0;
        if (Y)
            for (std::size_t c = 0; c < pY && keep[r]; ++c)
                if (std::isnan(Y->elemAsDouble(r + c * n))) keep[r] = 0;
        if (wts && keep[r] && !(wts->elemAsDouble(r) > 0.0)) keep[r] = 0;
    }
    gatherRows(X, Y, wts, keep, mr, Xo, Yo, wo);
}

// ── weighted / complex Pearson (the paths the simple compute fns
//    corr_xx / corr_xy do not cover) ───────────────────────────────────

// Centered-scaled column dot products. For real data this is the plain
// weighted Pearson matrix; for complex data the X side enters conjugated
// (x' * y in reference notation) and the output is complex.
Value pearsonMatrixFull(const Value &X, const Value &Y, const Value *wts,
                        bool corrXX, bool complexData,
                        std::pmr::memory_resource *mr)
{
    const std::size_t n = corrRowsOf(X);
    const std::size_t p = corrColsOf(X);
    const std::size_t q = corrColsOf(Y);
    const bool weighted = (wts != nullptr);
    auto wOf = [&](std::size_t k) {
        return weighted ? wts->elemAsDouble(k) : 1.0;
    };
    auto readX = [&](std::size_t k, std::size_t c) {
        if (X.type() == ValueType::COMPLEX)
            return std::complex<double>(X.complexData()[k + c * n]);
        return std::complex<double>(X.elemAsDouble(k + c * n), 0.0);
    };
    auto readY = [&](std::size_t k, std::size_t c) {
        if (Y.type() == ValueType::COMPLEX)
            return std::complex<double>(Y.complexData()[k + c * n]);
        return std::complex<double>(Y.elemAsDouble(k + c * n), 0.0);
    };

    // Weighted (or plain) column means; then the centered columns scaled
    // by sqrt(w).
    std::vector<std::complex<double>> mx(p), my(q);
    std::vector<double> sw(1, 0.0);
    if (weighted) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) s += wOf(k);
        sw[0] = s;
    } else {
        sw[0] = static_cast<double>(n);
    }
    for (std::size_t c = 0; c < p; ++c) {
        std::complex<double> s(0.0, 0.0);
        for (std::size_t k = 0; k < n; ++k) s += wOf(k) * readX(k, c);
        mx[c] = s / sw[0];
    }
    for (std::size_t c = 0; c < q; ++c) {
        std::complex<double> s(0.0, 0.0);
        for (std::size_t k = 0; k < n; ++k) s += wOf(k) * readY(k, c);
        my[c] = s / sw[0];
    }
    // coef(i,j) = (Σ conj(x̃) ỹ) / (|x̃| |ỹ|), x̃ = (x-mx)·sqrt(w).
    Value out = complexData ? Value::matrix(p, q, ValueType::COMPLEX, mr)
                            : Value::matrix(p, q, ValueType::DOUBLE, mr);
    std::vector<double> nx(p), ny(q);
    for (std::size_t i = 0; i < p; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const auto d = (readX(k, i) - mx[i]) * std::sqrt(wOf(k));
            s += std::norm(d);
        }
        nx[i] = std::sqrt(s);
    }
    for (std::size_t j = 0; j < q; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const auto d = (readY(k, j) - my[j]) * std::sqrt(wOf(k));
            s += std::norm(d);
        }
        ny[j] = std::sqrt(s);
    }
    for (std::size_t j = 0; j < q; ++j)
        for (std::size_t i = 0; i < p; ++i) {
            std::complex<double> s(0.0, 0.0);
            for (std::size_t k = 0; k < n; ++k)
                s += std::conj((readX(k, i) - mx[i]) * std::sqrt(wOf(k)))
                     * ((readY(k, j) - my[j]) * std::sqrt(wOf(k)));
            const std::complex<double> c = s / (nx[i] * ny[j]);
            if (complexData) {
                out.complexDataMut()[i + j * p] = Complex(c.real(), c.imag());
            } else {
                out.doubleDataMut()[i + j * p] = c.real();
            }
        }
    // Clamp |coef| > 1 to the unit circle (reference behavior); corrXX
    // diagonal becomes sign(diag) — for the complex path the diagonal is
    // x̃'x̃/|x̃|² = 1 exactly, and NaN columns stay NaN.
    if (complexData) {
        Complex *od = out.complexDataMut();
        for (std::size_t k = 0; k < p * q; ++k) {
            const std::complex<double> c(od[k].real(), od[k].imag());
            if (std::abs(c) > 1.0)
                od[k] = Complex(c.real() / std::abs(c), c.imag() / std::abs(c));
        }
        if (corrXX)
            for (std::size_t i = 0; i < p; ++i) {
                const std::complex<double> c(od[i + i * p].real(),
                                             od[i + i * p].imag());
                od[i + i * p] = std::isnan(c.real())
                    ? Complex(kNaN, 0.0) : Complex(1.0, 0.0);
            }
    } else {
        double *od = out.doubleDataMut();
        for (std::size_t k = 0; k < p * q; ++k)
            if (std::fabs(od[k]) > 1.0) od[k] = od[k] / std::fabs(od[k]);
        if (corrXX)
            for (std::size_t i = 0; i < p; ++i)
                od[i + i * p] = std::isnan(od[i + i * p])
                    ? od[i + i * p]
                    : (od[i + i * p] < 0.0 ? -1.0 : 1.0);
    }
    return out;
}

// Pairwise Pearson with per-pair NaN removal, optional weights, and the
// per-pair observation counts (nij — the p-value denominator for 'pairwise').
struct PairwisePearsonOut { Value coef, nij; };
PairwisePearsonOut corrPairwisePearson(const Value &X, const Value &Y,
                                       const Value *wts, bool corrXX,
                                       std::pmr::memory_resource *mr)
{
    const std::size_t n = corrRowsOf(X);
    const std::size_t p = corrColsOf(X);
    const std::size_t q = corrColsOf(Y);
    Value out = Value::matrix(p, q, ValueType::DOUBLE, mr);
    Value nij = Value::matrix(p, q, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    double *nd = nij.doubleDataMut();

    // One (xi, yj) pair over the rows where both are non-NaN.
    auto pairDot = [&](const Value &A, std::size_t ci,
                       const Value &B, std::size_t cj) {
        std::size_t m = 0;
        double sw = 0.0, sai = 0.0, sbj = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double a = A.elemAsDouble(k + ci * n);
            const double b = B.elemAsDouble(k + cj * n);
            if (std::isnan(a) || std::isnan(b)) continue;
            const double w = wts ? wts->elemAsDouble(k) : 1.0;
            sw += w; sai += w * a; sbj += w * b;
            ++m;
        }
        nd[ci + cj * p] = static_cast<double>(m);
        double rij = std::numeric_limits<double>::quiet_NaN();
        if (m >= 1 && sw > 0.0) {
            const double ma = sai / sw, mb = sbj / sw;
            double sab = 0.0, saa = 0.0, sbb = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                const double a = A.elemAsDouble(k + ci * n);
                const double b = B.elemAsDouble(k + cj * n);
                if (std::isnan(a) || std::isnan(b)) continue;
                const double w = wts ? wts->elemAsDouble(k) : 1.0;
                const double s = std::sqrt(w);
                const double da = (a - ma) * s, db = (b - mb) * s;
                sab += da * db; saa += da * da; sbb += db * db;
            }
            const double den = std::sqrt(saa * sbb);
            rij = (den > 0.0) ? sab / den
                              : std::numeric_limits<double>::quiet_NaN();
        }
        od[ci + cj * p] = rij;
    };

    for (std::size_t i = 0; i < p; ++i) {
        const std::size_t j1 = corrXX ? i : q;   // corrXX: lower triangle only
        for (std::size_t j = 0; j < j1; ++j)
            pairDot(X, i, Y, j);
        if (corrXX) {
            // Diagonal: 1 unless the column is NaN-poisoned or constant.
            std::size_t m = 0;
            for (std::size_t k = 0; k < n; ++k)
                if (!std::isnan(X.elemAsDouble(k + i * n))) ++m;
            nd[i + i * p] = static_cast<double>(m);
            od[i + i * p] = (m >= 1)
                ? 1.0 : std::numeric_limits<double>::quiet_NaN();
        }
    }
    if (corrXX) {
        for (std::size_t i = 0; i < p; ++i)
            for (std::size_t j = 0; j < i; ++j) {
                od[j + i * p] = od[i + j * p];
                nd[j + i * p] = nd[i + j * p];
            }
    }
    return { std::move(out), std::move(nij) };
}

// ── p-values ──────────────────────────────────────────────────────────
// Each correlation r_ij is tested against H0: no association. The p-value
// depends only on (r_ij, n, type, tail): Pearson via the t-distribution;
// Kendall via the EXACT permutation (Mahonian inversions) distribution;
// Spearman via the EXACT permutation distribution (small n) with a
// t-approximation fallback for large n. One-sided forms come straight
// from the same statistics (probe71b C08/C09/C15).

// Pearson: t = r·√((n-2)/(1-r²)), |r|==1 → t = ±Inf (analytic tail).
double corrPearsonTailP(CorrTail tail, double r, std::size_t n,
                        std::pmr::memory_resource *mr)
{
    if (n < 3 || std::isnan(r)) return kNaN;
    if (std::fabs(r) >= 1.0) {
        const bool pos = (r > 0.0);
        switch (tail) {
        case CorrTail::Both:  return 0.0;
        case CorrTail::Right: return pos ? 0.0 : 1.0;
        case CorrTail::Left:  return pos ? 1.0 : 0.0;
        }
    }
    const double df = static_cast<double>(n) - 2.0;
    const double t = r * std::sqrt(df / (1.0 - r * r));
    const double cl = tcdf(Value::scalar(t, mr), df, mr).toScalar();
    switch (tail) {
    case CorrTail::Both:  return std::min(1.0, 2.0 * tcdf(
        Value::scalar(-std::fabs(t), mr), df, mr).toScalar());
    case CorrTail::Right: return std::min(1.0, tcdf(
        Value::scalar(-t, mr), df, mr).toScalar());
    case CorrTail::Left:  return std::min(1.0, cl);
    }
    return kNaN;
}

// One-sided p from a discrete exact distribution: lower = P(D ≤ Dobs),
// upper = P(D ≥ Dobs). left = min(1, lower); right = min(1, upper);
// both = min(1, 2·min(lower, upper)) (doubling the more significant side).
double rankTailFromLU(CorrTail tail, double lower, double upper)
{
    switch (tail) {
    case CorrTail::Left:  return std::min(1.0, lower);
    case CorrTail::Right: return std::min(1.0, upper);
    case CorrTail::Both:  return std::min(1.0, 2.0 * std::min(lower, upper));
    }
    return kNaN;
}

// Exact Kendall {lower, upper} from the normalized inversions distribution
// (probabilities, so no factorial overflow). D = #discordant pairs. The DP
// ping-pongs two arena buffers sized to the max support (Dmax+1) — no
// per-iteration allocation.
std::pair<double, double> corrKendallExactLU(double tau, std::size_t n,
                                             std::pmr::memory_resource *mr)
{
    if (n < 2 || std::isnan(tau)) return { kNaN, kNaN };
    const std::size_t Dmax = n * (n - 1) / 2;
    long Dobs = std::lround(static_cast<double>(Dmax) * (1.0 - tau) / 2.0);
    if (Dobs < 0) Dobs = 0;
    if (Dobs > static_cast<long>(Dmax)) Dobs = static_cast<long>(Dmax);
    ScratchArena sc(mr);
    ScratchVec<double> prob(Dmax + 1, 0.0, &sc);      // P(#inversions == d)
    ScratchVec<double> next(Dmax + 1, 0.0, &sc);
    prob[0] = 1.0;
    std::size_t curLen = 1;                            // active prefix length
    for (std::size_t k = 1; k < n; ++k) {
        const std::size_t newLen = curLen + k;
        std::fill(next.begin(), next.begin() + newLen, 0.0);
        const double w = 1.0 / static_cast<double>(k + 1);
        for (std::size_t d = 0; d < curLen; ++d) {
            const double pv = prob[d] * w;
            for (std::size_t s = 0; s <= k; ++s) next[d + s] += pv;
        }
        prob.swap(next);                              // O(1), same arena
        curLen = newLen;
    }
    double lower = 0.0, upper = 0.0;
    for (long d = 0; d <= Dobs; ++d) lower += prob[static_cast<std::size_t>(d)];
    for (long d = Dobs; d < static_cast<long>(curLen); ++d)
        upper += prob[static_cast<std::size_t>(d)];
    return { lower, upper };
}

// Exact Spearman {lower, upper} by enumerating permutations (small n).
// D = Σ(rankX_i - rankY_i)²; under H0 it is the displacement of a random perm.
std::pair<double, double> corrSpearmanExactLU(double rho, std::size_t n,
                                              std::pmr::memory_resource *mr)
{
    const double scale = static_cast<double>(n) * (static_cast<double>(n) * n - 1.0) / 6.0;
    long Dobs = std::lround((1.0 - rho) * scale);
    if (Dobs < 0) Dobs = 0;
    ScratchArena sc(mr);
    ScratchVec<int> perm(n, &sc);
    for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<int>(i);
    std::size_t total = 0, leCount = 0, geCount = 0;
    do {
        long D = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const long d = perm[i] - static_cast<long>(i);
            D += d * d;
        }
        if (D <= Dobs) ++leCount;
        if (D >= Dobs) ++geCount;
        ++total;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return { static_cast<double>(leCount) / static_cast<double>(total),
             static_cast<double>(geCount) / static_cast<double>(total) };
}

// ── Spearman: the reference engine's two-branch contract ──────────────
// Coefficient: Pearson of the midranks when there are no ties (or no
// p-values / weights are involved); the TIE-CORRECTED form
// (meanD - D) / (sqrt(n-1) * stdD) with per-column tie adjustments when
// p-values are requested and ties exist. p-values: exact permutation
// enumeration for n < 10 (permutations of 1:n without ties, of the
// MIDRANKS with ties), the AS 89 Edgeworth expansion for n >= 10 without
// ties, a Student-t approximation with ties.

// AS 89: upper tail probability of Spearman's D (no ties), Edgeworth
// expansion with continuity correction (algorithm AS 89).
double as89Upper(double D, std::size_t n, double n3c)
{
    static const double c[12] = { .2274, .2531, .1745, .0758, .1033, .3932,
                                  .0879, .0151, .0072, .0831, .0131, .00046 };
    const double x = (2.0 * (D - 1.0) / n3c - 1.0)
                   * std::sqrt(static_cast<double>(n) - 1.0);
    const double y = x * x;
    const double u = x * (c[0] + (c[1] + c[2] / n) / n +
                          y * (-c[3] + (c[4] + c[5] / n) / n -
                          y * (c[6] + c[7] / n - y * (c[8] - c[9] / n +
                          y * (c[10] - c[11] * y) / n))) / n) / n;
    const double pv = u / std::exp(0.5 * y) + 0.5 * std::erfc(x / std::sqrt(2.0));
    return std::min(1.0, std::max(0.0, pv));
}

double as89Tail(CorrTail tail, double D, std::size_t n, double n3c)
{
    switch (tail) {
    case CorrTail::Both:
        return std::min(1.0, 2.0 * as89Upper(std::max(D, n3c - D), n, n3c));
    case CorrTail::Left:  return as89Upper(D, n, n3c);
    case CorrTail::Right: return as89Upper(n3c - D, n, n3c);
    }
    return kNaN;
}

// No-ties Spearman p from rho (rho -> D statistic -> exact / AS89).
double corrSpearmanNoTiesTailP(CorrTail tail, double rho, std::size_t n,
                               std::pmr::memory_resource *mr)
{
    if (n < 2 || std::isnan(rho)) return kNaN;
    if (n < 10) {
        const auto lo_up = corrSpearmanExactLU(rho, n, mr);
        return rankTailFromLU(tail, lo_up.first, lo_up.second);
    }
    const double n3c = (static_cast<double>(n) + 1.0) * n
                     * (static_cast<double>(n) - 1.0) / 3.0;
    const double scale = n3c / 2.0;
    const double D = scale - std::lround(rho * scale);
    return as89Tail(tail, D, n, n3c);
}

// Exact {lower, upper} tail by enumerating all n! permutations of the y
// MIDRANKS against a fixed x midrank column (the tied-data exact form).
std::pair<double, double> spearmanExactMidrankLU(const double *xr,
                                                 const double *yr,
                                                 std::size_t n, double Dobs,
                                                 std::pmr::memory_resource *mr)
{
    ScratchArena sc(mr);
    ScratchVec<int> perm(n, &sc);
    for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<int>(i);
    std::size_t total = 0, leCount = 0, geCount = 0;
    do {
        double D = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = xr[i] - yr[perm[i]];
            D += d * d;
        }
        if (D <= Dobs) ++leCount;
        if (D >= Dobs) ++geCount;
        ++total;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return { static_cast<double>(leCount) / static_cast<double>(total),
             static_cast<double>(geCount) / static_cast<double>(total) };
}

// Spearman coefficient (+ p-values). Computes coef into `coef` and, when
// wantP, the p matrix into `p` (left empty otherwise).
void spearmanFull(const Value &X, const Value &Y, bool corrXX, CorrTail tail,
                  bool wantP, std::pmr::memory_resource *mr,
                  Value &coef, Value &p)
{
    const std::size_t n = corrRowsOf(X);
    const std::size_t p1 = corrColsOf(X);
    const std::size_t p2 = corrXX ? p1 : corrColsOf(Y);
    auto xrP = tiedrank(X, mr);                 // ranks + per-column tieadj
    Value xr = xrP.first, xadjV = xrP.second;
    Value yr, yadjV;
    if (corrXX) {
        yr = xr;
        yadjV = xadjV;
    } else {
        auto yrP = tiedrank(Y, mr);
        yr = yrP.first;
        yadjV = yrP.second;
    }
    const double *xadj = xadjV.doubleData();
    const double *yadj = yadjV.doubleData();
    bool anyTies = false;
    for (std::size_t c = 0; c < p1; ++c) anyTies = anyTies || (xadj[c] > 0.0);
    if (!corrXX)
        for (std::size_t c = 0; c < p2; ++c) anyTies = anyTies || (yadj[c] > 0.0);

    if (!wantP || !anyTies) {
        coef = corrXX ? corr_xx(xr, mr) : corr_xy(xr, yr, mr);
        if (!wantP) return;
        p = Value::matrix(p1, p2, ValueType::DOUBLE, mr);
        double *pd = p.doubleDataMut();
        const double *rd = coef.doubleData();
        for (std::size_t j = 0; j < p2; ++j)
            for (std::size_t i = 0; i < p1; ++i) {
                const double rij = rd[i + j * p1];
                if (corrXX && i == j) {
                    pd[i + j * p1] = std::isnan(rij) ? kNaN : 1.0;
                    continue;
                }
                pd[i + j * p1] = corrSpearmanNoTiesTailP(tail, rij, n, mr);
            }
        return;
    }

    // Ties + p-values: the tie-corrected per-pair form.
    const double n3c = (static_cast<double>(n) + 1.0) * n
                     * (static_cast<double>(n) - 1.0) / 3.0;
    const double n3c2 = (static_cast<double>(n) + 1.0) * n
                      * (static_cast<double>(n) - 1.0) / 2.0;
    const double *xd = xr.doubleData();
    const double *yd = yr.doubleData();
    coef = Value::matrix(p1, p2, ValueType::DOUBLE, mr);
    double *cd = coef.doubleDataMut();
    p = Value::matrix(p1, p2, ValueType::DOUBLE, mr);
    double *pd = p.doubleDataMut();
    for (std::size_t i = 0; i < p1; ++i) {
        const std::size_t j1 = corrXX ? i + 1 : p2;
        for (std::size_t j = 0; j < j1; ++j) {
            double D = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                const double d = xd[k + i * n] - yd[k + j * n];
                D += d * d;
            }
            const double meanD = (n3c - (xadj[i] + yadj[j]) / 3.0) / 2.0;
            const double vx = n3c / 2.0 - xadj[i] / 3.0;
            const double vy = n3c / 2.0 - yadj[j] / 3.0;
            const bool constantCol = (xadj[i] == n3c2) || (yadj[j] == n3c2);
            const double stdD = constantCol ? 0.0
                              : std::sqrt(vx * vy / (n - 1.0));
            if (stdD == 0.0) {
                cd[i + j * p1] = kNaN;
                pd[i + j * p1] = kNaN;
                continue;
            }
            const double rho = (meanD - D) / (std::sqrt(n - 1.0) * stdD);
            cd[i + j * p1] = rho;
            if (corrXX && i == j) {
                pd[i + j * p1] = kNaN;   // overwritten by the diag pass below
                continue;
            }
            if (n < 10) {
                const auto lu = spearmanExactMidrankLU(xd + i * n, yd + j * n,
                                                       n, D, mr);
                pd[i + j * p1] = rankTailFromLU(tail, lu.first, lu.second);
            } else {
                // Student-t with the tie-corrected r.
                double t = std::copysign(kInf, rho);
                if (std::fabs(rho) < 1.0)
                    t = rho * std::sqrt((n - 2.0) / (1.0 - rho * rho));
                const double df = n - 2.0;
                switch (tail) {
                case CorrTail::Both:
                    pd[i + j * p1] = std::min(1.0, 2.0 * tcdf(
                        Value::scalar(-std::fabs(t), mr), df, mr).toScalar());
                    break;
                case CorrTail::Right:
                    pd[i + j * p1] = std::min(1.0, tcdf(
                        Value::scalar(-t, mr), df, mr).toScalar());
                    break;
                case CorrTail::Left:
                    pd[i + j * p1] = std::min(1.0, tcdf(
                        Value::scalar(t, mr), df, mr).toScalar());
                    break;
                }
            }
        }
    }
    // Clamp off-diagonal coefficients to [-1,1] (preserves NaNs).
    for (std::size_t k = 0; k < p1 * p2; ++k)
        if (std::fabs(cd[k]) > 1.0) cd[k] = cd[k] / std::fabs(cd[k]);
    if (corrXX) {
        for (std::size_t i = 0; i < p1; ++i)
            for (std::size_t j = 0; j < i; ++j) {
                cd[j + i * p1] = cd[i + j * p1];
                pd[j + i * p1] = pd[i + j * p1];
            }
        for (std::size_t i = 0; i < p1; ++i) {
            cd[i + i * p1] = std::isnan(cd[i + i * p1])
                ? cd[i + i * p1]
                : (cd[i + i * p1] < 0.0 ? -1.0 : 1.0);
            pd[i + i * p1] = std::isnan(cd[i + i * p1]) ? kNaN : 1.0;
        }
    }
}

// Large-n Kendall: tie-free normal approximation z = 3τ√(n(n-1)/2(2n+5)).
double corrKendallNormalP(CorrTail tail, double tau, std::size_t n)
{
    if (std::isnan(tau)) return kNaN;
    if (std::fabs(tau) >= 1.0)
        return rankTailFromLU(tail, tau > 0 ? 1.0 : 0.0, tau > 0 ? 0.0 : 1.0);
    const double z = 3.0 * tau * std::sqrt(static_cast<double>(n) * (n - 1))
                   / std::sqrt(2.0 * (2.0 * n + 5.0));
    // Φ(z) = erfc(-z/√2)/2
    const double phi = 0.5 * std::erfc(-z / std::sqrt(2.0));
    switch (tail) {
    case CorrTail::Left:  return std::min(1.0, phi);
    case CorrTail::Right: return std::min(1.0, 1.0 - phi);
    case CorrTail::Both:  return std::min(1.0, 2.0 * std::min(phi, 1.0 - phi));
    }
    return kNaN;
}

// Build the p-value matrix (same shape as r). corrXX → diagonal p is 1
// (or NaN when r is NaN). nij (pairwise) supplies the per-pair counts.
Value corrPValueMatrix(const Value &r, const Value *nij, std::size_t n,
                       CorrType ct, CorrTail tail, bool corrXX,
                       std::pmr::memory_resource *mr)
{
    const std::size_t p = static_cast<std::size_t>(r.dims().dim(0));
    const std::size_t q = static_cast<std::size_t>(r.dims().dim(1));
    Value P = Value::matrix(p, q, ValueType::DOUBLE, mr);
    double *pd = P.doubleDataMut();
    const double *rd = r.doubleData();
    const double *nd = nij ? nij->doubleData() : nullptr;
    const bool kendallExact = (ct == CorrType::Kendall && n <= 100);
    for (std::size_t j = 0; j < q; ++j)
        for (std::size_t i = 0; i < p; ++i) {
            const double rij = rd[i + j * p];
            if (corrXX && i == j) {
                pd[i + j * p] = std::isnan(rij) ? kNaN : 1.0;
                continue;
            }
            const std::size_t nn = nd ? static_cast<std::size_t>(nd[i + j * p]) : n;
            double pv;
            if (ct == CorrType::Pearson)
                pv = corrPearsonTailP(tail, rij, nn, mr);
            else if (kendallExact) {
                const auto [lo, up] = corrKendallExactLU(rij, n, mr);
                pv = rankTailFromLU(tail, lo, up);
            } else {
                pv = corrKendallNormalP(tail, rij, n);
            }
            pd[i + j * p] = pv;
        }
    return P;
}


// ── small shared bits ────────────────────────────────────────────────

bool allFinite(const Value &X)
{
    const std::size_t cnt = X.numel();
    for (std::size_t k = 0; k < cnt; ++k)
        if (!std::isfinite(X.elemAsDouble(k))) return false;
    return true;
}

Value nanMatrix(std::size_t r, std::size_t c, bool single,
                std::pmr::memory_resource *mr)
{
    Value out = single ? Value::matrix(r, c, ValueType::SINGLE, mr)
                       : Value::matrix(r, c, ValueType::DOUBLE, mr);
    const std::size_t cnt = r * c;
    for (std::size_t k = 0; k < cnt; ++k) {
        if (single) out.singleDataMut()[k] =
            std::numeric_limits<float>::quiet_NaN();
        else out.doubleDataMut()[k] = kNaN;
    }
    return out;
}

Value castToSingle(const Value &X, std::pmr::memory_resource *mr)
{
    const std::size_t r = static_cast<std::size_t>(X.dims().dim(0));
    const std::size_t c = static_cast<std::size_t>(X.dims().dim(1));
    Value out = Value::matrix(r, c, ValueType::SINGLE, mr);
    const std::size_t cnt = r * c;
    for (std::size_t k = 0; k < cnt; ++k)
        out.singleDataMut()[k] = static_cast<float>(X.elemAsDouble(k));
    return out;
}

// The compute kernels under these full entries are double/complex only;
// single (and integer-promoted) inputs materialize as double matrices —
// compute runs in double, the output is cast back down. Complex storage
// passes through untouched.
Value asComputeDouble(const Value &X, std::pmr::memory_resource *mr)
{
    if (X.type() == ValueType::DOUBLE || X.type() == ValueType::COMPLEX)
        return X;
    const std::size_t r = static_cast<std::size_t>(X.dims().dim(0));
    const std::size_t c = static_cast<std::size_t>(X.dims().dim(1));
    Value out = Value::matrix(r, c, ValueType::DOUBLE, mr);
    const std::size_t cnt = r * c;
    for (std::size_t k = 0; k < cnt; ++k)
        out.doubleDataMut()[k] = X.elemAsDouble(k);
    return out;
}

// Listwise NaN-row drop over 1-3 row-aligned matrices (partialcorr's
// 'complete' — and 'pairwise' in the combinedXZ form, where every column
// is a control for every pair so the two coincide).
void dropNaNRowsAny(std::vector<Value> &M, std::pmr::memory_resource *mr)
{
    const std::size_t n = static_cast<std::size_t>(M[0].dims().dim(0));
    std::vector<uint8_t> keep(n, 1);
    for (std::size_t t = 0; t < M.size(); ++t) {
        const std::size_t nn = static_cast<std::size_t>(M[t].dims().dim(0));
        const std::size_t p = static_cast<std::size_t>(M[t].dims().dim(1));
        for (std::size_t r = 0; r < std::min(nn, n); ++r)
            for (std::size_t c = 0; c < p && keep[r]; ++c)
                if (std::isnan(M[t].elemAsDouble(r + c * nn))) keep[r] = 0;
    }
    const std::size_t m = static_cast<std::size_t>(
        std::count(keep.begin(), keep.end(), 1));
    for (auto &Mt : M) {
        const std::size_t nn = static_cast<std::size_t>(Mt.dims().dim(0));
        const std::size_t p = static_cast<std::size_t>(Mt.dims().dim(1));
        Value out = Value::matrix(m, p, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (std::size_t c = 0; c < p; ++c)
            for (std::size_t r = 0, w = 0; r < nn; ++r)
                if (keep[r]) od[w++ + c * m] = Mt.elemAsDouble(r + c * nn);
        Mt = std::move(out);
    }
}

} // namespace

// ── corr full entry ───────────────────────────────────────────────────

CorrOut corr_full(const Value &x, const Value *Y, Span<const Value> nv,
                  bool wantP, std::pmr::memory_resource *mr)
{
    const std::size_t n = corrRowsOf(x);
    const std::size_t p1 = corrColsOf(x);
    const bool corrXX = (Y == nullptr);
    if (!corrXX && corrRowsOf(*Y) != n)
        throw Error("corr: X and Y must have the same number of rows "
                    "(X has " + std::to_string(n) + ", Y has "
                    + std::to_string(corrRowsOf(*Y)) + ").",
                    0, 0, "corr", "", "numkit:corr:inputSizeMismatch");
    const std::size_t p2 = corrXX ? p1 : corrColsOf(*Y);
    const bool anySingle = (x.type() == ValueType::SINGLE)
                           || (!corrXX && Y->type() == ValueType::SINGLE);
    // Materialize double copies for single / promoted inputs (the kernels
    // are double/complex only); complex storage passes through.
    const Value xD = asComputeDouble(x, mr);
    Value YDstore;
    const Value *YD = Y;
    if (!corrXX) {
        YDstore = asComputeDouble(*Y, mr);
        YD = &YDstore;
    }

    // Name-value parse: NAMES now; the Type/Tail VALUES are validated only
    // after the n<2 early-return (the reference's parseArgs validates names
    // only — a 1-row input never reaches the value checks).
    Value typeRaw, rowsRaw, tailRaw, wtsRaw;
    static constexpr std::string_view kNv[] = { "type", "rows", "tail",
                                                "weights" };
    argcheck::forEachNvPair(
        nv, 0, Span<const std::string_view>(kNv), "corr",
        "numkit:corr:badParamName", [&](std::size_t hit, const Value &v) {
            switch (hit) {
            case 0: typeRaw = v; break;
            case 1: rowsRaw = v; break;
            case 2: tailRaw = v; break;
            case 3: wtsRaw = v; break;
            default: break;
            }
        });

    // Weights: double/single N-by-1 nonnegative (NaN fails the >= test).
    bool weighted = false;
    Value wts;
    if (!wtsRaw.isEmpty()) {
        argcheck::requireClassSet(wtsRaw, kFloatReal, "corr", "Weights",
                                  argcheck::RejectionVoice{ "invalidWeight",
                                                            "", "" });
        if (wtsRaw.dims().rows() != n || wtsRaw.dims().cols() != 1)
            throw Error("corr: Weights must have size " + std::to_string(n)
                        + "x1, but the received array is "
                        + std::to_string(wtsRaw.dims().rows()) + "x"
                        + std::to_string(wtsRaw.dims().cols()) + ".",
                        0, 0, "corr", "", "numkit:corr:incorrectSize");
        for (std::size_t k = 0; k < n; ++k) {
            const double w = wtsRaw.elemAsDouble(k);
            if (!(w >= 0.0))
                throw Error("corr: expected Weights to be nonnegative (got "
                            + std::to_string(w) + " at position "
                            + std::to_string(k + 1) + ").",
                            0, 0, "corr", "", "numkit:corr:expectedNonnegative");
        }
        weighted = true;
        wts = wtsRaw;
    }

    // Rows value + per-mode row prep (zero-weight drop rules differ).
    CorrRows rows = CorrRows::All;
    if (!rowsRaw.isEmpty()) {
        static constexpr std::string_view kRows[] = { "all", "complete",
                                                      "pairwise" };
        const int m = argcheck::matchStringChoice(
            rowsRaw, Span<const std::string_view>(kRows));
        if (m < 0)
            throw Error("corr: the Rows parameter value must be 'all', "
                        "'complete', or 'pairwise', got '"
                        + rowsRaw.toString() + "'.",
                        0, 0, "corr", "", "numkit:corr:unknownRows");
        rows = static_cast<CorrRows>(m);
    }
    // gatherRows/dropCompleteRows read their inputs while writing the
    // outputs — pass DISTINCT in/out variables, then move into Xe/Ye/wEff.
    const Value *Yin = corrXX ? nullptr : YD;
    Value Xe, Ye, wEff;
    {
        Value Xo, Yo, Wo;
        if (rows == CorrRows::Complete) {
            dropCompleteRows(xD, Yin, weighted ? &wts : nullptr, mr, Xo,
                             Yin ? &Yo : nullptr, Wo);
        } else if (!weighted) {
            Xo = xD;
            if (Yin) Yo = *Yin;
        } else {
            // 'all' drops w==0 rows; 'pairwise' keeps w>0 (NaN weights were
            // already rejected at validation).
            std::vector<uint8_t> keep(n, 1);
            for (std::size_t k = 0; k < n; ++k)
                if (rows == CorrRows::All
                        ? (wts.elemAsDouble(k) == 0.0)
                        : !(wts.elemAsDouble(k) > 0.0))
                    keep[k] = 0;
            gatherRows(xD, Yin, &wts, keep, mr, Xo, Yin ? &Yo : nullptr, Wo);
        }
        Xe = std::move(Xo);
        if (Yin) Ye = std::move(Yo);
        wEff = std::move(Wo);
    }
    const bool wActive = weighted;
    const std::size_t nn = corrRowsOf(Xe);

    CorrOut out;
    if (nn < 2) {
        out.rho = nanMatrix(p1, p2, anySingle, mr);
        if (wantP) out.p = nanMatrix(p1, p2, anySingle, mr);
        return out;
    }

    // Type/Tail VALUES validated here — after the n<2 return.
    CorrType ct = CorrType::Pearson;
    if (!typeRaw.isEmpty()) {
        static constexpr std::string_view kTypes[] = { "pearson", "kendall",
                                                       "spearman" };
        const int m = argcheck::matchStringChoice(
            typeRaw, Span<const std::string_view>(kTypes));
        if (m < 0)
            throw Error("corr: the Type parameter value must be 'Pearson', "
                        "'Kendall', or 'Spearman', got '" + typeRaw.toString()
                        + "'.",
                        0, 0, "corr", "", "numkit:corr:unknownType");
        ct = static_cast<CorrType>(m);
    }
    CorrTail tail = CorrTail::Both;
    if (!tailRaw.isEmpty())
        tail = parseTailChoice(tailRaw, "corr");

    // Complex-data gates (probes C10/C11).
    const bool complexData = xD.isComplex() || (!corrXX && YD->isComplex());
    if (complexData) {
        if (wantP)
            throw Error("corr: p-values cannot be computed for complex "
                        "inputs (request only rho, or pass real data).",
                        0, 0, "corr", "", "numkit:corr:pValComplexData");
        if (ct != CorrType::Pearson)
            throw Error("corr: rank correlation types cannot be computed "
                        "for complex inputs (use the Pearson type).",
                        0, 0, "corr", "", "numkit:corr:rankCorComplexData");
        if (rows == CorrRows::Pairwise)
            throw Error("corr: complex inputs with pairwise row removal "
                        "are not supported in this revision (use 'all' or "
                        "'complete' Rows).",
                        0, 0, "corr", "", "numkit:corr:complexPairwiseUnsupported");
    }

    // Voiced capability gaps (filed in the family bug file).
    if (wActive && ct != CorrType::Pearson)
        throw Error("corr: Weights with the Kendall/Spearman types are not "
                    "supported in this revision (use the Pearson type or "
                    "omit Weights).",
                    0, 0, "corr", "", "numkit:corr:weightsRankUnsupported");
    if (rows == CorrRows::Pairwise && ct != CorrType::Pearson)
        throw Error("corr: 'pairwise' rows option is supported only for "
                    "the 'Pearson' type in this revision.",
                    0, 0, "corr", "", "numkit:corr:PairwiseType");

    // Compute rho (Spearman owns its p-values — its tie-corrected branch
    // needs the ranks, so coef and p come out of one function).
    Value r;
    Value pmat;
    bool pDone = false;
    Value nijStore;
    const Value *nij = nullptr;
    if (rows == CorrRows::Pairwise) {
        auto pw = corrPairwisePearson(Xe, corrXX ? Xe : Ye,
                                      wActive ? &wEff : nullptr, corrXX, mr);
        r = std::move(pw.coef);
        nijStore = std::move(pw.nij);
        nij = &nijStore;
    } else if (ct == CorrType::Spearman) {
        spearmanFull(Xe, corrXX ? Xe : Ye, corrXX, tail, wantP, mr, r, pmat);
        pDone = wantP;
    } else if (complexData) {
        r = pearsonMatrixFull(Xe, corrXX ? Xe : Ye, wActive ? &wEff : nullptr,
                              corrXX, true, mr);
    } else if (wActive) {
        r = pearsonMatrixFull(Xe, corrXX ? Xe : Ye, &wEff, corrXX, false, mr);
    } else {
        if (ct == CorrType::Kendall) {
            r = kendallMatrix(Xe, corrXX ? Xe : Ye, mr);
        } else {
            r = corrXX ? corr_xx(Xe, mr) : corr_xy(Xe, Ye, mr);
        }
    }

    // p-values: NaN by definition when weighted; otherwise per-type tails.
    if (wantP && !pDone) {
        if (wActive) {
            pmat = nanMatrix(p1, p2, anySingle, mr);
        } else {
            pmat = corrPValueMatrix(r, nij, nn, ct, tail, corrXX, mr);
        }
    }
    out.rho = std::move(r);
    out.p = std::move(pmat);
    if (anySingle) out.rho = castToSingle(out.rho, mr);
    if (anySingle && !out.p.isEmpty()) out.p = castToSingle(out.p, mr);
    return out;
}

// ── partialcorr full entry ────────────────────────────────────────────

Value partialcorr_full(const Value &x, const Value *y, const Value *z,
                       Span<const Value> nv, std::pmr::memory_resource *mr)
{
    const auto matricesErr = [](const std::string &what) {
        throw Error("partialcorr: all input arrays must be numeric matrices "
                    "(" + what + ").",
                    0, 0, "partialcorr", "",
                    "numkit:partialcorr:inputsMustBeMatrices");
    };
    const auto isMatrixNumeric = [](const Value &M) {
        return M.isNumeric() && M.dims().ndim() <= 2 && M.numel() > 0;
    };
    const auto mismatchErr = [](const std::string &which,
                                std::size_t got, std::size_t want) {
        throw Error("partialcorr: all input arrays must have the same number "
                    "of rows (" + which + " has " + std::to_string(got)
                    + " rows, expected " + std::to_string(want) + ").",
                    0, 0, "partialcorr", "",
                    "numkit:partialcorr:inputSizeMismatch");
    };
    const auto describe = [](const Value &M) {
        return std::string(mtypeName(M.type())) + " "
            + std::to_string(M.dims().dim(0)) + "x"
            + std::to_string(M.dims().dim(1));
    };

    if (!isMatrixNumeric(x)) matricesErr("X is " + describe(x));
    const std::size_t n = static_cast<std::size_t>(x.dims().dim(0));
    const std::size_t d = static_cast<std::size_t>(x.dims().dim(1));
    if (y) {
        // (X, Y, Z): Y's row count is blamed BEFORE Y's matrix-ness
        // (reference order), Z is checked after its form is known.
        if (static_cast<std::size_t>(y->dims().dim(0)) != n)
            mismatchErr("Y", static_cast<std::size_t>(y->dims().dim(0)), n);
        if (!isMatrixNumeric(*y)) matricesErr("Y is " + describe(*y));
        if (!isMatrixNumeric(*z)) matricesErr("Z is " + describe(*z));
        if (static_cast<std::size_t>(z->dims().dim(0)) != n)
            mismatchErr("Z", static_cast<std::size_t>(z->dims().dim(0)), n);
    } else if (z) {
        if (!isMatrixNumeric(*z)) matricesErr("Z is " + describe(*z));
        if (static_cast<std::size_t>(z->dims().dim(0)) != n)
            mismatchErr("Z", static_cast<std::size_t>(z->dims().dim(0)), n);
    }
    // Floats: the reference's superiorfloat step errors when no input is
    // float; single dominates double for the output class.
    const bool anyFloat = isFloatType(x.type())
                          || (y && isFloatType(y->type()))
                          || (z && isFloatType(z->type()));
    if (!anyFloat)
        throw Error("partialcorr: inputs must be floats, namely single or "
                    "double (integer classes need at least one float input "
                    "to promote; got none).",
                    0, 0, "partialcorr", "",
                    "numkit:partialcorr:inputsMustBeFloats");
    const bool wantSingle = (x.type() == ValueType::SINGLE)
                            || (y && y->type() == ValueType::SINGLE)
                            || (z && z->type() == ValueType::SINGLE);
    // Materialize double copies for single / promoted inputs (the
    // residualisation kernels are double-only); the output is cast back.
    const Value xD = asComputeDouble(x, mr);
    Value yD, zD;
    if (y) yD = asComputeDouble(*y, mr);
    if (z) zD = asComputeDouble(*z, mr);

    // NV: Rows / Type / Tail. Kendall is a parity error; Tail parses and
    // is inert (the p-value output is a filed missing output).
    CorrRows rows = CorrRows::All;
    CorrType ct = CorrType::Pearson;
    static constexpr std::string_view kNv[] = { "rows", "type", "tail" };
    argcheck::forEachNvPair(
        nv, 0, Span<const std::string_view>(kNv), "partialcorr",
        "numkit:partialcorr:badParamName", [&](std::size_t hit, const Value &v) {
            switch (hit) {
            case 0: {
                static constexpr std::string_view kRows[] = { "all",
                                                              "complete",
                                                              "pairwise" };
                const int m = argcheck::matchStringChoice(
                    v, Span<const std::string_view>(kRows));
                if (m < 0)
                    throw Error("partialcorr: the Rows parameter value must "
                                "be 'all', 'complete', or 'pairwise', got '"
                                + v.toString() + "'.",
                                0, 0, "partialcorr", "",
                                "numkit:partialcorr:unknownRows");
                rows = static_cast<CorrRows>(m);
                break;
            }
            case 1: {
                static constexpr std::string_view kTypes[] = { "pearson",
                                                               "kendall",
                                                               "spearman" };
                const int m = argcheck::matchStringChoice(
                    v, Span<const std::string_view>(kTypes));
                if (m < 0)
                    throw Error("partialcorr: the Type parameter value must "
                                "be 'Pearson', 'Kendall', or 'Spearman', "
                                "got '" + v.toString() + "'.",
                                0, 0, "partialcorr", "",
                                "numkit:partialcorr:unknownType");
                if (m == 1)
                    throw Error("partialcorr: Kendall's partial rank "
                                "correlation cannot be computed (the "
                                "reference engine refuses it too; use "
                                "Pearson or Spearman).",
                                0, 0, "partialcorr", "",
                                "numkit:partialcorr:kendall");
                ct = static_cast<CorrType>(m);
                break;
            }
            case 2:
                (void)parseTailChoice(v, "partialcorr");
                break;
            default: break;
            }
        });

    const bool combinedXZ = (z == nullptr);   // partialcorr(X, ...)
    const std::size_t dx = d;
    const std::size_t dy = y ? static_cast<std::size_t>(y->dims().dim(1)) : d;

    Value R;
    if (combinedXZ) {
        // 'complete' and 'pairwise' coincide here: every column of X is a
        // control for every pair, so row removal is necessarily listwise.
        std::vector<Value> M{ xD };
        if (rows != CorrRows::All) dropNaNRowsAny(M, mr);
        if (!allFinite(M[0]))
            return wantSingle ? nanMatrix(d, d, true, mr)
                              : nanMatrix(d, d, false, mr);
        if (ct == CorrType::Spearman) M[0] = tiedrank(M[0], mr).first;
        R = partialcorr_xx(M[0], mr);
    } else {
        if (rows == CorrRows::Pairwise)
            throw Error("partialcorr: the 'pairwise' rows option is not "
                        "supported for the (X, Z) / (X, Y, Z) forms in this "
                        "revision (use 'all' or 'complete').",
                        0, 0, "partialcorr", "",
                        "numkit:partialcorr:pairwiseUnsupported");
        Value Xv = xD, Yv, Zv = zD;
        if (y) Yv = yD;
        if (rows == CorrRows::Complete) {
            std::vector<Value> M{ xD };
            if (y) M.push_back(yD);
            M.push_back(zD);
            dropNaNRowsAny(M, mr);
            Xv = std::move(M[0]);
            if (y) Yv = std::move(M[1]);
            Zv = std::move(M.back());
        }
        if (!allFinite(Zv))
            return wantSingle ? nanMatrix(dx, dy, true, mr)
                              : nanMatrix(dx, dy, false, mr);
        if (ct == CorrType::Spearman) {
            Xv = tiedrank(Xv, mr).first;
            Zv = tiedrank(Zv, mr).first;
            if (y) Yv = tiedrank(Yv, mr).first;
        }
        R = y ? partialcorr_of(Xv, Yv, Zv, mr)
              : partialcorr_xz(Xv, Zv, mr);
    }
    return wantSingle ? castToSingle(R, mr) : R;
}

} // namespace numkit::stats
