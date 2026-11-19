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

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/string.h>
#include <numkit/stats/descriptive/descriptive.hpp>
#include "descriptive/descriptive_extras_detail.hpp"

#include <numkit/stats/distributions/students_t.hpp>   // tcdf
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

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
CorrTail parseTailChoice(const Value &v, keel::StringRef fn)
{
    static constexpr keel::StringRef kTail[] = {"left", "both", "right"};
    static constexpr keel::StringRef kAlias[] = {"lt", "ne", "gt"};
    int m = argcheck::matchStringChoice(v, keel::ArrayRef<keel::StringRef>(kTail));
    if (m >= 0) return static_cast<CorrTail>(m);
    m = argcheck::matchStringChoice(v, keel::ArrayRef<keel::StringRef>(kAlias));
    if (m >= 0) return static_cast<CorrTail>(m);
    keel::String vStr = v.toString();
    throw Error(keel::format("{}: Tail must be 'left', 'both', or 'right' "
                             "(aliases 'lt', 'ne', 'gt'), got '{}'.",
                             fn, vStr.as_ref()),
                0, 0, fn, "", keel::format("numkit:{}:unknownTail", fn));
}

// ── rank / Kendall kernels (moved from the register adapter) ──────────

// Replace each column of X (n×p) with its tied (average) ranks, or weighted
// midranks when wts is non-null.
Value rankColumnsWeighted(const Value &X, const Value *wts,
                          keel::Allocator *mr)
{
    if (!wts) return tiedrank(X, mr).first;
    const keel::usize n = static_cast<keel::usize>(X.dims().dim(0));
    const keel::usize p = (X.dims().ndim() >= 2)
                            ? static_cast<keel::usize>(X.dims().dim(1)) : 1;
    Value R = Value::matrix(n, p, ValueType::DOUBLE, mr);
    double *rd = R.doubleDataMut();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    for (keel::usize c = 0; c < p; ++c) {
        bool hasNaN = false;
        for (keel::usize i = 0; i < n; ++i)
            if (std::isnan(X.elemAsDouble(i + c * n))) { hasNaN = true; break; }
        if (hasNaN) {
            for (keel::usize i = 0; i < n; ++i) rd[i + c * n] = kNaN;
            continue;
        }
        keel::Array<keel::usize> idx(n, static_cast<keel::usize>(0), scratch.allocator());
        for (keel::usize i = 0; i < n; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](keel::usize a, keel::usize b) {
            return X.elemAsDouble(a + c * n) < X.elemAsDouble(b + c * n);
        });
        double wPrev = 0.0;
        keel::usize i = 0;
        while (i < n) {
            keel::usize j = i;
            const double v = X.elemAsDouble(idx[i] + c * n);
            double wGroup = wts->elemAsDouble(idx[i]);
            while (j + 1 < n && X.elemAsDouble(idx[j + 1] + c * n) == v) {
                ++j;
                wGroup += wts->elemAsDouble(idx[j]);
            }
            const double avg = wPrev + 0.5 * (1.0 + wGroup);
            for (keel::usize k = i; k <= j; ++k) rd[idx[k] + c * n] = avg;
            wPrev += wGroup;
            i = j + 1;
        }
    }
    return R;
}

// Kendall tau-b between column ci of X and column cj of Y (length n),
// with optional nonnegative observation weights wts.
double kendallTauB(const Value &X, keel::usize ci,
                   const Value &Y, keel::usize cj, keel::usize n,
                   const Value *wts = nullptr)
{
    double s = 0.0, n0 = 0.0, n1 = 0.0, n2 = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const double ai = X.elemAsDouble(i + ci * n);
        const double bi = Y.elemAsDouble(i + cj * n);
        if (std::isnan(ai) || std::isnan(bi)) return kNaN;
        const double wi = wts ? wts->elemAsDouble(i) : 1.0;
        for (keel::usize j = i + 1; j < n; ++j) {
            const double aj = X.elemAsDouble(j + ci * n);
            const double bj = Y.elemAsDouble(j + cj * n);
            if (std::isnan(aj) || std::isnan(bj)) return kNaN;
            const double wj = wts ? wts->elemAsDouble(j) : 1.0;
            const double wij = wi * wj;
            n0 += wij;
            const double da = aj - ai;
            const double db = bj - bi;
            const bool tiea = (da == 0.0);
            const bool tieb = (db == 0.0);
            if (tiea) n1 += wij;
            if (tieb) n2 += wij;
            if (!tiea && !tieb) {
                s += ((da > 0.0) == (db > 0.0)) ? wij : -wij;
            }
        }
    }
    const double denom = std::sqrt((n0 - n1) * (n0 - n2));
    if (denom <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return s / denom;
}

// p×q matrix of Kendall tau-b for every column pair of X (n×p), Y (n×q).
Value kendallMatrix(const Value &X, const Value &Y, const Value *wts,
                    keel::Allocator *mr)
{
    const keel::usize n = static_cast<keel::usize>(X.dims().dim(0));
    const keel::usize p = (X.dims().ndim() >= 2)
                            ? static_cast<keel::usize>(X.dims().dim(1)) : 1;
    if (static_cast<keel::usize>(Y.dims().dim(0)) != n)
        throw Error("corr: X and Y must have the same number of rows",
                    0, 0, "corr", "", "numkit:corr:rows");
    const keel::usize q = (Y.dims().ndim() >= 2)
                            ? static_cast<keel::usize>(Y.dims().dim(1)) : 1;
    Value out = Value::matrix(p, q, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (keel::usize j = 0; j < q; ++j)
        for (keel::usize i = 0; i < p; ++i)
            od[i + j * p] = kendallTauB(X, i, Y, j, n, wts);
    return out;
}

// ── row prep ──────────────────────────────────────────────────────────

keel::usize corrRowsOf(const Value &X)
{
    return static_cast<keel::usize>(X.dims().dim(0));
}
keel::usize corrColsOf(const Value &X)
{
    return (X.dims().ndim() >= 2) ? static_cast<keel::usize>(X.dims().dim(1)) : 1;
}

bool elemIsNaN(const Value &V, keel::usize idx)
{
    if (V.type() == ValueType::COMPLEX) {
        const keel::c64 c = V.complexData()[idx];
        return std::isnan(c.real()) || std::isnan(c.imag());
    }
    return std::isnan(V.elemAsDouble(idx));
}

// Gather X (and Y, sharing the row index) keeping only rows where keep
// is true; weights filtered alongside. keep has n entries.
void gatherRows(const Value &X, const Value *Y, const Value *wts,
                keel::ArrayRef<keel::u8> keep, keel::Allocator *mr,
                Value &Xo, Value *Yo, Value &wo)
{
    const keel::usize n = corrRowsOf(X);
    const keel::usize pX = corrColsOf(X);
    const keel::usize pY = Y ? corrColsOf(*Y) : 0;
    const keel::usize m = static_cast<keel::usize>(
        std::count(keep.begin(), keep.end(), 1));
    if (X.type() == ValueType::COMPLEX) {
        Xo = Value::matrix(m, pX, ValueType::COMPLEX, mr);
        keel::c64 *xo = Xo.complexDataMut();
        for (keel::usize c = 0; c < pX; ++c)
            for (keel::usize r = 0, w = 0; r < n; ++r)
                if (keep[r]) xo[w++ + c * m] = X.complexData()[r + c * n];
    } else {
        Xo = Value::matrix(m, pX, ValueType::DOUBLE, mr);
        double *xo = Xo.doubleDataMut();
        for (keel::usize c = 0; c < pX; ++c)
            for (keel::usize r = 0, w = 0; r < n; ++r)
                if (keep[r]) xo[w++ + c * m] = X.elemAsDouble(r + c * n);
    }
    if (Y && Yo) {
        if (Y->type() == ValueType::COMPLEX) {
            *Yo = Value::matrix(m, pY, ValueType::COMPLEX, mr);
            keel::c64 *yo = Yo->complexDataMut();
            for (keel::usize c = 0; c < pY; ++c)
                for (keel::usize r = 0, w = 0; r < n; ++r)
                    if (keep[r]) yo[w++ + c * m] = Y->complexData()[r + c * n];
        } else {
            *Yo = Value::matrix(m, pY, ValueType::DOUBLE, mr);
            double *yo = Yo->doubleDataMut();
            for (keel::usize c = 0; c < pY; ++c)
                for (keel::usize r = 0, w = 0; r < n; ++r)
                    if (keep[r]) yo[w++ + c * m] = Y->elemAsDouble(r + c * n);
        }
    }
    if (wts) {
        wo = Value::matrix(m, 1, ValueType::DOUBLE, mr);
        double *wd = wo.doubleDataMut();
        for (keel::usize r = 0, w = 0; r < n; ++r)
            if (keep[r]) wd[w++] = wts->elemAsDouble(r);
    }
}

// Listwise (complete-rows) deletion: keep rows with no NaN in any column
// of X (and Y, and no NaN weight — 'complete' turns zero weights into NaN
// so they drop here as well, exactly like the reference).
void dropCompleteRows(const Value &X, const Value *Y, const Value *wts,
                      keel::Allocator *mr,
                      Value &Xo, Value *Yo, Value &wo)
{
    const keel::usize n = corrRowsOf(X);
    const keel::usize pX = corrColsOf(X);
    const keel::usize pY = Y ? corrColsOf(*Y) : 0;
    keel::Array<keel::u8> keep(n, 1);
    for (keel::usize r = 0; r < n; ++r) {
        for (keel::usize c = 0; c < pX && keep[r]; ++c)
            if (elemIsNaN(X, r + c * n)) keep[r] = 0;
        if (Y)
            for (keel::usize c = 0; c < pY && keep[r]; ++c)
                if (elemIsNaN(*Y, r + c * n)) keep[r] = 0;
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
                        keel::Allocator *mr)
{
    const keel::usize n = corrRowsOf(X);
    const keel::usize p = corrColsOf(X);
    const keel::usize q = corrColsOf(Y);
    const bool weighted = (wts != nullptr);
    auto wOf = [&](keel::usize k) {
        return weighted ? wts->elemAsDouble(k) : 1.0;
    };
    auto readX = [&](keel::usize k, keel::usize c) {
        if (X.type() == ValueType::COMPLEX)
            return keel::c64(X.complexData()[k + c * n]);
        return keel::c64(X.elemAsDouble(k + c * n), 0.0);
    };
    auto readY = [&](keel::usize k, keel::usize c) {
        if (Y.type() == ValueType::COMPLEX)
            return keel::c64(Y.complexData()[k + c * n]);
        return keel::c64(Y.elemAsDouble(k + c * n), 0.0);
    };

    // Weighted (or plain) column means; then the centered columns scaled
    // by sqrt(w).
    keel::Array<keel::c64> mx(p), my(q);
    double sw = 0.0;
    if (weighted) {
        double s = 0.0;
        for (keel::usize k = 0; k < n; ++k) s += wOf(k);
        sw = s;
    } else {
        sw = static_cast<double>(n);
    }
    for (keel::usize c = 0; c < p; ++c) {
        keel::c64 s(0.0, 0.0);
        for (keel::usize k = 0; k < n; ++k) s += wOf(k) * readX(k, c);
        mx[c] = s / sw;
    }
    for (keel::usize c = 0; c < q; ++c) {
        keel::c64 s(0.0, 0.0);
        for (keel::usize k = 0; k < n; ++k) s += wOf(k) * readY(k, c);
        my[c] = s / sw;
    }
    // coef(i,j) = (Σ conj(x̃) ỹ) / (|x̃| |ỹ|), x̃ = (x-mx)·sqrt(w).
    Value out = complexData ? Value::matrix(p, q, ValueType::COMPLEX, mr)
                            : Value::matrix(p, q, ValueType::DOUBLE, mr);
    keel::Array<double> nx(p), ny(q);
    for (keel::usize i = 0; i < p; ++i) {
        double s = 0.0;
        for (keel::usize k = 0; k < n; ++k) {
            const auto d = (readX(k, i) - mx[i]) * std::sqrt(wOf(k));
            s += keel::norm(d);
        }
        nx[i] = std::sqrt(s);
    }
    for (keel::usize j = 0; j < q; ++j) {
        double s = 0.0;
        for (keel::usize k = 0; k < n; ++k) {
            const auto d = (readY(k, j) - my[j]) * std::sqrt(wOf(k));
            s += keel::norm(d);
        }
        ny[j] = std::sqrt(s);
    }
    for (keel::usize j = 0; j < q; ++j)
        for (keel::usize i = 0; i < p; ++i) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = 0; k < n; ++k)
                s += keel::conj((readX(k, i) - mx[i]) * std::sqrt(wOf(k)))
                     * ((readY(k, j) - my[j]) * std::sqrt(wOf(k)));
            const keel::c64 c = s / (nx[i] * ny[j]);
            if (complexData) {
                out.complexDataMut()[i + j * p] = keel::c64(c.real(), c.imag());
            } else {
                out.doubleDataMut()[i + j * p] = c.real();
            }
        }
    // Clamp |coef| > 1 to the unit circle (reference behavior); corrXX
    // diagonal becomes sign(diag) — for the complex path the diagonal is
    // x̃'x̃/|x̃|² = 1 exactly, and NaN columns stay NaN.
    if (complexData) {
        keel::c64 *od = out.complexDataMut();
        for (keel::usize k = 0; k < p * q; ++k) {
            const keel::c64 c(od[k].real(), od[k].imag());
            if (keel::abs(c) > 1.0)
                od[k] = keel::c64(c.real() / keel::abs(c), c.imag() / keel::abs(c));
        }
        if (corrXX)
            for (keel::usize i = 0; i < p; ++i) {
                const keel::c64 c(od[i + i * p].real(),
                                             od[i + i * p].imag());
                od[i + i * p] = std::isnan(c.real())
                    ? keel::c64(kNaN, 0.0) : keel::c64(1.0, 0.0);
            }
    } else {
        double *od = out.doubleDataMut();
        for (keel::usize k = 0; k < p * q; ++k)
            if (std::fabs(od[k]) > 1.0) od[k] = od[k] / std::fabs(od[k]);
        if (corrXX)
            for (keel::usize i = 0; i < p; ++i)
                od[i + i * p] = std::isnan(od[i + i * p])
                    ? od[i + i * p]
                    : (od[i + i * p] < 0.0 ? -1.0 : 1.0);
    }
    return out;
}

// Pairwise Pearson with per-pair NaN removal, optional weights, complex
// support, and the per-pair observation counts (nij — the p-value
// denominator for 'pairwise').
struct PairwisePearsonOut { Value coef, nij; };
PairwisePearsonOut corrPairwisePearson(const Value &X, const Value &Y,
                                       const Value *wts, bool corrXX,
                                       bool complexData,
                                       keel::Allocator *mr)
{
    const keel::usize n = corrRowsOf(X);
    const keel::usize p = corrColsOf(X);
    const keel::usize q = corrColsOf(Y);
    Value out = complexData ? Value::matrix(p, q, ValueType::COMPLEX, mr)
                            : Value::matrix(p, q, ValueType::DOUBLE, mr);
    Value nij = Value::matrix(p, q, ValueType::DOUBLE, mr);
    double *nd = nij.doubleDataMut();

    auto readC = [](const Value &V, keel::usize idx) -> keel::c64 {
        if (V.type() == ValueType::COMPLEX)
            return keel::c64(V.complexData()[idx]);
        return keel::c64(V.elemAsDouble(idx), 0.0);
    };

    // One (xi, yj) pair over the rows where both are non-NaN.
    auto pairDot = [&](const Value &A, keel::usize ci,
                       const Value &B, keel::usize cj) {
        keel::usize m = 0;
        double sw = 0.0;
        keel::c64 sai(0.0, 0.0), sbj(0.0, 0.0);
        for (keel::usize k = 0; k < n; ++k) {
            if (elemIsNaN(A, k + ci * n) || elemIsNaN(B, k + cj * n)) continue;
            const auto a = readC(A, k + ci * n);
            const auto b = readC(B, k + cj * n);
            const double w = wts ? wts->elemAsDouble(k) : 1.0;
            sw += w; sai += w * a; sbj += w * b;
            ++m;
        }
        nd[ci + cj * p] = static_cast<double>(m);
        keel::c64 rij(kNaN, 0.0);
        if (m >= 1 && sw > 0.0) {
            const auto ma = sai / sw, mb = sbj / sw;
            keel::c64 sab(0.0, 0.0);
            double saa = 0.0, sbb = 0.0;
            for (keel::usize k = 0; k < n; ++k) {
                if (elemIsNaN(A, k + ci * n) || elemIsNaN(B, k + cj * n)) continue;
                const auto a = readC(A, k + ci * n);
                const auto b = readC(B, k + cj * n);
                const double w = wts ? wts->elemAsDouble(k) : 1.0;
                const double s = std::sqrt(w);
                const auto da = (a - ma) * s, db = (b - mb) * s;
                sab += keel::conj(da) * db;
                saa += keel::norm(da);
                sbb += keel::norm(db);
            }
            const double den = std::sqrt(saa * sbb);
            if (den > 0.0) {
                rij = sab / den;
                if (keel::abs(rij) > 1.0) rij /= keel::abs(rij);
            }
        }
        if (complexData) {
            out.complexDataMut()[ci + cj * p] = keel::c64(rij.real(), rij.imag());
        } else {
            out.doubleDataMut()[ci + cj * p] = rij.real();
        }
    };

    for (keel::usize i = 0; i < p; ++i) {
        const keel::usize j1 = corrXX ? i : q;   // corrXX: lower triangle only
        for (keel::usize j = 0; j < j1; ++j)
            pairDot(X, i, Y, j);
        if (corrXX) {
            // Diagonal: 1 unless the column is all-NaN.
            keel::usize m = 0;
            for (keel::usize k = 0; k < n; ++k)
                if (!elemIsNaN(X, k + i * n)) ++m;
            nd[i + i * p] = static_cast<double>(m);
            if (complexData) {
                out.complexDataMut()[i + i * p] = (m >= 1)
                    ? keel::c64(1.0, 0.0) : keel::c64(kNaN, 0.0);
            } else {
                out.doubleDataMut()[i + i * p] = (m >= 1) ? 1.0 : kNaN;
            }
        }
    }
    if (corrXX) {
        for (keel::usize i = 0; i < p; ++i)
            for (keel::usize j = 0; j < i; ++j) {
                if (complexData) {
                    const keel::c64 c = out.complexData()[i + j * p];
                    out.complexDataMut()[j + i * p] = keel::c64(c.real(), -c.imag());
                } else {
                    out.doubleDataMut()[j + i * p] = out.doubleData()[i + j * p];
                }
                nd[j + i * p] = nd[i + j * p];
            }
    }
    return { std::move(out), std::move(nij) };
}

// Pairwise Kendall tau-b with per-pair NaN removal and optional weights.
PairwisePearsonOut corrPairwiseKendall(const Value &X, const Value &Y,
                                       const Value *wts, bool corrXX,
                                       keel::Allocator *mr)
{
    const keel::usize n = corrRowsOf(X);
    const keel::usize p = corrColsOf(X);
    const keel::usize q = corrColsOf(Y);
    Value out = Value::matrix(p, q, ValueType::DOUBLE, mr);
    Value nij = Value::matrix(p, q, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    double *nd = nij.doubleDataMut();

    for (keel::usize i = 0; i < p; ++i) {
        const keel::usize j1 = corrXX ? (i + 1) : q;
        for (keel::usize j = 0; j < j1; ++j) {
            keel::Array<keel::u8> keep(n, 0);
            keel::usize m = 0;
            for (keel::usize k = 0; k < n; ++k) {
                if (!std::isnan(X.elemAsDouble(k + i * n)) &&
                    !std::isnan(Y.elemAsDouble(k + j * n))) {
                    keep[k] = 1;
                    ++m;
                }
            }
            nd[i + j * p] = static_cast<double>(m);
            if (m < 2) {
                od[i + j * p] = kNaN;
                continue;
            }
            Value Xa = Value::matrix(m, 1, ValueType::DOUBLE, mr);
            Value Yb = Value::matrix(m, 1, ValueType::DOUBLE, mr);
            Value Wa;
            if (wts) Wa = Value::matrix(m, 1, ValueType::DOUBLE, mr);
            for (keel::usize k = 0, w = 0; k < n; ++k) {
                if (keep[k]) {
                    Xa.doubleDataMut()[w] = X.elemAsDouble(k + i * n);
                    Yb.doubleDataMut()[w] = Y.elemAsDouble(k + j * n);
                    if (wts) Wa.doubleDataMut()[w] = wts->elemAsDouble(k);
                    ++w;
                }
            }
            const double tau = kendallTauB(Xa, 0, Yb, 0, m, wts ? &Wa : nullptr);
            od[i + j * p] = (corrXX && i == j && !std::isnan(tau)) ? 1.0 : tau;
        }
    }
    if (corrXX) {
        for (keel::usize i = 0; i < p; ++i)
            for (keel::usize j = 0; j < i; ++j) {
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
double corrPearsonTailP(CorrTail tail, double r, keel::usize n,
                        keel::Allocator *mr)
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
std::pair<double, double> corrKendallExactLU(double tau, keel::usize n,
                                             keel::Allocator *mr)
{
    if (n < 2 || std::isnan(tau)) return { kNaN, kNaN };
    const keel::usize Dmax = n * (n - 1) / 2;
    long Dobs = std::lround(static_cast<double>(Dmax) * (1.0 - tau) / 2.0);
    if (Dobs < 0) Dobs = 0;
    if (Dobs > static_cast<long>(Dmax)) Dobs = static_cast<long>(Dmax);
    keel::ArenaTemp sc = keel::get_scratch(mr);
    keel::Array<double> prob(Dmax + 1, 0.0, sc.allocator());      // P(#inversions == d)
    keel::Array<double> next(Dmax + 1, 0.0, sc.allocator());
    prob[0] = 1.0;
    keel::usize curLen = 1;                            // active prefix length
    for (keel::usize k = 1; k < n; ++k) {
        const keel::usize newLen = curLen + k;
        std::fill(next.begin(), next.begin() + newLen, 0.0);
        const double w = 1.0 / static_cast<double>(k + 1);
        for (keel::usize d = 0; d < curLen; ++d) {
            const double pv = prob[d] * w;
            for (keel::usize s = 0; s <= k; ++s) next[d + s] += pv;
        }
        keel::swap(prob, next);                       // O(1), same arena
        curLen = newLen;
    }
    double lower = 0.0, upper = 0.0;
    for (long d = 0; d <= Dobs; ++d) lower += prob[static_cast<keel::usize>(d)];
    for (long d = Dobs; d < static_cast<long>(curLen); ++d)
        upper += prob[static_cast<keel::usize>(d)];
    return { lower, upper };
}

// Exact Spearman {lower, upper} by enumerating permutations (small n).
// D = Σ(rankX_i - rankY_i)²; under H0 it is the displacement of a random perm.
std::pair<double, double> corrSpearmanExactLU(double rho, keel::usize n,
                                              keel::Allocator *mr)
{
    const double scale = static_cast<double>(n) * (static_cast<double>(n) * n - 1.0) / 6.0;
    long Dobs = std::lround((1.0 - rho) * scale);
    if (Dobs < 0) Dobs = 0;
    keel::ArenaTemp sc = keel::get_scratch(mr);
    keel::Array<int> perm(n, sc.allocator());
    for (keel::usize i = 0; i < n; ++i) perm[i] = static_cast<int>(i);
    keel::usize total = 0, leCount = 0, geCount = 0;
    do {
        long D = 0;
        for (keel::usize i = 0; i < n; ++i) {
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
double as89Upper(double D, keel::usize n, double n3c)
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

double as89Tail(CorrTail tail, double D, keel::usize n, double n3c)
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
double corrSpearmanNoTiesTailP(CorrTail tail, double rho, keel::usize n,
                               keel::Allocator *mr)
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
                                                 keel::usize n, double Dobs,
                                                 keel::Allocator *mr)
{
    keel::ArenaTemp sc = keel::get_scratch(mr);
    keel::Array<int> perm(n, sc.allocator());
    for (keel::usize i = 0; i < n; ++i) perm[i] = static_cast<int>(i);
    keel::usize total = 0, leCount = 0, geCount = 0;
    do {
        double D = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
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
                  bool wantP, keel::Allocator *mr,
                  Value &coef, Value &p)
{
    const keel::usize n = corrRowsOf(X);
    const keel::usize p1 = corrColsOf(X);
    const keel::usize p2 = corrXX ? p1 : corrColsOf(Y);
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
    for (keel::usize c = 0; c < p1; ++c) anyTies = anyTies || (xadj[c] > 0.0);
    if (!corrXX)
        for (keel::usize c = 0; c < p2; ++c) anyTies = anyTies || (yadj[c] > 0.0);

    if (!wantP || !anyTies) {
        coef = corrXX ? corr_xx(xr, mr) : corr_xy(xr, yr, mr);
        if (!wantP) return;
        p = Value::matrix(p1, p2, ValueType::DOUBLE, mr);
        double *pd = p.doubleDataMut();
        const double *rd = coef.doubleData();
        for (keel::usize j = 0; j < p2; ++j)
            for (keel::usize i = 0; i < p1; ++i) {
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
    for (keel::usize i = 0; i < p1; ++i) {
        const keel::usize j1 = corrXX ? i + 1 : p2;
        for (keel::usize j = 0; j < j1; ++j) {
            double D = 0.0;
            for (keel::usize k = 0; k < n; ++k) {
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
    for (keel::usize k = 0; k < p1 * p2; ++k)
        if (std::fabs(cd[k]) > 1.0) cd[k] = cd[k] / std::fabs(cd[k]);
    if (corrXX) {
        for (keel::usize i = 0; i < p1; ++i)
            for (keel::usize j = 0; j < i; ++j) {
                cd[j + i * p1] = cd[i + j * p1];
                pd[j + i * p1] = pd[i + j * p1];
            }
        for (keel::usize i = 0; i < p1; ++i) {
            cd[i + i * p1] = std::isnan(cd[i + i * p1])
                ? cd[i + i * p1]
                : (cd[i + i * p1] < 0.0 ? -1.0 : 1.0);
            pd[i + i * p1] = std::isnan(cd[i + i * p1]) ? kNaN : 1.0;
        }
    }
}

// Pairwise Spearman with per-pair NaN removal, per-pair midranks, optional
// weights, and per-pair p-values.
void corrPairwiseSpearman(const Value &X, const Value &Y, const Value *wts,
                          bool corrXX, CorrTail tail, bool wantP,
                          keel::Allocator *mr,
                          Value &coef, Value &p)
{
    const keel::usize n = corrRowsOf(X);
    const keel::usize p1 = corrColsOf(X);
    const keel::usize p2 = corrXX ? p1 : corrColsOf(Y);
    coef = Value::matrix(p1, p2, ValueType::DOUBLE, mr);
    double *cd = coef.doubleDataMut();
    if (wantP) p = Value::matrix(p1, p2, ValueType::DOUBLE, mr);
    double *pd = wantP ? p.doubleDataMut() : nullptr;

    for (keel::usize i = 0; i < p1; ++i) {
        const keel::usize j1 = corrXX ? (i + 1) : p2;
        for (keel::usize j = 0; j < j1; ++j) {
            keel::Array<keel::u8> keep(n, 0);
            keel::usize m = 0;
            for (keel::usize k = 0; k < n; ++k) {
                if (!std::isnan(X.elemAsDouble(k + i * n)) &&
                    !std::isnan(Y.elemAsDouble(k + j * n))) {
                    keep[k] = 1;
                    ++m;
                }
            }
            if (m < 2) {
                cd[i + j * p1] = kNaN;
                if (pd) pd[i + j * p1] = kNaN;
                continue;
            }
            Value Xa = Value::matrix(m, 1, ValueType::DOUBLE, mr);
            Value Yb = Value::matrix(m, 1, ValueType::DOUBLE, mr);
            Value Wa;
            if (wts) Wa = Value::matrix(m, 1, ValueType::DOUBLE, mr);
            for (keel::usize k = 0, w = 0; k < n; ++k) {
                if (keep[k]) {
                    Xa.doubleDataMut()[w] = X.elemAsDouble(k + i * n);
                    Yb.doubleDataMut()[w] = Y.elemAsDouble(k + j * n);
                    if (wts) Wa.doubleDataMut()[w] = wts->elemAsDouble(k);
                    ++w;
                }
            }
            if (wts) {
                Value xr = rankColumnsWeighted(Xa, &Wa, mr);
                Value yr = (corrXX && i == j) ? xr : rankColumnsWeighted(Yb, &Wa, mr);
                Value r1 = pearsonMatrixFull(xr, yr, &Wa, corrXX && (i == j), false, mr);
                cd[i + j * p1] = r1.elemAsDouble(0);
                if (pd) pd[i + j * p1] = kNaN;
            } else {
                Value r1, pv1;
                spearmanFull(Xa, Yb, corrXX && (i == j), tail, wantP, mr, r1, pv1);
                cd[i + j * p1] = r1.elemAsDouble(0);
                if (pd) pd[i + j * p1] = pv1.elemAsDouble(0);
            }
        }
    }
    if (corrXX) {
        for (keel::usize i = 0; i < p1; ++i)
            for (keel::usize j = 0; j < i; ++j) {
                cd[j + i * p1] = cd[i + j * p1];
                if (pd) pd[j + i * p1] = pd[i + j * p1];
            }
    }
}

// Large-n Kendall: tie-free normal approximation z = 3τ√(n(n-1)/2(2n+5)).
double corrKendallNormalP(CorrTail tail, double tau, keel::usize n)
{
    if (n < 2 || std::isnan(tau)) return kNaN;
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
Value corrPValueMatrix(const Value &r, const Value *nij, keel::usize n,
                       CorrType ct, CorrTail tail, bool corrXX,
                       keel::Allocator *mr)
{
    const keel::usize p = static_cast<keel::usize>(r.dims().dim(0));
    const keel::usize q = static_cast<keel::usize>(r.dims().dim(1));
    Value P = Value::matrix(p, q, ValueType::DOUBLE, mr);
    double *pd = P.doubleDataMut();
    const double *rd = r.doubleData();
    const double *nd = nij ? nij->doubleData() : nullptr;
    for (keel::usize j = 0; j < q; ++j)
        for (keel::usize i = 0; i < p; ++i) {
            const double rij = rd[i + j * p];
            if (corrXX && i == j) {
                pd[i + j * p] = std::isnan(rij) ? kNaN : 1.0;
                continue;
            }
            const keel::usize nn = nd ? static_cast<keel::usize>(nd[i + j * p]) : n;
            double pv;
            if (ct == CorrType::Pearson)
                pv = corrPearsonTailP(tail, rij, nn, mr);
            else if (ct == CorrType::Kendall && nn <= 100) {
                const auto [lo, up] = corrKendallExactLU(rij, nn, mr);
                pv = rankTailFromLU(tail, lo, up);
            } else {
                pv = corrKendallNormalP(tail, rij, nn);
            }
            pd[i + j * p] = pv;
        }
    return P;
}


// ── small shared bits ────────────────────────────────────────────────

bool allFinite(const Value &X)
{
    const keel::usize cnt = X.numel();
    for (keel::usize k = 0; k < cnt; ++k)
        if (!std::isfinite(X.elemAsDouble(k))) return false;
    return true;
}

Value nanMatrix(keel::usize r, keel::usize c, bool single,
                keel::Allocator *mr)
{
    Value out = single ? Value::matrix(r, c, ValueType::SINGLE, mr)
                       : Value::matrix(r, c, ValueType::DOUBLE, mr);
    const keel::usize cnt = r * c;
    for (keel::usize k = 0; k < cnt; ++k) {
        if (single) out.singleDataMut()[k] =
            std::numeric_limits<float>::quiet_NaN();
        else out.doubleDataMut()[k] = kNaN;
    }
    return out;
}

Value castToSingle(const Value &X, keel::Allocator *mr)
{
    const keel::usize r = static_cast<keel::usize>(X.dims().dim(0));
    const keel::usize c = static_cast<keel::usize>(X.dims().dim(1));
    Value out = Value::matrix(r, c, ValueType::SINGLE, mr);
    const keel::usize cnt = r * c;
    for (keel::usize k = 0; k < cnt; ++k)
        out.singleDataMut()[k] = static_cast<float>(X.elemAsDouble(k));
    return out;
}

// The compute kernels under these full entries are double/complex only;
// single (and integer-promoted) inputs materialize as double matrices —
// compute runs in double, the output is cast back down. Complex storage
// passes through untouched.
Value asComputeDouble(const Value &X, keel::Allocator *mr)
{
    if (X.type() == ValueType::DOUBLE || X.type() == ValueType::COMPLEX)
        return X;
    const keel::usize r = static_cast<keel::usize>(X.dims().dim(0));
    const keel::usize c = static_cast<keel::usize>(X.dims().dim(1));
    Value out = Value::matrix(r, c, ValueType::DOUBLE, mr);
    const keel::usize cnt = r * c;
    for (keel::usize k = 0; k < cnt; ++k)
        out.doubleDataMut()[k] = X.elemAsDouble(k);
    return out;
}

// Listwise NaN-row drop over 1-3 row-aligned matrices (partialcorr's
// 'complete' — and 'pairwise' in the combinedXZ form, where every column
// is a control for every pair so the two coincide).
void dropNaNRowsAny(keel::Array<Value> &M, keel::Allocator *mr)
{
    const keel::usize n = static_cast<keel::usize>(M[0].dims().dim(0));
    keel::Array<keel::u8> keep(n, 1);
    for (keel::usize t = 0; t < M.size(); ++t) {
        const keel::usize nn = static_cast<keel::usize>(M[t].dims().dim(0));
        const keel::usize p = static_cast<keel::usize>(M[t].dims().dim(1));
        for (keel::usize r = 0; r < std::min(nn, n); ++r)
            for (keel::usize c = 0; c < p && keep[r]; ++c)
                if (std::isnan(M[t].elemAsDouble(r + c * nn))) keep[r] = 0;
    }
    const keel::usize m = static_cast<keel::usize>(
        std::count(keep.begin(), keep.end(), 1));
    for (auto &Mt : M) {
        const keel::usize nn = static_cast<keel::usize>(Mt.dims().dim(0));
        const keel::usize p = static_cast<keel::usize>(Mt.dims().dim(1));
        Value out = Value::matrix(m, p, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize c = 0; c < p; ++c)
            for (keel::usize r = 0, w = 0; r < nn; ++r)
                if (keep[r]) od[w++ + c * m] = Mt.elemAsDouble(r + c * nn);
        Mt = std::move(out);
    }
}

// Numerical column rank of centered Z (m × pZ), used to compute df = m - dz0 - 2
// in partialcorr p-values.
keel::usize centeredRank(const Value &Z, keel::Allocator *mr)
{
    const keel::usize m = static_cast<keel::usize>(Z.dims().dim(0));
    const keel::usize pZ = static_cast<keel::usize>(Z.dims().dim(1));
    if (m < 2 || pZ == 0) return 0;
    keel::ArenaTemp sc = keel::get_scratch(mr);
    keel::Array<double> A(m * pZ, 0.0, sc.allocator());
    const double *zd = Z.doubleData();
    double maxNorm = 0.0;
    for (keel::usize j = 0; j < pZ; ++j) {
        double mean = 0.0;
        for (keel::usize i = 0; i < m; ++i) mean += zd[i + j * m];
        mean /= static_cast<double>(m);
        double colNorm2 = 0.0;
        for (keel::usize i = 0; i < m; ++i) {
            const double d = zd[i + j * m] - mean;
            A[i + j * m] = d;
            colNorm2 += d * d;
        }
        maxNorm = std::max(maxNorm, std::sqrt(colNorm2));
    }
    if (maxNorm == 0.0) return 0;
    const double tol = static_cast<double>(std::max(m, pZ))
                     * std::numeric_limits<double>::epsilon() * maxNorm;
    // Modified Gram-Schmidt / column-pivoted QR rank count.
    keel::usize rnk = 0;
    keel::Array<keel::u8> used(pZ, 0, sc.allocator());
    for (keel::usize step = 0; step < std::min(m, pZ); ++step) {
        keel::usize bestCol = 0;
        double bestNorm = -1.0;
        for (keel::usize j = 0; j < pZ; ++j) {
            if (used[j]) continue;
            double s = 0.0;
            for (keel::usize i = 0; i < m; ++i) s += A[i + j * m] * A[i + j * m];
            const double nrm = std::sqrt(s);
            if (nrm > bestNorm) { bestNorm = nrm; bestCol = j; }
        }
        if (bestNorm <= tol) break;
        used[bestCol] = 1;
        ++rnk;
        for (keel::usize i = 0; i < m; ++i) A[i + bestCol * m] /= bestNorm;
        for (keel::usize j = 0; j < pZ; ++j) {
            if (used[j]) continue;
            double dot = 0.0;
            for (keel::usize i = 0; i < m; ++i)
                dot += A[i + bestCol * m] * A[i + j * m];
            for (keel::usize i = 0; i < m; ++i)
                A[i + j * m] -= dot * A[i + bestCol * m];
        }
    }
    return rnk;
}

void corr_full_argcheck(const Value &x, const Value *Y)
{
    const keel::usize n = corrRowsOf(x);
    if (Y != nullptr && corrRowsOf(*Y) != n)
        throw Error(keel::format("corr: X and Y must have the same number of rows "
                                 "(X has {}, Y has {}).", n, corrRowsOf(*Y)),
                    0, 0, "corr", "", "numkit:corr:inputSizeMismatch");
}

} // namespace

// ── corr full entry ───────────────────────────────────────────────────

CorrOut corr_full(const Value &x, const Value *Y, keel::ArrayRef<Value> nv,
                  bool wantP, keel::Allocator *mr)
{
    corr_full_argcheck(x, Y);
    const keel::usize n = corrRowsOf(x);
    const keel::usize p1 = corrColsOf(x);
    const bool corrXX = (Y == nullptr);
    const keel::usize p2 = corrXX ? p1 : corrColsOf(*Y);
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
    static constexpr keel::StringRef kNv[] = { "type", "rows", "tail",
                                               "weights" };
    argcheck::forEachNvPair(
        nv, 0, keel::ArrayRef<keel::StringRef>(kNv), "corr",
        "numkit:corr:badParamName", [&](keel::usize hit, const Value &v) {
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
            throw Error(keel::format("corr: Weights must have size {}x1, "
                                     "but the received array is {}x{}.",
                                     n, wtsRaw.dims().rows(), wtsRaw.dims().cols()),
                        0, 0, "corr", "", "numkit:corr:incorrectSize");
        for (keel::usize k = 0; k < n; ++k) {
            const double w = wtsRaw.elemAsDouble(k);
            if (!(w >= 0.0))
                throw Error(keel::format("corr: expected Weights to be nonnegative (got {} "
                                         "at position {}).", w, k + 1),
                            0, 0, "corr", "", "numkit:corr:expectedNonnegative");
        }
        weighted = true;
        wts = wtsRaw;
    }

    // Rows value + per-mode row prep (zero-weight drop rules differ).
    CorrRows rows = CorrRows::All;
    if (!rowsRaw.isEmpty()) {
        static constexpr keel::StringRef kRows[] = { "all", "complete",
                                                     "pairwise" };
        const int m = argcheck::matchStringChoice(
            rowsRaw, keel::ArrayRef<keel::StringRef>(kRows));
        if (m < 0) {
            keel::String rowsStr = rowsRaw.toString();
            throw Error(keel::format("corr: the Rows parameter value must be 'all', "
                                     "'complete', or 'pairwise', got '{}'.",
                                     rowsStr.as_ref()),
                        0, 0, "corr", "", "numkit:corr:unknownRows");
        }
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
            keel::Array<keel::u8> keep(n, 1);
            for (keel::usize k = 0; k < n; ++k)
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
    const keel::usize nn = corrRowsOf(Xe);

    CorrOut out;
    if (nn < 2) {
        out.rho = nanMatrix(p1, p2, anySingle, mr);
        if (wantP) out.p = nanMatrix(p1, p2, anySingle, mr);
        return out;
    }

    // Type/Tail VALUES validated here — after the n<2 return.
    CorrType ct = CorrType::Pearson;
    if (!typeRaw.isEmpty()) {
        static constexpr keel::StringRef kTypes[] = { "pearson", "kendall",
                                                      "spearman" };
        const int m = argcheck::matchStringChoice(
            typeRaw, keel::ArrayRef<keel::StringRef>(kTypes));
        if (m < 0) {
            keel::String typeStr = typeRaw.toString();
            throw Error(keel::format("corr: the Type parameter value must be 'Pearson', "
                                     "'Kendall', or 'Spearman', got '{}'.",
                                     typeStr.as_ref()),
                        0, 0, "corr", "", "numkit:corr:unknownType");
        }
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
    }

    // Compute rho (Spearman owns its p-values — its tie-corrected branch
    // needs the ranks, so coef and p come out of one function).
    Value r;
    Value pmat;
    bool pDone = false;
    Value nijStore;
    const Value *nij = nullptr;
    if (rows == CorrRows::Pairwise) {
        if (ct == CorrType::Pearson) {
            auto pw = corrPairwisePearson(Xe, corrXX ? Xe : Ye,
                                          wActive ? &wEff : nullptr, corrXX,
                                          complexData, mr);
            r = std::move(pw.coef);
            nijStore = std::move(pw.nij);
            nij = &nijStore;
        } else if (ct == CorrType::Kendall) {
            auto pw = corrPairwiseKendall(Xe, corrXX ? Xe : Ye,
                                          wActive ? &wEff : nullptr, corrXX, mr);
            r = std::move(pw.coef);
            nijStore = std::move(pw.nij);
            nij = &nijStore;
        } else {
            corrPairwiseSpearman(Xe, corrXX ? Xe : Ye,
                                 wActive ? &wEff : nullptr, corrXX,
                                 tail, wantP, mr, r, pmat);
            pDone = wantP;
        }
    } else if (ct == CorrType::Spearman) {
        if (wActive) {
            Value xr = rankColumnsWeighted(Xe, &wEff, mr);
            Value yr = corrXX ? xr : rankColumnsWeighted(Ye, &wEff, mr);
            r = pearsonMatrixFull(xr, yr, &wEff, corrXX, false, mr);
        } else {
            spearmanFull(Xe, corrXX ? Xe : Ye, corrXX, tail, wantP, mr, r, pmat);
            pDone = wantP;
        }
    } else if (complexData) {
        r = pearsonMatrixFull(Xe, corrXX ? Xe : Ye, wActive ? &wEff : nullptr,
                              corrXX, true, mr);
    } else if (wActive) {
        if (ct == CorrType::Kendall) {
            r = kendallMatrix(Xe, corrXX ? Xe : Ye, &wEff, mr);
        } else {
            r = pearsonMatrixFull(Xe, corrXX ? Xe : Ye, &wEff, corrXX, false, mr);
        }
    } else {
        if (ct == CorrType::Kendall) {
            r = kendallMatrix(Xe, corrXX ? Xe : Ye, nullptr, mr);
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
    if (anySingle && !out.rho.isComplex()) out.rho = castToSingle(out.rho, mr);
    if (anySingle && !out.p.isEmpty() && !out.p.isComplex())
        out.p = castToSingle(out.p, mr);
    return out;
}

// ── partialcorr full entry ────────────────────────────────────────────

CorrOut partialcorr_full(const Value &x, const Value *y, const Value *z,
                         keel::ArrayRef<Value> nv, bool wantP,
                         keel::Allocator *mr)
{
    const auto matricesErr = [](keel::StringRef what) {
        throw Error(keel::format("partialcorr: all input arrays must be numeric matrices ({}).",
                                 what),
                    0, 0, "partialcorr", "",
                    "numkit:partialcorr:inputsMustBeMatrices");
    };
    const auto isMatrixNumeric = [](const Value &M) {
        return M.isNumeric() && M.dims().ndim() <= 2 && M.numel() > 0;
    };
    const auto mismatchErr = [](keel::StringRef which,
                                keel::usize got, keel::usize want) {
        throw Error(keel::format("partialcorr: all input arrays must have the same number "
                                 "of rows ({} has {} rows, expected {}).",
                                 which, got, want),
                    0, 0, "partialcorr", "",
                    "numkit:partialcorr:inputSizeMismatch");
    };
    const auto describe = [](const Value &M) {
        return keel::format("{} {}x{}", mtypeName(M.type()),
                            M.dims().dim(0), M.dims().dim(1));
    };

    if (!isMatrixNumeric(x)) {
        keel::String dsc = describe(x);
        matricesErr(keel::format("X is {}", dsc.as_ref()).as_ref());
    }
    const keel::usize n = static_cast<keel::usize>(x.dims().dim(0));
    const keel::usize d = static_cast<keel::usize>(x.dims().dim(1));
    if (y) {
        // (X, Y, Z): Y's row count is blamed BEFORE Y's matrix-ness
        // (reference order), Z is checked after its form is known.
        if (static_cast<keel::usize>(y->dims().dim(0)) != n)
            mismatchErr("Y", static_cast<keel::usize>(y->dims().dim(0)), n);
        if (!isMatrixNumeric(*y)) {
            keel::String dsc = describe(*y);
            matricesErr(keel::format("Y is {}", dsc.as_ref()).as_ref());
        }
        if (!isMatrixNumeric(*z)) {
            keel::String dsc = describe(*z);
            matricesErr(keel::format("Z is {}", dsc.as_ref()).as_ref());
        }
        if (static_cast<keel::usize>(z->dims().dim(0)) != n)
            mismatchErr("Z", static_cast<keel::usize>(z->dims().dim(0)), n);
    } else if (z) {
        if (!isMatrixNumeric(*z)) {
            keel::String dsc = describe(*z);
            matricesErr(keel::format("Z is {}", dsc.as_ref()).as_ref());
        }
        if (static_cast<keel::usize>(z->dims().dim(0)) != n)
            mismatchErr("Z", static_cast<keel::usize>(z->dims().dim(0)), n);
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

    // NV: Rows / Type / Tail. Kendall is a parity error.
    CorrRows rows = CorrRows::All;
    CorrType ct = CorrType::Pearson;
    CorrTail tail = CorrTail::Both;
    static constexpr keel::StringRef kNv[] = { "rows", "type", "tail" };
    argcheck::forEachNvPair(
        nv, 0, keel::ArrayRef<keel::StringRef>(kNv), "partialcorr",
        "numkit:partialcorr:badParamName", [&](keel::usize hit, const Value &v) {
            switch (hit) {
            case 0: {
                static constexpr keel::StringRef kRows[] = { "all",
                                                             "complete",
                                                             "pairwise" };
                const int m = argcheck::matchStringChoice(
                    v, keel::ArrayRef<keel::StringRef>(kRows));
                if (m < 0) {
                    keel::String vStr = v.toString();
                    throw Error(keel::format("partialcorr: the Rows parameter value must "
                                             "be 'all', 'complete', or 'pairwise', got '{}'.",
                                             vStr.as_ref()),
                                0, 0, "partialcorr", "",
                                "numkit:partialcorr:unknownRows");
                }
                rows = static_cast<CorrRows>(m);
                break;
            }
            case 1: {
                static constexpr keel::StringRef kTypes[] = { "pearson",
                                                              "kendall",
                                                              "spearman" };
                const int m = argcheck::matchStringChoice(
                    v, keel::ArrayRef<keel::StringRef>(kTypes));
                if (m < 0) {
                    keel::String vStr = v.toString();
                    throw Error(keel::format("partialcorr: the Type parameter value must "
                                             "be 'Pearson', 'Kendall', or 'Spearman', "
                                             "got '{}'.", vStr.as_ref()),
                                0, 0, "partialcorr", "",
                                "numkit:partialcorr:unknownType");
                }
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
                tail = parseTailChoice(v, "partialcorr");
                break;
            default: break;
            }
        });

    const bool combinedXZ = (z == nullptr);   // partialcorr(X, ...)
    const bool crossCorr = (y != nullptr);
    const keel::usize dx = d;
    const keel::usize dy = y ? static_cast<keel::usize>(y->dims().dim(1)) : d;

    auto makeNanOut = [&]() -> CorrOut {
        CorrOut co;
        co.rho = nanMatrix(dx, dy, wantSingle, mr);
        if (wantP) co.p = nanMatrix(dx, dy, wantSingle, mr);
        return co;
    };

    auto buildPMatrix = [&](const Value &R, long df) -> Value {
        Value P = Value::matrix(dx, dy, ValueType::DOUBLE, mr);
        double *pd = P.doubleDataMut();
        const double *rd = R.doubleData();
        for (keel::usize j = 0; j < dy; ++j) {
            for (keel::usize i = 0; i < dx; ++i) {
                const double rij = rd[i + j * dx];
                if (!crossCorr && i == j) {
                    pd[i + j * dx] = std::isnan(rij) ? kNaN : 1.0;
                    continue;
                }
                if (df < 1 || std::isnan(rij)) {
                    pd[i + j * dx] = kNaN;
                } else {
                    pd[i + j * dx] = corrPearsonTailP(
                        tail, rij, static_cast<keel::usize>(df) + 2, mr);
                }
            }
        }
        return P;
    };

    Value R, P;
    if (combinedXZ) {
        // 'complete' and 'pairwise' coincide here: every column of X is a
        // control for every pair, so row removal is necessarily listwise.
        keel::Array<Value> M;
        M.push_back(xD);
        if (rows != CorrRows::All) dropNaNRowsAny(M, mr);
        const keel::usize m = static_cast<keel::usize>(M[0].dims().dim(0));
        const long dz = (d >= 2) ? static_cast<long>(d - 2) : 0;
        const long df = static_cast<long>(m) - dz - 2;
        if (!allFinite(M[0]) || m < 2 || df < 0)
            return makeNanOut();
        if (ct == CorrType::Spearman) M[0] = tiedrank(M[0], mr).first;
        R = partialcorr_xx(M[0], mr);
        if (wantP) P = buildPMatrix(R, df);
    } else if (rows == CorrRows::Pairwise) {
        const keel::usize dz = static_cast<keel::usize>(zD.dims().dim(1));
        const Value &yRef = y ? yD : xD;
        R = Value::matrix(dx, dy, ValueType::DOUBLE, mr);
        double *rd = R.doubleDataMut();
        if (wantP) P = Value::matrix(dx, dy, ValueType::DOUBLE, mr);
        double *pd = wantP ? P.doubleDataMut() : nullptr;

        for (keel::usize i = 0; i < dx; ++i) {
            const keel::usize j1 = crossCorr ? dy : (i + 1);
            for (keel::usize j = 0; j < j1; ++j) {
                keel::Array<keel::u8> keep(n, 1);
                keel::usize m = 0;
                for (keel::usize r = 0; r < n; ++r) {
                    if (std::isnan(xD.elemAsDouble(r + i * n)) ||
                        std::isnan(yRef.elemAsDouble(r + j * n))) {
                        keep[r] = 0;
                        continue;
                    }
                    for (keel::usize c = 0; c < dz; ++c) {
                        if (std::isnan(zD.elemAsDouble(r + c * n))) {
                            keep[r] = 0;
                            break;
                        }
                    }
                    if (keep[r]) ++m;
                }
                if (m < 2) {
                    rd[i + j * dx] = kNaN;
                    if (pd) pd[i + j * dx] = kNaN;
                    continue;
                }
                Value xSub = Value::matrix(m, 1, ValueType::DOUBLE, mr);
                Value ySub = Value::matrix(m, 1, ValueType::DOUBLE, mr);
                Value zSub = Value::matrix(m, dz, ValueType::DOUBLE, mr);
                for (keel::usize r = 0, w = 0; r < n; ++r) {
                    if (!keep[r]) continue;
                    xSub.doubleDataMut()[w] = xD.elemAsDouble(r + i * n);
                    ySub.doubleDataMut()[w] = yRef.elemAsDouble(r + j * n);
                    for (keel::usize c = 0; c < dz; ++c)
                        zSub.doubleDataMut()[w + c * m] = zD.elemAsDouble(r + c * n);
                    ++w;
                }
                if (!allFinite(xSub) || !allFinite(ySub) || !allFinite(zSub)) {
                    rd[i + j * dx] = kNaN;
                    if (pd) pd[i + j * dx] = kNaN;
                    continue;
                }
                if (ct == CorrType::Spearman) {
                    xSub = tiedrank(xSub, mr).first;
                    ySub = tiedrank(ySub, mr).first;
                    zSub = tiedrank(zSub, mr).first;
                }
                const keel::usize dz0 = centeredRank(zSub, mr);
                const long df = static_cast<long>(m) - static_cast<long>(dz0) - 2;
                if (df < 0) {
                    rd[i + j * dx] = kNaN;
                    if (pd) pd[i + j * dx] = kNaN;
                    continue;
                }
                if (!crossCorr && i == j) {
                    rd[i + i * dx] = 1.0;
                    if (pd) pd[i + i * dx] = 1.0;
                    continue;
                }
                Value r1 = partialcorr_of(xSub, ySub, zSub, mr);
                double rij = r1.elemAsDouble(0);
                if (std::fabs(rij) > 1.0) rij = (rij > 0.0) ? 1.0 : -1.0;
                rd[i + j * dx] = rij;
                if (pd) {
                    pd[i + j * dx] = (df >= 1 && !std::isnan(rij))
                        ? corrPearsonTailP(tail, rij, static_cast<keel::usize>(df) + 2, mr)
                        : kNaN;
                }
            }
        }
        if (!crossCorr) {
            for (keel::usize i = 0; i < dx; ++i)
                for (keel::usize j = 0; j < i; ++j) {
                    rd[j + i * dx] = rd[i + j * dx];
                    if (pd) pd[j + i * dx] = pd[i + j * dx];
                }
        }
    } else {
        Value Xv = xD, Yv, Zv = zD;
        if (y) Yv = yD;
        if (rows == CorrRows::Complete) {
            keel::Array<Value> M;
            M.push_back(xD);
            if (y) M.push_back(yD);
            M.push_back(zD);
            dropNaNRowsAny(M, mr);
            Xv = std::move(M[0]);
            if (y) Yv = std::move(M[1]);
            Zv = std::move(M.back());
        }
        const keel::usize m = static_cast<keel::usize>(Zv.dims().dim(0));
        if (!allFinite(Zv) || m < 2)
            return makeNanOut();
        if (ct == CorrType::Spearman) {
            Xv = tiedrank(Xv, mr).first;
            Zv = tiedrank(Zv, mr).first;
            if (y) Yv = tiedrank(Yv, mr).first;
        }
        const keel::usize dz0 = centeredRank(Zv, mr);
        const long df = static_cast<long>(m) - static_cast<long>(dz0) - 2;
        if (df < 0)
            return makeNanOut();
        R = y ? partialcorr_of(Xv, Yv, Zv, mr)
              : partialcorr_xz(Xv, Zv, mr);
        if (wantP) P = buildPMatrix(R, df);
    }

    CorrOut out;
    out.rho = wantSingle ? castToSingle(R, mr) : std::move(R);
    if (wantP) out.p = wantSingle ? castToSingle(P, mr) : std::move(P);
    return out;
}

} // namespace numkit::stats
