// toolboxes/stats/src/descriptive/cleaning.cpp
//
// Data-cleaning family: the full parse+validate+compute entries for
// detrend / isoutlier / rmoutliers / fillmissing / rmmissing /
// standardizeMissing. Contracts were read from the reference engine's
// datafun sources (detrend.m, parseIsOutlierInput.m, locateoutliers.m,
// rmMissingOutliers.m, ismissingKernel.m, fillmissing.m) and frozen by
// the probe batteries build/probe70{,b,c,d}:
//
//   * detrend takes float arrays ONLY (single/double, complex allowed);
//     the degree is a nonneg integer / logical scalar or a prefix string
//     ('cons'.. 'constant', 'l'.. 'linear'); breakpoints are sample
//     positions (the reference shifts numeric bp by -1 onto its default
//     0-based abscissa) and may be fractional or logical; nanflags
//     select per-run NaN-aware fitting (default: any NaN poisons the
//     run); 'Continuous' (default true) selects piecewise vs
//     segment-wise fitting; SINGLE results stay SINGLE (computed in
//     double, narrowed — residuals differ from the single-computed
//     reference at single-eps scale).
//   * isoutlier / rmoutliers take REAL single/double only; methods
//     match case-insensitive prefixes (validatestring semantics, NV
//     names participate); grubbs/gesd report thresholds as a STATIC
//     bound rule from the final / selected iteration (locateoutliers.m)
//     — not the removal set; movmedian/movmean thresholds and centers
//     are per-element arrays ('omitnan' moving stats, shrunk windows).
//   * rmmissing removes ROWS (or columns with dim=2 / row-vector input)
//     and returns the removed-rows mask as 2nd output; integer, logical
//     and char inputs pass through unchanged (no representable
//     missing); single stays single; complex is supported.
//   * fillmissing 'constant' REQUIRES its value (scalar or per-slice
//     vector); 'movmean'/'movmedian' take a window and fill gaps from
//     the NaN-omitted moving statistic; EndValues accepts
//     extrap/previous/next/nearest/none or a numeric constant, is legal
//     with every fill method, and only touches the edges a direction
//     can supply; the 2nd output is the logical map of filled
//     locations; single stays single.
//   * standardizeMissing indicators are a numeric/logical VECTOR (any
//     compatible element matches, NaN indicators also match NaN);
//     integer/logical/char arrays flow through UNCHANGED; single stays
//     single.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/format.h>
#include <keel/function_ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/builtin/datatypes.hpp>   // standardizeMissing (live bare-name impl)
#include <numkit/stats/distributions/normal.hpp>     // norminv (via descriptive_detail.hpp)
#include <numkit/stats/distributions/students_t.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>

#include <algorithm>
#include <cstdio>

#include <cmath>
#include <limits>

#include "descriptive_extras_detail.hpp"
#include "descriptive_detail.hpp"   // narrowToSingle

namespace numkit::stats {
namespace {

keel::String lowerCopy(keel::StringRef sv)
{
    keel::String s = keel::String::from(sv);
    for (char &c : s)
        c = keel::to_ascii_lower(c);
    return s;
}

bool startsWith(keel::StringRef text, keel::StringRef prefix)
{
    return prefix.size() <= text.size() && text.substr(0, prefix.size()) == prefix;
}

bool isStringValue(const Value &v) { return v.isChar() || v.isString(); }

// ── shared input gates ────────────────────────────────────────────────
// The class contracts route through the numkit::argcheck class-set gate
// (consolidation Phase A, 2026-09-19): mechanism in argcheck.hpp, the
// probed masks stay here beside the tests that pin them.
//   * detrend: single/double, complex allowed (probe70 D18/D19)
//   * isoutlier / rmoutliers: single/double, REAL (probe70 I01/I17)
using argcheck::ClsDbl;
using argcheck::ClsSgl;
using argcheck::ClsCpx;
using argcheck::classMaskOf;

inline constexpr argcheck::ClassMask kFloatComplex =
    classMaskOf(ClsDbl) | classMaskOf(ClsSgl) | classMaskOf(ClsCpx);
inline constexpr argcheck::ClassMask kFloatReal =
    classMaskOf(ClsDbl) | classMaskOf(ClsSgl);

// Window semantics shared by isoutlier('mov*') and fillmissing('mov*'):
// scalar k > 0 → centered (odd k: (k-1)/2 each side; even: one more in
// back); 2-element [back forward] → as given.
void parseWindowHalfSpans(const Value &wl, const char *fn, long &hb, long &hf,
                          keel::f64 *hbSpan = nullptr, keel::f64 *hfSpan = nullptr,
                          bool *twoElem = nullptr)
{
    const bool numeric = wl.isNumeric() && !wl.isComplex();
    auto bad = [&]() {
        throw Error(keel::format("{}: window length must be a positive "
                                 "scalar or a 2-element nonnegative [back forward] "
                                 "vector of finite values", fn),
                    0, 0, fn, "", keel::format("numkit:{}:invalidWindow", fn));
    };
    if (!(numeric || wl.isLogical())) bad();
    if (wl.numel() == 1) {
        const keel::f64 k = wl.elemAsDouble(0);
        if (!(k > 0.0) || !std::isfinite(k)) bad();
        const long ki = keel::as<long>(k);
        if (ki % 2 == 1) { hb = hf = (ki - 1) / 2; }
        else             { hb = ki / 2; hf = ki / 2 - 1; }
        if (hbSpan) *hbSpan = 0.5 * k;
        if (hfSpan) *hfSpan = 0.5 * k;
        if (twoElem) *twoElem = false;
    } else if (wl.numel() == 2) {
        const keel::f64 b = wl.elemAsDouble(0), f = wl.elemAsDouble(1);
        if (b < 0.0 || f < 0.0 || !std::isfinite(b) || !std::isfinite(f)) bad();
        hb = keel::as<long>(b);
        hf = keel::as<long>(f);
        if (hbSpan) *hbSpan = b;
        if (hfSpan) *hfSpan = f;
        if (twoElem) *twoElem = true;
    } else {
        bad();
    }
}

keel::Array<keel::f64> validateSamplePoints(const Value &sp, keel::usize expectedLen,
                                         bool arrayIsEmpty, bool floatOnly,
                                         const char *fn)
{
    const bool isFloat = (sp.type() == ValueType::DOUBLE ||
                          sp.type() == ValueType::SINGLE);
    if (floatOnly && !isFloat)
        throw Error(keel::format("{}: 'SamplePoints' must be a floating-point vector", fn),
                    0, 0, fn, "",
                    keel::format("numkit:{}:samplePointsInvalidDatatype", fn));
    if (arrayIsEmpty) {
        if (sp.isNumeric() && !sp.isEmpty())
            throw Error(keel::format("{}: 'SamplePoints' length must match the operating dimension", fn),
                        0, 0, fn, "",
                        keel::format("numkit:{}:samplePointsLength", fn));
        return {};
    }
    if ((!sp.dims().isVector() && !sp.isEmpty()) || !sp.isNumeric())
        throw Error(keel::format("{}: 'SamplePoints' must be a numeric vector", fn),
                    0, 0, fn, "",
                    keel::format("numkit:{}:samplePointsInvalidDatatype", fn));
    if (sp.numel() != expectedLen)
        throw Error(keel::format("{}: 'SamplePoints' length ({}) must match the operating dimension ({})",
                                 fn, sp.numel(), expectedLen),
                    0, 0, fn, "",
                    keel::format("numkit:{}:samplePointsLength", fn));
    if (sp.isComplex())
        throw Error(keel::format("{}: 'SamplePoints' must be real", fn),
                    0, 0, fn, "",
                    keel::format("numkit:{}:samplePointsComplex", fn));
    keel::Array<keel::f64> out(expectedLen);
    for (keel::usize j = 0; j < expectedLen; ++j) {
        const keel::f64 v = sp.elemAsDouble(j);
        if (!std::isfinite(v))
            throw Error(keel::format("{}: 'SamplePoints' must contain finite values", fn),
                        0, 0, fn, "",
                        keel::format("numkit:{}:samplePointsNonFinite", fn));
        out[j] = v;
    }
    for (keel::usize j = 1; j < expectedLen; ++j) {
        if (out[j] == out[j - 1])
            throw Error(keel::format("{}: 'SamplePoints' must contain unique values", fn),
                        0, 0, fn, "",
                        keel::format("numkit:{}:samplePointsDuplicate", fn));
        if (out[j] < out[j - 1])
            throw Error(keel::format("{}: 'SamplePoints' must be sorted in strictly increasing order", fn),
                        0, 0, fn, "",
                        keel::format("numkit:{}:samplePointsSorted", fn));
    }
    return out;
}

// ── outlier detection (mirrors locateoutliers.m) ──────────────────────

struct ColStats {
    keel::Array<keel::u8> mask;
    keel::f64 center = std::numeric_limits<keel::f64>::quiet_NaN();
    keel::f64 lo = -std::numeric_limits<keel::f64>::infinity();
    keel::f64 hi = std::numeric_limits<keel::f64>::infinity();
};

// Iterative Grubbs test; tf is the significance level alpha. The final
// bounds are those of the last TESTED state (post-removal) and the mask
// is the STATIC rule against them — exactly locateoutliers 'grubbs'.
ColStats grubbsColumn(const keel::f64 *x, keel::usize n, keel::f64 alpha,
                      keel::Allocator *mr)
{
    ColStats r;
    r.mask.assign(n, 0);
    keel::Array<keel::usize> idx;
    keel::Array<keel::f64> v;
    idx.reserve(n); v.reserve(n);
    for (keel::usize i = 0; i < n; ++i)
        if (!std::isnan(x[i])) { v.push_back(x[i]); idx.push_back(i); }
    if (v.empty()) return r;
    keel::f64 astd = 0.0, thr = 0.0, cen = 0.0;
    while (v.size() >= 3) {
        const keel::usize m = v.size();
        keel::f64 s = 0.0;
        for (keel::f64 e : v) s += e;
        cen = s / keel::f64(m);
        keel::f64 ss = 0.0;
        for (keel::f64 e : v) ss += (e - cen) * (e - cen);
        astd = std::sqrt(ss / keel::f64(m - 1));
        if (!(astd > 0.0)) break;
        keel::usize jmax = 0; keel::f64 amax = -1.0;
        for (keel::usize j = 0; j < m; ++j) {
            const keel::f64 ad = std::fabs(v[j] - cen);
            if (ad > amax) { amax = ad; jmax = j; }
        }
        const keel::f64 dof = keel::f64(m) - 2.0;
        const keel::f64 t = tinv(Value::scalar(alpha / (2.0 * keel::f64(m)), mr),
                              dof, mr).toScalar();
        thr = ((keel::f64(m) - 1.0) / std::sqrt(keel::f64(m))) *
              std::fabs(t) / std::sqrt(dof + t * t);
        if (amax / astd > thr) {
            v.remove_at(jmax);
            idx.remove_at(jmax);
        } else {
            break;
        }
    }
    r.center = cen;
    r.lo = cen - astd * thr;
    r.hi = cen + astd * thr;
    for (keel::usize i = 0; i < n; ++i)
        r.mask[i] = (std::isnan(x[i]) ? 0
                    : ((x[i] > r.hi || x[i] < r.lo) ? 1 : 0));
    return r;
}

// Generalized ESD; tf is the significance level alpha. Thresholds come
// from iteration tindex = (last exceeding R)+1 clamped; the mask is the
// STATIC rule at tindex — exactly locateoutliers 'gesd'.
ColStats gesdColumn(const keel::f64 *x, keel::usize n, keel::f64 alpha,
                    long maxOutArg, keel::Allocator *mr)
{
    ColStats r;
    r.mask.assign(n, 0);
    keel::Array<keel::f64> v;
    v.reserve(n);
    for (keel::usize i = 0; i < n; ++i)
        if (!std::isnan(x[i])) v.push_back(x[i]);
    const keel::usize N = v.size();
    if (N < 2) return r;
    const keel::usize maxo = (maxOutArg >= 0)
        ? keel::as<keel::usize>(maxOutArg)
        : keel::as<keel::usize>(std::ceil(0.1 * keel::f64(N)));
    const keel::usize rounds = std::min(N, maxo);
    keel::Array<keel::f64> amean(rounds, 0.0), astd(rounds, 0.0),
                        lam(rounds, 0.0), R(rounds, 0.0);
    for (keel::usize it = 0; it < rounds; ++it) {
        const keel::usize m = v.size();
        const keel::usize nm1 = N - it;       // n-i+1 (ref 1-based)
        keel::f64 s = 0.0;
        for (keel::f64 e : v) s += e;
        amean[it] = s / keel::f64(m);
        keel::f64 ss = 0.0;
        for (keel::f64 e : v) ss += (e - amean[it]) * (e - amean[it]);
        astd[it] = (m >= 2) ? std::sqrt(ss / keel::f64(m - 1)) : 0.0;
        keel::usize jmax = 0; keel::f64 amax = -1.0;
        for (keel::usize j = 0; j < m; ++j) {
            const keel::f64 ad = std::fabs(v[j] - amean[it]);
            if (ad > amax) { amax = ad; jmax = j; }
        }
        R[it] = (astd[it] > 0.0) ? amax / astd[it]
                                : std::numeric_limits<keel::f64>::infinity();
        v.remove_at(jmax);
        const keel::f64 dof = keel::f64(N) - keel::f64(it) - 1.0;   // n-i-1
        const keel::f64 pp = 1.0 - alpha / (2.0 * keel::f64(nm1));
        const keel::f64 t = tinv(Value::scalar(pp, mr), dof, mr).toScalar();
        lam[it] = (keel::f64(nm1) - 1.0) * t / std::sqrt((dof + t * t) * keel::f64(nm1));
    }
    long last = -1;
    for (long it = 0; it < keel::as<long>(rounds); ++it)
        if (R[keel::as<keel::usize>(it)] > lam[keel::as<keel::usize>(it)])
            last = it;
    const keel::usize tindex = (last < 0)
        ? 0
        : std::min<keel::usize>(keel::as<keel::usize>(last) + 1, rounds - 1);
    r.center = amean[tindex];
    r.lo = amean[tindex] - astd[tindex] * lam[tindex];
    r.hi = amean[tindex] + astd[tindex] * lam[tindex];
    for (keel::usize i = 0; i < n; ++i)
        r.mask[i] = (std::isnan(x[i]) ? 0
                    : ((x[i] > r.hi || x[i] < r.lo) ? 1 : 0));
    return r;
}

// One detection run. Moving methods fill per-element center/lo/hi
// arrays instead of the scalar bounds.
ColStats detectRun(keel::ArrayRef<keel::f64> x, keel::StringRef method,
                   keel::f64 tf, long hb, long hf, keel::f64 loP, keel::f64 hiP,
                   long maxOut, keel::Array<keel::f64> *movCenter,
                   keel::Array<keel::f64> *movLo, keel::Array<keel::f64> *movHi,
                   keel::Allocator *mr,
                   const keel::f64 *sp = nullptr, keel::f64 hbSpan = 0.0,
                   keel::f64 hfSpan = 0.0, bool winTwoElem = false)
{
    const keel::usize n = x.size();
    ColStats r;
    r.mask.assign(n, 0);
    if (method == "grubbs") return grubbsColumn(x.data(), n, tf, mr);
    if (method == "gesd")   return gesdColumn(x.data(), n, tf, maxOut, mr);
    if (method == "percentiles") {
        FoDetect d = detect_percentile_column(x.data(), n, loP, hiP);
        r.center = d.center; r.lo = d.lo; r.hi = d.hi;
        for (keel::usize i = 0; i < n; ++i) r.mask[i] = d.mask[i];
        return r;
    }
    if (method == "movmedian" || method == "movmean") {
        const bool isMed = (method == "movmedian");
        const keel::f64 nanv = std::numeric_limits<keel::f64>::quiet_NaN();
        if (movCenter) {
            movCenter->assign(n, nanv);
            movLo->assign(n, nanv);
            movHi->assign(n, nanv);
        }
        auto medOf = [](const keel::Array<keel::f64> &w) {
            const keel::usize k = w.size();
            return (k % 2 == 1) ? w[k / 2] : 0.5 * (w[k / 2 - 1] + w[k / 2]);
        };
        keel::Array<keel::f64> win, dev, sorted;
        for (keel::usize i = 0; i < n; ++i) {
            win.clear();
            if (sp) {
                const keel::f64 loS = sp[i] - hbSpan;
                const keel::f64 hiS = sp[i] + hfSpan;
                for (keel::usize j = 0; j < n; ++j) {
                    const bool inWin = winTwoElem
                                           ? (sp[j] >= loS && sp[j] <= hiS)
                                           : (sp[j] >= loS && sp[j] < hiS);
                    if (inWin && !std::isnan(x[j])) win.push_back(x[j]);
                }
            } else {
                long lo = keel::as<long>(i) - hb; if (lo < 0) lo = 0;
                long hi = keel::as<long>(i) + hf;
                if (hi > keel::as<long>(n) - 1) hi = keel::as<long>(n) - 1;
                for (long j = lo; j <= hi; ++j)
                    if (!std::isnan(x[keel::as<keel::usize>(j)]))
                        win.push_back(x[keel::as<keel::usize>(j)]);
            }
            if (win.empty()) continue;
            keel::f64 center, scale;
            if (isMed) {
                sorted.assign(win.as_ref());
                std::sort(sorted.begin(), sorted.end());
                center = medOf(sorted);
                dev.clear();
                for (keel::f64 vv : win) dev.push_back(std::fabs(vv - center));
                std::sort(dev.begin(), dev.end());
                scale = 1.4826022185056 * medOf(dev);
            } else {
                keel::f64 s = 0.0;
                for (keel::f64 vv : win) s += vv;
                center = s / keel::f64(win.size());
                keel::f64 ss = 0.0;
                for (keel::f64 vv : win) ss += (vv - center) * (vv - center);
                scale = (win.size() >= 2)
                            ? std::sqrt(ss / keel::f64(win.size() - 1)) : 0.0;
            }
            const keel::f64 loB = center - tf * scale, hiB = center + tf * scale;
            if (movCenter) {
                (*movCenter)[i] = center;
                (*movLo)[i] = loB;
                (*movHi)[i] = hiB;
            }
            if (!std::isnan(x[i]))
                r.mask[i] = (x[i] > hiB || x[i] < loB) ? 1 : 0;
        }
        r.center = std::numeric_limits<keel::f64>::quiet_NaN();
        r.lo = -std::numeric_limits<keel::f64>::infinity();
        r.hi = std::numeric_limits<keel::f64>::infinity();
        return r;
    }
    // median / mean / quartiles — static detectors.
    FoDetect d = detect_one_column(x.data(), n, method, tf);
    r.center = d.center; r.lo = d.lo; r.hi = d.hi;
    for (keel::usize i = 0; i < n; ++i) r.mask[i] = d.mask[i];
    return r;
}

} // anonymous namespace

// ── isoutlier ─────────────────────────────────────────────────────────

namespace {

// parseIsOutlierInput port. `in` excludes A itself. Throws nk-voiced
// errors with the probed rejection classes; unknown method / NV names
// participate in prefix matching exactly like the reference
// validatestring call (an NV-name match falls through to the NV loop).
void isoutlierParse(const Value &A, keel::ArrayRef<Value> in, bool forRmoutliers,
                    IsOutlierPlan &p, const Value *&outlierLocations)
{
    static constexpr keel::StringRef kMethods[] = {
        "median", "mean", "quartiles", "grubbs", "gesd",
        "movmedian", "movmean", "percentiles"};
    static constexpr keel::StringRef kNvRm[] = {
        "samplepoints", "datavariables", "thresholdfactor",
        "outlierlocations", "maxnumoutliers", "minnumoutliers"};
    static constexpr keel::StringRef kNvIso[] = {
        "samplepoints", "datavariables", "thresholdfactor",
        "maxnumoutliers", "outputformat"};
    const keel::ArrayRef<keel::StringRef> nvNames =
        forRmoutliers ? keel::ArrayRef<keel::StringRef>(kNvRm)
                      : keel::ArrayRef<keel::StringRef>(kNvIso);
    keel::usize i = 0;
    if (!in.empty() && isStringValue(in[0])) {
        keel::Array<keel::StringRef> all;
        all.reserve(std::size(kMethods) + nvNames.size());
        for (keel::StringRef s : kMethods) all.push_back(s);
        for (keel::StringRef s : nvNames) all.push_back(s);
        const int hit = argcheck::matchStringChoice(in[0], all.as_ref());
        if (hit < 0) {
            keel::String in0Str = in[0].toString();
            throw Error(keel::format("isoutlier: unrecognized method or parameter name '{}' "
                                     "(expected median, mean, quartiles, grubbs, gesd, "
                                     "movmedian, movmean, or percentiles)", in0Str.as_ref()),
                        0, 0, "isoutlier", "", "numkit:isoutlier:badMethod");
        }
        const keel::StringRef m = all[keel::as<keel::usize>(hit)];
        if (m == "movmedian" || m == "movmean") {
            p.method.assign(m);
            if (in.size() < 2)
                throw Error(keel::format("isoutlier: specify a window length after '{}'", m),
                            0, 0, "isoutlier", "", "numkit:isoutlier:missingWindow");
            parseWindowHalfSpans(in[1], "isoutlier", p.hb, p.hf,
                                 &p.hbSpan, &p.hfSpan, &p.winTwoElem);
            i = 2;
        } else if (m == "percentiles") {
            p.method.assign("percentiles");
            const Value &lu = in[1];
            const bool row2 = lu.isNumeric() && !lu.isComplex() &&
                              lu.dims().ndim() <= 2 && lu.dims().dim(0) == 1 &&
                              lu.dims().dim(1) == 2;
            if (!row2 || std::isnan(lu.elemAsDouble(0)) ||
                std::isnan(lu.elemAsDouble(1)) ||
                lu.elemAsDouble(0) < 0.0 || lu.elemAsDouble(0) > 100.0 ||
                lu.elemAsDouble(1) < 0.0 || lu.elemAsDouble(1) > 100.0 ||
                lu.elemAsDouble(0) > lu.elemAsDouble(1))
                throw Error("isoutlier: 'Percentiles' value must be a sorted "
                            "2-element numeric row vector with entries in "
                            "[0, 100]",
                            0, 0, "isoutlier", "",
                            "numkit:isoutlier:badPercentiles");
            p.loP = lu.elemAsDouble(0);
            p.hiP = lu.elemAsDouble(1);
            i = 2;
        } else if (m != "samplepoints" && m != "datavariables" &&
                   m != "thresholdfactor" && m != "maxnumoutliers" &&
                   m != "minnumoutliers" && m != "outlierlocations" &&
                   m != "outputformat") {
            p.method.assign(m);
            i = 1;
        }
        // else: an NV name — keep i = 0 so the NV loop sees it.
    }

    if (i < in.size() && !isStringValue(in[i])) {
        const Value &dv = in[i];
        const keel::f64 dd = (dv.isNumeric() && !dv.isComplex() && dv.numel() == 1)
                              ? dv.elemAsDouble(0) : -1.0;
        if (!(dd >= 1.0) || std::floor(dd) != dd)
            throw Error("isoutlier: dim must be a positive integer scalar",
                        0, 0, "isoutlier", "", "numkit:isoutlier:badDim");
        p.dim = keel::as<int>(dd);
        if (p.dim > 2 && A.dims().ndim() > 2)
            throw Error("isoutlier: reducing beyond dimension 2 is not "
                        "supported for N-D arrays in this revision",
                        0, 0, "isoutlier", "", "numkit:isoutlier:dimND");
        p.dropIdx.push_back(i);   // rmoutliers: detection always runs at
                                  // the default dim; dim selects removal
        ++i;
    }

    if ((in.size() - i) % 2 != 0)
        throw Error("isoutlier: each parameter name needs a value "
                    "(unmatched name in the name-value list)",
                    0, 0, "isoutlier", "", "numkit:isoutlier:nvMismatch");
    for (; i + 1 < in.size(); i += 2) {
        if (!isStringValue(in[i])) continue;
        const int hit = argcheck::matchStringChoice(in[i], nvNames);
        if (hit < 0) {
            keel::String iniStr = in[i].toString();
            throw Error(keel::format("isoutlier: unrecognized parameter name '{}'",
                                     iniStr.as_ref()),
                        0, 0, "isoutlier", "", "numkit:isoutlier:badParam");
        }
        const keel::StringRef name = nvNames[keel::as<keel::usize>(hit)];
        const Value &v = in[i + 1];
        if (name == "thresholdfactor") {
            if (p.method.as_ref() == "percentiles")
                throw Error("isoutlier: 'ThresholdFactor' is not supported "
                            "with the 'percentiles' method",
                            0, 0, "isoutlier", "",
                            "numkit:isoutlier:unsupportedThreshold");
            const bool scalar = v.isNumeric() && !v.isComplex() && v.numel() == 1;
            if (!scalar)
                throw Error("isoutlier: ThresholdFactor must be a real "
                            "numeric scalar",
                            0, 0, "isoutlier", "", "numkit:isoutlier:tfScalar");
            if (std::isnan(v.elemAsDouble(0)))
                throw Error("isoutlier: ThresholdFactor must not be NaN",
                            0, 0, "isoutlier", "", "numkit:isoutlier:tfNaN");
            if (v.elemAsDouble(0) < 0.0)
                throw Error(keel::format("isoutlier: ThresholdFactor must be nonnegative "
                                         "(got {})", v.elemAsDouble(0)),
                            0, 0, "isoutlier", "", "numkit:isoutlier:tfNonneg");
            p.tf = v.elemAsDouble(0);
            p.haveTf = true;
        } else if (name == "maxnumoutliers") {
            const keel::f64 mo = (v.isNumeric() && !v.isComplex() && v.numel() == 1)
                                  ? v.elemAsDouble(0) : -1.0;
            if (!(mo >= 1.0) || std::floor(mo) != mo)
                throw Error("isoutlier: MaxNumOutliers must be a positive "
                            "integer scalar",
                            0, 0, "isoutlier", "", "numkit:isoutlier:badMaxNum");
            p.maxOut = keel::as<long>(mo);
        } else if (name == "minnumoutliers") {
            const keel::f64 mo = (v.isNumeric() && !v.isComplex() && v.numel() == 1)
                                  ? v.elemAsDouble(0) : -1.0;
            if (!(mo >= 0.0) || std::floor(mo) != mo)
                throw Error("rmoutliers: MinNumOutliers must be a nonnegative "
                            "integer scalar",
                            0, 0, "rmoutliers", "", "numkit:rmoutliers:badMinNum");
            p.minNum = keel::as<long>(mo);
            p.dropIdx.push_back(i);
            p.dropIdx.push_back(i + 1);
        } else if (name == "outlierlocations") {
            outlierLocations = &v;
            p.dropIdx.push_back(i);
            p.dropIdx.push_back(i + 1);
        } else if (name == "samplepoints") {
            p.samplePointsRaw = &v;
        } else if (name == "datavariables") {
            throw Error("isoutlier: 'DataVariables' is supported only for "
                        "table inputs",
                        0, 0, "isoutlier", "", "numkit:isoutlier:dataVariables");
        } else if (name == "outputformat") {
            throw Error("isoutlier: 'OutputFormat' is supported only for "
                        "table inputs",
                        0, 0, "isoutlier", "", "numkit:isoutlier:outputFormat");
        }
    }
}

void isoutlierApplyTfDefaults(IsOutlierPlan &p, const Value &A)
{
    const keel::StringRef mref = p.method.as_ref();
    if (!p.haveTf) {
        if (mref == "quartiles") p.tf = 1.5;
        else if (mref == "grubbs" || mref == "gesd") p.tf = 0.05;
        else p.tf = 3.0;
    } else if (mref == "grubbs" || mref == "gesd") {
        if (p.tf > 1.0)
            throw Error(keel::format("isoutlier: 'ThresholdFactor' value must be between "
                                     "0 and 1 when the method is '{}'", mref),
                        0, 0, "isoutlier", "", "numkit:isoutlier:alphaRange");
    }
    if (p.maxOut >= 0) {
        if (mref != "gesd")
            throw Error("isoutlier: 'MaxNumOutliers' is supported only for "
                        "the 'gesd' method",
                        0, 0, "isoutlier", "", "numkit:isoutlier:maxNumGesdOnly");
        keel::usize dlen = 1;
        if (p.dim > 0) {
            dlen = keel::as<keel::usize>(A.dims().dim(p.dim - 1));
        } else {
            // first non-singleton dim (row vectors operate along dim 2)
            for (int k = 0; k < A.dims().ndim(); ++k) {
                if (A.dims().dim(k) > 1) {
                    dlen = keel::as<keel::usize>(A.dims().dim(k));
                    break;
                }
            }
        }
        if (keel::as<keel::usize>(p.maxOut) > dlen)
            throw Error("isoutlier: MaxNumOutliers must not exceed the "
                        "length of the operating dimension",
                        0, 0, "isoutlier", "", "numkit:isoutlier:maxNumTooLarge");
    }
}

void isoutlierFull_argcheck(const Value &A)
{
    argcheck::requireClassSet(A, kFloatReal, "isoutlier");
}

void rmoutliersFull_argcheck(const Value &A)
{
    argcheck::requireClassSet(A, kFloatReal, "rmoutliers");
    if (A.dims().ndim() > 2)
        throw Error("rmoutliers: N-D arrays are not supported",
                    0, 0, "rmoutliers", "", "numkit:rmoutliers:ndArrays");
}

} // anonymous namespace

IsOutlierOut isoutlier_full(const Value &A, keel::ArrayRef<Value> in,
                            bool wantThresholds, keel::Allocator *mr)
{
    isoutlierFull_argcheck(A);

    IsOutlierPlan p;
    const Value *locs = nullptr;
    isoutlierParse(A, in, false, p, locs);
    if (locs)
        throw Error("isoutlier: 'OutlierLocations' is supported only in "
                    "rmoutliers",
                    0, 0, "isoutlier", "", "numkit:isoutlier:outlierLocations");
    isoutlierApplyTfDefaults(p, A);

    int dim = p.dim;
    if (dim == 0) {
        dim = 1;
        for (int k = 0; k < A.dims().ndim(); ++k) {
            if (A.dims().dim(k) > 1) { dim = k + 1; break; }
        }
        if (dim > 2) dim = 2;   // N-D beyond dim 2: reduce in the dim-2 view
    }
    const keel::usize r = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize cols = (A.dims().ndim() >= 2)
                                 ? A.numel() / std::max<keel::usize>(r, 1) : 1;
    // Runs along dim: dim 1 → contiguous columns; dim 2 → rows (stride r).
    const bool along2 = (dim == 2 && A.dims().ndim() >= 2);
    const keel::usize runLen = along2 ? cols : r;
    const keel::usize nRuns = along2 ? r : cols;

    if (p.samplePointsRaw) {
        p.samplePoints = validateSamplePoints(
            *p.samplePointsRaw, runLen, A.numel() == 0, false, "isoutlier");
    }

    IsOutlierOut out;
    if (A.numel() == 0) {
        out.tf = Value::matrix(0, 0, ValueType::LOGICAL, mr);
        if (wantThresholds)
            out.lo = out.hi = out.center =
                Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return out;
    }

    Value mask = Value::matrix(r, cols, ValueType::LOGICAL, mr);
    keel::u8 *md = mask.logicalDataMut();

    const bool isMoving = (p.method.as_ref() == "movmedian" || p.method.as_ref() == "movmean");
    Value loM, hiM, ceM;
    keel::f64 *lod = nullptr, *hid = nullptr, *ced = nullptr;
    if (wantThresholds) {
        if (isMoving) {
            loM = Value::matrix(r, cols, ValueType::DOUBLE, mr);
            hiM = Value::matrix(r, cols, ValueType::DOUBLE, mr);
            ceM = Value::matrix(r, cols, ValueType::DOUBLE, mr);
        } else {
            const keel::usize tq = along2 ? r : cols;
            loM = Value::matrix(1, tq, ValueType::DOUBLE, mr);
            hiM = Value::matrix(1, tq, ValueType::DOUBLE, mr);
            ceM = Value::matrix(1, tq, ValueType::DOUBLE, mr);
        }
        lod = loM.doubleDataMut();
        hid = hiM.doubleDataMut();
        ced = ceM.doubleDataMut();
    }

    const keel::f64 *spPtr = p.samplePoints.empty() ? nullptr : p.samplePoints.data();
    keel::Array<keel::f64> run(runLen);
    keel::Array<keel::f64> movC, movL, movH;
    for (keel::usize k = 0; k < nRuns; ++k) {
        for (keel::usize j = 0; j < runLen; ++j)
            run[j] = A.elemAsDouble(along2 ? k + j * r : k * runLen + j);
        ColStats st = detectRun(run.as_ref(), p.method.as_ref(), p.tf, p.hb, p.hf, p.loP, p.hiP,
                                p.maxOut,
                                isMoving ? &movC : nullptr,
                                isMoving ? &movL : nullptr,
                                isMoving ? &movH : nullptr, mr,
                                spPtr, p.hbSpan, p.hfSpan, p.winTwoElem);
        for (keel::usize j = 0; j < runLen; ++j) {
            const keel::usize lin = along2 ? k + j * r : k * runLen + j;
            md[lin] = st.mask[j];
            if (wantThresholds) {
                if (isMoving) {
                    lod[lin] = movL[j]; hid[lin] = movH[j]; ced[lin] = movC[j];
                }
            }
        }
        if (wantThresholds && !isMoving) {
            lod[k] = st.lo; hid[k] = st.hi; ced[k] = st.center;
        }
    }
    out.tf = std::move(mask);
    out.lo = std::move(loM);
    out.hi = std::move(hiM);
    out.center = std::move(ceM);
    if (wantThresholds && A.type() == ValueType::SINGLE) {
        out.lo = narrowToSingle(std::move(out.lo), mr);
        out.hi = narrowToSingle(std::move(out.hi), mr);
        out.center = narrowToSingle(std::move(out.center), mr);
    }
    return out;
}

// ── rmoutliers ────────────────────────────────────────────────────────

RmOutliersOut rmoutliers_full(const Value &A, keel::ArrayRef<Value> in,
                              keel::Allocator *mr)
{
    rmoutliersFull_argcheck(A);

    IsOutlierPlan p;
    const Value *locs = nullptr;
    isoutlierParse(A, in, true, p, locs);
    isoutlierApplyTfDefaults(p, A);

    const keel::usize r = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize c = (A.dims().ndim() >= 2)
                              ? keel::as<keel::usize>(A.dims().dim(1)) : 1;
    const bool single = (A.type() == ValueType::SINGLE);
    RmOutliersOut out;

    // Outlier map: explicit override or the detector. The 3rd..6th
    // outputs mirror isoutlier's (NaN bounds with an explicit override).
    Value map;
    int dim = p.dim;
    if (dim == 0) dim = 1;
    IsOutlierOut iso;
    if (locs) {
        if (!locs->isLogical() || locs->numel() != A.numel() ||
            locs->dims().dim(0) != A.dims().dim(0))
            throw Error("rmoutliers: 'OutlierLocations' must be a logical "
                        "array the same size as the input",
                        0, 0, "rmoutliers", "",
                        "numkit:rmoutliers:badOutlierLocations");
        map = *locs;
        Value nanRow = Value::matrix(
            1, (A.dims().ndim() >= 2) ? A.numel() /
                std::max<keel::usize>(A.dims().dim(0), 1) : 1,
            ValueType::DOUBLE, mr);
        keel::f64 *nd = nanRow.doubleDataMut();
        std::fill_n(nd, nanRow.numel(),
                    std::numeric_limits<keel::f64>::quiet_NaN());
        iso.lo = iso.hi = iso.center = nanRow;
    } else {
        // Detection sees the SAME tail minus the rmoutliers-only NV
        // pairs (isoutlier's own parse rejects those names).
        keel::Array<Value> filtered;
        keel::ArrayRef<Value> detect = in;
        if (!p.dropIdx.empty()) {
            filtered.reserve(in.size() - p.dropIdx.size());
            for (keel::usize q = 0; q < in.size(); ++q)
                if (!std::count(p.dropIdx.begin(), p.dropIdx.end(), q))
                    filtered.push_back(in[q]);
            detect = filtered.as_ref();
        }
        iso = isoutlier_full(A, detect, true, mr);
        map = std::move(iso.tf);
    }
    const keel::u8 *m = map.logicalData();

    if (A.numel() == 0) {
        out.B = Value::matrix(0, 0, single ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        out.ind = Value::matrix(0, 0, ValueType::LOGICAL, mr);
        return out;
    }

    const bool isVec = (A.dims().isVector() || A.isScalar());
    const long minNum = std::max<long>(p.minNum, 1);
    auto putB = [&](Value &B, keel::usize lin, keel::f64 v) {
        if (single) B.singleDataMut()[lin] = keel::as<keel::f32>(v);
        else        B.doubleDataMut()[lin] = v;
    };
    if (isVec) {
        // Vector (either orientation): drop flagged entries.
        const keel::usize n = A.numel();
        keel::usize kept = 0;
        for (keel::usize i = 0; i < n; ++i) kept += m[i] ? 0u : 1u;
        const bool colOrient = (r != 1);
        Value B = colOrient
            ? Value::matrix(kept, 1, single ? ValueType::SINGLE : ValueType::DOUBLE, mr)
            : Value::matrix(1, kept, single ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        keel::usize w = 0;
        for (keel::usize i = 0; i < n; ++i)
            if (!m[i]) putB(B, w++, A.elemAsDouble(i));
        Value ind = colOrient ? Value::matrix(n, 1, ValueType::LOGICAL, mr)
                              : Value::matrix(1, n, ValueType::LOGICAL, mr);
        std::copy(m, m + n, ind.logicalDataMut());
        out.B = std::move(B);
        out.ind = std::move(ind);
        out.tf = std::move(map);
        out.lo = std::move(iso.lo);
        out.hi = std::move(iso.hi);
        out.center = std::move(iso.center);
        return out;
    }

    if (dim == 1) {
        keel::Array<keel::usize> cnt(r, 0);
        for (keel::usize col = 0; col < c; ++col)
            for (keel::usize i = 0; i < r; ++i)
                if (m[col * r + i]) ++cnt[i];
        keel::Array<keel::u8> rm(r, 0);
        keel::usize kept = 0;
        for (keel::usize i = 0; i < r; ++i) {
            rm[i] = (cnt[i] >= keel::as<keel::usize>(minNum)) ? 1 : 0;
            kept += rm[i] ? 0u : 1u;
        }
        Value B = Value::matrix(kept, c,
                                single ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        keel::usize orow = 0;
        for (keel::usize i = 0; i < r; ++i) {
            if (rm[i]) continue;
            for (keel::usize col = 0; col < c; ++col)
                putB(B, col * kept + orow, A.elemAsDouble(col * r + i));
            ++orow;
        }
        Value ind = Value::matrix(r, 1, ValueType::LOGICAL, mr);
        std::copy(rm.begin(), rm.end(), ind.logicalDataMut());
        out.B = std::move(B);
        out.ind = std::move(ind);
        out.tf = std::move(map);
        out.lo = std::move(iso.lo);
        out.hi = std::move(iso.hi);
        out.center = std::move(iso.center);
        return out;
    }
    // dim == 2: remove columns.
    keel::Array<keel::usize> cnt(c, 0);
    for (keel::usize col = 0; col < c; ++col)
        for (keel::usize i = 0; i < r; ++i)
            if (m[col * r + i]) ++cnt[col];
    keel::Array<keel::u8> rm(c, 0);
    keel::usize kept = 0;
    for (keel::usize col = 0; col < c; ++col) {
        rm[col] = (cnt[col] >= keel::as<keel::usize>(minNum)) ? 1 : 0;
        kept += rm[col] ? 0u : 1u;
    }
    Value B = Value::matrix(r, kept, single ? ValueType::SINGLE : ValueType::DOUBLE, mr);
    keel::usize ocol = 0;
    for (keel::usize col = 0; col < c; ++col) {
        if (rm[col]) continue;
        for (keel::usize i = 0; i < r; ++i)
            putB(B, ocol * r + i, A.elemAsDouble(col * r + i));
        ++ocol;
    }
    Value ind = Value::matrix(1, c, ValueType::LOGICAL, mr);
    std::copy(rm.begin(), rm.end(), ind.logicalDataMut());
    out.B = std::move(B);
    out.ind = std::move(ind);
    out.tf = std::move(map);
    out.lo = std::move(iso.lo);
    out.hi = std::move(iso.hi);
    out.center = std::move(iso.center);
    return out;
}

// ── rmmissing ─────────────────────────────────────────────────────────

RmMissingOut rmmissing_full(const Value &A, keel::ArrayRef<Value> in,
                            keel::Allocator *mr)
{
    // rmmissingFull_argcheck: numeric / logical / char arrays (strings
    // and cells are a voiced gap — the reference supports them).
    const ValueType t = A.type();
    if (!(A.isNumeric() || A.isLogical() || t == ValueType::CHAR))
        throw Error(keel::format("rmmissing: unsupported input type {} "
                                 "(string and cell inputs are not supported in this revision)",
                                 mtypeName(t)),
                    0, 0, "rmmissing", "", "numkit:rmmissing:typeUnsupported");
    if (A.dims().ndim() > 2)
        throw Error("rmmissing: N-D arrays are not supported",
                    0, 0, "rmmissing", "", "numkit:rmmissing:ndArrays");

    int dim = 0;
    long minNum = 1;
    if (!in.empty() && !isStringValue(in[0])) {
        const Value &dv = in[0];
        const keel::f64 dd = (dv.isNumeric() && !dv.isComplex() && dv.numel() == 1)
                              ? dv.elemAsDouble(0) : 0.0;
        if (std::floor(dd) != dd || dd < 1.0 || dd > 2.0)
            throw Error("rmmissing: dimension must be 1 or 2",
                        0, 0, "rmmissing", "", "numkit:rmmissing:badDim");
        dim = keel::as<int>(dd);
    }
    for (keel::usize i = isStringValue(in.empty() ? Value() : in[0]) || in.empty()
                             ? 0 : 1;
         i + 1 < in.size() || (i + 1 == in.size() && false); i += 2) {
        if (i >= in.size() || !isStringValue(in[i])) continue;
        static constexpr keel::StringRef kRmNv[] = {"minnummissing",
                                                    "datavariables"};
        const int hit = argcheck::matchStringChoice(
            in[i], keel::ArrayRef<keel::StringRef>(kRmNv));
        if (hit == 0) {
            const Value &v = in[i + 1];
            const keel::f64 mo = (v.isNumeric() && !v.isComplex() && v.numel() == 1)
                                  ? v.elemAsDouble(0) : -1.0;
            if (!(mo >= 0.0) || std::floor(mo) != mo)
                throw Error("rmmissing: MinNumMissing must be a nonnegative "
                            "integer scalar",
                            0, 0, "rmmissing", "", "numkit:rmmissing:badMinNum");
            minNum = keel::as<long>(mo);
        } else if (hit == 1) {
            throw Error("rmmissing: 'DataVariables' is supported only for "
                        "table inputs",
                        0, 0, "rmmissing", "", "numkit:rmmissing:dataVariables");
        } else {
            keel::String iniStr = in[i].toString();
            throw Error(keel::format("rmmissing: unrecognized parameter name '{}'",
                                     iniStr.as_ref()),
                        0, 0, "rmmissing", "", "numkit:rmmissing:badParam");
        }
    }

    const bool floaty = (t == ValueType::DOUBLE || t == ValueType::SINGLE ||
                         t == ValueType::COMPLEX);
    const bool single = (t == ValueType::SINGLE);
    const keel::usize r = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize c = (A.dims().ndim() >= 2)
                              ? keel::as<keel::usize>(A.dims().dim(1)) : 1;
    RmMissingOut out;

    // Integer / logical / char arrays cannot hold a standard missing
    // value — everything passes through unchanged.
    if (!floaty) {
        out.B = A;
        const bool isVec = (A.dims().isVector() || A.isScalar());
        Value ind = isVec
            ? ((r == 1 || A.isScalar())
                   ? Value::matrix(1, A.numel(), ValueType::LOGICAL, mr)
                   : Value::matrix(A.numel(), 1, ValueType::LOGICAL, mr))
            : ((dim == 2) ? Value::matrix(1, c, ValueType::LOGICAL, mr)
                          : Value::matrix(r, 1, ValueType::LOGICAL, mr));
        std::fill_n(ind.logicalDataMut(), ind.numel(), keel::u8{0});
        out.ind = std::move(ind);
        return out;
    }

    // NaN map (complex: NaN in either part).
    keel::Array<keel::u8> miss(A.numel(), 0);
    if (t == ValueType::COMPLEX) {
        const keel::c64 *z = A.complexData();
        for (keel::usize i = 0; i < A.numel(); ++i)
            miss[i] = (std::isnan(z[i].real()) || std::isnan(z[i].imag())) ? 1 : 0;
    } else if (single) {
        const keel::f32 *xd = A.singleData();
        for (keel::usize i = 0; i < A.numel(); ++i)
            miss[i] = std::isnan(xd[i]) ? 1 : 0;
    } else {
        const keel::f64 *xd = A.doubleData();
        for (keel::usize i = 0; i < A.numel(); ++i)
            miss[i] = std::isnan(xd[i]) ? 1 : 0;
    }

    auto copyIn = [&](keel::usize lin, Value &B, keel::usize lout) {
        if (t == ValueType::COMPLEX)
            B.complexDataMut()[lout] = A.complexData()[lin];
        else if (single)
            B.singleDataMut()[lout] = A.singleData()[lin];
        else
            B.doubleDataMut()[lout] = A.doubleData()[lin];
    };

    const bool isVec = (A.dims().isVector() || A.isScalar());
    const bool byRows = isVec ? true : (dim != 2);
    const auto outT = t;
    if (isVec) {
        const keel::usize n = A.numel();
        keel::Array<keel::usize> keep;
        keep.reserve(n);
        for (keel::usize i = 0; i < n; ++i)
            if (!miss[i]) keep.push_back(i);
        const bool colOrient = (r != 1);
        Value B = colOrient ? Value::matrix(keep.size(), 1, outT, mr)
                            : Value::matrix(1, keep.size(), outT, mr);
        for (keel::usize w = 0; w < keep.size(); ++w) copyIn(keep[w], B, w);
        Value ind = colOrient ? Value::matrix(n, 1, ValueType::LOGICAL, mr)
                              : Value::matrix(1, n, ValueType::LOGICAL, mr);
        std::copy(miss.begin(), miss.end(), ind.logicalDataMut());
        out.B = std::move(B);
        out.ind = std::move(ind);
        return out;
    }

    if (byRows) {
        keel::Array<keel::usize> cnt(r, 0);
        for (keel::usize col = 0; col < c; ++col)
            for (keel::usize i = 0; i < r; ++i)
                if (miss[col * r + i]) ++cnt[i];
        keel::Array<keel::u8> rm(r, 0);
        keel::Array<keel::usize> keep;
        for (keel::usize i = 0; i < r; ++i) {
            rm[i] = (cnt[i] >= keel::as<keel::usize>(minNum)) ? 1 : 0;
            if (!rm[i]) keep.push_back(i);
        }
        Value B = Value::matrix(keep.size(), c, outT, mr);
        for (keel::usize col = 0; col < c; ++col)
            for (keel::usize w = 0; w < keep.size(); ++w)
                copyIn(col * r + keep[w], B, col * keep.size() + w);
        Value ind = Value::matrix(r, 1, ValueType::LOGICAL, mr);
        std::copy(rm.begin(), rm.end(), ind.logicalDataMut());
        out.B = std::move(B);
        out.ind = std::move(ind);
        return out;
    }
    keel::Array<keel::usize> cnt(c, 0);
    for (keel::usize col = 0; col < c; ++col)
        for (keel::usize i = 0; i < r; ++i)
            if (miss[col * r + i]) ++cnt[col];
    keel::Array<keel::u8> rm(c, 0);
    keel::Array<keel::usize> keep;
    for (keel::usize col = 0; col < c; ++col) {
        rm[col] = (cnt[col] >= keel::as<keel::usize>(minNum)) ? 1 : 0;
        if (!rm[col]) keep.push_back(col);
    }
    Value B = Value::matrix(r, keep.size(), outT, mr);
    for (keel::usize w = 0; w < keep.size(); ++w)
        for (keel::usize i = 0; i < r; ++i)
            copyIn(keep[w] * r + i, B, w * r + i);
    Value ind = Value::matrix(1, c, ValueType::LOGICAL, mr);
    std::copy(rm.begin(), rm.end(), ind.logicalDataMut());
    out.B = std::move(B);
    out.ind = std::move(ind);
    return out;
}

// ── standardizeMissing ────────────────────────────────────────────────

Value standardizeMissing_full(const Value &A, const Value &indicators,
                              keel::Allocator *mr)
{
    // The live interpreter path resolves the bare name to the builtin
    // implementation (numkit::builtin::standardizeMissing); this entry
    // delegates to the SAME contract so the namespaced/direct-C++
    // paths cannot diverge from it.
    return ::numkit::builtin::standardizeMissing(A, indicators, mr);
}

// ── fillmissing ───────────────────────────────────────────────────────

namespace {

// Numeric scalar / per-slice vector constant (checkConstantsSize).
keel::Array<keel::f64> parseConstants(const Value &C, keel::usize nVects,
                                   const char *fn, const char *eidSuffix,
                                   keel::Array<keel::f64> *imagParts = nullptr)
{
    if (!C.isNumeric())
        throw Error(keel::format("{}: fill constant must be numeric (got {})",
                                 fn, mtypeName(C.type())),
                    0, 0, fn, "",
                    keel::format("numkit:{}:constantNumeric", fn));
    // Complex constants are legal for complex inputs (the parts fill
    // the real / imaginary buffers separately).
    const bool cplx = C.isComplex();
    if (cplx && !imagParts)
        throw Error(keel::format("{}: complex fill constants are "
                                 "supported only for complex inputs", fn),
                    0, 0, fn, "",
                    keel::format("numkit:{}:constantNumeric", fn));
    auto partOf = [&](keel::usize i) {
        if (!cplx) return C.elemAsDouble(i);
        const keel::c64 z = C.complexData()[i];
        if (imagParts) (*imagParts)[i] = z.imag();
        return z.real();
    };
    if (C.numel() == 1) {
        if (imagParts && imagParts->size() < 1) imagParts->resize(1, 0.0);
        keel::Array<keel::f64> one(1);
        one[0] = partOf(0);
        return one;
    }
    if (C.numel() != nVects)
        throw Error(keel::format("{}: fill constant must be a scalar or "
                                 "a {}-element vector{}",
                                 fn, nVects,
                                 nVects <= 1 ? "" : " (one per slice)"),
                    0, 0, fn, "",
                    keel::format("numkit:{}:SizeConstant{}", fn, eidSuffix));
    if (imagParts) imagParts->resize(C.numel(), 0.0);
    keel::Array<keel::f64> v(C.numel());
    for (keel::usize i = 0; i < C.numel(); ++i) v[i] = partOf(i);
    return v;
}

} // anonymous namespace

FillMissingOut fillmissing_full(
    const Value &A, keel::ArrayRef<Value> in, bool wantFilled,
    keel::Allocator *mr,
    keel::FunctionRef<Value(const Value &, keel::ArrayRef<Value>)> callFn)
{
    // fillmissingFull_argcheck: supported-array gate + method contract.
    const char *fn = "fillmissing";
    const ValueType at = A.type();
    if (!(A.isNumeric() || A.isLogical() || at == ValueType::CHAR))
        throw Error(keel::format("{}: unsupported input type {} "
                                 "(string and cell inputs are not supported in this revision)",
                                 fn, mtypeName(at)),
                    0, 0, fn, keel::format("numkit:{}:firstInputInvalid", fn));
    if (A.dims().ndim() > 2)
        throw Error(keel::format("{}: N-D arrays are not supported", fn),
                    0, 0, fn, "", keel::format("numkit:{}:ndArrays", fn));

    static constexpr keel::StringRef kFmMethods[] = {
        "constant", "previous", "next", "nearest", "linear",
        "spline", "pchip", "movmean", "movmedian", "makima", "knn"};
    const keel::ArrayRef<keel::StringRef> methods =
        keel::ArrayRef<keel::StringRef>(kFmMethods);
    if (in.empty())
        throw Error(keel::format("{}: specify a fill method", fn),
                    0, 0, fn, "", keel::format("numkit:{}:methodInvalid", fn));

    bool isFnHandle = false;
    const Value *fhVal = nullptr;
    keel::f64 fhA = 0.0, fhB = 0.0;
    bool fhTwoElem = false;
    keel::String m;
    long hb = 0, hf = 0;
    keel::f64 hbSpan = 0.0, hfSpan = 0.0;
    bool winTwoElem = false;
    keel::usize knnK = 1;
    keel::usize i = 1;

    if (in[0].isFuncHandle()) {
        isFnHandle = true;
        fhVal = &in[0];
        if (in.size() < 2)
            throw Error(keel::format("{}: specify a window length after "
                                     "the function handle", fn),
                        0, 0, fn, "",
                        keel::format("numkit:{}:functionHandleInput", fn));
        const Value &wl = in[1];
        const bool okNum = wl.isNumeric() && !wl.isComplex() &&
                           (wl.numel() == 1 || wl.numel() == 2);
        bool okVal = okNum;
        if (okNum) {
            bool anyPos = false;
            for (keel::usize q = 0; q < wl.numel(); ++q) {
                const keel::f64 wv = wl.elemAsDouble(q);
                if (!std::isfinite(wv) || wv < 0.0) okVal = false;
                if (wv > 0.0) anyPos = true;
            }
            if (!anyPos) okVal = false;
        }
        if (!okVal)
            throw Error(keel::format("{}: function-handle window must be "
                                     "a positive scalar or 2-element nonnegative finite vector", fn),
                        0, 0, fn, "",
                        keel::format("numkit:{}:functionHandleInvalidWindow", fn));
        if (wl.numel() == 2) {
            fhA = wl.elemAsDouble(0);
            fhB = wl.elemAsDouble(1);
            fhTwoElem = true;
        } else {
            fhA = fhB = wl.elemAsDouble(0);
            fhTwoElem = false;
        }
        i = 2;
    } else if (isStringValue(in[0])) {
        const int mhit = argcheck::matchStringChoice(in[0], methods);
        if (mhit < 0) {
            keel::String in0Str = in[0].toString();
            throw Error(keel::format("{}: fill method must be one of "
                                     "'constant', 'previous', 'next', 'nearest', 'linear', "
                                     "'spline', 'pchip', 'makima', 'movmean', 'movmedian', "
                                     "'knn', or a function handle (got '{}'; ambiguous or unrecognized)",
                                     fn, in0Str.as_ref()),
                        0, 0, fn, "", keel::format("numkit:{}:methodInvalid", fn));
        }
        m.assign(methods[keel::as<keel::usize>(mhit)]);
        if (m.as_ref() == "constant") {
            if (in.size() < 2)
                throw Error(keel::format("{}: specify a fill value after the "
                                         "'constant' method", fn),
                            0, 0, fn, "", keel::format("numkit:{}:constantInput", fn));
            ++i;
        } else if (m.as_ref() == "movmean" || m.as_ref() == "movmedian") {
            if (in.size() < 2)
                throw Error(keel::format("{}: specify a window length after "
                                         "the '{}' method", fn, m.as_ref()),
                            0, 0, fn, "", keel::format("numkit:{}:windowInput", fn));
            parseWindowHalfSpans(in[1], fn, hb, hf, &hbSpan, &hfSpan, &winTwoElem);
            ++i;
        } else if (m.as_ref() == "knn") {
            if (in.size() >= 2 && in[1].isNumeric()) {
                const Value &kv = in[1];
                const keel::f64 kd = (kv.numel() == 1 && !kv.isComplex())
                                      ? kv.elemAsDouble(0) : -1.0;
                if (!(kd >= 1.0) || std::floor(kd) != kd || !std::isfinite(kd))
                    throw Error(keel::format("{}: number of nearest neighbors k "
                                             "must be a positive integer scalar", fn),
                                0, 0, fn, "", keel::format("numkit:{}:invalidK", fn));
                knnK = keel::as<keel::usize>(kd);
                ++i;
            }
        }
    } else {
        throw Error(keel::format("{}: fill method must be a string or function handle", fn),
                    0, 0, fn, "", keel::format("numkit:{}:methodInvalid", fn));
    }

    // Optional positional dim.
    int dim = 0;
    if (i < in.size() && !isStringValue(in[i]) &&
        (in[i].isNumeric() || in[i].isLogical())) {
        const Value &dv = in[i];
        const keel::f64 dd = (dv.numel() == 1 && !dv.isComplex())
                              ? dv.elemAsDouble(0) : 0.0;
        if (std::floor(dd) != dd || !(dd >= 1.0) || !std::isfinite(dd))
            throw Error(keel::format("{}: dimension must be a positive "
                                     "integer scalar", fn),
                        0, 0, fn, "",
                        keel::format("numkit:{}:dimensionInvalid", fn));
        dim = keel::as<int>(dd);
        ++i;
    }

    // Name-value tail.
    static constexpr keel::StringRef kFmNv[] = {
        "endvalues", "datavariables", "replacevalues", "samplepoints",
        "missinglocations", "maxgap", "distance"};
    const keel::ArrayRef<keel::StringRef> nvNames =
        keel::ArrayRef<keel::StringRef>(kFmNv);
    const Value *constRaw = (m.as_ref() == "constant") ? &in[1] : nullptr;
    const Value *endRaw = nullptr;
    int endMode = 0;   // 0 extrap / 1 none / 2 nearest / 3 prevEnd / 4 nextEnd / 5 const
    bool haveEnd = false;
    keel::f64 endConst = 0.0;
    const Value *missingLocations = nullptr;
    const Value *samplePointsRaw = nullptr;
    keel::f64 maxGap = std::numeric_limits<keel::f64>::infinity();
    bool hasMaxGap = false;
    keel::String knnDist = keel::String::from("euclidean");
    const Value *distFnVal = nullptr;

    if ((in.size() - i) % 2 != 0)
        throw Error(keel::format("{}: each parameter name needs a value "
                                 "(unmatched value in the name-value list)", fn),
                    0, 0, fn, "", keel::format("numkit:{}:nameValuePairs", fn));
    for (; i + 1 < in.size(); i += 2) {
        if (!isStringValue(in[i]))
            throw Error(keel::format("{}: expected a parameter name", fn),
                        0, 0, fn, "", keel::format("numkit:{}:nameValuePairs", fn));
        const int hit = argcheck::matchStringChoice(in[i], nvNames);
        if (hit < 0) {
            keel::String iniStr = in[i].toString();
            throw Error(keel::format("{}: unrecognized parameter name '{}'",
                                     fn, iniStr.as_ref()),
                        0, 0, fn, "", keel::format("numkit:{}:nameValueNames", fn));
        }
        const keel::StringRef name = nvNames[keel::as<keel::usize>(hit)];
        const Value &v = in[i + 1];
        if (name == "endvalues") {
            if (m.as_ref() == "knn")
                throw Error(keel::format("{}: 'EndValues' is not supported "
                                         "for the 'knn' method", fn),
                            0, 0, fn, "",
                            keel::format("numkit:{}:unsupportedNVPair", fn));
            haveEnd = true;
            if (isStringValue(v)) {
                static constexpr keel::StringRef kEndModes[] = {
                    "extrap", "previous", "next", "nearest", "none"};
                const int eh = argcheck::matchStringChoice(
                    v, keel::ArrayRef<keel::StringRef>(kEndModes));
                if (eh < 0)
                    throw Error(keel::format("{}: 'EndValues' must be "
                                             "'extrap', 'previous', 'next', 'nearest', "
                                             "'none', or a numeric constant", fn),
                                0, 0, fn, "",
                                keel::format("numkit:{}:endValuesInvalid", fn));
                endMode = (eh == 0) ? 0 : (eh == 4) ? 1 : (eh == 3) ? 2
                          : (eh == 1) ? 3 : 4;
            } else {
                endMode = 5;
                endRaw = &v;
            }
        } else if (name == "missinglocations") {
            if (!v.isLogical() || v.numel() != A.numel() ||
                v.dims().dim(0) != A.dims().dim(0))
                throw Error(keel::format("{}: 'MissingLocations' must be "
                                         "a logical array the same size as the input", fn),
                            0, 0, fn, "",
                            keel::format("numkit:{}:badMissingLocations", fn));
            missingLocations = &v;
        } else if (name == "samplepoints") {
            if (m.as_ref() == "knn")
                throw Error(keel::format("{}: 'SamplePoints' is not supported "
                                         "for the 'knn' method", fn),
                            0, 0, fn, "",
                            keel::format("numkit:{}:unsupportedNVPair", fn));
            samplePointsRaw = &v;
        } else if (name == "maxgap") {
            if (m.as_ref() == "knn")
                throw Error(keel::format("{}: 'MaxGap' is not supported "
                                         "for the 'knn' method", fn),
                            0, 0, fn, "",
                            keel::format("numkit:{}:unsupportedNVPair", fn));
            if (!v.isNumeric() || v.isComplex() || v.numel() != 1 ||
                std::isnan(v.elemAsDouble(0)) || !(v.elemAsDouble(0) > 0.0))
                throw Error(keel::format("{}: 'MaxGap' must be a positive "
                                         "real scalar", fn),
                            0, 0, fn, "",
                            keel::format("numkit:{}:maxGapInvalid", fn));
            maxGap = v.elemAsDouble(0);
            hasMaxGap = true;
        } else if (name == "distance") {
            if (m.as_ref() != "knn")
                throw Error(keel::format("{}: 'Distance' is supported only "
                                         "for the 'knn' method", fn),
                            0, 0, fn, "",
                            keel::format("numkit:{}:distanceNonKNNMethod", fn));
            if (isStringValue(v)) {
                static constexpr keel::StringRef kDist[] = {
                    "euclidean", "seuclidean"};
                const int dh = argcheck::matchStringChoice(
                    v, keel::ArrayRef<keel::StringRef>(kDist));
                if (dh < 0)
                    throw Error(keel::format("{}: 'Distance' must be "
                                             "'euclidean', 'seuclidean', or a function handle", fn),
                                0, 0, fn, "",
                                keel::format("numkit:{}:invalidDistance", fn));
                knnDist.assign(kDist[keel::as<keel::usize>(dh)]);
                distFnVal = nullptr;
            } else if (v.isFuncHandle()) {
                distFnVal = &v;
            } else {
                throw Error(keel::format("{}: 'Distance' must be "
                                         "'euclidean', 'seuclidean', or a function handle", fn),
                            0, 0, fn, "",
                            keel::format("numkit:{}:invalidDistance", fn));
            }
        } else {
            throw Error(keel::format("{}: '{}' is supported only for table inputs",
                                     fn, name),
                        0, 0, fn, "", keel::format("numkit:{}:tableOnly", fn));
        }
    }

    if (missingLocations && hasMaxGap)
        throw Error(keel::format("{}: 'MaxGap' and 'MissingLocations' "
                                 "cannot be specified together", fn),
                    0, 0, fn, "",
                    keel::format("numkit:{}:maxGapMissingLocations", fn));

    // Direction: default = first non-singleton (row vectors along 2).
    if (dim == 0) {
        dim = 1;
        if (A.dims().ndim() >= 2 && A.dims().dim(0) == 1 &&
            A.dims().dim(1) > 1)
            dim = 2;
    }
    const keel::usize r = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize c = (A.dims().ndim() >= 2)
                              ? keel::as<keel::usize>(A.dims().dim(1)) : 1;
    const bool along2 = (dim == 2 && A.dims().ndim() >= 2);
    const keel::usize runLen = along2 ? c : r;
    const keel::usize nRuns = along2 ? r : c;

    keel::Array<keel::f64> samplePoints;
    if (samplePointsRaw) {
        samplePoints = validateSamplePoints(
            *samplePointsRaw, runLen, A.numel() == 0, false, fn);
    }

    const bool complex = (at == ValueType::COMPLEX);
    const bool outComplex = complex ||
                            (constRaw && constRaw->isComplex()) ||
                            (endRaw && endRaw->isComplex());
    keel::Array<keel::f64> constVals, constImag;
    if (constRaw) {
        if (outComplex) constImag.resize(1, 0.0);
        constVals = parseConstants(*constRaw, nRuns, fn, "",
                                   outComplex ? &constImag : nullptr);
    }
    keel::Array<keel::f64> endVals, endImag;
    if (endRaw) {
        if (outComplex) endImag.resize(1, 0.0);
        endVals = parseConstants(*endRaw, nRuns, fn, "Extrap",
                                 outComplex ? &endImag : nullptr);
        endConst = endVals[0];
    }

    // Integer / logical / char inputs cannot represent NaN: flow-through
    // unless MissingLocations is explicitly provided or knn rejects non-float.
    const bool floaty = (at == ValueType::DOUBLE || at == ValueType::SINGLE ||
                         at == ValueType::COMPLEX);
    if (m.as_ref() == "knn" && !floaty && !distFnVal)
        throw Error(keel::format("{}: built-in 'knn' distances require "
                                 "floating-point input", fn),
                    0, 0, fn, "",
                    keel::format("numkit:{}:distanceNonFloatsNotSupported", fn));
    FillMissingOut out;
    if (!floaty || A.numel() == 0) {
        out.B = A;
        if (wantFilled) {
            Value f = Value::matrix(r, c, ValueType::LOGICAL, mr);
            std::fill_n(f.logicalDataMut(), f.numel(), keel::u8{0});
            out.filled = std::move(f);
        }
        return out;
    }

    const bool single = (at == ValueType::SINGLE) && !outComplex;
    Value B = outComplex ? Value::matrix(r, c, ValueType::COMPLEX, mr)
             : single    ? Value::matrix(r, c, ValueType::SINGLE, mr)
                         : Value::matrix(r, c, ValueType::DOUBLE, mr);
    Value filledMap = wantFilled ? Value::matrix(r, c, ValueType::LOGICAL, mr)
                                 : Value();
    keel::u8 *fmd = wantFilled ? filledMap.logicalDataMut() : nullptr;
    if (fmd) std::fill_n(fmd, A.numel(), keel::u8{0});

    // Original missing map (or the explicit override).
    keel::Array<keel::u8> missMap(A.numel(), 0);
    if (missingLocations) {
        const keel::u8 *lm = missingLocations->logicalData();
        std::copy(lm, lm + A.numel(), missMap.begin());
    } else if (complex) {
        const keel::c64 *z = A.complexData();
        for (keel::usize j = 0; j < A.numel(); ++j)
            missMap[j] = (std::isnan(z[j].real()) ||
                          std::isnan(z[j].imag())) ? 1 : 0;
    } else {
        for (keel::usize j = 0; j < A.numel(); ++j)
            missMap[j] = std::isnan(A.elemAsDouble(j)) ? 1 : 0;
    }

    // ── knn fill path ─────────────────────────────────────────────────
    if (m.as_ref() == "knn") {
        // Initialize B from A.
        for (keel::usize j = 0; j < A.numel(); ++j) {
            if (outComplex) {
                B.complexDataMut()[j] = complex ? A.complexData()[j]
                                                : keel::c64(A.elemAsDouble(j), 0.0);
            } else if (single) {
                B.singleDataMut()[j] = keel::as<keel::f32>(A.elemAsDouble(j));
            } else {
                B.doubleDataMut()[j] = A.elemAsDouble(j);
            }
        }
        if (dim > 2) {
            out.B = std::move(B);
            out.filled = std::move(filledMap);
            return out;
        }
        // When dim == 1, vectors are rows (M = r, N = c); when dim == 2,
        // vectors are columns (M = c, N = r).
        const bool transpose = (dim == 2);
        const keel::usize M = transpose ? c : r;
        const keel::usize N = transpose ? r : c;
        auto linAt = [&](keel::usize row, keel::usize col) -> keel::usize {
            return transpose ? (col + row * r) : (row + col * r);
        };
        auto reAt = [&](keel::usize row, keel::usize col) -> keel::f64 {
            const keel::usize idx = linAt(row, col);
            return complex ? A.complexData()[idx].real() : A.elemAsDouble(idx);
        };
        auto imAt = [&](keel::usize row, keel::usize col) -> keel::f64 {
            if (!complex) return 0.0;
            return A.complexData()[linAt(row, col)].imag();
        };

        keel::Array<keel::f64> scale(N, 1.0);
        if (!distFnVal && knnDist.as_ref() == "seuclidean") {
            for (keel::usize col = 0; col < N; ++col) {
                keel::f64 mRe = 0.0, mIm = 0.0;
                keel::usize cnt = 0;
                for (keel::usize row = 0; row < M; ++row) {
                    if (!missMap[linAt(row, col)]) {
                        mRe += reAt(row, col);
                        mIm += imAt(row, col);
                        ++cnt;
                    }
                }
                if (cnt < 2) {
                    scale[col] = 0.0;
                } else {
                    mRe /= keel::f64(cnt);
                    mIm /= keel::f64(cnt);
                    keel::f64 ss = 0.0;
                    for (keel::usize row = 0; row < M; ++row) {
                        if (!missMap[linAt(row, col)]) {
                            const keel::f64 dr = reAt(row, col) - mRe;
                            const keel::f64 di = imAt(row, col) - mIm;
                            ss += dr * dr + di * di;
                        }
                    }
                    scale[col] = std::sqrt(ss / keel::f64(cnt - 1));
                }
            }
        }

        const keel::f64 nanv = std::numeric_limits<keel::f64>::quiet_NaN();
        keel::Array<keel::f64> dists(M, nanv);
        keel::Array<keel::usize> ord(M);
        for (keel::usize ii = 0; ii < M; ++ii) {
            bool anyMiss = false, allMiss = true;
            for (keel::usize col = 0; col < N; ++col) {
                if (missMap[linAt(ii, col)]) anyMiss = true;
                else                         allMiss = false;
            }
            if (!anyMiss || allMiss) continue;

            for (keel::usize jj = 0; jj < M; ++jj) {
                if (jj == ii) {
                    dists[jj] = nanv;
                    continue;
                }
                if (distFnVal) {
                    if (!callFn)
                        throw Error(keel::format("{}: custom distance function "
                                                 "handle requires an execution context", fn),
                                    0, 0, fn, "",
                                    keel::format("numkit:{}:invalidCustomDistance", fn));
                    Value X2 = outComplex ? Value::matrix(2, N, ValueType::COMPLEX, mr)
                                          : Value::matrix(2, N, ValueType::DOUBLE, mr);
                    Value M2 = Value::matrix(2, N, ValueType::LOGICAL, mr);
                    for (keel::usize col = 0; col < N; ++col) {
                        if (outComplex) {
                            X2.complexDataMut()[0 + col * 2] =
                                keel::c64(reAt(ii, col), imAt(ii, col));
                            X2.complexDataMut()[1 + col * 2] =
                                keel::c64(reAt(jj, col), imAt(jj, col));
                        } else {
                            X2.doubleDataMut()[0 + col * 2] = reAt(ii, col);
                            X2.doubleDataMut()[1 + col * 2] = reAt(jj, col);
                        }
                        M2.logicalDataMut()[0 + col * 2] = missMap[linAt(ii, col)];
                        M2.logicalDataMut()[1 + col * 2] = missMap[linAt(jj, col)];
                    }
                    Value fargs[2] = {std::move(X2), std::move(M2)};
                    Value dv = callFn(*distFnVal, keel::ArrayRef<Value>(fargs, 2));
                    if (!dv.isNumeric() || dv.isComplex() || dv.numel() != 1)
                        throw Error(keel::format("{}: custom distance function "
                                                 "must return a real scalar", fn),
                                    0, 0, fn, "",
                                    keel::format("numkit:{}:invalidCustomDistance", fn));
                    dists[jj] = dv.elemAsDouble(0);
                } else {
                    keel::f64 sumSq = 0.0;
                    for (keel::usize col = 0; col < N; ++col) {
                        if (missMap[linAt(ii, col)]) continue;
                        if (missMap[linAt(jj, col)]) {
                            sumSq = nanv;
                            break;
                        }
                        const keel::f64 dr = reAt(ii, col) - reAt(jj, col);
                        const keel::f64 di = imAt(ii, col) - imAt(jj, col);
                        const keel::f64 d2 = dr * dr + di * di;
                        if (knnDist.as_ref() == "seuclidean") {
                            const keel::f64 sc = scale[col];
                            sumSq += d2 / (sc * sc);
                        } else {
                            sumSq += d2;
                        }
                    }
                    dists[jj] = std::sqrt(sumSq);
                }
            }

            ord.clear();
            for (keel::usize jj = 0; jj < M; ++jj)
                if (!std::isnan(dists[jj])) ord.push_back(jj);
            std::stable_sort(ord.begin(), ord.end(),
                             [&](keel::usize a, keel::usize b) {
                                 return dists[a] < dists[b];
                             });
            if (ord.empty()) continue;

            for (keel::usize col = 0; col < N; ++col) {
                const keel::usize dstLin = linAt(ii, col);
                if (!missMap[dstLin]) continue;
                keel::f64 sumRe = 0.0, sumIm = 0.0;
                keel::usize used = 0;
                for (keel::usize cand : ord) {
                    if (!missMap[linAt(cand, col)]) {
                        sumRe += reAt(cand, col);
                        sumIm += imAt(cand, col);
                        if (++used == knnK) break;
                    }
                }
                if (used > 0) {
                    const keel::f64 fRe = sumRe / keel::f64(used);
                    const keel::f64 fIm = sumIm / keel::f64(used);
                    if (outComplex) {
                        B.complexDataMut()[dstLin] = keel::c64(fRe, fIm);
                    } else if (single) {
                        B.singleDataMut()[dstLin] = keel::as<keel::f32>(fRe);
                    } else {
                        B.doubleDataMut()[dstLin] = fRe;
                    }
                    if (fmd) fmd[dstLin] = 1;
                }
            }
        }
        out.B = std::move(B);
        out.filled = std::move(filledMap);
        return out;
    }

    // ── standard 1-D slice fill path ──────────────────────────────────
    const keel::f64 nanv = std::numeric_limits<keel::f64>::quiet_NaN();
    const bool useSp = !samplePoints.empty();
    keel::Array<keel::f64> sVec(runLen);
    for (keel::usize j = 0; j < runLen; ++j)
        sVec[j] = useSp ? samplePoints[j] : keel::as<keel::f64>(j + 1);
    const keel::f64 *spPtr = useSp ? sVec.data() : nullptr;

    keel::Array<keel::f64> re(runLen), im(runLen, 0.0), origRe(runLen), origIm(runLen, 0.0);
    keel::Array<keel::u8> ma(runLen), maBeforeInterp(runLen), origMiss(runLen);

    auto removeLargeGapsSlice = [&](keel::Array<keel::u8> &mask) {
        if (!std::isfinite(maxGap) || runLen == 0) return;
        keel::usize j = 0;
        while (j < runLen) {
            if (!mask[j]) { ++j; continue; }
            keel::usize k = j;
            while (k + 1 < runLen && mask[k + 1]) ++k;
            const keel::f64 leftX = (j == 0) ? sVec[0] : sVec[j - 1];
            const keel::f64 rightX = (k + 1 == runLen) ? sVec[runLen - 1] : sVec[k + 1];
            if (!(leftX + maxGap >= rightX)) {
                for (keel::usize q = j; q <= k; ++q) mask[q] = 0;
            }
            j = k + 1;
        }
    };

    for (keel::usize k = 0; k < nRuns; ++k) {
        const keel::f64 cv = (constVals.size() > 1) ? constVals[k]
                          : (constVals.empty() ? 0.0 : constVals[0]);
        const keel::f64 cim = (outComplex && constImag.size() > 1)
                               ? constImag[k]
                               : (outComplex && !constImag.empty()
                                      ? constImag[0] : 0.0);
        const keel::f64 evRe = (endVals.size() > 1) ? endVals[k]
                            : (endVals.empty() ? endConst : endVals[0]);
        const keel::f64 evIm = (outComplex && endImag.size() > 1)
                                ? endImag[k]
                                : (outComplex && !endImag.empty()
                                       ? endImag[0] : 0.0);

        keel::usize numNonMissing = 0;
        for (keel::usize j = 0; j < runLen; ++j) {
            const keel::usize lin = along2 ? k + j * r : k * runLen + j;
            if (complex) {
                const keel::c64 z = A.complexData()[lin];
                re[j] = z.real();
                im[j] = z.imag();
            } else {
                re[j] = A.elemAsDouble(lin);
                im[j] = 0.0;
            }
            origRe[j] = re[j];
            origIm[j] = im[j];
            origMiss[j] = missMap[lin];
            ma[j] = missMap[lin];
            if (!ma[j]) ++numNonMissing;
        }

        // All-missing column special case (matches fillArrayColumn lines 413-434).
        if (numNonMissing == 0) {
            if (!std::isfinite(maxGap)) {
                if (isFnHandle && endMode == 0 && callFn) {
                    Value xs = Value::matrix(0, 1, ValueType::DOUBLE, mr);
                    Value ts = Value::matrix(0, 1, ValueType::DOUBLE, mr);
                    Value tq = Value::matrix(runLen, 1, ValueType::DOUBLE, mr);
                    for (keel::usize j = 0; j < runLen; ++j)
                        tq.doubleDataMut()[j] = sVec[j];
                    Value fargs[3] = {std::move(xs), std::move(ts), std::move(tq)};
                    Value ytmp = callFn(*fhVal, keel::ArrayRef<Value>(fargs, 3));
                    if (ytmp.numel() == 1 || (ytmp.dims().isVector() && ytmp.numel() == runLen)) {
                        for (keel::usize j = 0; j < runLen; ++j) {
                            const keel::usize yi = (ytmp.numel() == 1) ? 0 : j;
                            re[j] = ytmp.elemAsDouble(yi);
                            if (ytmp.isComplex()) im[j] = ytmp.complexData()[yi].imag();
                        }
                    } else {
                        throw Error(keel::format("{}: function handle output "
                                                 "must be a scalar or a vector matching the gap length", fn),
                                    0, 0, fn, "",
                                    keel::format("numkit:{}:functionHandleInvalidOutputSize", fn));
                    }
                } else if (endMode == 5) {
                    for (keel::usize j = 0; j < runLen; ++j) {
                        re[j] = evRe;
                        im[j] = evIm;
                    }
                } else if (m.as_ref() == "constant" && endMode == 0) {
                    for (keel::usize j = 0; j < runLen; ++j) {
                        re[j] = cv;
                        im[j] = cim;
                    }
                } else {
                    std::fill(ma.begin(), ma.end(), keel::u8{0});
                }
            } else {
                std::fill(ma.begin(), ma.end(), keel::u8{0});
            }
            for (keel::usize j = 0; j < runLen; ++j) {
                const keel::usize lin = along2 ? k + j * r : k * runLen + j;
                if (outComplex) {
                    B.complexDataMut()[lin] = keel::c64(re[j], im[j]);
                } else if (single) {
                    B.singleDataMut()[lin] = keel::as<keel::f32>(re[j]);
                } else {
                    B.doubleDataMut()[lin] = re[j];
                }
                if (fmd) {
                    const bool stillNaN = std::isnan(re[j]) ||
                                          (outComplex && std::isnan(im[j]));
                    fmd[lin] = (ma[j] && !stillNaN) ? 1 : 0;
                }
            }
            continue;
        }

        // Ignore gaps larger than MaxGap.
        removeLargeGapsSlice(ma);
        maBeforeInterp.assign(ma.as_ref());

        // Find leading/trailing gap boundaries from maBeforeInterp.
        keel::usize indBeg = 0;
        if (maBeforeInterp[0]) {
            while (indBeg < runLen && maBeforeInterp[indBeg]) ++indBeg;
        }
        keel::usize indEnd = runLen - 1;
        if (maBeforeInterp[runLen - 1]) {
            while (indEnd > 0 && maBeforeInterp[indEnd]) --indEnd;
        }

        if (isFnHandle) {
            if (!callFn)
                throw Error(keel::format("{}: function handle fill requires "
                                         "an execution context", fn),
                            0, 0, fn, "",
                            keel::format("numkit:{}:functionHandleInput", fn));
            const bool isExtrap = (endMode == 0);
            if (!isExtrap) {
                for (keel::usize j = 0; j < indBeg; ++j) ma[j] = 0;
                for (keel::usize j = indEnd + 1; j < runLen; ++j) ma[j] = 0;
            }
            const keel::f64 aSpan = fhTwoElem ? fhA : (useSp ? 0.5 * fhA : std::floor(0.5 * fhA));
            const keel::f64 bSpan = fhTwoElem ? fhB : (useSp ? 0.5 * fhB : std::floor(0.5 * fhB));
            keel::usize j0 = 0;
            while (j0 < runLen) {
                if (!ma[j0]) { ++j0; continue; }
                keel::usize j1 = j0;
                while (j1 + 1 < runLen && ma[j1 + 1]) ++j1;
                keel::Array<keel::usize> ind;
                for (keel::usize q = 0; q < runLen; ++q) {
                    if (origMiss[q]) continue;
                    bool inW = false;
                    if (useSp) {
                        inW = ((sVec[q] >= sVec[j0] - aSpan) && (sVec[q] < sVec[j0])) ||
                              ((sVec[q] > sVec[j1]) && (sVec[q] <= sVec[j1] + bSpan));
                    } else {
                        const keel::f64 q1 = keel::as<keel::f64>(q + 1);
                        const keel::f64 g0 = keel::as<keel::f64>(j0 + 1);
                        const keel::f64 g1 = keel::as<keel::f64>(j1 + 1);
                        inW = (q1 >= std::ceil(g0 - aSpan) && q1 <= g0 - 1.0) ||
                              (q1 >= g1 + 1.0 && q1 <= std::floor(g1 + bSpan));
                    }
                    if (inW) ind.push_back(q);
                }
                const keel::usize gapLen = j1 - j0 + 1;
                Value xs = outComplex ? Value::matrix(ind.size(), 1, ValueType::COMPLEX, mr)
                                      : Value::matrix(ind.size(), 1, ValueType::DOUBLE, mr);
                Value ts = Value::matrix(ind.size(), 1, ValueType::DOUBLE, mr);
                Value tq = Value::matrix(gapLen, 1, ValueType::DOUBLE, mr);
                for (keel::usize q = 0; q < ind.size(); ++q) {
                    if (outComplex)
                        xs.complexDataMut()[q] = keel::c64(origRe[ind[q]], origIm[ind[q]]);
                    else
                        xs.doubleDataMut()[q] = origRe[ind[q]];
                    ts.doubleDataMut()[q] = sVec[ind[q]];
                }
                for (keel::usize q = 0; q < gapLen; ++q)
                    tq.doubleDataMut()[q] = sVec[j0 + q];
                Value fargs[3] = {std::move(xs), std::move(ts), std::move(tq)};
                Value ytmp = callFn(*fhVal, keel::ArrayRef<Value>(fargs, 3));
                if (!(ytmp.numel() == 1 ||
                      (ytmp.dims().isVector() && ytmp.numel() == gapLen)))
                    throw Error(keel::format("{}: function handle output "
                                             "must be a scalar or a vector matching the gap length", fn),
                                0, 0, fn, "",
                                keel::format("numkit:{}:functionHandleInvalidOutputSize", fn));
                for (keel::usize q = 0; q < gapLen; ++q) {
                    const keel::usize yi = (ytmp.numel() == 1) ? 0 : q;
                    re[j0 + q] = ytmp.elemAsDouble(yi);
                    if (ytmp.isComplex())
                        im[j0 + q] = ytmp.complexData()[yi].imag();
                }
                j0 = j1 + 1;
            }
        } else if (m.as_ref() == "constant") {
            for (keel::usize j = 0; j < runLen; ++j) {
                if (ma[j]) {
                    re[j] = cv;
                    im[j] = cim;
                }
            }
        } else if (m.as_ref() == "movmean" || m.as_ref() == "movmedian") {
            // The moving statistic is computed on the ORIGINAL data
            // (NaN-omitted shrunk windows) — earlier fills must not
            // feed later windows.
            const bool isMean = (m.as_ref() == "movmean");
            keel::Array<keel::f64> winRe, winIm;
            for (keel::usize j = 0; j < runLen; ++j) {
                if (!ma[j]) continue;
                winRe.clear();
                winIm.clear();
                if (useSp) {
                    const keel::f64 loS = sVec[j] - hbSpan;
                    const keel::f64 hiS = sVec[j] + hfSpan;
                    for (keel::usize q = 0; q < runLen; ++q) {
                        const bool inW = winTwoElem
                                             ? (sVec[q] >= loS && sVec[q] <= hiS)
                                             : (sVec[q] >= loS && sVec[q] < hiS);
                        if (inW && !origMiss[q]) {
                            winRe.push_back(origRe[q]);
                            if (outComplex) winIm.push_back(origIm[q]);
                        }
                    }
                } else {
                    long lo = keel::as<long>(j) - hb; if (lo < 0) lo = 0;
                    long hi = keel::as<long>(j) + hf;
                    if (hi > keel::as<long>(runLen) - 1)
                        hi = keel::as<long>(runLen) - 1;
                    for (long q = lo; q <= hi; ++q) {
                        const keel::usize qi = keel::as<keel::usize>(q);
                        if (!origMiss[qi]) {
                            winRe.push_back(origRe[qi]);
                            if (outComplex) winIm.push_back(origIm[qi]);
                        }
                    }
                }
                if (winRe.empty()) continue;
                if (isMean) {
                    keel::f64 s = 0.0;
                    for (keel::f64 vv : winRe) s += vv;
                    re[j] = s / keel::f64(winRe.size());
                    if (outComplex) {
                        keel::f64 si = 0.0;
                        for (keel::f64 vv : winIm) si += vv;
                        im[j] = si / keel::f64(winIm.size());
                    }
                } else {
                    std::sort(winRe.begin(), winRe.end());
                    const keel::usize k2 = winRe.size();
                    re[j] = (k2 % 2 == 1) ? winRe[k2 / 2]
                                          : 0.5 * (winRe[k2 / 2 - 1] + winRe[k2 / 2]);
                    if (outComplex) {
                        std::sort(winIm.begin(), winIm.end());
                        im[j] = (k2 % 2 == 1) ? winIm[k2 / 2]
                                              : 0.5 * (winIm[k2 / 2 - 1] + winIm[k2 / 2]);
                    }
                }
            }
        } else {
            // Prepare working copy where origMiss positions are NaN so
            // fill_one_column sees true known points, then copy back only
            // at ma[j] == 1 (respecting MaxGap).
            keel::Array<keel::f64> workRe(runLen), workIm(runLen, 0.0);
            for (keel::usize j = 0; j < runLen; ++j) {
                workRe[j] = origMiss[j] ? nanv : origRe[j];
                if (outComplex) workIm[j] = origMiss[j] ? nanv : origIm[j];
            }
            fill_one_column(workRe.data(), runLen, m.as_ref(), 0.0, spPtr, mr);
            if (outComplex)
                fill_one_column(workIm.data(), runLen, m.as_ref(), 0.0, spPtr, mr);
            for (keel::usize j = 0; j < runLen; ++j) {
                if (ma[j]) {
                    re[j] = workRe[j];
                    if (outComplex) im[j] = workIm[j];
                }
            }
        }

        // EndValues post-pass on the leading [0..indBeg-1] and trailing
        // [indEnd+1..runLen-1] gaps that survived MaxGap.
        if (haveEnd && (indBeg > 0 || indEnd + 1 < runLen)) {
            auto applyEndAt = [&](keel::usize j, bool leading) {
                if (endMode == 1) { // 'none'
                    re[j] = origRe[j];
                    im[j] = origIm[j];
                    ma[j] = 0;
                } else if (endMode == 2) { // 'nearest'
                    const keel::usize src = leading ? indBeg : indEnd;
                    re[j] = origRe[src];
                    im[j] = origIm[src];
                    ma[j] = 1;
                } else if (endMode == 3) { // 'previous'
                    if (leading) {
                        re[j] = origRe[j];
                        im[j] = origIm[j];
                        ma[j] = 0;
                    } else {
                        re[j] = origRe[indEnd];
                        im[j] = origIm[indEnd];
                        ma[j] = 1;
                    }
                } else if (endMode == 4) { // 'next'
                    if (leading) {
                        re[j] = origRe[indBeg];
                        im[j] = origIm[indBeg];
                        ma[j] = 1;
                    } else {
                        re[j] = origRe[j];
                        im[j] = origIm[j];
                        ma[j] = 0;
                    }
                } else if (endMode == 5) { // constant
                    re[j] = evRe;
                    im[j] = evIm;
                    ma[j] = 1;
                }
            };
            for (keel::usize j = 0; j < indBeg; ++j) applyEndAt(j, true);
            for (keel::usize j = indEnd + 1; j < runLen; ++j) applyEndAt(j, false);
        }

        for (keel::usize j = 0; j < runLen; ++j) {
            const keel::usize lin = along2 ? k + j * r : k * runLen + j;
            if (outComplex) {
                B.complexDataMut()[lin] = keel::c64(re[j], im[j]);
            } else if (single) {
                B.singleDataMut()[lin] = keel::as<keel::f32>(re[j]);
            } else {
                B.doubleDataMut()[lin] = re[j];
            }
            if (fmd) {
                const bool stillNaN = std::isnan(re[j]) ||
                                      (outComplex && std::isnan(im[j]));
                fmd[lin] = (ma[j] && !stillNaN) ? 1 : 0;
            }
        }
    }
    out.B = std::move(B);
    out.filled = std::move(filledMap);
    return out;
}

// ── detrend (full) ────────────────────────────────────────────────────

namespace {

// Solve the small least-squares system for the design W (column-major
// n×k) against one data run: dst = src − W·coef. Rank-deficient pivots
// get a free variable fixed at 0 (a basic solution) so degenerate
// designs still subtract an exact fit through the data.
void detrendDesignFit(const keel::f64 *W, keel::usize n, keel::usize k,
                      const keel::f64 *src, keel::f64 *dst,
                      keel::Allocator *mr)
{
    keel::ArenaTemp sc = keel::get_scratch(mr);
    keel::Array<keel::f64> G(k * k, 0.0, sc.allocator()), rhs(k, 0.0, sc.allocator());
    for (keel::usize a = 0; a < k; ++a) {
        for (keel::usize b = a; b < k; ++b) {
            keel::f64 s = 0.0;
            for (keel::usize r = 0; r < n; ++r) s += W[a * n + r] * W[b * n + r];
            G[a * k + b] = s;
            G[b * k + a] = s;
        }
        keel::f64 s = 0.0;
        for (keel::usize r = 0; r < n; ++r) s += W[a * n + r] * src[r];
        rhs[a] = s;
    }
    // Gaussian elimination; a zero pivot pins that variable to 0.
    keel::Array<char> freeVar(k, 0, sc.allocator());
    for (keel::usize col = 0; col < k; ++col) {
        keel::usize piv = col;
        keel::f64 best = std::fabs(G[col * k + col]);
        for (keel::usize r2 = col + 1; r2 < k; ++r2) {
            const keel::f64 v = std::fabs(G[r2 * k + col]);
            if (v > best) { best = v; piv = r2; }
        }
        if (best <= 0.0) { freeVar[col] = 1; continue; }
        if (piv != col) {
            for (keel::usize j = 0; j < k; ++j)
                std::swap(G[col * k + j], G[piv * k + j]);
            std::swap(rhs[col], rhs[piv]);
        }
        const keel::f64 d = G[col * k + col];
        for (keel::usize r2 = col + 1; r2 < k; ++r2) {
            const keel::f64 f = G[r2 * k + col] / d;
            if (f == 0.0) continue;
            for (keel::usize j = col; j < k; ++j) G[r2 * k + j] -= f * G[col * k + j];
            rhs[r2] -= f * rhs[col];
        }
    }
    keel::Array<keel::f64> coef(k, 0.0, sc.allocator());
    for (keel::usize ii = k; ii-- > 0;) {
        if (freeVar[ii]) { coef[ii] = 0.0; continue; }
        keel::f64 s = rhs[ii];
        for (keel::usize j = ii + 1; j < k; ++j) s -= G[ii * k + j] * coef[j];
        coef[ii] = s / G[ii * k + ii];
    }
    for (keel::usize r = 0; r < n; ++r) {
        keel::f64 fit = 0.0;
        for (keel::usize a = 0; a < k; ++a) fit += W[a * n + r] * coef[a];
        dst[r] = src[r] - fit;
    }
}

} // anonymous namespace

Value detrend_full(const Value &x, keel::ArrayRef<Value> in,
                   keel::Allocator *mr)
{
    // detrendFull_argcheck: float-only gate (single/double, complex ok),
    // degree / bp / nanflag / NV contracts.
    argcheck::requireClassSet(x, kFloatComplex, "detrend");

    int polyDeg = 1;
    keel::Array<keel::f64> bp;                  // sample-axis positions (raw)
    bool bpLogical = false;
    keel::Array<keel::usize> bpLogIdx;
    bool omitNan = false;
    bool continuity = true;
    const Value *samplePointsRaw = nullptr;

    keel::usize i = 0;
    if (!in.empty()) {
        const Value &a0 = in[0];
        if (isStringValue(a0)) {
            const keel::String lowStr = lowerCopy(a0.toString().as_ref());
            const keel::StringRef low = lowStr.as_ref();
            if (low.size() >= 4 && startsWith("constant", low)) {
                polyDeg = 0;
                i = 1;
            } else if (!low.empty() && startsWith("linear", low)) {
                polyDeg = 1;
                i = 1;
            }
            // else: not a degree string → NV-name fallback (i stays 0).
        } else if (a0.isNumeric() || a0.isLogical()) {
            if (a0.isComplex() || a0.numel() != 1 ||
                (!a0.isLogical() &&
                 (a0.elemAsDouble(0) < 0.0 ||
                  std::floor(a0.elemAsDouble(0)) != a0.elemAsDouble(0))))
                throw Error("detrend: polynomial degree must be a "
                            "non-negative integer scalar, 'constant', or "
                            "'linear'",
                            0, 0, "detrend", "", "numkit:detrend:invalidTrendInput");
            polyDeg = keel::as<int>(a0.elemAsDouble(0));
            i = 1;
        } else {
            throw Error("detrend: polynomial degree must be a non-negative "
                        "integer scalar, 'constant', or 'linear'",
                        0, 0, "detrend", "", "numkit:detrend:invalidTrendInput");
        }
    }

    if (i < in.size() && !isStringValue(in[i])) {
        const Value &b = in[i];
        if (b.isLogical()) {
            bpLogical = true;
            for (keel::usize j = 0; j < b.numel(); ++j)
                if (b.logicalData()[j]) bpLogIdx.push_back(j);
        } else if (b.isNumeric() && !b.isComplex()) {
            bp.reserve(b.numel());
            for (keel::usize j = 0; j < b.numel(); ++j)
                bp.push_back(b.elemAsDouble(j));
        } else {
            throw Error("detrend: breakpoints must be a real numeric or "
                        "logical vector of sample positions",
                        0, 0, "detrend", "", "numkit:detrend:breakpointsInvalid");
        }
        ++i;
    }
    if (i < in.size() && isStringValue(in[i])) {
        const keel::String fStr = lowerCopy(in[i].toString().as_ref());
        const keel::StringRef f = fStr.as_ref();
        if (startsWith("omitnan", f) || startsWith("omitmissing", f)) {
            omitNan = true;
            ++i;
        } else if (startsWith("includenan", f) || startsWith("includemissing", f)) {
            ++i;
        }
    }

    if ((in.size() - i) % 2 != 0)
        throw Error("detrend: each parameter name needs a value "
                    "(unmatched name in the name-value list)",
                    0, 0, "detrend", "", "numkit:detrend:keyWithoutValue");
    for (; i + 1 < in.size(); i += 2) {
        if (!isStringValue(in[i]))
            throw Error("detrend: each parameter name needs a value",
                        0, 0, "detrend", "", "numkit:detrend:keyWithoutValue");
        const keel::String rawNm = in[i].toString();
        const keel::String nmStr = lowerCopy(rawNm.as_ref());
        const keel::StringRef nm = nmStr.as_ref();
        const Value &v = in[i + 1];
        if (nm.size() >= 4 && startsWith("continuous", nm)) {
            continuity = !((v.isLogical() || v.isNumeric()) && v.numel() == 1 &&
                           v.elemAsDouble(0) == 0.0);
        } else if (!nm.empty() && startsWith("samplepoints", nm)) {
            samplePointsRaw = &v;
        } else if ((!nm.empty() && startsWith("datavariables", nm)) ||
                   (!nm.empty() && startsWith("replacevalues", nm))) {
            throw Error(keel::format("detrend: '{}' is supported only for table inputs",
                                     rawNm.as_ref()),
                        0, 0, "detrend", "", "numkit:detrend:tableOnly");
        } else {
            throw Error(keel::format("detrend: unrecognized parameter name '{}'",
                                     rawNm.as_ref()),
                        0, 0, "detrend", "", "numkit:detrend:parseFlags");
        }
    }

    // Geometry: row vectors run whole; N-D flattens to dim0 × rest.
    const auto &d = x.dims();
    const bool isRow = (d.ndim() == 2 && d.dim(0) == 1);
    const keel::usize N = isRow ? keel::as<keel::usize>(x.numel())
                                : keel::as<keel::usize>(d.dim(0));
    const keel::usize ncols = isRow ? 1
        : keel::as<keel::usize>(d.numel() / std::max<keel::usize>(N, 1));
    const bool single = (x.type() == ValueType::SINGLE);
    const bool complex = (x.type() == ValueType::COMPLEX);

    keel::Array<keel::f64> sVec(N);
    keel::f64 minS = 1.0;
    if (samplePointsRaw) {
        keel::Array<keel::f64> sp = validateSamplePoints(
            *samplePointsRaw, N, x.isEmpty(), true, "detrend");
        minS = sp.empty() ? 0.0 : sp[0];
        for (keel::usize j = 0; j < N; ++j) sVec[j] = sp[j] - minS;
    } else {
        for (keel::usize j = 0; j < N; ++j) sVec[j] = keel::as<keel::f64>(j);
    }

    auto detrendRun = [&](const keel::f64 *src, keel::f64 *dst, keel::usize n) {
        const keel::f64 nanv = std::numeric_limits<keel::f64>::quiet_NaN();
        if (n == 0) return;
        // Knots on the centered abscissa sVec (0..sVec.back()).
        keel::Array<keel::f64> knots;
        if (bpLogical) {
            for (keel::usize j : bpLogIdx)
                if (j < n) knots.push_back(sVec[j]);
        } else {
            for (keel::f64 v : bp) knots.push_back(v - minS);
        }
        knots.push_back(0.0);                               // bookend s(1)
        if (n > 1) {
            keel::Array<keel::f64> keep;
            for (keel::f64 v : knots)
                if (v >= 0.0 && v < sVec[n - 1]) keep.push_back(v);
            knots = std::move(keep);
        }
        std::sort(knots.begin(), knots.end());
        knots.resize(std::unique(knots.begin(), knots.end()) - knots.begin());

        // NaN policy.
        keel::Array<keel::usize> orig;
        keel::Array<keel::f64> xs, ys;
        for (keel::usize j = 0; j < n; ++j) dst[j] = nanv;
        {
            bool anyNaN = false;
            for (keel::usize j = 0; j < n; ++j) anyNaN = anyNaN || std::isnan(src[j]);
            if (anyNaN && !omitNan) return;   // includenan poisons the run
            for (keel::usize j = 0; j < n; ++j) {
                if (omitNan && std::isnan(src[j])) continue;
                xs.push_back(sVec[j]);
                ys.push_back(src[j]);
                orig.push_back(j);
            }
        }
        const keel::usize m = xs.size();
        if (m == 0) return;

        const bool plainPoly = knots.size() <= 1 && !bpLogical && bp.empty();
        if (plainPoly || (knots.size() <= 1 && continuity)) {
            // Single global polynomial (no interior breakpoints).
            if (polyDeg == 0) {
                keel::f64 s = 0.0;
                for (keel::f64 v : ys) s += v;
                const keel::f64 mean = s / keel::f64(m);
                for (keel::usize j = 0; j < m; ++j) dst[orig[j]] = ys[j] - mean;
                return;
            }
            // Vandermonde on the normalized abscissa.
            const keel::f64 scale = (xs.back() != 0.0) ? xs.back() : 1.0;
            const keel::usize nb = keel::as<keel::usize>(polyDeg) + 1;
            keel::ArenaTemp sc = keel::get_scratch(mr);
            keel::Array<keel::f64> V(m * nb, 0.0, sc.allocator());
            for (keel::usize j = 0; j < m; ++j) {
                const keel::f64 a = xs[j] / scale;
                for (int pw = polyDeg; pw >= 1; --pw)
                    V[keel::as<keel::usize>(polyDeg - pw) * m + j] =
                        std::pow(a, keel::f64(pw));
                V[keel::as<keel::usize>(polyDeg) * m + j] = 1.0;
            }
            keel::Array<keel::f64> fitted(m, sc.allocator());
            detrendDesignFit(V.data(), m, nb, ys.data(), fitted.data(), mr);
            for (keel::usize j = 0; j < m; ++j) dst[orig[j]] = fitted[j];
            return;
        }

        if (!continuity) {
            // Segment-wise: segment k covers s in [knots[k], knots[k+1]).
            for (keel::usize j = 0; j < m; ++j) dst[orig[j]] = ys[j];
            for (keel::usize k = 0; k < knots.size(); ++k) {
                const keel::f64 lo = knots[k];
                const keel::f64 hi = (k + 1 < knots.size())
                                      ? knots[k + 1] : (sVec.back() + 1.0);
                keel::Array<keel::usize> seg;
                for (keel::usize j = 0; j < m; ++j)
                    if (xs[j] >= lo && xs[j] < hi) seg.push_back(j);
                if (seg.empty()) continue;
                if (polyDeg == 0 || seg.size() == 1) {
                    keel::f64 s = 0.0;
                    for (keel::usize j : seg) s += ys[j];
                    const keel::f64 mean = s / keel::f64(seg.size());
                    for (keel::usize j : seg) dst[orig[j]] = ys[j] - mean;
                    continue;
                }
                keel::f64 ms = 0.0;
                for (keel::usize j : seg) ms += xs[j];
                ms /= keel::f64(seg.size());
                keel::f64 ss = 0.0;
                for (keel::usize j : seg) ss += (xs[j] - ms) * (xs[j] - ms);
                const keel::f64 sd = std::sqrt(ss / keel::f64(seg.size() - 1));
                const keel::usize nb = keel::as<keel::usize>(polyDeg) + 1;
                keel::ArenaTemp sc = keel::get_scratch(mr);
                keel::Array<keel::f64> V(seg.size() * nb, 0.0, sc.allocator());
                keel::Array<keel::f64> yseg(seg.size(), sc.allocator());
                for (keel::usize q = 0; q < seg.size(); ++q) {
                    const keel::f64 a = (sd > 0.0) ? (xs[seg[q]] - ms) / sd : 0.0;
                    for (int pw = polyDeg; pw >= 1; --pw)
                        V[keel::as<keel::usize>(polyDeg - pw) * seg.size() + q] =
                            std::pow(a, keel::f64(pw));
                    V[keel::as<keel::usize>(polyDeg) * seg.size() + q] = 1.0;
                    yseg[q] = ys[seg[q]];
                }
                keel::Array<keel::f64> fitted(seg.size(), sc.allocator());
                detrendDesignFit(V.data(), seg.size(), nb, yseg.data(),
                                 fitted.data(), mr);
                for (keel::usize q = 0; q < seg.size(); ++q)
                    dst[orig[seg[q]]] = fitted[q];
            }
            return;
        }

        // Continuous piecewise design (reference b-basis).
        if (polyDeg == 0) {
            // Subtract the mean of the first segment.
            keel::usize b = 0, e = m - 1;
            if (knots.size() >= 2) {
                keel::f64 bestE = 1e300;
                for (keel::usize j = 0; j < m; ++j) {
                    if (std::fabs(xs[j] - knots[1]) < bestE) {
                        bestE = std::fabs(xs[j] - knots[1]);
                        e = j;
                    }
                }
            }
            keel::f64 s = 0.0;
            keel::usize cnt = 0;
            for (keel::usize j = 0; j < m; ++j)
                if (j >= b && j <= e) { s += ys[j]; ++cnt; }
            const keel::f64 mean = cnt ? s / keel::f64(cnt) : 0.0;
            for (keel::usize j = 0; j < m; ++j) dst[orig[j]] = ys[j] - mean;
            return;
        }
        const keel::f64 scaleS = (xs.back() != 0.0) ? xs.back() : 1.0;
        const keel::usize nRamp = knots.size();
        const keel::usize nb = nRamp * keel::as<keel::usize>(polyDeg) + 1;
        keel::ArenaTemp sc = keel::get_scratch(mr);
        keel::Array<keel::f64> W(m * nb, 0.0, sc.allocator());
        for (keel::usize rr = 0; rr < nRamp; ++rr) {
            const keel::f64 bk = knots[rr] / scaleS;
            for (int pw = polyDeg; pw >= 1; --pw) {
                keel::f64 *col = W.data() +
                    (rr * keel::as<keel::usize>(polyDeg) +
                     keel::as<keel::usize>(polyDeg - pw)) * m;
                for (keel::usize j = 0; j < m; ++j) {
                    const keel::f64 bb = xs[j] / scaleS - bk;
                    col[j] = (bb > 0.0) ? std::pow(bb, keel::f64(pw)) : 0.0;
                }
            }
        }
        for (keel::usize j = 0; j < m; ++j) W[(nb - 1) * m + j] = 1.0;
        keel::Array<keel::f64> fitted(m, sc.allocator());
        detrendDesignFit(W.data(), m, nb, ys.data(), fitted.data(), mr);
        for (keel::usize j = 0; j < m; ++j) dst[orig[j]] = fitted[j];
    };

    Value out = createForDims(d, complex ? ValueType::COMPLEX
                                : single ? ValueType::SINGLE
                                         : ValueType::DOUBLE, mr);

    keel::Array<keel::f64> re(N), im, dst(N);
    if (complex) im.resize(N);
    const keel::usize nRuns = isRow ? 1 : ncols;
    for (keel::usize k = 0; k < nRuns; ++k) {
        if (complex) {
            const keel::c64 *z = x.complexData() + (isRow ? 0 : k * N);
            for (keel::usize j = 0; j < N; ++j) {
                re[j] = z[j].real();
                im[j] = z[j].imag();
            }
        } else {
            for (keel::usize j = 0; j < N; ++j)
                re[j] = x.elemAsDouble(isRow ? j : k * N + j);
        }
        detrendRun(re.data(), dst.data(), N);
        if (complex) {
            keel::Array<keel::f64> dst2(N);
            detrendRun(im.data(), dst2.data(), N);
            keel::c64 *od = out.complexDataMut() + (isRow ? 0 : k * N);
            for (keel::usize j = 0; j < N; ++j) od[j] = keel::c64(dst[j], dst2[j]);
        } else if (single) {
            keel::f32 *od = out.singleDataMut() + (isRow ? 0 : k * N);
            for (keel::usize j = 0; j < N; ++j) od[j] = keel::as<keel::f32>(dst[j]);
        } else {
            keel::f64 *od = out.doubleDataMut() + (isRow ? 0 : k * N);
            for (keel::usize j = 0; j < N; ++j) od[j] = dst[j];
        }
    }
    return out;
}

} // namespace numkit::stats
