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

#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/builtin/datatypes.hpp>   // standardizeMissing (live bare-name impl)
#include <numkit/stats/distributions/normal.hpp>     // norminv (via descriptive_detail.hpp)
#include <numkit/stats/distributions/students_t.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "descriptive_extras_detail.hpp"
#include "descriptive_detail.hpp"   // narrowToSingle

namespace numkit::stats {
namespace {

std::string lowerCopy(std::string s)
{
    for (char &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
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
void parseWindowHalfSpans(const Value &wl, const char *fn, long &hb, long &hf)
{
    const bool numeric = wl.isNumeric() && !wl.isComplex();
    auto bad = [&]() {
        throw Error(std::string(fn) + ": window length must be a positive "
                    "scalar or a 2-element nonnegative [back forward] "
                    "vector of finite values",
                    0, 0, fn, "", std::string("nk:") + fn + ":invalidWindow");
    };
    if (!(numeric || wl.isLogical())) bad();
    if (wl.numel() == 1) {
        const double k = wl.elemAsDouble(0);
        if (!(k > 0.0) || !std::isfinite(k)) bad();
        const long ki = static_cast<long>(k);
        if (ki % 2 == 1) { hb = hf = (ki - 1) / 2; }
        else             { hb = ki / 2; hf = ki / 2 - 1; }
    } else if (wl.numel() == 2) {
        const double b = wl.elemAsDouble(0), f = wl.elemAsDouble(1);
        if (b < 0.0 || f < 0.0 || !std::isfinite(b) || !std::isfinite(f)) bad();
        hb = static_cast<long>(b);
        hf = static_cast<long>(f);
    } else {
        bad();
    }
}

// ── outlier detection (mirrors locateoutliers.m) ──────────────────────

struct ColStats {
    std::vector<uint8_t> mask;
    double center = std::numeric_limits<double>::quiet_NaN();
    double lo = -std::numeric_limits<double>::infinity();
    double hi = std::numeric_limits<double>::infinity();
};

// Iterative Grubbs test; tf is the significance level alpha. The final
// bounds are those of the last TESTED state (post-removal) and the mask
// is the STATIC rule against them — exactly locateoutliers 'grubbs'.
ColStats grubbsColumn(const double *x, std::size_t n, double alpha,
                      std::pmr::memory_resource *mr)
{
    ColStats r;
    r.mask.assign(n, 0);
    std::vector<std::size_t> idx;
    std::vector<double> v;
    idx.reserve(n); v.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isnan(x[i])) { v.push_back(x[i]); idx.push_back(i); }
    if (v.empty()) return r;
    double astd = 0.0, thr = 0.0, cen = 0.0;
    while (v.size() >= 3) {
        const std::size_t m = v.size();
        double s = 0.0;
        for (double e : v) s += e;
        cen = s / double(m);
        double ss = 0.0;
        for (double e : v) ss += (e - cen) * (e - cen);
        astd = std::sqrt(ss / double(m - 1));
        if (!(astd > 0.0)) break;
        std::size_t jmax = 0; double amax = -1.0;
        for (std::size_t j = 0; j < m; ++j) {
            const double ad = std::fabs(v[j] - cen);
            if (ad > amax) { amax = ad; jmax = j; }
        }
        const double dof = double(m) - 2.0;
        const double t = tinv(Value::scalar(alpha / (2.0 * double(m)), mr),
                              dof, mr).toScalar();
        thr = ((double(m) - 1.0) / std::sqrt(double(m))) *
              std::fabs(t) / std::sqrt(dof + t * t);
        if (amax / astd > thr) {
            v.erase(v.begin() + static_cast<std::ptrdiff_t>(jmax));
            idx.erase(idx.begin() + static_cast<std::ptrdiff_t>(jmax));
        } else {
            break;
        }
    }
    r.center = cen;
    r.lo = cen - astd * thr;
    r.hi = cen + astd * thr;
    for (std::size_t i = 0; i < n; ++i)
        r.mask[i] = (std::isnan(x[i]) ? 0
                    : ((x[i] > r.hi || x[i] < r.lo) ? 1 : 0));
    return r;
}

// Generalized ESD; tf is the significance level alpha. Thresholds come
// from iteration tindex = (last exceeding R)+1 clamped; the mask is the
// STATIC rule at tindex — exactly locateoutliers 'gesd'.
ColStats gesdColumn(const double *x, std::size_t n, double alpha,
                    long maxOutArg, std::pmr::memory_resource *mr)
{
    ColStats r;
    r.mask.assign(n, 0);
    std::vector<double> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isnan(x[i])) v.push_back(x[i]);
    const std::size_t N = v.size();
    if (N < 2) return r;
    const std::size_t maxo = (maxOutArg >= 0)
        ? static_cast<std::size_t>(maxOutArg)
        : static_cast<std::size_t>(std::ceil(0.1 * double(N)));
    const std::size_t rounds = std::min(N, maxo);
    std::vector<double> amean(rounds, 0.0), astd(rounds, 0.0),
                        lam(rounds, 0.0), R(rounds, 0.0);
    for (std::size_t it = 0; it < rounds; ++it) {
        const std::size_t m = v.size();
        const std::size_t nm1 = N - it;       // n-i+1 (ref 1-based)
        double s = 0.0;
        for (double e : v) s += e;
        amean[it] = s / double(m);
        double ss = 0.0;
        for (double e : v) ss += (e - amean[it]) * (e - amean[it]);
        astd[it] = (m >= 2) ? std::sqrt(ss / double(m - 1)) : 0.0;
        std::size_t jmax = 0; double amax = -1.0;
        for (std::size_t j = 0; j < m; ++j) {
            const double ad = std::fabs(v[j] - amean[it]);
            if (ad > amax) { amax = ad; jmax = j; }
        }
        R[it] = (astd[it] > 0.0) ? amax / astd[it]
                                : std::numeric_limits<double>::infinity();
        v.erase(v.begin() + static_cast<std::ptrdiff_t>(jmax));
        const double dof = double(N) - double(it) - 1.0;   // n-i-1
        const double pp = 1.0 - alpha / (2.0 * double(nm1));
        const double t = tinv(Value::scalar(pp, mr), dof, mr).toScalar();
        lam[it] = (double(nm1) - 1.0) * t / std::sqrt((dof + t * t) * double(nm1));
    }
    long last = -1;
    for (long it = 0; it < static_cast<long>(rounds); ++it)
        if (R[static_cast<std::size_t>(it)] > lam[static_cast<std::size_t>(it)])
            last = it;
    const std::size_t tindex = (last < 0)
        ? 0
        : std::min<std::size_t>(static_cast<std::size_t>(last) + 1, rounds - 1);
    r.center = amean[tindex];
    r.lo = amean[tindex] - astd[tindex] * lam[tindex];
    r.hi = amean[tindex] + astd[tindex] * lam[tindex];
    for (std::size_t i = 0; i < n; ++i)
        r.mask[i] = (std::isnan(x[i]) ? 0
                    : ((x[i] > r.hi || x[i] < r.lo) ? 1 : 0));
    return r;
}

// One detection run. Moving methods fill per-element center/lo/hi
// arrays instead of the scalar bounds.
ColStats detectRun(const std::vector<double> &x, const std::string &method,
                   double tf, long hb, long hf, double loP, double hiP,
                   long maxOut, std::vector<double> *movCenter,
                   std::vector<double> *movLo, std::vector<double> *movHi,
                   std::pmr::memory_resource *mr)
{
    const std::size_t n = x.size();
    ColStats r;
    r.mask.assign(n, 0);
    if (method == "grubbs") return grubbsColumn(x.data(), n, tf, mr);
    if (method == "gesd")   return gesdColumn(x.data(), n, tf, maxOut, mr);
    if (method == "percentiles") {
        FoDetect d = detect_percentile_column(x.data(), n, loP, hiP);
        r.center = d.center; r.lo = d.lo; r.hi = d.hi;
        for (std::size_t i = 0; i < n; ++i) r.mask[i] = d.mask[i];
        return r;
    }
    if (method == "movmedian" || method == "movmean") {
        const bool isMed = (method == "movmedian");
        const double nanv = std::numeric_limits<double>::quiet_NaN();
        if (movCenter) {
            movCenter->assign(n, nanv);
            movLo->assign(n, nanv);
            movHi->assign(n, nanv);
        }
        auto medOf = [](std::vector<double> &w) {
            const std::size_t k = w.size();
            return (k % 2 == 1) ? w[k / 2] : 0.5 * (w[k / 2 - 1] + w[k / 2]);
        };
        std::vector<double> win, dev, sorted;
        for (std::size_t i = 0; i < n; ++i) {
            long lo = static_cast<long>(i) - hb; if (lo < 0) lo = 0;
            long hi = static_cast<long>(i) + hf;
            if (hi > static_cast<long>(n) - 1) hi = static_cast<long>(n) - 1;
            win.clear();
            for (long j = lo; j <= hi; ++j)
                if (!std::isnan(x[static_cast<std::size_t>(j)]))
                    win.push_back(x[static_cast<std::size_t>(j)]);
            if (win.empty()) continue;
            double center, scale;
            if (isMed) {
                sorted = win;
                std::sort(sorted.begin(), sorted.end());
                center = medOf(sorted);
                dev.clear();
                for (double vv : win) dev.push_back(std::fabs(vv - center));
                std::sort(dev.begin(), dev.end());
                scale = 1.4826022185056 * medOf(dev);
            } else {
                double s = 0.0;
                for (double vv : win) s += vv;
                center = s / double(win.size());
                double ss = 0.0;
                for (double vv : win) ss += (vv - center) * (vv - center);
                scale = (win.size() >= 2)
                            ? std::sqrt(ss / double(win.size() - 1)) : 0.0;
            }
            const double loB = center - tf * scale, hiB = center + tf * scale;
            if (movCenter) {
                (*movCenter)[i] = center;
                (*movLo)[i] = loB;
                (*movHi)[i] = hiB;
            }
            if (!std::isnan(x[i]))
                r.mask[i] = (x[i] > hiB || x[i] < loB) ? 1 : 0;
        }
        r.center = std::numeric_limits<double>::quiet_NaN();
        r.lo = -std::numeric_limits<double>::infinity();
        r.hi = std::numeric_limits<double>::infinity();
        return r;
    }
    // median / mean / quartiles — static detectors.
    FoDetect d = detect_one_column(x.data(), n, method, tf);
    r.center = d.center; r.lo = d.lo; r.hi = d.hi;
    for (std::size_t i = 0; i < n; ++i) r.mask[i] = d.mask[i];
    return r;
}

} // anonymous namespace

// ── isoutlier ─────────────────────────────────────────────────────────

namespace {

// parseIsOutlierInput port. `in` excludes A itself. Throws nk-voiced
// errors with the probed rejection classes; unknown method / NV names
// participate in prefix matching exactly like the reference
// validatestring call (an NV-name match falls through to the NV loop).
void isoutlierParse(const Value &A, Span<const Value> in, bool forRmoutliers,
                    IsOutlierPlan &p, const Value *&outlierLocations)
{
    static constexpr std::string_view kMethods[] = {
        "median", "mean", "quartiles", "grubbs", "gesd",
        "movmedian", "movmean", "percentiles"};
    static constexpr std::string_view kNvRm[] = {
        "samplepoints", "datavariables", "thresholdfactor",
        "outlierlocations", "maxnumoutliers", "minnumoutliers"};
    static constexpr std::string_view kNvIso[] = {
        "samplepoints", "datavariables", "thresholdfactor",
        "maxnumoutliers", "outputformat"};
    const Span<const std::string_view> nvNames =
        forRmoutliers ? Span<const std::string_view>(kNvRm)
                      : Span<const std::string_view>(kNvIso);
    std::size_t i = 0;
    if (!in.empty() && isStringValue(in[0])) {
        std::vector<std::string_view> all(std::begin(kMethods),
                                          std::end(kMethods));
        all.insert(all.end(), nvNames.begin(), nvNames.end());
        const int hit = argcheck::matchStringChoice(
            in[0], Span<const std::string_view>(all));
        if (hit < 0)
            throw Error("isoutlier: unrecognized method or parameter name '" +
                        in[0].toString() + "' (expected median, mean, "
                        "quartiles, grubbs, gesd, movmedian, movmean, or "
                        "percentiles)",
                        0, 0, "isoutlier", "", "nk:isoutlier:badMethod");
        const std::string_view m = all[static_cast<std::size_t>(hit)];
        if (m == "movmedian" || m == "movmean") {
            p.method = m;
            if (in.size() < 2)
                throw Error("isoutlier: specify a window length after '" +
                            std::string(m) + "'",
                            0, 0, "isoutlier", "", "nk:isoutlier:missingWindow");
            parseWindowHalfSpans(in[1], "isoutlier", p.hb, p.hf);
            i = 2;
        } else if (m == "percentiles") {
            p.method = "percentiles";
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
                            "nk:isoutlier:badPercentiles");
            p.loP = lu.elemAsDouble(0);
            p.hiP = lu.elemAsDouble(1);
            i = 2;
        } else if (m != "samplepoints" && m != "datavariables" &&
                   m != "thresholdfactor" && m != "maxnumoutliers" &&
                   m != "minnumoutliers" && m != "outlierlocations" &&
                   m != "outputformat") {
            p.method = m;
            i = 1;
        }
        // else: an NV name — keep i = 0 so the NV loop sees it.
    }

    if (i < in.size() && !isStringValue(in[i])) {
        const Value &dv = in[i];
        const double dd = (dv.isNumeric() && !dv.isComplex() && dv.numel() == 1)
                              ? dv.elemAsDouble(0) : -1.0;
        if (!(dd >= 1.0) || std::floor(dd) != dd)
            throw Error("isoutlier: dim must be a positive integer scalar",
                        0, 0, "isoutlier", "", "nk:isoutlier:badDim");
        p.dim = static_cast<int>(dd);
        if (p.dim > 2 && A.dims().ndim() > 2)
            throw Error("isoutlier: reducing beyond dimension 2 is not "
                        "supported for N-D arrays in this revision",
                        0, 0, "isoutlier", "", "nk:isoutlier:dimND");
        p.dropIdx.push_back(i);   // rmoutliers: detection always runs at
                                  // the default dim; dim selects removal
        ++i;
    }

    if ((in.size() - i) % 2 != 0)
        throw Error("isoutlier: each parameter name needs a value "
                    "(unmatched name in the name-value list)",
                    0, 0, "isoutlier", "", "nk:isoutlier:nvMismatch");
    for (; i + 1 < in.size(); i += 2) {
        if (!isStringValue(in[i])) continue;
        const int hit = argcheck::matchStringChoice(
            in[i], Span<const std::string_view>(nvNames));
        if (hit < 0)
            throw Error("isoutlier: unrecognized parameter name '" +
                        in[i].toString() + "'",
                        0, 0, "isoutlier", "", "nk:isoutlier:badParam");
        const std::string_view name = nvNames[static_cast<std::size_t>(hit)];
        const Value &v = in[i + 1];
        if (name == "thresholdfactor") {
            if (p.method == "percentiles")
                throw Error("isoutlier: 'ThresholdFactor' is not supported "
                            "with the 'percentiles' method",
                            0, 0, "isoutlier", "",
                            "nk:isoutlier:unsupportedThreshold");
            const bool scalar = v.isNumeric() && !v.isComplex() && v.numel() == 1;
            if (!scalar)
                throw Error("isoutlier: ThresholdFactor must be a real "
                            "numeric scalar",
                            0, 0, "isoutlier", "", "nk:isoutlier:tfScalar");
            if (std::isnan(v.elemAsDouble(0)))
                throw Error("isoutlier: ThresholdFactor must not be NaN",
                            0, 0, "isoutlier", "", "nk:isoutlier:tfNaN");
            if (v.elemAsDouble(0) < 0.0)
                throw Error("isoutlier: ThresholdFactor must be nonnegative "
                            "(got " + std::to_string(v.elemAsDouble(0)) + ")",
                            0, 0, "isoutlier", "", "nk:isoutlier:tfNonneg");
            p.tf = v.elemAsDouble(0);
            p.haveTf = true;
        } else if (name == "maxnumoutliers") {
            const double mo = (v.isNumeric() && !v.isComplex() && v.numel() == 1)
                                  ? v.elemAsDouble(0) : -1.0;
            if (!(mo >= 1.0) || std::floor(mo) != mo)
                throw Error("isoutlier: MaxNumOutliers must be a positive "
                            "integer scalar",
                            0, 0, "isoutlier", "", "nk:isoutlier:badMaxNum");
            p.maxOut = static_cast<long>(mo);
        } else if (name == "minnumoutliers") {
            const double mo = (v.isNumeric() && !v.isComplex() && v.numel() == 1)
                                  ? v.elemAsDouble(0) : -1.0;
            if (!(mo >= 0.0) || std::floor(mo) != mo)
                throw Error("rmoutliers: MinNumOutliers must be a nonnegative "
                            "integer scalar",
                            0, 0, "rmoutliers", "", "nk:rmoutliers:badMinNum");
            p.minNum = static_cast<long>(mo);
            p.dropIdx.push_back(i);
            p.dropIdx.push_back(i + 1);
        } else if (name == "outlierlocations") {
            outlierLocations = &v;
            p.dropIdx.push_back(i);
            p.dropIdx.push_back(i + 1);
        } else if (name == "samplepoints") {
            throw Error("isoutlier: 'SamplePoints' is not supported in this "
                        "revision",
                        0, 0, "isoutlier", "", "nk:isoutlier:samplePoints");
        } else if (name == "datavariables") {
            throw Error("isoutlier: 'DataVariables' is supported only for "
                        "table inputs",
                        0, 0, "isoutlier", "", "nk:isoutlier:dataVariables");
        } else if (name == "outputformat") {
            throw Error("isoutlier: 'OutputFormat' is supported only for "
                        "table inputs",
                        0, 0, "isoutlier", "", "nk:isoutlier:outputFormat");
        }
    }
}

void isoutlierApplyTfDefaults(IsOutlierPlan &p, const Value &A)
{
    if (!p.haveTf) {
        if (p.method == "quartiles") p.tf = 1.5;
        else if (p.method == "grubbs" || p.method == "gesd") p.tf = 0.05;
        else p.tf = 3.0;
    } else if (p.method == "grubbs" || p.method == "gesd") {
        if (p.tf > 1.0)
            throw Error("isoutlier: 'ThresholdFactor' value must be between "
                        "0 and 1 when the method is '" + p.method + "'",
                        0, 0, "isoutlier", "", "nk:isoutlier:alphaRange");
    }
    if (p.maxOut >= 0) {
        if (p.method != "gesd")
            throw Error("isoutlier: 'MaxNumOutliers' is supported only for "
                        "the 'gesd' method",
                        0, 0, "isoutlier", "", "nk:isoutlier:maxNumGesdOnly");
        std::size_t dlen = 1;
        if (p.dim > 0) {
            dlen = static_cast<std::size_t>(A.dims().dim(p.dim - 1));
        } else {
            // first non-singleton dim (row vectors operate along dim 2)
            for (std::size_t k = 0; k < A.dims().ndim(); ++k) {
                if (A.dims().dim(k) > 1) {
                    dlen = static_cast<std::size_t>(A.dims().dim(k));
                    break;
                }
            }
        }
        if (static_cast<std::size_t>(p.maxOut) > dlen)
            throw Error("isoutlier: MaxNumOutliers must not exceed the "
                        "length of the operating dimension",
                        0, 0, "isoutlier", "", "nk:isoutlier:maxNumTooLarge");
    }
}

} // anonymous namespace

IsOutlierOut isoutlier_full(const Value &A, Span<const Value> in,
                            bool wantThresholds, std::pmr::memory_resource *mr)
{
    // isoutlierFull_argcheck: the real-float gate crossed by every call
    // path (interpreter and direct C++).
    argcheck::requireClassSet(A, kFloatReal, "isoutlier");

    IsOutlierPlan p;
    const Value *locs = nullptr;
    isoutlierParse(A, in, false, p, locs);
    if (locs)
        throw Error("isoutlier: 'OutlierLocations' is supported only in "
                    "rmoutliers",
                    0, 0, "isoutlier", "", "nk:isoutlier:outlierLocations");
    isoutlierApplyTfDefaults(p, A);

    int dim = p.dim;
    if (dim == 0) {
        dim = 1;
        for (std::size_t k = 0; k < A.dims().ndim(); ++k) {
            if (A.dims().dim(k) > 1) { dim = static_cast<int>(k) + 1; break; }
        }
        if (dim > 2) dim = 2;   // N-D beyond dim 2: reduce in the dim-2 view
    }
    const std::size_t r = static_cast<std::size_t>(A.dims().dim(0));
    const std::size_t cols = (A.dims().ndim() >= 2)
                                 ? A.numel() / std::max<std::size_t>(r, 1) : 1;
    // Runs along dim: dim 1 → contiguous columns; dim 2 → rows (stride r).
    const bool along2 = (dim == 2 && A.dims().ndim() >= 2);
    const std::size_t runLen = along2 ? cols : r;
    const std::size_t nRuns = along2 ? r : cols;

    IsOutlierOut out;
    if (A.numel() == 0) {
        out.tf = Value::matrix(0, 0, ValueType::LOGICAL, mr);
        if (wantThresholds)
            out.lo = out.hi = out.center =
                Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return out;
    }

    Value mask = Value::matrix(r, cols, ValueType::LOGICAL, mr);
    uint8_t *md = mask.logicalDataMut();

    const bool isMoving = (p.method == "movmedian" || p.method == "movmean");
    Value loM, hiM, ceM;
    double *lod = nullptr, *hid = nullptr, *ced = nullptr;
    if (wantThresholds) {
        if (isMoving) {
            loM = Value::matrix(r, cols, ValueType::DOUBLE, mr);
            hiM = Value::matrix(r, cols, ValueType::DOUBLE, mr);
            ceM = Value::matrix(r, cols, ValueType::DOUBLE, mr);
        } else {
            const std::size_t tq = along2 ? r : cols;
            loM = Value::matrix(1, tq, ValueType::DOUBLE, mr);
            hiM = Value::matrix(1, tq, ValueType::DOUBLE, mr);
            ceM = Value::matrix(1, tq, ValueType::DOUBLE, mr);
        }
        lod = loM.doubleDataMut();
        hid = hiM.doubleDataMut();
        ced = ceM.doubleDataMut();
    }

    std::vector<double> run(runLen);
    std::vector<double> movC, movL, movH;
    for (std::size_t k = 0; k < nRuns; ++k) {
        for (std::size_t j = 0; j < runLen; ++j)
            run[j] = A.elemAsDouble(along2 ? k + j * r : k * runLen + j);
        ColStats st = detectRun(run, p.method, p.tf, p.hb, p.hf, p.loP, p.hiP,
                                p.maxOut,
                                isMoving ? &movC : nullptr,
                                isMoving ? &movL : nullptr,
                                isMoving ? &movH : nullptr, mr);
        for (std::size_t j = 0; j < runLen; ++j) {
            const std::size_t lin = along2 ? k + j * r : k * runLen + j;
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

RmOutliersOut rmoutliers_full(const Value &A, Span<const Value> in,
                              std::pmr::memory_resource *mr)
{
    // rmoutliersFull_argcheck: the same real-float gate as isoutlier.
    argcheck::requireClassSet(A, kFloatReal, "rmoutliers");
    if (A.dims().ndim() > 2)
        throw Error("rmoutliers: N-D arrays are not supported",
                    0, 0, "rmoutliers", "", "nk:rmoutliers:ndArrays");

    IsOutlierPlan p;
    const Value *locs = nullptr;
    isoutlierParse(A, in, true, p, locs);
    isoutlierApplyTfDefaults(p, A);

    const std::size_t r = static_cast<std::size_t>(A.dims().dim(0));
    const std::size_t c = (A.dims().ndim() >= 2)
                              ? static_cast<std::size_t>(A.dims().dim(1)) : 1;
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
                        "nk:rmoutliers:badOutlierLocations");
        map = *locs;
        Value nanRow = Value::matrix(
            1, (A.dims().ndim() >= 2) ? A.numel() /
                std::max<std::size_t>(A.dims().dim(0), 1) : 1,
            ValueType::DOUBLE, mr);
        double *nd = nanRow.doubleDataMut();
        std::fill_n(nd, nanRow.numel(),
                    std::numeric_limits<double>::quiet_NaN());
        iso.lo = iso.hi = iso.center = nanRow;
    } else {
        // Detection sees the SAME tail minus the rmoutliers-only NV
        // pairs (isoutlier's own parse rejects those names).
        std::vector<Value> filtered;
        Span<const Value> detect = in;
        if (!p.dropIdx.empty()) {
            filtered.reserve(in.size() - p.dropIdx.size());
            for (std::size_t q = 0; q < in.size(); ++q)
                if (!std::count(p.dropIdx.begin(), p.dropIdx.end(), q))
                    filtered.push_back(in[q]);
            detect = Span<const Value>(filtered.data(), filtered.size());
        }
        iso = isoutlier_full(A, detect, true, mr);
        map = std::move(iso.tf);
    }
    const uint8_t *m = map.logicalData();

    if (A.numel() == 0) {
        out.B = Value::matrix(0, 0, single ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        out.ind = Value::matrix(0, 0, ValueType::LOGICAL, mr);
        return out;
    }

    const bool isVec = (A.dims().isVector() || A.isScalar());
    const long minNum = std::max<long>(p.minNum, 1);
    auto putB = [&](Value &B, std::size_t lin, double v) {
        if (single) B.singleDataMut()[lin] = static_cast<float>(v);
        else        B.doubleDataMut()[lin] = v;
    };
    if (isVec) {
        // Vector (either orientation): drop flagged entries.
        const std::size_t n = A.numel();
        std::size_t kept = 0;
        for (std::size_t i = 0; i < n; ++i) kept += m[i] ? 0u : 1u;
        const bool colOrient = (r != 1);
        Value B = colOrient
            ? Value::matrix(kept, 1, single ? ValueType::SINGLE : ValueType::DOUBLE, mr)
            : Value::matrix(1, kept, single ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        std::size_t w = 0;
        for (std::size_t i = 0; i < n; ++i)
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
        std::vector<std::size_t> cnt(r, 0);
        for (std::size_t col = 0; col < c; ++col)
            for (std::size_t i = 0; i < r; ++i)
                if (m[col * r + i]) ++cnt[i];
        std::vector<uint8_t> rm(r, 0);
        std::size_t kept = 0;
        for (std::size_t i = 0; i < r; ++i) {
            rm[i] = (cnt[i] >= static_cast<std::size_t>(minNum)) ? 1 : 0;
            kept += rm[i] ? 0u : 1u;
        }
        Value B = Value::matrix(kept, c,
                                single ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        std::size_t orow = 0;
        for (std::size_t i = 0; i < r; ++i) {
            if (rm[i]) continue;
            for (std::size_t col = 0; col < c; ++col)
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
    std::vector<std::size_t> cnt(c, 0);
    for (std::size_t col = 0; col < c; ++col)
        for (std::size_t i = 0; i < r; ++i)
            if (m[col * r + i]) ++cnt[col];
    std::vector<uint8_t> rm(c, 0);
    std::size_t kept = 0;
    for (std::size_t col = 0; col < c; ++col) {
        rm[col] = (cnt[col] >= static_cast<std::size_t>(minNum)) ? 1 : 0;
        kept += rm[col] ? 0u : 1u;
    }
    Value B = Value::matrix(r, kept, single ? ValueType::SINGLE : ValueType::DOUBLE, mr);
    std::size_t ocol = 0;
    for (std::size_t col = 0; col < c; ++col) {
        if (rm[col]) continue;
        for (std::size_t i = 0; i < r; ++i)
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

RmMissingOut rmmissing_full(const Value &A, Span<const Value> in,
                            std::pmr::memory_resource *mr)
{
    // rmmissingFull_argcheck: numeric / logical / char arrays (strings
    // and cells are a voiced gap — the reference supports them).
    const ValueType t = A.type();
    if (!(A.isNumeric() || A.isLogical() || t == ValueType::CHAR))
        throw Error("rmmissing: unsupported input type " +
                    std::string(mtypeName(t)) +
                    " (string and cell inputs are not supported in this "
                    "revision)",
                    0, 0, "rmmissing", "", "nk:rmmissing:typeUnsupported");
    if (A.dims().ndim() > 2)
        throw Error("rmmissing: N-D arrays are not supported",
                    0, 0, "rmmissing", "", "nk:rmmissing:ndArrays");

    int dim = 0;
    long minNum = 1;
    if (!in.empty() && !isStringValue(in[0])) {
        const Value &dv = in[0];
        const double dd = (dv.isNumeric() && !dv.isComplex() && dv.numel() == 1)
                              ? dv.elemAsDouble(0) : 0.0;
        if (std::floor(dd) != dd || dd < 1.0 || dd > 2.0)
            throw Error("rmmissing: dimension must be 1 or 2",
                        0, 0, "rmmissing", "", "nk:rmmissing:badDim");
        dim = static_cast<int>(dd);
    }
    for (std::size_t i = isStringValue(in.empty() ? Value() : in[0]) || in.empty()
                             ? 0 : 1;
         i + 1 < in.size() || (i + 1 == in.size() && false); i += 2) {
        if (i >= in.size() || !isStringValue(in[i])) continue;
        static constexpr std::string_view kRmNv[] = {"minnummissing",
                                                     "datavariables"};
        const int hit = argcheck::matchStringChoice(
            in[i], Span<const std::string_view>(kRmNv));
        if (hit == 0) {
            const Value &v = in[i + 1];
            const double mo = (v.isNumeric() && !v.isComplex() && v.numel() == 1)
                                  ? v.elemAsDouble(0) : -1.0;
            if (!(mo >= 0.0) || std::floor(mo) != mo)
                throw Error("rmmissing: MinNumMissing must be a nonnegative "
                            "integer scalar",
                            0, 0, "rmmissing", "", "nk:rmmissing:badMinNum");
            minNum = static_cast<long>(mo);
        } else if (hit == 1) {
            throw Error("rmmissing: 'DataVariables' is supported only for "
                        "table inputs",
                        0, 0, "rmmissing", "", "nk:rmmissing:dataVariables");
        } else {
            throw Error("rmmissing: unrecognized parameter name '" +
                        in[i].toString() + "'",
                        0, 0, "rmmissing", "", "nk:rmmissing:badParam");
        }
    }

    const bool floaty = (t == ValueType::DOUBLE || t == ValueType::SINGLE ||
                         t == ValueType::COMPLEX);
    const bool single = (t == ValueType::SINGLE);
    const std::size_t r = static_cast<std::size_t>(A.dims().dim(0));
    const std::size_t c = (A.dims().ndim() >= 2)
                              ? static_cast<std::size_t>(A.dims().dim(1)) : 1;
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
        std::fill_n(ind.logicalDataMut(), ind.numel(), 0);
        out.ind = std::move(ind);
        return out;
    }

    // NaN map (complex: NaN in either part).
    std::vector<uint8_t> miss(A.numel(), 0);
    if (t == ValueType::COMPLEX) {
        const Complex *z = A.complexData();
        for (std::size_t i = 0; i < A.numel(); ++i)
            miss[i] = (std::isnan(z[i].real()) || std::isnan(z[i].imag())) ? 1 : 0;
    } else if (single) {
        const float *xd = A.singleData();
        for (std::size_t i = 0; i < A.numel(); ++i)
            miss[i] = std::isnan(xd[i]) ? 1 : 0;
    } else {
        const double *xd = A.doubleData();
        for (std::size_t i = 0; i < A.numel(); ++i)
            miss[i] = std::isnan(xd[i]) ? 1 : 0;
    }

    auto copyIn = [&](std::size_t lin, Value &B, std::size_t lout) {
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
        const std::size_t n = A.numel();
        std::vector<std::size_t> keep;
        keep.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            if (!miss[i]) keep.push_back(i);
        const bool colOrient = (r != 1);
        Value B = colOrient ? Value::matrix(keep.size(), 1, outT, mr)
                            : Value::matrix(1, keep.size(), outT, mr);
        for (std::size_t w = 0; w < keep.size(); ++w) copyIn(keep[w], B, w);
        Value ind = colOrient ? Value::matrix(n, 1, ValueType::LOGICAL, mr)
                              : Value::matrix(1, n, ValueType::LOGICAL, mr);
        std::copy(miss.begin(), miss.end(), ind.logicalDataMut());
        out.B = std::move(B);
        out.ind = std::move(ind);
        return out;
    }

    if (byRows) {
        std::vector<std::size_t> cnt(r, 0);
        for (std::size_t col = 0; col < c; ++col)
            for (std::size_t i = 0; i < r; ++i)
                if (miss[col * r + i]) ++cnt[i];
        std::vector<uint8_t> rm(r, 0);
        std::vector<std::size_t> keep;
        for (std::size_t i = 0; i < r; ++i) {
            rm[i] = (cnt[i] >= static_cast<std::size_t>(minNum)) ? 1 : 0;
            if (!rm[i]) keep.push_back(i);
        }
        Value B = Value::matrix(keep.size(), c, outT, mr);
        for (std::size_t col = 0; col < c; ++col)
            for (std::size_t w = 0; w < keep.size(); ++w)
                copyIn(col * r + keep[w], B, col * keep.size() + w);
        Value ind = Value::matrix(r, 1, ValueType::LOGICAL, mr);
        std::copy(rm.begin(), rm.end(), ind.logicalDataMut());
        out.B = std::move(B);
        out.ind = std::move(ind);
        return out;
    }
    std::vector<std::size_t> cnt(c, 0);
    for (std::size_t col = 0; col < c; ++col)
        for (std::size_t i = 0; i < r; ++i)
            if (miss[col * r + i]) ++cnt[col];
    std::vector<uint8_t> rm(c, 0);
    std::vector<std::size_t> keep;
    for (std::size_t col = 0; col < c; ++col) {
        rm[col] = (cnt[col] >= static_cast<std::size_t>(minNum)) ? 1 : 0;
        if (!rm[col]) keep.push_back(col);
    }
    Value B = Value::matrix(r, keep.size(), outT, mr);
    for (std::size_t w = 0; w < keep.size(); ++w)
        for (std::size_t i = 0; i < r; ++i)
            copyIn(keep[w] * r + i, B, w * r + i);
    Value ind = Value::matrix(1, c, ValueType::LOGICAL, mr);
    std::copy(rm.begin(), rm.end(), ind.logicalDataMut());
    out.B = std::move(B);
    out.ind = std::move(ind);
    return out;
}

// ── standardizeMissing ────────────────────────────────────────────────

Value standardizeMissing_full(const Value &A, const Value &indicators,
                              std::pmr::memory_resource *mr)
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
std::vector<double> parseConstants(const Value &C, std::size_t nVects,
                                   const char *fn, const char *eidSuffix,
                                   std::vector<double> *imagParts = nullptr)
{
    if (!C.isNumeric())
        throw Error(std::string(fn) + ": fill constant must be numeric "
                    "(got " + mtypeName(C.type()) + ")",
                    0, 0, fn, "",
                    std::string("nk:") + fn + ":constantNumeric");
    // Complex constants are legal for complex inputs (the parts fill
    // the real / imaginary buffers separately).
    const bool cplx = C.isComplex();
    if (cplx && !imagParts)
        throw Error(std::string(fn) + ": complex fill constants are "
                    "supported only for complex inputs",
                    0, 0, fn, "",
                    std::string("nk:") + fn + ":constantNumeric");
    auto partOf = [&](std::size_t i) {
        if (!cplx) return C.elemAsDouble(i);
        const Complex z = C.complexData()[i];
        if (imagParts) (*imagParts)[i] = z.imag();
        return z.real();
    };
    if (C.numel() == 1) {
        if (imagParts && imagParts->size() < 1) imagParts->resize(1, 0.0);
        return {partOf(0)};
    }
    if (C.numel() != nVects)
        throw Error(std::string(fn) + ": fill constant must be a scalar or "
                    "a " + std::to_string(nVects) + "-element vector" +
                    (nVects <= 1 ? "" : " (one per slice)"),
                    0, 0, fn, "",
                    std::string("nk:") + fn + ":SizeConstant" + eidSuffix);
    if (imagParts) imagParts->resize(C.numel(), 0.0);
    std::vector<double> v(C.numel());
    for (std::size_t i = 0; i < C.numel(); ++i) v[i] = partOf(i);
    return v;
}

} // anonymous namespace

FillMissingOut fillmissing_full(const Value &A, Span<const Value> in,
                                bool wantFilled, std::pmr::memory_resource *mr)
{
    // fillmissingFull_argcheck: supported-array gate + method contract.
    const char *fn = "fillmissing";
    const ValueType at = A.type();
    if (!(A.isNumeric() || A.isLogical() || at == ValueType::CHAR))
        throw Error(std::string(fn) + ": unsupported input type " +
                    std::string(mtypeName(at)) + " (string and cell inputs "
                    "are not supported in this revision)",
                    0, 0, fn, std::string("nk:") + fn + ":firstInputInvalid");
    if (A.dims().ndim() > 2)
        throw Error(std::string(fn) + ": N-D arrays are not supported",
                    0, 0, fn, "", std::string("nk:") + fn + ":ndArrays");

    static constexpr std::string_view kFmMethods[] = {
        "constant", "previous", "next", "nearest", "linear",
        "spline", "pchip", "movmean", "movmedian", "makima", "knn"};
    const Span<const std::string_view> methods =
        Span<const std::string_view>(kFmMethods);
    if (in.empty() || !isStringValue(in[0]))
        throw Error(std::string(fn) + ": fill method must be one of "
                    "'constant', 'previous', 'next', 'nearest', 'linear', "
                    "'movmean', 'movmedian' (spline/pchip/makima/knn and "
                    "custom function handles are deferred)",
                    0, 0, fn, "", std::string("nk:") + fn + ":methodInvalid");
    const int mhit = argcheck::matchStringChoice(in[0], methods);
    if (mhit < 0)
        throw Error(std::string(fn) + ": fill method must be one of "
                    "'constant', 'previous', 'next', 'nearest', 'linear', "
                    "'movmean', 'movmedian' (got '" + in[0].toString() +
                    "'; ambiguous or unrecognized)",
                    0, 0, fn, "", std::string("nk:") + fn + ":methodInvalid");
    const std::string m(methods[static_cast<std::size_t>(mhit)]);
    if (m == "spline" || m == "pchip" || m == "makima" || m == "knn")
        throw Error(std::string(fn) + ": the '" + m + "' fill method is not "
                    "supported in this revision (constant, previous, next, "
                    "nearest, linear, movmean, movmedian)",
                    0, 0, fn, "", std::string("nk:") + fn + ":methodDeferred");

    long hb = 0, hf = 0;
    std::size_t i = 1;
    if (m == "constant") {
        if (in.size() < 2)
            throw Error(std::string(fn) + ": specify a fill value after the "
                        "'constant' method",
                        0, 0, fn, "", std::string("nk:") + fn + ":constantInput");
        ++i;
    } else if (m == "movmean" || m == "movmedian") {
        if (in.size() < 2)
            throw Error(std::string(fn) + ": specify a window length after "
                        "the '" + m + "' method",
                        0, 0, fn, "", std::string("nk:") + fn + ":windowInput");
        parseWindowHalfSpans(in[1], fn, hb, hf);
        ++i;
    }

    // Optional positional dim.
    int dim = 0;
    if (i < in.size() && !isStringValue(in[i]) &&
        (in[i].isNumeric() || in[i].isLogical())) {
        const Value &dv = in[i];
        const double dd = (dv.numel() == 1 && !dv.isComplex())
                              ? dv.elemAsDouble(0) : 0.0;
        if (std::floor(dd) != dd || !(dd >= 1.0) || !std::isfinite(dd))
            throw Error(std::string(fn) + ": dimension must be a positive "
                        "integer scalar",
                        0, 0, fn, "",
                        std::string("nk:") + fn + ":dimensionInvalid");
        dim = static_cast<int>(dd);
        ++i;
    }

    // Name-value tail.
    static constexpr std::string_view kFmNv[] = {
        "endvalues", "datavariables", "replacevalues", "samplepoints",
        "missinglocations", "maxgap", "distance"};
    const Span<const std::string_view> nvNames =
        Span<const std::string_view>(kFmNv);
    const Value *constRaw = (m == "constant") ? &in[1] : nullptr;
    const Value *endRaw = nullptr;
    int endMode = 0;   // 0 extrap / 1 none / 2 nearest / 3 prevEnd / 4 nextEnd / 5 const
    bool haveEnd = false;
    double endConst = 0.0;
    const Value *missingLocations = nullptr;
    if ((in.size() - i) % 2 != 0)
        throw Error(std::string(fn) + ": each parameter name needs a value "
                    "(unmatched value in the name-value list)",
                    0, 0, fn, "", std::string("nk:") + fn + ":nameValuePairs");
    for (; i + 1 < in.size(); i += 2) {
        if (!isStringValue(in[i]))
            throw Error(std::string(fn) + ": expected a parameter name",
                        0, 0, fn, "", std::string("nk:") + fn + ":nameValuePairs");
        const int hit = argcheck::matchStringChoice(
            in[i], Span<const std::string_view>(nvNames));
        if (hit < 0)
            throw Error(std::string(fn) + ": unrecognized parameter name '" +
                        in[i].toString() + "'",
                        0, 0, fn, "", std::string("nk:") + fn + ":nameValueNames");
        const std::string_view name = nvNames[static_cast<std::size_t>(hit)];
        const Value &v = in[i + 1];
        if (name == "endvalues") {
            haveEnd = true;
            if (isStringValue(v)) {
                static constexpr std::string_view kEndModes[] = {
                    "extrap", "previous", "next", "nearest", "none"};
                const int eh = argcheck::matchStringChoice(
                    v, Span<const std::string_view>(kEndModes));
                if (eh < 0)
                    throw Error(std::string(fn) + ": 'EndValues' must be "
                                "'extrap', 'previous', 'next', 'nearest', "
                                "'none', or a numeric constant",
                                0, 0, fn, "",
                                std::string("nk:") + fn + ":endValuesInvalid");
                endMode = (eh == 0) ? 0 : (eh == 4) ? 1 : (eh == 3) ? 2
                          : (eh == 1) ? 3 : 4;
            } else {
                endMode = 5;
                endRaw = &v;
            }
        } else if (name == "missinglocations") {
            if (!v.isLogical() || v.numel() != A.numel() ||
                v.dims().dim(0) != A.dims().dim(0))
                throw Error(std::string(fn) + ": 'MissingLocations' must be "
                            "a logical array the same size as the input",
                            0, 0, fn, "",
                            std::string("nk:") + fn + ":badMissingLocations");
            missingLocations = &v;
        } else if (name == "samplepoints" || name == "maxgap" ||
                   name == "distance") {
            throw Error(std::string(fn) + ": '" + std::string(name) +
                        "' is not supported " "in this revision",
                        0, 0, fn, "",
                        std::string("nk:") + fn + ":optionDeferred");
        } else {
            throw Error(std::string(fn) + ": '" + std::string(name) +
                        "' is supported " "only for table inputs",
                        0, 0, fn, "", std::string("nk:") + fn + ":tableOnly");
        }
    }

    // Direction: default = first non-singleton (row vectors along 2).
    if (dim == 0) {
        dim = 1;
        if (A.dims().ndim() >= 2 && A.dims().dim(0) == 1 &&
            A.dims().dim(1) > 1)
            dim = 2;
    }
    const std::size_t r = static_cast<std::size_t>(A.dims().dim(0));
    const std::size_t c = (A.dims().ndim() >= 2)
                              ? static_cast<std::size_t>(A.dims().dim(1)) : 1;
    const bool along2 = (dim == 2 && A.dims().ndim() >= 2);
    const std::size_t runLen = along2 ? c : r;
    const std::size_t nRuns = along2 ? r : c;

    std::vector<double> constVals, constImag;
    if (constRaw) {
        if (at == ValueType::COMPLEX) constImag.resize(1, 0.0);
        constVals = parseConstants(*constRaw, nRuns, fn, "",
                                   at == ValueType::COMPLEX ? &constImag
                                                            : nullptr);
    }
    std::vector<double> endImag;
    if (endRaw) {
        if (at == ValueType::COMPLEX) endImag.resize(1, 0.0);
        endConst = parseConstants(*endRaw, nRuns, fn, "Extrap",
                                  at == ValueType::COMPLEX ? &endImag
                                                           : nullptr)[0];
    }

    // Integer / logical / char inputs cannot represent NaN: flow-through.
    const bool floaty = (at == ValueType::DOUBLE || at == ValueType::SINGLE ||
                         at == ValueType::COMPLEX);
    FillMissingOut out;
    if (!floaty || A.numel() == 0) {
        out.B = A;
        if (wantFilled) {
            Value f = Value::matrix(r, c, ValueType::LOGICAL, mr);
            std::fill_n(f.logicalDataMut(), f.numel(), 0);
            out.filled = std::move(f);
        }
        return out;
    }

    const bool single = (at == ValueType::SINGLE);
    const bool complex = (at == ValueType::COMPLEX);
    Value B = complex ? Value::matrix(r, c, ValueType::COMPLEX, mr)
             : single ? Value::matrix(r, c, ValueType::SINGLE, mr)
                      : Value::matrix(r, c, ValueType::DOUBLE, mr);
    Value filledMap = wantFilled ? Value::matrix(r, c, ValueType::LOGICAL, mr)
                                 : Value();
    uint8_t *fmd = wantFilled ? filledMap.logicalDataMut() : nullptr;

    // Original missing map (or the explicit override).
    std::vector<uint8_t> missMap(A.numel(), 0);
    if (missingLocations) {
        const uint8_t *lm = missingLocations->logicalData();
        std::copy(lm, lm + A.numel(), missMap.begin());
    } else if (complex) {
        const Complex *z = A.complexData();
        for (std::size_t j = 0; j < A.numel(); ++j)
            missMap[j] = (std::isnan(z[j].real()) ||
                          std::isnan(z[j].imag())) ? 1 : 0;
    } else {
        for (std::size_t j = 0; j < A.numel(); ++j)
            missMap[j] = std::isnan(A.elemAsDouble(j)) ? 1 : 0;
    }

    const double nanv = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> re(runLen), im, origRe(runLen);
    if (complex) im.resize(runLen);
    for (std::size_t k = 0; k < nRuns; ++k) {
        const double cv = (constVals.size() > 1) ? constVals[k]
                          : (constVals.empty() ? 0.0 : constVals[0]);
        const double cim = (complex && constVals.size() > 1)
                               ? constImag[k]
                               : (complex && !constImag.empty()
                                      ? constImag[0] : 0.0);
        for (std::size_t j = 0; j < runLen; ++j) {
            const std::size_t lin = along2 ? k + j * r : k * runLen + j;
            if (complex) {
                const Complex z = A.complexData()[lin];
                re[j] = z.real();
                im[j] = z.imag();
            } else {
                re[j] = A.elemAsDouble(lin);
            }
            origRe[j] = re[j];
        }
        // Main fill (on the real part; complex positional methods split).
        if (m == "constant") {
            if (complex) {
                // A complex element is missing when EITHER part is NaN
                // (the other part may be a stored 0).
                for (std::size_t j = 0; j < runLen; ++j)
                    if (std::isnan(re[j]) || std::isnan(im[j])) {
                        re[j] = cv;
                        im[j] = cim;
                    }
            } else {
                for (std::size_t j = 0; j < runLen; ++j)
                    if (std::isnan(re[j])) re[j] = cv;
            }
        } else if (m == "movmean" || m == "movmedian") {
            // The moving statistic is computed on the ORIGINAL data
            // (NaN-omitted shrunk windows) — earlier fills must not
            // feed later windows.
            const bool isMean = (m == "movmean");
            std::vector<double> win;
            for (std::size_t j = 0; j < runLen; ++j) {
                if (!std::isnan(re[j])) continue;
                long lo = static_cast<long>(j) - hb; if (lo < 0) lo = 0;
                long hi = static_cast<long>(j) + hf;
                if (hi > static_cast<long>(runLen) - 1)
                    hi = static_cast<long>(runLen) - 1;
                win.clear();
                for (long q = lo; q <= hi; ++q)
                    if (!std::isnan(origRe[static_cast<std::size_t>(q)]))
                        win.push_back(origRe[static_cast<std::size_t>(q)]);
                if (win.empty()) continue;
                if (isMean) {
                    double s = 0.0;
                    for (double vv : win) s += vv;
                    re[j] = s / double(win.size());
                } else {
                    std::sort(win.begin(), win.end());
                    const std::size_t k2 = win.size();
                    re[j] = (k2 % 2 == 1) ? win[k2 / 2]
                                          : 0.5 * (win[k2 / 2 - 1] + win[k2 / 2]);
                }
            }
            if (complex)
                for (std::size_t j = 0; j < runLen; ++j)
                    if (std::isnan(im[j])) im[j] = 0.0;
        } else {
            fill_one_column(re.data(), runLen, m, 0.0);
            if (complex)
                fill_one_column(im.data(), runLen, m, 0.0);
        }
        // EndValues post-pass on the leading/trailing gap runs.
        if (haveEnd) {
            const FmEndMode mode = (endMode == 0) ? FmEndMode::Extrap
                               : (endMode == 1) ? FmEndMode::None
                               : (endMode == 2) ? FmEndMode::Nearest
                               : (endMode == 3) ? FmEndMode::PrevEnd
                               : (endMode == 4) ? FmEndMode::NextEnd
                                                : FmEndMode::Const;
            apply_end_values_column(origRe.data(), re.data(), runLen, mode,
                                    endConst);
            if (complex)
                apply_end_values_column(origRe.data(), im.data(), runLen,
                                        mode, endMode == 5 ? 0.0 : endConst);
        }
        for (std::size_t j = 0; j < runLen; ++j) {
            const std::size_t lin = along2 ? k + j * r : k * runLen + j;
            if (complex) {
                B.complexDataMut()[lin] = Complex(re[j], im[j]);
            } else if (single) {
                B.singleDataMut()[lin] = static_cast<float>(re[j]);
            } else {
                B.doubleDataMut()[lin] = re[j];
            }
            if (fmd) fmd[lin] = missMap[lin];
        }
    }
    (void)nanv;
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
void detrendDesignFit(const double *W, std::size_t n, std::size_t k,
                      const double *src, double *dst,
                      std::pmr::memory_resource *mr)
{
    ScratchArena sc(mr);
    ScratchVec<double> G(k * k, 0.0, &sc), rhs(k, 0.0, &sc);
    for (std::size_t a = 0; a < k; ++a) {
        for (std::size_t b = a; b < k; ++b) {
            double s = 0.0;
            for (std::size_t r = 0; r < n; ++r) s += W[a * n + r] * W[b * n + r];
            G[a * k + b] = s;
            G[b * k + a] = s;
        }
        double s = 0.0;
        for (std::size_t r = 0; r < n; ++r) s += W[a * n + r] * src[r];
        rhs[a] = s;
    }
    // Gaussian elimination; a zero pivot pins that variable to 0.
    std::vector<char> freeVar(k, 0);
    for (std::size_t col = 0; col < k; ++col) {
        std::size_t piv = col;
        double best = std::fabs(G[col * k + col]);
        for (std::size_t r2 = col + 1; r2 < k; ++r2) {
            const double v = std::fabs(G[r2 * k + col]);
            if (v > best) { best = v; piv = r2; }
        }
        if (best <= 0.0) { freeVar[col] = 1; continue; }
        if (piv != col) {
            for (std::size_t j = 0; j < k; ++j)
                std::swap(G[col * k + j], G[piv * k + j]);
            std::swap(rhs[col], rhs[piv]);
        }
        const double d = G[col * k + col];
        for (std::size_t r2 = col + 1; r2 < k; ++r2) {
            const double f = G[r2 * k + col] / d;
            if (f == 0.0) continue;
            for (std::size_t j = col; j < k; ++j) G[r2 * k + j] -= f * G[col * k + j];
            rhs[r2] -= f * rhs[col];
        }
    }
    ScratchVec<double> coef(k, 0.0, &sc);
    for (std::size_t ii = k; ii-- > 0;) {
        if (freeVar[ii]) { coef[ii] = 0.0; continue; }
        double s = rhs[ii];
        for (std::size_t j = ii + 1; j < k; ++j) s -= G[ii * k + j] * coef[j];
        coef[ii] = s / G[ii * k + ii];
    }
    for (std::size_t r = 0; r < n; ++r) {
        double fit = 0.0;
        for (std::size_t a = 0; a < k; ++a) fit += W[a * n + r] * coef[a];
        dst[r] = src[r] - fit;
    }
}

} // anonymous namespace

Value detrend_full(const Value &x, Span<const Value> in,
                   std::pmr::memory_resource *mr)
{
    // detrendFull_argcheck: float-only gate (single/double, complex ok),
    // degree / bp / nanflag / NV contracts.
    argcheck::requireClassSet(x, kFloatComplex, "detrend");

    int polyDeg = 1;
    std::vector<double> bp;                  // sample-axis positions (raw)
    bool bpLogical = false;
    std::vector<std::size_t> bpLogIdx;
    bool omitNan = false;
    bool continuity = true;

    std::size_t i = 0;
    if (!in.empty()) {
        const Value &a0 = in[0];
        if (isStringValue(a0)) {
            const std::string low = lowerCopy(a0.toString());
            if (low.size() >= 4 && std::string("constant").rfind(low, 0) == 0) {
                polyDeg = 0;
                i = 1;
            } else if (!low.empty() &&
                       std::string("linear").rfind(low, 0) == 0) {
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
                            0, 0, "detrend", "", "nk:detrend:invalidTrendInput");
            polyDeg = static_cast<int>(a0.elemAsDouble(0));
            i = 1;
        } else {
            throw Error("detrend: polynomial degree must be a non-negative "
                        "integer scalar, 'constant', or 'linear'",
                        0, 0, "detrend", "", "nk:detrend:invalidTrendInput");
        }
    }

    if (i < in.size() && !isStringValue(in[i])) {
        const Value &b = in[i];
        if (b.isLogical()) {
            bpLogical = true;
            for (std::size_t j = 0; j < b.numel(); ++j)
                if (b.logicalData()[j]) bpLogIdx.push_back(j);
        } else if (b.isNumeric() && !b.isComplex()) {
            bp.reserve(b.numel());
            for (std::size_t j = 0; j < b.numel(); ++j)
                bp.push_back(b.elemAsDouble(j));
        } else {
            throw Error("detrend: breakpoints must be a real numeric or "
                        "logical vector of sample positions",
                        0, 0, "detrend", "", "nk:detrend:breakpointsInvalid");
        }
        ++i;
    }
    if (i < in.size() && isStringValue(in[i])) {
        const std::string f = lowerCopy(in[i].toString());
        if (std::string("omitnan").rfind(f, 0) == 0 ||
            std::string("omitmissing").rfind(f, 0) == 0) {
            omitNan = true;
            ++i;
        } else if (std::string("includenan").rfind(f, 0) == 0 ||
                   std::string("includemissing").rfind(f, 0) == 0) {
            ++i;
        }
    }

    if ((in.size() - i) % 2 != 0)
        throw Error("detrend: each parameter name needs a value "
                    "(unmatched name in the name-value list)",
                    0, 0, "detrend", "", "nk:detrend:keyWithoutValue");
    for (; i + 1 < in.size(); i += 2) {
        if (!isStringValue(in[i]))
            throw Error("detrend: each parameter name needs a value",
                        0, 0, "detrend", "", "nk:detrend:keyWithoutValue");
        const std::string nm = lowerCopy(in[i].toString());
        const Value &v = in[i + 1];
        if (nm.size() >= 4 && std::string("continuous").rfind(nm, 0) == 0) {
            continuity = !((v.isLogical() || v.isNumeric()) && v.numel() == 1 &&
                           v.elemAsDouble(0) == 0.0);
        } else if (std::string("samplepoints").rfind(nm, 0) == 0) {
            throw Error("detrend: 'SamplePoints' is not supported in this "
                        "revision",
                        0, 0, "detrend", "", "nk:detrend:samplePoints");
        } else if (std::string("datavariables").rfind(nm, 0) == 0 ||
                   std::string("replacevalues").rfind(nm, 0) == 0) {
            throw Error("detrend: '" + in[i].toString() + "' is supported "
                        "only for table inputs",
                        0, 0, "detrend", "", "nk:detrend:tableOnly");
        } else {
            throw Error("detrend: unrecognized parameter name '" +
                        in[i].toString() + "'",
                        0, 0, "detrend", "", "nk:detrend:parseFlags");
        }
    }

    // Geometry: row vectors run whole; N-D flattens to dim0 × rest.
    const auto &d = x.dims();
    const bool isRow = (d.ndim() == 2 && d.dim(0) == 1);
    const std::size_t N = isRow ? static_cast<std::size_t>(x.numel())
                                : static_cast<std::size_t>(d.dim(0));
    const std::size_t ncols = isRow ? 1
        : static_cast<std::size_t>(d.numel() / std::max<std::size_t>(N, 1));
    const bool single = (x.type() == ValueType::SINGLE);
    const bool complex = (x.type() == ValueType::COMPLEX);

    auto detrendRun = [&](const double *src, double *dst, std::size_t n) {
        const double nanv = std::numeric_limits<double>::quiet_NaN();
        if (n == 0) return;
        // Knots on the default abscissa s = 0..n-1.
        std::vector<double> knots;
        if (bpLogical) {
            for (std::size_t j : bpLogIdx)
                if (j < n) knots.push_back(double(j));
        } else {
            for (double v : bp) knots.push_back(v - 1.0);   // ref shift
        }
        knots.push_back(0.0);                               // bookend s(1)
        if (n > 1) {
            std::vector<double> keep;
            for (double v : knots)
                if (v >= 0.0 && v < double(n - 1)) keep.push_back(v);
            knots = keep;
        }
        std::sort(knots.begin(), knots.end());
        knots.erase(std::unique(knots.begin(), knots.end()), knots.end());

        // NaN policy.
        std::vector<std::size_t> orig;
        std::vector<double> xs, ys;
        for (std::size_t j = 0; j < n; ++j) dst[j] = nanv;
        {
            bool anyNaN = false;
            for (std::size_t j = 0; j < n; ++j) anyNaN = anyNaN || std::isnan(src[j]);
            if (anyNaN && !omitNan) return;   // includenan poisons the run
            for (std::size_t j = 0; j < n; ++j) {
                if (omitNan && std::isnan(src[j])) continue;
                xs.push_back(double(j));
                ys.push_back(src[j]);
                orig.push_back(j);
            }
        }
        const std::size_t m = xs.size();
        if (m == 0) return;

        const bool plainPoly = knots.size() <= 1 && !bpLogical && bp.empty();
        if (plainPoly || (knots.size() <= 1 && continuity)) {
            // Single global polynomial (no interior breakpoints).
            if (polyDeg == 0) {
                double s = 0.0;
                for (double v : ys) s += v;
                const double mean = s / double(m);
                for (std::size_t j = 0; j < m; ++j) dst[orig[j]] = ys[j] - mean;
                return;
            }
            // Vandermonde on the normalized abscissa.
            const double scale = (xs.back() != 0.0) ? xs.back() : 1.0;
            const std::size_t nb = static_cast<std::size_t>(polyDeg) + 1;
            ScratchArena sc(mr);
            ScratchVec<double> V(m * nb, 0.0, &sc);
            for (std::size_t j = 0; j < m; ++j) {
                const double a = xs[j] / scale;
                for (int pw = polyDeg; pw >= 1; --pw)
                    V[static_cast<std::size_t>(polyDeg - pw) * m + j] =
                        std::pow(a, double(pw));
                V[static_cast<std::size_t>(polyDeg) * m + j] = 1.0;
            }
            std::vector<double> fitted(m);
            detrendDesignFit(V.data(), m, nb, ys.data(), fitted.data(), mr);
            for (std::size_t j = 0; j < m; ++j) dst[orig[j]] = fitted[j];
            return;
        }

        if (!continuity) {
            // Segment-wise: segment k covers s in [knots[k], knots[k+1]).
            for (std::size_t j = 0; j < m; ++j) dst[orig[j]] = ys[j];
            for (std::size_t k = 0; k < knots.size(); ++k) {
                const double lo = knots[k];
                const double hi = (k + 1 < knots.size())
                                      ? knots[k + 1] : double(n);
                std::vector<std::size_t> seg;
                for (std::size_t j = 0; j < m; ++j)
                    if (xs[j] >= lo && xs[j] < hi) seg.push_back(j);
                if (seg.empty()) continue;
                if (polyDeg == 0 || seg.size() == 1) {
                    double s = 0.0;
                    for (std::size_t j : seg) s += ys[j];
                    const double mean = s / double(seg.size());
                    for (std::size_t j : seg) dst[orig[j]] = ys[j] - mean;
                    continue;
                }
                double ms = 0.0;
                for (std::size_t j : seg) ms += xs[j];
                ms /= double(seg.size());
                double ss = 0.0;
                for (std::size_t j : seg) ss += (xs[j] - ms) * (xs[j] - ms);
                const double sd = std::sqrt(ss / double(seg.size() - 1));
                const std::size_t nb = static_cast<std::size_t>(polyDeg) + 1;
                ScratchArena sc(mr);
                ScratchVec<double> V(seg.size() * nb, 0.0, &sc);
                std::vector<double> yseg(seg.size());
                for (std::size_t q = 0; q < seg.size(); ++q) {
                    const double a = (sd > 0.0) ? (xs[seg[q]] - ms) / sd : 0.0;
                    for (int pw = polyDeg; pw >= 1; --pw)
                        V[static_cast<std::size_t>(polyDeg - pw) * seg.size() + q] =
                            std::pow(a, double(pw));
                    V[static_cast<std::size_t>(polyDeg) * seg.size() + q] = 1.0;
                    yseg[q] = ys[seg[q]];
                }
                std::vector<double> fitted(seg.size());
                detrendDesignFit(V.data(), seg.size(), nb, yseg.data(),
                                 fitted.data(), mr);
                for (std::size_t q = 0; q < seg.size(); ++q)
                    dst[orig[seg[q]]] = fitted[q];
            }
            return;
        }

        // Continuous piecewise design (reference b-basis).
        if (polyDeg == 0) {
            // Subtract the mean of the first segment.
            std::size_t b = 0, e = m - 1;
            if (knots.size() >= 2) {
                double bestE = 1e300;
                for (std::size_t j = 0; j < m; ++j) {
                    if (std::fabs(xs[j] - knots[1]) < bestE) {
                        bestE = std::fabs(xs[j] - knots[1]);
                        e = j;
                    }
                }
            }
            double s = 0.0;
            std::size_t cnt = 0;
            for (std::size_t j = 0; j < m; ++j)
                if (j >= b && j <= e) { s += ys[j]; ++cnt; }
            const double mean = cnt ? s / double(cnt) : 0.0;
            for (std::size_t j = 0; j < m; ++j) dst[orig[j]] = ys[j] - mean;
            return;
        }
        const double scaleS = (xs.back() != 0.0) ? xs.back() : 1.0;
        const std::size_t nRamp = knots.size();
        const std::size_t nb = nRamp * static_cast<std::size_t>(polyDeg) + 1;
        ScratchArena sc(mr);
        ScratchVec<double> W(m * nb, 0.0, &sc);
        for (std::size_t rr = 0; rr < nRamp; ++rr) {
            const double bk = knots[rr] / scaleS;
            for (int pw = polyDeg; pw >= 1; --pw) {
                double *col = W.data() +
                    (rr * static_cast<std::size_t>(polyDeg) +
                     static_cast<std::size_t>(polyDeg - pw)) * m;
                for (std::size_t j = 0; j < m; ++j) {
                    const double bb = xs[j] / scaleS - bk;
                    col[j] = (bb > 0.0) ? std::pow(bb, double(pw)) : 0.0;
                }
            }
        }
        for (std::size_t j = 0; j < m; ++j) W[(nb - 1) * m + j] = 1.0;
        std::vector<double> fitted(m);
        detrendDesignFit(W.data(), m, nb, ys.data(), fitted.data(), mr);
        for (std::size_t j = 0; j < m; ++j) dst[orig[j]] = fitted[j];
    };

    Value out = createForDims(d, complex ? ValueType::COMPLEX
                                : single ? ValueType::SINGLE
                                         : ValueType::DOUBLE, mr);

    std::vector<double> re(N), im, dst(N);
    if (complex) im.resize(N);
    const std::size_t nRuns = isRow ? 1 : ncols;
    for (std::size_t k = 0; k < nRuns; ++k) {
        if (complex) {
            const Complex *z = x.complexData() + (isRow ? 0 : k * N);
            for (std::size_t j = 0; j < N; ++j) {
                re[j] = z[j].real();
                im[j] = z[j].imag();
            }
        } else {
            for (std::size_t j = 0; j < N; ++j)
                re[j] = x.elemAsDouble(isRow ? j : k * N + j);
        }
        detrendRun(re.data(), dst.data(), N);
        if (complex) {
            std::vector<double> dst2(N);
            detrendRun(im.data(), dst2.data(), N);
            Complex *od = out.complexDataMut() + (isRow ? 0 : k * N);
            for (std::size_t j = 0; j < N; ++j) od[j] = Complex(dst[j], dst2[j]);
        } else if (single) {
            float *od = out.singleDataMut() + (isRow ? 0 : k * N);
            for (std::size_t j = 0; j < N; ++j) od[j] = static_cast<float>(dst[j]);
        } else {
            double *od = out.doubleDataMut() + (isRow ? 0 : k * N);
            for (std::size_t j = 0; j < N; ++j) od[j] = dst[j];
        }
    }
    return out;
}

} // namespace numkit::stats
