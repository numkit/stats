// toolboxes/stats/src/regress/regress_full.cpp
//
// Full parse+validate+compute contracts for the regression entries —
// ridge / nlparci / glmval (probed against the reference engine R2025b:
// build/probe76.m R/N/G batteries). The register adapters marshal only;
// the numeric kernels (ridge's normal equations, nlparci's covariance,
// glmval's inverse link) live in their existing TUs and are reused.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/regress/regress.hpp>   // ridge / nlparci / glmval + full entries

#include <numkit/stats/distributions/chi2.hpp>         // chi2inv (glmval simultaneous)
#include <numkit/stats/distributions/fisher_f.hpp>     // finv (glmval simultaneous)
#include <numkit/stats/distributions/normal.hpp>       // norminv (glmval bounds)
#include <numkit/stats/distributions/students_t.hpp>   // tinv (nlparci, glmval bounds)
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::stats {

namespace {

GlmLink parseGlmLinkName(const Value &v, const char *fn)
{
    keel::String raw = v.toString();
    keel::String s = raw.clone();
    for (char &c : s)
        c = keel::to_ascii_lower(c);
    if (s.empty() || s == "canonical") return GlmLink::Identity;
    if (s == "identity")    return GlmLink::Identity;
    if (s == "logit")       return GlmLink::Logit;
    if (s == "log" || s == "exp") return GlmLink::Log;
    if (s == "reciprocal")  return GlmLink::Reciprocal;
    if (s == "probit")      return GlmLink::Probit;
    throw Error(keel::format("{}: unrecognized link function name '{}' "
                             "(expected identity, logit, log, "
                             "probit, reciprocal, or canonical).", fn, raw),
                0, 0, fn, "",
                keel::format("numkit:{}:unrecognizedLink", fn));
}

} // namespace

// ── ridge ─────────────────────────────────────────────────────────────

namespace {
void ridge_full_argcheck(const Value &y, const Value &X, const Value &flag, bool &scaled)
{
    scaled = true;
    if (!flag.isEmpty()) {
        const double f = flag.elemAsDouble(0);
        if (f == 0.0) scaled = false;
        else if (f != 1.0)
            throw Error(keel::format("ridge: the scaling flag must be 0 or 1 (got {}).", f),
                        0, 0, "ridge", "", "numkit:ridge:badScalingFlag");
    }

    if (X.dims().ndim() != 2)
        throw Error("ridge: X must be a matrix.",
                    0, 0, "ridge", "", "numkit:ridge:invalidData");
    const keel::usize n = static_cast<keel::usize>(X.dims().dim(0));
    if (static_cast<keel::usize>(y.dims().dim(0)) != n
        || y.dims().ndim() != 2)
        throw Error(keel::format("ridge: the number of rows in Y must equal the number "
                                 "of rows in X (Y has {}, X has {}).",
                                 y.dims().dim(0), n),
                    0, 0, "ridge", "", "numkit:ridge:inputSizeMismatch");
    if (y.dims().dim(1) != 1)
        throw Error(keel::format("ridge: Y must be a column vector (got {}x{}).",
                                 y.dims().dim(0), y.dims().dim(1)),
                    0, 0, "ridge", "", "numkit:ridge:invalidData");
}
} // namespace

Value ridge_full(const Value &y, const Value &X, const Value &kVec,
                 const Value &flag, keel::Allocator *mr)
{
    bool scaled = true;
    ridge_full_argcheck(y, X, flag, scaled);
    const keel::usize n = static_cast<keel::usize>(X.dims().dim(0));

    // Missing-value removal: any row with a NaN in y or X (probe76 R07).
    const keel::usize p = static_cast<keel::usize>(X.dims().dim(1));
    keel::Array<keel::usize> keep;
    keep.reserve(n);
    for (keel::usize i = 0; i < n; ++i) {
        if (std::isnan(y.elemAsDouble(i))) continue;
        bool ok = true;
        for (keel::usize j = 0; j < p && ok; ++j)
            if (std::isnan(X.elemAsDouble(i + j * n))) ok = false;
        if (ok) keep.push_back(i);
    }
    const keel::usize m = keep.size();
    if (m == 0)
        throw Error("ridge: no observations remain after removing rows "
                    "with missing values.",
                    0, 0, "ridge", "", "numkit:ridge:invalidData");
    Value yc = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    Value Xc = Value::matrix(m, p, ValueType::DOUBLE, mr);
    for (keel::usize k = 0; k < m; ++k) {
        yc.doubleDataMut()[k] = y.elemAsDouble(keep[k]);
        for (keel::usize j = 0; j < p; ++j)
            Xc.doubleDataMut()[k + j * m] =
                X.elemAsDouble(keep[k] + j * n);
    }
    return ridge(yc, Xc, kVec, scaled, mr);
}

// ── nlparci ───────────────────────────────────────────────────────────

Value nlparci_full(const Value &beta, const Value &resid,
                   keel::ArrayRef<Value> tail, keel::Allocator *mr)
{
    // Tail forms (reference): (beta, resid, J[, alpha]) or the NV form
    // (beta, resid, 'covar', Sigma[, 'alpha', a]) — the named-argument
    // path keys off a char/string 3rd argument.
    Value J, Sigma;
    double alpha = 0.05;
    if (!tail.empty() && (tail[0].isChar() || tail[0].isString())) {
        for (keel::usize i = 0; i + 1 < tail.size(); i += 2) {
            static constexpr keel::StringRef kNv[] = { "jacobian",
                                                       "covariance",
                                                       "alpha" };
            const int hit = argcheck::matchStringChoice(
                tail[i], keel::ArrayRef<keel::StringRef>(kNv));
            if (hit == 0) J = tail[i + 1];
            else if (hit == 1) Sigma = tail[i + 1];
            else if (hit == 2) alpha = tail[i + 1].elemAsDouble(0);
            else {
                keel::String rawName = tail[i].toString();
                throw Error(keel::format("nlparci: unrecognized parameter name '{}'.", rawName),
                            0, 0, "nlparci", "",
                            "numkit:nlparci:badParamName");
            }
        }
    } else {
        if (tail.size() >= 1) J = tail[0];
        if (tail.size() >= 2 && !tail[1].isEmpty())
            alpha = tail[1].elemAsDouble(0);
    }
    if (resid.isEmpty() || (J.isEmpty() && Sigma.isEmpty()))
        throw Error("nlparci: requires (BETA, RESID, J) or the 'covar' "
                    "form.",
                    0, 0, "nlparci", "", "numkit:nlparci:tooFewInputs");
    if (beta.isComplex() || J.isComplex())
        throw Error("nlparci: cannot compute confidence intervals for "
                    "complex parameters.",
                    0, 0, "nlparci", "", "numkit:nlparci:complexParams");
    if (!std::isnan(alpha) && !(alpha > 0.0 && alpha < 1.0))
        throw Error(keel::format("nlparci: ALPHA must be a scalar between 0 and 1 (got {}).", alpha),
                    0, 0, "nlparci", "", "numkit:nlparci:badAlpha");

    // Drop NaN residuals (and their Jacobian rows).
    const keel::usize nAll = static_cast<keel::usize>(resid.numel());
    keel::Array<keel::usize> keep;
    keep.reserve(nAll);
    for (keel::usize i = 0; i < nAll; ++i)
        if (!std::isnan(resid.elemAsDouble(i))) keep.push_back(i);
    const keel::usize n = keep.size();
    const keel::usize pp = static_cast<keel::usize>(beta.numel());
    if (n <= pp)
        throw Error(keel::format("nlparci: the degrees of freedom v = n - p must be "
                                 "positive (n = {}, p = {}).", n, pp),
                    0, 0, "nlparci", "", "numkit:nlparci:noDOF");

    keel::Array<double> se(pp, 0.0);
    if (!Sigma.isEmpty()) {
        // se = sqrt(diag(Sigma)) (probe76 N03).
        for (keel::usize j = 0; j < pp; ++j)
            se[j] = std::sqrt(Sigma.elemAsDouble(j + j * pp));
    } else {
        if (static_cast<keel::usize>(J.dims().dim(0)) != nAll
            || static_cast<keel::usize>(J.dims().dim(1)) != pp)
            throw Error(keel::format("nlparci: the size of J does not match the sizes "
                                     "of BETA and RESID (J is {}x{}, expected {}x{}).",
                                     J.dims().dim(0), J.dims().dim(1), nAll, pp),
                        0, 0, "nlparci", "", "numkit:nlparci:inputSizeMismatch");
        Value Jc = Value::matrix(n, pp, ValueType::DOUBLE, mr);
        for (keel::usize k = 0; k < n; ++k)
            for (keel::usize j = 0; j < pp; ++j)
                Jc.doubleDataMut()[k + j * n] =
                    J.elemAsDouble(keep[k] + j * nAll);
        Value residC = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        for (keel::usize k = 0; k < n; ++k)
            residC.doubleDataMut()[k] = resid.elemAsDouble(keep[k]);
        // Reuse the kernel: se_j = sqrt(mse · [(J'J)^{-1}]_jj).
        Value ci = nlparci(beta, residC, Jc, alpha, mr);
        // The kernel's half-widths: recompute via its covariance route —
        // extract se from ci directly (ci = beta ± se·t).
        const double tcrit = tinv(Value::scalar(1.0 - alpha / 2.0, mr),
                                  static_cast<double>(n - pp), mr).toScalar();
        Value out = Value::matrix(pp, 2, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize j = 0; j < pp; ++j) {
            const double half = (ci.elemAsDouble(j + pp) - ci.elemAsDouble(j)) / 2.0;
            const double sej = tcrit != 0.0 ? half / tcrit : 0.0;
            se[j] = sej;
            od[j] = beta.elemAsDouble(j) - half;
            od[j + pp] = beta.elemAsDouble(j) + half;
        }
        return out;
    }

    const double tcrit = tinv(Value::scalar(1.0 - alpha / 2.0, mr),
                              static_cast<double>(n - pp), mr).toScalar();
    Value out = Value::matrix(pp, 2, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (keel::usize j = 0; j < pp; ++j) {
        const double d = se[j] * tcrit;
        od[j] = beta.elemAsDouble(j) - d;
        od[j + pp] = beta.elemAsDouble(j) + d;
    }
    return out;
}

// ── glmval ────────────────────────────────────────────────────────────

GlmvalResult glmval_full_bounds(const Value &b, const Value &X, const Value &link,
                                keel::ArrayRef<Value> tail, bool wantBounds,
                                FnHandle customInvLink,
                                keel::Allocator *mr)
{
    GlmLink lk = GlmLink::Identity;
    bool useCustomCallback = false;
    if (link.isChar() || link.isString()) {
        lk = parseGlmLinkName(link, "glmval");
    } else if (link.isStruct()) {
        Value invField;
        if (link.hasField("Inverse")) invField = link.field("Inverse");
        else if (link.hasField("inverse")) invField = link.field("inverse");
        else
            throw Error("glmval: custom link struct must contain an 'Inverse' field.",
                        0, 0, "glmval", "", "numkit:glmval:badLink");
        const bool isInvFh = invField.isFuncHandle();
        if (isInvFh && static_cast<bool>(customInvLink)) {
            useCustomCallback = true;
        } else if (invField.isChar() || invField.isString()) {
            lk = parseGlmLinkName(invField, "glmval");
        } else {
            throw Error("glmval: custom link 'Inverse' field must be a function handle or link name.",
                        0, 0, "glmval", "", "numkit:glmval:badLink");
        }
    } else if (link.isCell() && link.numel() >= 3) {
        const Value &invField = link.cellAt(2);
        const bool isInvFh = invField.isFuncHandle();
        if (isInvFh && static_cast<bool>(customInvLink)) {
            useCustomCallback = true;
        } else if (invField.isChar() || invField.isString()) {
            lk = parseGlmLinkName(invField, "glmval");
        } else {
            throw Error("glmval: custom link cell must have a function handle or link name as 3rd element.",
                        0, 0, "glmval", "", "numkit:glmval:badLink");
        }
    } else {
        throw Error("glmval: link must be a char row, string scalar, or custom link struct.",
                    0, 0, "glmval", "", "numkit:glmval:badLink");
    }

    // STATS: a non-string 4th argument is the stats struct (used by the bounds output).
    bool haveStats = false;
    Value statsVal;
    keel::usize i = 0;
    if (!tail.empty() && !tail[0].isChar() && !tail[0].isString()) {
        haveStats = !tail[0].isEmpty();
        statsVal = tail[0];
        i = 1;
    }

    // Option parse: the NV form when the next argument is a string,
    // else the legacy positional (clev, N, offset, const).
    double clev = 0.95;
    Value Nv = Value::scalar(1.0, mr);
    Value offset = Value::scalar(0.0, mr);
    bool isconst = true;
    bool simultaneous = false;
    if (i < tail.size() && (tail[i].isChar() || tail[i].isString())) {
        bool haveConfidence = false, haveAlpha = false;
        bool haveSize = false, haveBinomSize = false;
        for (; i + 1 < tail.size(); i += 2) {
            keel::String rawName = tail[i].toString();
            keel::String name = rawName.clone();
            for (char &c : name)
                c = keel::to_ascii_lower(c);
            const Value &v = tail[i + 1];
            if      (name == "confidence")   { clev = v.elemAsDouble(0); haveConfidence = true; }
            else if (name == "alpha")        { clev = 1.0 - v.elemAsDouble(0); haveAlpha = true; }
            else if (name == "size")         { Nv = v; haveSize = true; }
            else if (name == "binomialsize") { Nv = v; haveBinomSize = true; }
            else if (name == "offset")       offset = v;
            else if (name == "constant")     isconst = (v.toString() == "on");
            else if (name == "simultaneous") {
                if (v.isLogical()) simultaneous = v.toBool();
                else if (v.isChar() || v.isString()) {
                    keel::String sv = v.toString();
                    for (char &c : sv)
                        c = keel::to_ascii_lower(c);
                    simultaneous = (sv == "true" || sv == "on");
                } else {
                    simultaneous = (v.elemAsDouble(0) != 0.0);
                }
            } else
                throw Error(keel::format("glmval: unrecognized parameter name '{}'.", rawName),
                            0, 0, "glmval", "", "numkit:glmval:badParamName");
        }
        if (haveConfidence && haveAlpha)
            throw Error("glmval: cannot supply both the 'Confidence' and "
                        "'Alpha' parameters.",
                        0, 0, "glmval", "", "numkit:glmval:argCombination");
        if (haveSize && haveBinomSize)
            throw Error("glmval: cannot supply both the 'Size' and "
                        "'BinomialSize' parameters.",
                        0, 0, "glmval", "", "numkit:glmval:argCombination");
    } else {
        // Legacy positional: (clev, N, offset, const, simultaneous).
        if (i < tail.size() && !tail[i].isEmpty())
            clev = tail[i].elemAsDouble(0);
        if (i + 1 < tail.size() && !tail[i + 1].isEmpty())
            Nv = tail[i + 1];
        if (i + 2 < tail.size() && !tail[i + 2].isEmpty())
            offset = tail[i + 2];
        if (i + 3 < tail.size() && !tail[i + 3].isEmpty())
            isconst = (tail[i + 3].toString() == "on");
        if (i + 4 < tail.size() && !tail[i + 4].isEmpty())
            simultaneous = (tail[i + 4].elemAsDouble(0) != 0.0);
    }

    if (wantBounds && (!haveStats || !statsVal.isStruct()))
        throw Error("glmval: the STATS parameter is required to compute "
                    "confidence bounds.",
                    0, 0, "glmval", "", "numkit:glmval:needSTATS");

    // Row X with a 2-coefficient (or scalar-no-const) model is a column
    // of points (probe76 V02).
    Value Xw = X;
    if (X.dims().dim(0) == 1 && X.dims().dim(1) > 1
        && ((b.numel() == 2 && isconst) || (b.numel() == 1 && !isconst))) {
        Xw = Value::matrix(static_cast<keel::usize>(X.numel()), 1,
                           ValueType::DOUBLE, mr);
        for (keel::usize k = 0; k < static_cast<keel::usize>(X.numel()); ++k)
            Xw.doubleDataMut()[k] = X.elemAsDouble(k);
    }

    // BinomialSize / Offset shape gates (probe76: WrongArgSize classes).
    const keel::usize m = static_cast<keel::usize>(Xw.dims().dim(0));
    if (!Nv.isScalar()
        && (Nv.dims().dim(1) != 1
            || static_cast<keel::usize>(Nv.numel()) != m))
        throw Error("glmval: BinomialSize must be a scalar or a column "
                    "vector with one entry per row of X.",
                    0, 0, "glmval", "", "numkit:glmval:wrongArgSizeBinomialSize");
    if (!offset.isScalar()
        && (offset.dims().dim(1) != 1
            || static_cast<keel::usize>(offset.numel()) != m))
        throw Error("glmval: Offset must be a scalar or a column vector "
                    "with one entry per row of X.",
                    0, 0, "glmval", "", "numkit:glmval:wrongArgSizeOffset");

    auto invLink = [&](double eta) -> double {
        if (useCustomCallback) {
            Value arg = Value::scalar(eta, mr);
            Value res;
            customInvLink(keel::ArrayRef<Value>(&arg, 1), keel::MutArrayRef<Value>(&res, 1), mr);
            return res.elemAsDouble(0);
        }
        switch (lk) {
        case GlmLink::Identity:   return eta;
        case GlmLink::Logit:      return 1.0 / (1.0 + std::exp(-eta));
        case GlmLink::Log:        return std::exp(eta);
        case GlmLink::Reciprocal: return 1.0 / eta;
        case GlmLink::Probit:     return 0.5 * std::erfc(-eta / std::sqrt(2.0));
        }
        return eta;
    };

    const keel::usize rows = static_cast<keel::usize>(Xw.dims().dim(0));
    const keel::usize cols = static_cast<keel::usize>(Xw.dims().dim(1));
    const bool scalarBeta = b.numel() == 1;
    const keel::usize wantCoef = isconst ? cols + 1 : cols;
    if (!scalarBeta
        && static_cast<keel::usize>(b.numel()) != wantCoef)
        throw Error(keel::format("glmval: length(beta) must equal the number of "
                                 "columns of X plus one when the constant is on (got {} coefficients for {} columns).",
                                 b.numel(), cols),
                    0, 0, "glmval", "", "numkit:glmval:shapeMismatch");

    if (scalarBeta && !wantBounds) {
        const keel::usize width = isconst ? cols + 1 : cols;
        Value out = Value::matrix(rows, width, ValueType::DOUBLE, mr);
        for (keel::usize r = 0; r < rows; ++r)
            for (keel::usize j = 0; j < width; ++j) {
                const double xj = isconst && j == 0 ? 1.0
                    : Xw.elemAsDouble(r + (isconst ? j - 1 : j) * rows);
                const double eta = b.elemAsDouble(0) * xj
                    + (offset.isScalar() ? offset.elemAsDouble(0)
                                         : offset.elemAsDouble(r));
                out.doubleDataMut()[r + j * rows] =
                    (Nv.isScalar() ? Nv.elemAsDouble(0) : Nv.elemAsDouble(r))
                    * invLink(eta);
            }
        return { std::move(out), Value(), Value() };
    }

    Value yhat = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
    keel::Array<double> etaVec(rows, 0.0);
    for (keel::usize r = 0; r < rows; ++r) {
        double eta = isconst ? b.elemAsDouble(0) : 0.0;
        for (keel::usize j = 0; j < cols; ++j)
            eta += Xw.elemAsDouble(r + j * rows)
                 * b.elemAsDouble(j + (isconst ? 1 : 0));
        eta += offset.isScalar() ? offset.elemAsDouble(0)
                                 : offset.elemAsDouble(r);
        etaVec[r] = eta;
        yhat.doubleDataMut()[r] =
            (Nv.isScalar() ? Nv.elemAsDouble(0) : Nv.elemAsDouble(r))
            * invLink(eta);
    }

    if (!wantBounds)
        return { std::move(yhat), Value(), Value() };

    // Extract covariance matrix C (wantCoef x wantCoef) from statsVal.
    keel::Array<double> covMat(wantCoef * wantCoef, 0.0);
    if (statsVal.hasField("covb")
        && static_cast<keel::usize>(statsVal.field("covb").numel()) == wantCoef * wantCoef) {
        const Value &cb = statsVal.field("covb");
        for (keel::usize idx = 0; idx < wantCoef * wantCoef; ++idx)
            covMat[idx] = cb.elemAsDouble(idx);
    } else if (statsVal.hasField("se")
               && static_cast<keel::usize>(statsVal.field("se").numel()) == wantCoef) {
        const Value &se = statsVal.field("se");
        const bool hasCorr = statsVal.hasField("coeffcorr")
            && static_cast<keel::usize>(statsVal.field("coeffcorr").numel()) == wantCoef * wantCoef;
        for (keel::usize j = 0; j < wantCoef; ++j)
            for (keel::usize k = 0; k < wantCoef; ++k) {
                const double corr_jk = hasCorr
                    ? statsVal.field("coeffcorr").elemAsDouble(j + k * wantCoef)
                    : (j == k ? 1.0 : 0.0);
                covMat[j + k * wantCoef] = se.elemAsDouble(j) * se.elemAsDouble(k) * corr_jk;
            }
    } else {
        throw Error("glmval: STATS must contain 'covb' or ('se' and 'coeffcorr') matching length(b).",
                    0, 0, "glmval", "", "numkit:glmval:needSTATS");
    }

    const double dfe = statsVal.hasField("dfe")
        ? statsVal.field("dfe").elemAsDouble(0)
        : std::numeric_limits<double>::infinity();
    const bool estdisp = statsVal.hasField("estdisp")
        ? (statsVal.field("estdisp").elemAsDouble(0) != 0.0)
        : std::isfinite(dfe);
    const double alpha = 1.0 - clev;
    const double p_df = static_cast<double>(wantCoef);
    double crit = 0.0;
    if (!simultaneous) {
        crit = (estdisp && std::isfinite(dfe))
            ? tinv(Value::scalar(1.0 - alpha / 2.0, mr), dfe, mr).toScalar()
            : norminv(Value::scalar(1.0 - alpha / 2.0, mr), 0.0, 1.0, mr).toScalar();
    } else {
        crit = (estdisp && std::isfinite(dfe))
            ? std::sqrt(p_df * finv(Value::scalar(1.0 - alpha, mr), p_df, dfe, mr).toScalar())
            : std::sqrt(chi2inv(Value::scalar(1.0 - alpha, mr), p_df, mr).toScalar());
    }

    Value dylo = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
    Value dyhi = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
    keel::Array<double> xr(wantCoef, 0.0);
    for (keel::usize r = 0; r < rows; ++r) {
        if (isconst) {
            xr[0] = 1.0;
            for (keel::usize j = 0; j < cols; ++j)
                xr[j + 1] = Xw.elemAsDouble(r + j * rows);
        } else {
            for (keel::usize j = 0; j < cols; ++j)
                xr[j] = Xw.elemAsDouble(r + j * rows);
        }
        double varEta = 0.0;
        for (keel::usize j = 0; j < wantCoef; ++j)
            for (keel::usize k = 0; k < wantCoef; ++k)
                varEta += xr[j] * covMat[j + k * wantCoef] * xr[k];
        const double seEta = std::sqrt(std::max(0.0, varEta));
        const double nr = Nv.isScalar() ? Nv.elemAsDouble(0) : Nv.elemAsDouble(r);
        const double y1 = nr * invLink(etaVec[r] - crit * seEta);
        const double y2 = nr * invLink(etaVec[r] + crit * seEta);
        const double ymin = std::min(y1, y2);
        const double ymax = std::max(y1, y2);
        const double yh = yhat.doubleData()[r];
        dylo.doubleDataMut()[r] = yh - ymin;
        dyhi.doubleDataMut()[r] = ymax - yh;
    }

    return { std::move(yhat), std::move(dylo), std::move(dyhi) };
}

Value glmval_full(const Value &b, const Value &X, const Value &link,
                  keel::ArrayRef<Value> tail, bool wantBounds,
                  keel::Allocator *mr)
{
    return glmval_full_bounds(b, X, link, tail, wantBounds, FnHandle{}, mr).yhat;
}

} // namespace numkit::stats
