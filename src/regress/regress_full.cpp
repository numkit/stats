// toolboxes/stats/src/regress/regress_full.cpp
//
// Full parse+validate+compute contracts for the regression entries —
// ridge / nlparci / glmval (probed against the reference engine R2025b:
// build/probe76.m R/N/G batteries). The register adapters marshal only;
// the numeric kernels (ridge's normal equations, nlparci's covariance,
// glmval's inverse link) live in their existing TUs and are reused.

#include <numkit/stats/regress/regress.hpp>   // ridge / nlparci / glmval + full entries

#include <numkit/stats/distributions/students_t.hpp>   // tinv (nlparci)
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <string>
#include <vector>

namespace numkit::stats {

namespace {

GlmLink parseGlmLinkName(const Value &v, const char *fn)
{
    std::string s = v.toString();
    for (char &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s.empty() || s == "canonical") return GlmLink::Identity;
    if (s == "identity")    return GlmLink::Identity;
    if (s == "logit")       return GlmLink::Logit;
    if (s == "log")         return GlmLink::Log;
    if (s == "reciprocal")  return GlmLink::Reciprocal;
    if (s == "probit")      return GlmLink::Probit;
    throw Error(std::string(fn) + ": unrecognized link function name '"
                + v.toString() + "' (expected identity, logit, log, "
                "probit, reciprocal, or canonical).",
                0, 0, fn, "",
                std::string("nk:") + fn + ":unrecognizedLink");
}

} // namespace

// ── ridge ─────────────────────────────────────────────────────────────

Value ridge_full(const Value &y, const Value &X, const Value &kVec,
                 const Value &flag, std::pmr::memory_resource *mr)
{
    // Scaling flag: empty or 1 → scaled; 0 → unscaled; else reject
    // (probe76 R04/R08).
    bool scaled = true;
    if (!flag.isEmpty()) {
        const double f = flag.elemAsDouble(0);
        if (f == 0.0) scaled = false;
        else if (f != 1.0)
            throw Error("ridge: the scaling flag must be 0 or 1 (got "
                        + std::to_string(f) + ").",
                        0, 0, "ridge", "", "nk:ridge:badScalingFlag");
    }

    // Shape gates (probe76 R05/R06): row count then column-vector y.
    if (X.dims().ndim() != 2)
        throw Error("ridge: X must be a matrix.",
                    0, 0, "ridge", "", "nk:ridge:invalidData");
    const std::size_t n = static_cast<std::size_t>(X.dims().dim(0));
    if (static_cast<std::size_t>(y.dims().dim(0)) != n
        || y.dims().ndim() != 2)
        throw Error("ridge: the number of rows in Y must equal the number "
                    "of rows in X (Y has "
                    + std::to_string(y.dims().dim(0)) + ", X has "
                    + std::to_string(n) + ").",
                    0, 0, "ridge", "", "nk:ridge:inputSizeMismatch");
    if (y.dims().dim(1) != 1)
        throw Error("ridge: Y must be a column vector (got "
                    + std::to_string(y.dims().dim(0)) + "x"
                    + std::to_string(y.dims().dim(1)) + ").",
                    0, 0, "ridge", "", "nk:ridge:invalidData");

    // Missing-value removal: any row with a NaN in y or X (probe76 R07).
    const std::size_t p = static_cast<std::size_t>(X.dims().dim(1));
    std::vector<std::size_t> keep;
    keep.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(y.elemAsDouble(i))) continue;
        bool ok = true;
        for (std::size_t j = 0; j < p && ok; ++j)
            if (std::isnan(X.elemAsDouble(i + j * n))) ok = false;
        if (ok) keep.push_back(i);
    }
    const std::size_t m = keep.size();
    if (m == 0)
        throw Error("ridge: no observations remain after removing rows "
                    "with missing values.",
                    0, 0, "ridge", "", "nk:ridge:invalidData");
    Value yc = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    Value Xc = Value::matrix(m, p, ValueType::DOUBLE, mr);
    for (std::size_t k = 0; k < m; ++k) {
        yc.doubleDataMut()[k] = y.elemAsDouble(keep[k]);
        for (std::size_t j = 0; j < p; ++j)
            Xc.doubleDataMut()[k + j * m] =
                X.elemAsDouble(keep[k] + j * n);
    }
    return ridge(yc, Xc, kVec, scaled, mr);
}

// ── nlparci ───────────────────────────────────────────────────────────

Value nlparci_full(const Value &beta, const Value &resid,
                   Span<const Value> tail, std::pmr::memory_resource *mr)
{
    // Tail forms (reference): (beta, resid, J[, alpha]) or the NV form
    // (beta, resid, 'covar', Sigma[, 'alpha', a]) — the named-argument
    // path keys off a char/string 3rd argument.
    Value J, Sigma;
    double alpha = 0.05;
    if (!tail.empty() && (tail[0].isChar() || tail[0].isString())) {
        for (std::size_t i = 0; i + 1 < tail.size(); i += 2) {
            std::string name = tail[i].toString();
            for (char &c : name)
                c = static_cast<char>(std::tolower(
                    static_cast<unsigned char>(c)));
            static constexpr std::string_view kNv[] = { "jacobian",
                                                        "covariance",
                                                        "alpha" };
            const int hit = argcheck::matchStringChoice(
                tail[i], Span<const std::string_view>(kNv));
            if (hit == 0) J = tail[i + 1];
            else if (hit == 1) Sigma = tail[i + 1];
            else if (hit == 2) alpha = tail[i + 1].elemAsDouble(0);
            else
                throw Error("nlparci: unrecognized parameter name '"
                            + tail[i].toString() + "'.",
                            0, 0, "nlparci", "",
                            "nk:nlparci:badParamName");
        }
    } else {
        if (tail.size() >= 1) J = tail[0];
        if (tail.size() >= 2 && !tail[1].isEmpty())
            alpha = tail[1].elemAsDouble(0);
    }
    if (resid.isEmpty() || (J.isEmpty() && Sigma.isEmpty()))
        throw Error("nlparci: requires (BETA, RESID, J) or the 'covar' "
                    "form.",
                    0, 0, "nlparci", "", "nk:nlparci:tooFewInputs");
    if (beta.isComplex() || J.isComplex())
        throw Error("nlparci: cannot compute confidence intervals for "
                    "complex parameters.",
                    0, 0, "nlparci", "", "nk:nlparci:complexParams");
    if (!std::isnan(alpha) && !(alpha > 0.0 && alpha < 1.0))
        throw Error("nlparci: ALPHA must be a scalar between 0 and 1 "
                    "(got " + std::to_string(alpha) + ").",
                    0, 0, "nlparci", "", "nk:nlparci:badAlpha");

    // Drop NaN residuals (and their Jacobian rows).
    const std::size_t nAll = static_cast<std::size_t>(resid.numel());
    std::vector<std::size_t> keep;
    keep.reserve(nAll);
    for (std::size_t i = 0; i < nAll; ++i)
        if (!std::isnan(resid.elemAsDouble(i))) keep.push_back(i);
    const std::size_t n = keep.size();
    const std::size_t pp = static_cast<std::size_t>(beta.numel());
    if (n <= pp)
        throw Error("nlparci: the degrees of freedom v = n - p must be "
                    "positive (n = " + std::to_string(n) + ", p = "
                    + std::to_string(pp) + ").",
                    0, 0, "nlparci", "", "nk:nlparci:noDOF");

    std::vector<double> se(pp, 0.0);
    if (!Sigma.isEmpty()) {
        // se = sqrt(diag(Sigma)) (probe76 N03).
        for (std::size_t j = 0; j < pp; ++j)
            se[j] = std::sqrt(Sigma.elemAsDouble(j + j * pp));
    } else {
        if (static_cast<std::size_t>(J.dims().dim(0)) != nAll
            || static_cast<std::size_t>(J.dims().dim(1)) != pp)
            throw Error("nlparci: the size of J does not match the sizes "
                        "of BETA and RESID (J is "
                        + std::to_string(J.dims().dim(0)) + "x"
                        + std::to_string(J.dims().dim(1)) + ", expected "
                        + std::to_string(nAll) + "x" + std::to_string(pp)
                        + ").",
                        0, 0, "nlparci", "", "nk:nlparci:inputSizeMismatch");
        Value Jc = Value::matrix(n, pp, ValueType::DOUBLE, mr);
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t j = 0; j < pp; ++j)
                Jc.doubleDataMut()[k + j * n] =
                    J.elemAsDouble(keep[k] + j * nAll);
        Value residC = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        for (std::size_t k = 0; k < n; ++k)
            residC.doubleDataMut()[k] = resid.elemAsDouble(keep[k]);
        // Reuse the kernel: se_j = sqrt(mse · [(J'J)^{-1}]_jj).
        Value ci = nlparci(beta, residC, Jc, alpha, mr);
        // The kernel's half-widths: recompute via its covariance route —
        // extract se from ci directly (ci = beta ± se·t).
        const double tcrit = tinv(Value::scalar(1.0 - alpha / 2.0, mr),
                                  static_cast<double>(n - pp), mr).toScalar();
        Value out = Value::matrix(pp, 2, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (std::size_t j = 0; j < pp; ++j) {
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
    for (std::size_t j = 0; j < pp; ++j) {
        const double d = se[j] * tcrit;
        od[j] = beta.elemAsDouble(j) - d;
        od[j + pp] = beta.elemAsDouble(j) + d;
    }
    return out;
}

// ── glmval ────────────────────────────────────────────────────────────

Value glmval_full(const Value &b, const Value &X, const Value &link,
                  Span<const Value> tail, bool wantBounds,
                  std::pmr::memory_resource *mr)
{
    if (!(link.isChar() || link.isString()))
        throw Error("glmval: link must be a char row or string scalar.",
                    0, 0, "glmval", "", "nk:glmval:badLink");
    const GlmLink lk = parseGlmLinkName(link, "glmval");

    // STATS: a non-string 4th argument is the stats struct (used only
    // by the bounds output).
    bool haveStats = false;
    std::size_t i = 0;
    if (!tail.empty() && !tail[0].isChar() && !tail[0].isString()) {
        haveStats = true;
        i = 1;
    }

    // Option parse: the NV form when the next argument is a string,
    // else the legacy positional (clev, N, offset, const).
    double clev = 0.95;
    Value Nv = Value::scalar(1.0, mr);
    Value offset = Value::scalar(0.0, mr);
    bool isconst = true;
    if (i < tail.size() && (tail[i].isChar() || tail[i].isString())) {
        bool haveConfidence = false, haveAlpha = false;
        bool haveSize = false, haveBinomSize = false;
        Value binomN;
        for (; i + 1 < tail.size(); i += 2) {
            std::string name = tail[i].toString();
            for (char &c : name)
                c = static_cast<char>(std::tolower(
                    static_cast<unsigned char>(c)));
            const Value &v = tail[i + 1];
            if      (name == "confidence")  { clev = v.elemAsDouble(0); haveConfidence = true; }
            else if (name == "alpha")       { clev = 1.0 - v.elemAsDouble(0); haveAlpha = true; }
            else if (name == "size")        { Nv = v; haveSize = true; }
            else if (name == "binomialsize") { Nv = v; haveBinomSize = true; }
            else if (name == "offset")      offset = v;
            else if (name == "constant")    isconst = (v.toString() == "on");
            else if (name == "simultaneous") { /* bounds option — filed */ }
            else
                throw Error("glmval: unrecognized parameter name '"
                            + tail[i].toString() + "'.",
                            0, 0, "glmval", "", "nk:glmval:badParamName");
        }
        if (haveConfidence && haveAlpha)
            throw Error("glmval: cannot supply both the 'Confidence' and "
                        "'Alpha' parameters.",
                        0, 0, "glmval", "", "nk:glmval:argCombination");
        if (haveSize && haveBinomSize)
            throw Error("glmval: cannot supply both the 'Size' and "
                        "'BinomialSize' parameters.",
                        0, 0, "glmval", "", "nk:glmval:argCombination");
    } else {
        // Legacy positional: (clev, N, offset, const).
        if (i < tail.size() && !tail[i].isEmpty())
            clev = tail[i].elemAsDouble(0);
        if (i + 1 < tail.size() && !tail[i + 1].isEmpty())
            Nv = tail[i + 1];
        if (i + 2 < tail.size() && !tail[i + 2].isEmpty())
            offset = tail[i + 2];
        if (i + 3 < tail.size() && !tail[i + 3].isEmpty())
            isconst = (tail[i + 3].toString() == "on");
    }

    if (wantBounds)
        throw Error(haveStats
            ? "glmval: the confidence-bounds output is not supported in "
              "this revision (filed: bugs/opened/stats/"
              "regression-deferred-forms.md)."
            : "glmval: the STATS parameter is required to compute "
              "confidence bounds.",
            0, 0, "glmval", "",
            haveStats ? "nk:glmval:boundsUnsupported" : "nk:glmval:needSTATS");

    // Row X with a 2-coefficient (or scalar-no-const) model is a column
    // of points (probe76 V02).
    Value Xw = X;
    if (X.dims().dim(0) == 1 && X.dims().dim(1) > 1
        && ((b.numel() == 2 && isconst) || (b.numel() == 1 && !isconst))) {
        Xw = Value::matrix(static_cast<std::size_t>(X.numel()), 1,
                           ValueType::DOUBLE, mr);
        for (std::size_t k = 0; k < static_cast<std::size_t>(X.numel()); ++k)
            Xw.doubleDataMut()[k] = X.elemAsDouble(k);
    }

    // BinomialSize / Offset shape gates (probe76: WrongArgSize classes).
    const std::size_t m = static_cast<std::size_t>(Xw.dims().dim(0));
    if (!Nv.isScalar()
        && (Nv.dims().dim(1) != 1
            || static_cast<std::size_t>(Nv.numel()) != m))
        throw Error("glmval: BinomialSize must be a scalar or a column "
                    "vector with one entry per row of X.",
                    0, 0, "glmval", "", "nk:glmval:wrongArgSizeBinomialSize");
    if (!offset.isScalar()
        && (offset.dims().dim(1) != 1
            || static_cast<std::size_t>(offset.numel()) != m))
        throw Error("glmval: Offset must be a scalar or a column vector "
                    "with one entry per row of X.",
                    0, 0, "glmval", "", "nk:glmval:wrongArgSizeOffset");

    // eta = x·beta + offset, then yhat = N·ilink(eta) — computed
    // directly (the offset enters INSIDE the inverse link). A scalar
    // beta broadcasts over the const-prepended x exactly like the
    // reference's x*beta (probe76 V02).
    auto invLink = [lk](double eta) -> double {
        switch (lk) {
        case GlmLink::Identity:   return eta;
        case GlmLink::Logit:      return 1.0 / (1.0 + std::exp(-eta));
        case GlmLink::Log:        return std::exp(eta);
        case GlmLink::Reciprocal: return 1.0 / eta;
        case GlmLink::Probit:     return 0.5 * std::erfc(-eta / std::sqrt(2.0));
        }
        return eta;
    };
    const std::size_t rows = static_cast<std::size_t>(Xw.dims().dim(0));
    const std::size_t cols = static_cast<std::size_t>(Xw.dims().dim(1));
    const bool scalarBeta = b.numel() == 1;
    const std::size_t wantCoef = isconst ? cols + 1 : cols;
    if (!scalarBeta
        && static_cast<std::size_t>(b.numel()) != wantCoef)
        throw Error("glmval: length(beta) must equal the number of "
                    "columns of X plus one when the constant is on (got "
                    + std::to_string(b.numel()) + " coefficients for "
                    + std::to_string(cols) + " columns).",
                    0, 0, "glmval", "", "nk:glmval:shapeMismatch");
    Value out;
    if (scalarBeta) {
        // x*beta semantics: every column (including the intercept
        // column when const is on) times the scalar.
        const std::size_t width = isconst ? cols + 1 : cols;
        out = Value::matrix(rows, width, ValueType::DOUBLE, mr);
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < width; ++j) {
                const double xj = isconst && j == 0 ? 1.0
                    : Xw.elemAsDouble(r + (isconst ? j - 1 : j) * rows);
                const double eta = b.elemAsDouble(0) * xj
                    + (offset.isScalar() ? offset.elemAsDouble(0)
                                         : offset.elemAsDouble(r));
                out.doubleDataMut()[r + j * rows] =
                    (Nv.isScalar() ? Nv.elemAsDouble(0) : Nv.elemAsDouble(r))
                    * invLink(eta);
            }
        return out;
    }
    out = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
    for (std::size_t r = 0; r < rows; ++r) {
        double eta = isconst ? b.elemAsDouble(0) : 0.0;
        for (std::size_t j = 0; j < cols; ++j)
            eta += Xw.elemAsDouble(r + j * rows)
                 * b.elemAsDouble(j + (isconst ? 1 : 0));
        eta += offset.isScalar() ? offset.elemAsDouble(0)
                                 : offset.elemAsDouble(r);
        out.doubleDataMut()[r] =
            (Nv.isScalar() ? Nv.elemAsDouble(0) : Nv.elemAsDouble(r))
            * invLink(eta);
    }
    return out;
}

} // namespace numkit::stats
