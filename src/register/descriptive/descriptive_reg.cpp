// toolboxes/signal/src/descriptive/descriptive_reg.cpp
//
// CallContext register half of descriptive/descriptive.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/stats/distributions/normal.hpp>     // norminv for corrcoef conf bounds
#include <numkit/stats/distributions/students_t.hpp> // tcdf for corrcoef p-values
#include <numkit/stats/nan_aware/nan_aware.hpp>  // var_reg / std_reg / median_reg dispatch into stats:: when 'omitnan' is given
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include "descriptive/descriptive_detail.hpp"
#include "support/reduce_vecdim.hpp"
#include <numkit/ops/helpers.hpp>
#include <numkit/builtin/detail/datafun/var_reduction.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <memory>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::stats {

namespace detail {

// Helper: weighted variance / std on a flat 1-D array.
//   normFlag = 0  â†’ divide by Î£w - 0 (default for scalar w=0; sample)
//   normFlag = 1  â†’ divide by Î£w     (population; ML estimate)
//
// MATLAB's documented denominator for weighted variance is Î£w (the
// "default" for vector weights â€” equivalent to normFlag=1 semantics).
// var(A, W, ...) with vector W therefore implies normFlag=1.
double weightedVarFlat(const double *x, const double *w, size_t n,
                       bool sqrtIt, bool omitNan)
{
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    double sw = 0.0, sxw = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double xi = x[i], wi = w[i];
        if (omitNan && std::isnan(xi)) continue;
        if (wi < 0.0)
            throw Error("var/std: weights must be non-negative",
                        0, 0, "var/std", "", "numkit:varstd:negWeight");
        sw  += wi;
        sxw += wi * xi;
    }
    if (sw == 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double mean = sxw / sw;
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double xi = x[i], wi = w[i];
        if (omitNan && std::isnan(xi)) continue;
        const double d = xi - mean;
        ss += wi * d * d;
    }
    const double v = ss / sw;
    return sqrtIt ? std::sqrt(v) : v;
}

// Helper: scalar-normFlag variance/std on a flat 1-D array.
double scalarVarFlat(const double *x, size_t n, int normFlag,
                     bool sqrtIt, bool omitNan)
{
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    double s = 0.0;
    size_t cnt = 0;
    for (size_t i = 0; i < n; ++i) {
        if (omitNan && std::isnan(x[i])) continue;
        s += x[i]; ++cnt;
    }
    if (cnt == 0) return std::numeric_limits<double>::quiet_NaN();
    if (cnt == 1) return (normFlag == 1) ? 0.0
                                         : std::numeric_limits<double>::quiet_NaN();
    const double mean = s / static_cast<double>(cnt);
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (omitNan && std::isnan(x[i])) continue;
        const double d = x[i] - mean;
        ss += d * d;
    }
    const double denom = (normFlag == 1) ? static_cast<double>(cnt)
                                         : static_cast<double>(cnt - 1);
    const double v = ss / denom;
    return sqrtIt ? std::sqrt(v) : v;
}

// Convert a Value to a flat row (used for 'all' / vecdim full-flatten /
// weight-vector inputs). Skips no elements.
std::vector<double> flatten(const Value &x)
{
    std::vector<double> out(x.numel());
    for (size_t i = 0; i < x.numel(); ++i) out[i] = x.elemAsDouble(i);
    return out;
}

// Common var/std driver â€” handles every variant (scalar w + scalar dim,
// 'all', vecdim full-flatten, weight vector).
//
// `sqrtIt = true` makes this `std`.
Value varStdDispatch(Span<const Value> args, bool sqrtIt, const char *fn,
                     std::pmr::memory_resource *mr, Value *wOut = nullptr)
{
    bool omitNan = false;
    size_t n = stripNanFlag(args, omitNan, fn);

    const Value &x = args[0];
    // Probed R2025b: the first input must be single or double
    // (logical, complex and even char pass â€” integers reject).
    if (isIntegerType(x.type()))
        throw Error(std::string(fn) + ": invalid data type. First input "
                    "argument must be single or double (got "
                    + std::string(mtypeName(x.type())) + ").",
                    0, 0, fn, "", std::string("numkit:") + fn + ":integerClass");

    // Parse args[1] (w) and args[2] (dim/'all'/vecdim) into a normalised
    // shape: scalar normFlag (0 or 1), or weight vector + flatten flag.
    int normFlag = 0;
    bool isWeightVec = false;
    const Value *wVec = nullptr;

    if (n >= 2 && !args[1].isEmpty()) {
        if (args[1].isChar() || args[1].isString())
            // var(X,'all') is NOT a form â€” flags only live in the dim
            // position (probed R2025b: unknownFlag).
            throw Error(std::string(fn)
                            + ": invalid option. Option must be 'omitnan', "
                              "'includenan', or a weight vector.",
                        0, 0, fn, "",
                        std::string("numkit:") + fn + ":unknownFlag");
        if (args[1].numel() == 1 && args[1].isNumeric()) {
            const double wf = argcheck::expectedReal(args[1], 2, "w", fn);
            if (wf != 0.0 && wf != 1.0)
                throw Error(std::string(fn) + ": W must be a double or "
                            "single vector of nonnegative weights.",
                            0, 0, fn, "",
                            std::string("numkit:") + fn + ":invalidWgts");
            normFlag = static_cast<int>(wf);
        } else {
            isWeightVec = true;
            wVec = &args[1];
        }
    }

    int dim = 0;
    bool flattenAll = false;
    std::vector<int> vecdimSet;
    if (n >= 3 && !args[2].isEmpty()) {
        const Value &a = args[2];
        if (a.isChar() || a.isString()) {
            std::string s = a.toString();
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (s == "all") flattenAll = true;
            else throw Error(std::string(fn) + ": unknown dim flag '" + s + "'",
                              0, 0, fn, "", std::string("numkit:") + fn + ":dim");
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, fn);
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            std::vector<int> set = vecdim::parseDimSet(a, fn);
            if (vecdim::coversAllDims(set, x))
                flattenAll = true;
            else {
                if (isWeightVec)
                    throw Error(std::string(fn) + ": weight vector along a "
                                "dimension set is supported for 2-D inputs "
                                "only (N-D deferred)",
                                0, 0, fn, "", std::string("numkit:") + fn + ":wND");
                vecdimSet = std::move(set);
            }
        }
    }

    if (wOut) {
        if (isWeightVec) {
            *wOut = *wVec;
        } else {
            size_t wLen = 1;
            if (flattenAll) {
                wLen = x.numel();
            } else if (!vecdimSet.empty()) {
                for (int d : vecdimSet) {
                    if (d >= 1 && static_cast<size_t>(d) <= x.dims().ndim())
                        wLen *= x.dims().dim(static_cast<size_t>(d - 1));
                }
            } else {
                const int effDim = (dim == 0) ? firstNonSingletonDim(x) : dim;
                if (effDim >= 1 && static_cast<size_t>(effDim) <= x.dims().ndim())
                    wLen = x.dims().dim(static_cast<size_t>(effDim - 1));
            }
            Value wv = Value::matrix(1, wLen, ValueType::DOUBLE, mr);
            double *wd = wv.doubleDataMut();
            for (size_t i = 0; i < wLen; ++i) wd[i] = 1.0;
            *wOut = std::move(wv);
        }
    }

    // â”€â”€ Weighted-vector path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (isWeightVec) {
        const bool xIsVec = x.dims().isVector() || x.isScalar();
        if (flattenAll || (dim == 0 && xIsVec)) {
            // Vector input or 'all': flatten + run weightedVarFlat. The
            // weight length must match the element count.
            auto xv = flatten(x);
            auto wv = flatten(*wVec);
            if (xv.size() != wv.size())
                throw Error(std::string(fn) + ": weight vector length must "
                            "match number of elements",
                            0, 0, fn, "", std::string("numkit:") + fn + ":invalidSizeWgts");
            const double v = weightedVarFlat(xv.data(), wv.data(),
                                             xv.size(), sqrtIt, omitNan);
            return Value::scalar(v, mr);
        }
        // Matrix (or explicit dim): MATLAB applies the weight vector along
        // the operating dimension, computing one weighted variance per
        // slice. Weight length must equal size(x, dim). 2-D only (N-D
        // weighted reduction is deferred).
        if (x.dims().ndim() > 2)
            throw Error(std::string(fn) + ": weight vector along a dimension "
                        "is supported for 2-D inputs only (N-D deferred)",
                        0, 0, fn, "", std::string("numkit:") + fn + ":wND");
        const int d = (dim == 0) ? firstNonSingletonDim(x) : dim;
        if (d != 1 && d != 2)
            throw Error(std::string(fn) + ": dim out of range",
                        0, 0, fn, "", std::string("numkit:") + fn + ":dim");
        const size_t r = static_cast<size_t>(x.dims().dim(0));
        const size_t c = static_cast<size_t>(x.dims().dim(1));
        auto wv = flatten(*wVec);
        const size_t need = (d == 1) ? r : c;
        if (wv.size() != need)
            throw Error(std::string(fn) + ": weight vector length must match "
                        "the length of the operating dimension",
                        0, 0, fn, "", std::string("numkit:") + fn + ":invalidSizeWgts");
        if (d == 1) {
            auto out = Value::matrix(1, c, ValueType::DOUBLE, mr);
            double *od = out.doubleDataMut();
            std::vector<double> col(r);
            for (size_t j = 0; j < c; ++j) {
                for (size_t i = 0; i < r; ++i) col[i] = x.elemAsDouble(j * r + i);
                od[j] = weightedVarFlat(col.data(), wv.data(), r, sqrtIt, omitNan);
            }
            return out;
        }
        auto out = Value::matrix(r, 1, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        std::vector<double> row(c);
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) row[j] = x.elemAsDouble(j * r + i);
            od[i] = weightedVarFlat(row.data(), wv.data(), c, sqrtIt, omitNan);
        }
        return out;
    }

    // â”€â”€ Partial vecdim: collapse the listed dims and reduce along the
    // last dim of the collapsed view (same kernels as the scalar-dim
    // path, so every flag combination carries over).
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(x, vecdimSet, mr);
        const int lastd = static_cast<int>(c.dims().ndim());
        if (omitNan) {
            if (x.type() == ValueType::COMPLEX)
                return vecdim::expandReduced(
                    varianceComplex(c, normFlag, lastd, mr, sqrtIt, true),
                    x, vecdimSet, mr);
            Value r = sqrtIt
                        ? ::numkit::stats::nanstdev(c, normFlag, lastd, mr)
                        : ::numkit::stats::nanvar  (c, normFlag, lastd, mr);
            if (x.type() == ValueType::SINGLE)
                r = narrowToSingle(std::move(r), mr);
            return vecdim::expandReduced(std::move(r), x, vecdimSet, mr);
        }
        Value r = sqrtIt ? stdev(c, normFlag, lastd, mr)
                         : var(c, normFlag, lastd, mr);
        return vecdim::expandReduced(std::move(r), x, vecdimSet, mr);
    }

    // â”€â”€ Flatten 'all' / vecdim path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (flattenAll) {
        auto xv = flatten(x);
        const double v = scalarVarFlat(xv.data(), xv.size(), normFlag,
                                       sqrtIt, omitNan);
        return Value::scalar(v, mr);
    }

    // â”€â”€ Standard scalar-dim path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (omitNan) {
        if (x.type() == ValueType::COMPLEX) {
            return varianceComplex(x, normFlag, dim, mr, sqrtIt, true);
        }
        Value r = sqrtIt
                    ? ::numkit::stats::nanstdev(x, normFlag, dim, mr)
                    : ::numkit::stats::nanvar  (x, normFlag, dim, mr);
        if (x.type() == ValueType::SINGLE)
            r = narrowToSingle(std::move(r), mr);
        return r;
    }
    return sqrtIt ? stdev(x, normFlag, dim, mr) : var(x, normFlag, dim, mr);
}

void var_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
             CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "var");
    Value wOut;
    outs[0] = varStdDispatch(args, /*sqrtIt=*/false, "var", ctx.engine->resource(),
                             (nargout >= 2) ? &wOut : nullptr);
    if (nargout >= 2) outs[1] = std::move(wOut);
}

void std_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
             CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "std");
    Value wOut;
    outs[0] = varStdDispatch(args, /*sqrtIt=*/true, "std", ctx.engine->resource(),
                             (nargout >= 2) ? &wOut : nullptr);
    if (nargout >= 2) outs[1] = std::move(wOut);
}

void median_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "median");  // no max: flag/option tail in the reference
    // MATLAB narrows an all-zero-imaginary complex median back to a real double.
    // The scalar / dim / 'all' / omitnan branches each assign outs[0] and return
    // separately, so a scope guard narrows whichever one fires (no-op unless the
    // result is left complex with all-zero imag).
    struct NarrowGuard {
        Value *v;
        std::pmr::memory_resource *mr;
        ~NarrowGuard()
        {
            if (v->isComplex())
                *v = numkit::narrowComplex(std::move(*v), mr);
        }
    } narrowGuard{&outs[0], ctx.engine->resource()};
    // Probed R2025b: median preserves the LOGICAL class (result is the
    // majority/rounded median re-narrowed to logical); integers already
    // round-and-saturate in the kernel. Runs on every exit path like
    // narrowGuard above.
    struct LogicalGuard {
        Value *v;
        ~LogicalGuard()
        {
            if (v->type() == ValueType::DOUBLE) {
                Value lg = Value::matrix(v->dims().rows(), v->dims().cols(),
                                         ValueType::LOGICAL);
                for (std::size_t i = 0; i < v->numel(); ++i)
                    lg.logicalDataMut()[i] =
                        v->doubleData()[i] != 0.0 ? 1 : 0;
                *v = std::move(lg);
            }
        }
    };
    std::unique_ptr<LogicalGuard> logicalGuard;
    if (args[0].type() == ValueType::LOGICAL)
        logicalGuard = std::make_unique<LogicalGuard>(
            LogicalGuard{&outs[0]});
    bool omitNan = false;
    size_t n = stripNanFlag(args, omitNan, "median");
    int dim = 0;
    bool isAll = false;
    std::vector<int> vecdimSet;
    if (n >= 2 && !args[1].isEmpty()) {
        const Value &a = args[1];
        if (a.isChar() || a.isString()) {
            std::string s = a.toString();
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (s == "all") isAll = true;
            else throw Error("median: unknown option '" + s + "'",
                              0, 0, "median", "", "numkit:median:unknownOption");
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, "median");
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            std::vector<int> set = vecdim::parseDimSet(a, "median");
            if (vecdim::coversAllDims(set, args[0]))
                isAll = true;
            else
                vecdimSet = std::move(set);
        }
    }
    if (!vecdimSet.empty()) {
        // Partial vecdim â€” same kernels as the scalar-dim path (the
        // NarrowGuard above still narrows an all-zero-imaginary result).
        Value c = vecdim::collapseDims(args[0], vecdimSet,
                                       ctx.engine->resource());
        const int lastd = static_cast<int>(c.dims().ndim());
        if (omitNan) {
            rejectComplexOmitNan(args[0], "median");
            Value r = ::numkit::stats::nanmedian(c, lastd, ctx.engine->resource());
            if (args[0].type() == ValueType::SINGLE)
                r = narrowToSingle(std::move(r), ctx.engine->resource());
            outs[0] = vecdim::expandReduced(std::move(r), args[0], vecdimSet,
                                            ctx.engine->resource());
            return;
        }
        outs[0] = vecdim::expandReduced(median(c, lastd, ctx.engine->resource()),
                                        args[0], vecdimSet,
                                        ctx.engine->resource());
        return;
    }
    if (isAll) {
        // 'all' â†’ flatten + median over all elements (skipping NaN if omitnan).
        if (args[0].type() == ValueType::COMPLEX) {
            const size_t total = args[0].numel();
            ScratchArena scratch(ctx.engine->resource());
            auto buf = ScratchVec<Complex>(total, &scratch);
            const Complex *src = args[0].complexData();
            size_t k = 0;
            for (size_t i = 0; i < total; ++i) {
                if (omitNan && isComplexNaNStats(src[i])) continue;
                buf[k++] = src[i];
            }
            outs[0] = Value::complexScalar(complexMedianFromSlice(buf.data(), k),
                                           ctx.engine->resource());
            return;
        }
        const size_t total = args[0].numel();
        ScratchArena scratch(ctx.engine->resource());
        auto buf = ScratchVec<double>(total, &scratch);
        const bool fastDouble = (args[0].type() == ValueType::DOUBLE);
        if (fastDouble)
            std::copy(args[0].doubleData(), args[0].doubleData() + total, buf.data());
        else
            for (size_t i = 0; i < total; ++i) buf[i] = args[0].elemAsDouble(i);
        size_t k = total;
        if (omitNan) k = compactNonNan(buf.data(), total);
        double v = medianFromSlice(buf.data(), k);
        Value r = Value::scalar(v, ctx.engine->resource());
        if (args[0].type() == ValueType::SINGLE)
            r = narrowToSingle(std::move(r), ctx.engine->resource());
        outs[0] = std::move(r);
        return;
    }
    if (omitNan) {
        rejectComplexOmitNan(args[0], "median");
        Value r = ::numkit::stats::nanmedian(args[0], dim, ctx.engine->resource());
        if (args[0].type() == ValueType::SINGLE)
            r = narrowToSingle(std::move(r), ctx.engine->resource());
        outs[0] = std::move(r);
        return;
    }
    outs[0] = median(args[0], dim, ctx.engine->resource());
}

// Common parser for the trailing args of quantile/prctile:
//   fn(X, p[, dim] [, Method=method])
//
// Handles:
//   * 'all' (string) â€” flatten input
//   * scalar dim
//   * vecdim â€” only [1 2] / [1 2 3] / etc. covering ALL dims is supported
//     as a synonym for 'all'; partial reductions throw a documented error.
//   * Method N-V pair: 'midpoint' (default) | 'inclusive' | 'exclusive'
//                       | 'approximate' (falls back to midpoint).
struct QArgs {
    int dim = 0;
    bool flatten = false;
    QMethod method = QMethod::Midpoint;
};

QArgs parseQArgs(Span<const Value> args, size_t start, const Value &x,
                 const char *fn)
{
    QArgs q;
    size_t i = start;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()
        && !args[i].isEmpty()) {
        if (args[i].numel() == 1) {
            q.dim = argcheck::dimArg(args[i], "median");
        } else {
            // Vecdim: covers-all-dims â†’ flatten; otherwise error.
            std::vector<int> dims;
            for (size_t j = 0; j < args[i].numel(); ++j)
                dims.push_back(argcheck::dimArg(args[i].elemAsDouble(j), "median"));
            const int rank = x.dims().is3D() ? 3
                              : (x.dims().isVector() || x.isScalar() ? 1 : 2);
            std::vector<bool> seen(rank + 1, false);
            for (int d : dims) {
                if (d < 1 || d > rank)
                    throw Error(std::string(fn) + ": vecdim entries out of range",
                                0, 0, fn, "", std::string("numkit:") + fn + ":vecdim");
                seen[d] = true;
            }
            bool allCovered = true;
            for (int d = 1; d <= rank; ++d) if (!seen[d]) allCovered = false;
            if (!allCovered)
                throw Error(std::string(fn) + ": partial vecdim reduction is "
                            "not yet supported in nk (only full-flatten "
                            "vecdim like [1 2] or 'all')",
                            0, 0, fn, "", std::string("numkit:") + fn + ":vecdim");
            q.flatten = true;
        }
        ++i;
    } else if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        std::string s = args[i].toString();
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (s == "all") {
            q.flatten = true;
            ++i;
        }
        // else: leave for the Name-Value loop below ("Method=...")
    }

    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error(std::string(fn) + ": expected Name-Value pair",
                        0, 0, fn, "", std::string("numkit:") + fn + ":nv");
        std::string name = args[i].toString();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (name == "method") {
            if (!args[i + 1].isChar() && !args[i + 1].isString())
                throw Error(std::string(fn) + ": Method must be a string",
                            0, 0, fn, "", std::string("numkit:") + fn + ":method");
            std::string m = args[i + 1].toString();
            std::transform(m.begin(), m.end(), m.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            // MATLAB's documented values are 'exact' (default) and
            // 'approximate'. 'exact' is the linear-interpolation order-
            // statistic method, which is nk's Midpoint. The
            // midpoint/inclusive/exclusive names are kept for compatibility.
            if      (m == "exact")       q.method = QMethod::Midpoint;
            else if (m == "midpoint")    q.method = QMethod::Midpoint;
            else if (m == "inclusive")   q.method = QMethod::Inclusive;
            else if (m == "exclusive")   q.method = QMethod::Exclusive;
            else if (m == "approximate") q.method = QMethod::Approximate;
            else
                throw Error(std::string(fn) + ": Method must be 'exact' or "
                            "'approximate'",
                            0, 0, fn, "", std::string("numkit:") + fn + ":method");
        } else {
            throw Error(std::string(fn) + ": unknown Name-Value '" + name + "'",
                        0, 0, fn, "", std::string("numkit:") + fn + ":nv");
        }
        i += 2;
    }
    return q;
}

void quantile_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "quantile");  // no max: name-value tail in the reference
    auto q = parseQArgs(args, 2, args[0], "quantile");
    outs[0] = quantileWithOpts(args[0], args[1], q.dim, q.flatten, q.method, 1.0, "quantile", ctx.engine->resource());
}

void prctile_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "prctile");  // no max: name-value tail in the reference
    auto q = parseQArgs(args, 2, args[0], "prctile");
    outs[0] = quantileWithOpts(args[0], args[1], q.dim, q.flatten, q.method, 0.01, "prctile", ctx.engine->resource());
}

// 3rd mode output C: a cell array of the modal values. Each cell holds a
// column vector of the values that attain the modal frequency in that
// slice, sorted ascending; MATLAB ignores NaN. Supported for a real DOUBLE
// vector, 2-D matrix, or 'all'-flattened input.
Value computeModeCell(const Value &x, int dim, bool flatten,
                      std::pmr::memory_resource *mr)
{
    if (x.type() != ValueType::DOUBLE || x.dims().is3D() || x.dims().ndim() > 2)
        throw Error("mode: the 3rd output C is currently supported only for a "
                    "real double vector or 2-D matrix input",
                    0, 0, "mode", "", "numkit:mode:cellNd");
    auto tiedCol = [&](const double *s, size_t stride, size_t n) -> Value {
        std::vector<double> v;
        v.reserve(n);
        for (size_t k = 0; k < n; ++k) {
            const double d = s[k * stride];
            if (!std::isnan(d)) v.push_back(d);
        }
        std::sort(v.begin(), v.end());
        const size_t N = v.size();
        size_t maxc = 0;
        for (size_t i = 0; i < N;) {
            size_t j = i;
            while (j < N && v[j] == v[i]) ++j;
            if (j - i > maxc) maxc = j - i;
            i = j;
        }
        std::vector<double> tied;
        for (size_t i = 0; i < N;) {
            size_t j = i;
            while (j < N && v[j] == v[i]) ++j;
            if (j - i == maxc) tied.push_back(v[i]);
            i = j;
        }
        Value col = Value::matrix(tied.size(), 1, ValueType::DOUBLE, mr);
        for (size_t k = 0; k < tied.size(); ++k) col.doubleDataMut()[k] = tied[k];
        return col;
    };
    const double *data = x.numel() ? x.doubleData() : nullptr;
    if (flatten || x.isScalar() || x.dims().isVector()) {
        Value cell = Value::cell(1, 1, mr);
        cell.cellAt(0) = tiedCol(data, 1, x.numel());
        return cell;
    }
    const size_t R = x.dims().rows(), Cn = x.dims().cols();
    const int rdim = (dim == 0) ? 1 : dim;
    if (rdim == 1) {
        Value cell = Value::cell(1, Cn, mr);
        for (size_t c = 0; c < Cn; ++c) cell.cellAt(c) = tiedCol(data + c * R, 1, R);
        return cell;
    }
    Value cell = Value::cell(R, 1, mr);
    for (size_t r = 0; r < R; ++r) cell.cellAt(r) = tiedCol(data + r, R, Cn);
    return cell;
}

void mode_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "mode");
    int dim = 0;
    bool flatten = false;
    std::vector<int> vecdimSet;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &a = args[1];
        if (a.isChar() || a.isString()) {
            std::string s = a.toString();
            std::transform(s.begin(), s.end(), s.begin(),
                [](unsigned char c){ return std::tolower(c); });
            if (s == "all") flatten = true;
            else throw Error("mode: unknown flag '" + s + "'",
                              0, 0, "mode", "", "numkit:mode:badFlag");
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, "mode");
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            std::vector<int> set = vecdim::parseDimSet(a, "mode");
            if (vecdim::coversAllDims(set, args[0]))
                flatten = true;
            else
                vecdimSet = std::move(set);
        }
    }
    auto *mr = ctx.engine->resource();
    if (!vecdimSet.empty()) {
        // Partial vecdim â€” same kernels as the scalar-dim path; the
        // 3rd output's cell array reshapes through the same mechanism
        // (permute/reshape support cell values).
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        const int lastd = static_cast<int>(c.dims().ndim());
        auto [v, cnt] = mode(c, lastd, mr);
        outs[0] = vecdim::expandReduced(std::move(v), args[0], vecdimSet, mr);
        if (nargout > 1)
            outs[1] = vecdim::expandReduced(std::move(cnt), args[0], vecdimSet, mr);
        if (nargout > 2)
            outs[2] = vecdim::expandReduced(computeModeCell(c, lastd, false, mr),
                                            args[0], vecdimSet, mr);
        return;
    }
    if (flatten) {
        Value flat = Value::matrix(1, args[0].numel(), ValueType::DOUBLE, mr);
        if (args[0].numel() > 0) {
            const double *src = args[0].doubleData();
            std::copy(src, src + args[0].numel(), flat.doubleDataMut());
        }
        auto [v, c] = mode(flat, 2, mr);
        outs[0] = std::move(v);
        if (nargout > 1) outs[1] = std::move(c);
        if (nargout > 2) outs[2] = computeModeCell(args[0], 0, true, mr);
        return;
    }
    auto [v, c] = mode(args[0], dim, mr);
    outs[0] = std::move(v);
    if (nargout > 1)
        outs[1] = std::move(c);
    if (nargout > 2)
        outs[2] = computeModeCell(args[0], dim, false, mr);
}

// skewness_reg / kurtosis_reg moved to toolboxes/stats/src/moments/moments.cpp

void cov_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
             CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "cov");
    std::pmr::memory_resource *mr = ctx.engine->resource();

    // A trailing NaN-policy string ('includenan'|'omitrows'|'partialrows')
    // may follow any signature. Strip it; the remaining args are numeric.
    CovNan nanMode = CovNan::Include;
    std::size_t nargs = args.size();
    if (nargs >= 2 && parseCovNanFlag(args[nargs - 1], nanMode))
        --nargs;

    if (nanMode != CovNan::Include) {
        // 'omitrows' / 'partialrows' over the numeric arg(s).
        //   1 numeric arg : cov(X, flag)
        //   2 numeric args: cov(X, w, flag) [w scalar 0/1] | cov(x, y, flag)
        //   3 numeric args: cov(x, y, w, flag)
        if (nargs == 1) {
            outs[0] = covNanAware(args[0], 0, nanMode, mr);
            return;
        }
        if (nargs == 2) {
            if (args[1].isScalar()) {
                const double v = args[1].toScalar();
                if (v == 0.0 || v == 1.0) {
                    outs[0] = covNanAware(args[0], static_cast<int>(v),
                                          nanMode, mr);
                    return;
                }
            }
            outs[0] = covNanAware(assembleXY(args[0], args[1], mr), 0,
                                  nanMode, mr);
            return;
        }
        const double wv = argcheck::expectedReal(args[2], 3, "w", "cov");
        argcheck::expectedFlag01(wv, 3, "w", "cov");
        outs[0] = covNanAware(assembleXY(args[0], args[1], mr),
                              static_cast<int>(wv), nanMode, mr);
        return;
    }

    // 'includenan' (default) â†’ plain cov over the numeric args.
    if (nargs == 1) {
        outs[0] = cov(args[0], 0, mr);
        return;
    }
    // 2-arg form is ambiguous: cov(x, normFlag) vs cov(x, y).
    // Disambiguate exactly the way MATLAB does: if the second arg is a
    // scalar (0 or 1), it's normFlag; otherwise it's y.
    if (nargs == 2) {
        if (args[1].isScalar()) {
            const double v = args[1].toScalar();
            if (v == 0.0 || v == 1.0) {
                outs[0] = cov(args[0], static_cast<int>(v), mr);
                return;
            }
        }
        outs[0] = cov(args[0], args[1], 0, mr);
        return;
    }
    // 3-arg form: cov(x, y, normFlag).
    const double wv = argcheck::expectedReal(args[2], 3, "w", "cov");
    argcheck::expectedFlag01(wv, 3, "w", "cov");
    outs[0] = cov(args[0], args[1], static_cast<int>(wv), mr);
}

// Widen a single matrix to double (stats kernels are double-based; the
// result narrows back â€” parity predicates cover the fp difference).
static Value widenSingleToDouble(const Value &v, std::pmr::memory_resource *mr)
{
    Value d = Value::matrix(v.dims().rows(), v.dims().cols(), ValueType::DOUBLE, mr);
    const float *sv = v.singleData();
    double *dv = d.doubleDataMut();
    for (std::size_t i = 0; i < v.numel(); ++i)
        dv[i] = static_cast<double>(sv[i]);
    return d;
}
void corrcoef_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "corrcoef");  // no max: name-value tail in the reference
    std::pmr::memory_resource *mr = ctx.engine->resource();
    // Probed R2025b: integer inputs reject (mixedClasses); single
    // contagion; complex input is fine for R alone but rejects with a
    // ComplexInputs-class error once p-values are requested.
    for (std::size_t i = 0; i < args.size(); ++i) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) break;  // NV tail begins
        if (isIntegerType(a.type()))
            throw Error("corrcoef: integers can only be combined with "
                        "integers of the same class, or scalar doubles.",
                        0, 0, "corrcoef", "", "numkit:corrcoef:mixedClasses");
    }

    const bool twoArg =
        (args.size() >= 2 && !args[1].isChar() && !args[1].isString());
    const std::size_t nvStart = twoArg ? 2 : 1;
    // Single inputs widen to double for the kernels; the result narrows
    // back to single (the reference computes single end-to-end â€” parity
    // uses closeness predicates).
    const CorrcoefRows rows = parseCorrcoefRows(args, nvStart);
    const double alpha = parseCorrcoefAlpha(args, nvStart);
    const bool hasComplex = args[0].isComplex() || (twoArg && args[1].isComplex());
    if (hasComplex && nargout >= 2) {
        throw Error("corrcoef: cannot compute p-values for complex inputs.",
                    0, 0, "corrcoef", "", "numkit:corrcoef:ComplexInputs");
    }

    if (rows == CorrcoefRows::All) {
        double n;
        if (!twoArg) {
            const Value &x = args[0];
            n = (x.dims().isVector() || x.isScalar())
                    ? static_cast<double>(x.numel())          // single variable
                    : static_cast<double>(x.dims().rows());   // n obs × p vars
            Value xd = (!x.isComplex() && x.type() == ValueType::SINGLE)
                ? widenSingleToDouble(x, mr) : Value(x);
            Value R = corrcoef(xd, mr);
            if (nargout >= 2) {
                outs[1] = corrcoefPValues(R, n, mr);
            }
            outs[0] = std::move(R);
            if (!outs[0].isComplex() && x.type() == ValueType::SINGLE)
                outs[0] = narrowToSingle(std::move(outs[0]), mr);
        } else {
            n = static_cast<double>(args[0].numel());         // corrcoef(x, y)
            Value x0 = (!args[0].isComplex() && args[0].type() == ValueType::SINGLE)
                ? widenSingleToDouble(args[0], mr) : Value(args[0]);
            Value x1 = (!args[1].isComplex() && args[1].type() == ValueType::SINGLE)
                ? widenSingleToDouble(args[1], mr) : Value(args[1]);
            Value R = corrcoef(x0, x1, mr);
            if (nargout >= 2) {
                outs[1] = corrcoefPValues(R, n, mr);
            }
            outs[0] = std::move(R);
            if (!outs[0].isComplex() &&
                (args[0].type() == ValueType::SINGLE || args[1].type() == ValueType::SINGLE))
                outs[0] = narrowToSingle(std::move(outs[0]), mr);
        }
        if (nargout >= 3) {
            Value RL, RU;
            corrcoefConfBounds(outs[0], n, alpha, RL, RU, mr);
            outs[2] = std::move(RL);
            if (nargout >= 4) outs[3] = std::move(RU);
        }
        return;
    }

    // NaN-aware paths. Assemble the working matrix ([x(:) y(:)] for the
    // two-vector form) and apply listwise/pairwise deletion.
    const Value M = twoArg ? assembleXY(args[0], args[1], mr) : args[0];
    const bool isVec = M.dims().isVector() || M.isScalar();

    if (rows == CorrcoefRows::Pairwise) {
        if (isVec) {
            const std::size_t nValid = countCompleteRows(M);
            Value rMat = Value::matrix(1, 1, ValueType::DOUBLE, mr);
            rMat.doubleDataMut()[0] = (nValid <= 1) ? std::nan("") : 1.0;
            outs[0] = rMat;
            if (nargout >= 2) outs[1] = rMat;
            if (nargout >= 3) {
                Value RL, RU;
                corrcoefConfBounds(outs[0], static_cast<double>(nValid), alpha, RL, RU, mr);
                outs[2] = std::move(RL);
                if (nargout >= 4) outs[3] = std::move(RU);
            }
            return;
        }
        Value MCounts;
        outs[0] = corrcoefPairwiseWithCounts(M, (nargout >= 2) ? &MCounts : nullptr, mr);
        if (nargout >= 2) {
            outs[1] = corrcoefPValuesPairwise(outs[0], MCounts, mr);
            if (nargout >= 3) {
                Value RL, RU;
                corrcoefConfBoundsPairwise(outs[0], MCounts, alpha, RL, RU, mr);
                outs[2] = std::move(RL);
                if (nargout >= 4) outs[3] = std::move(RU);
            }
        }
        return;
    }

    // Complete (listwise deletion).
    if (isVec) {
        const std::size_t nValid = countCompleteRows(M);
        Value rMat = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        rMat.doubleDataMut()[0] = (nValid <= 1) ? std::nan("") : 1.0;
        outs[0] = rMat;
        if (nargout >= 2) outs[1] = rMat;
        if (nargout >= 3) {
            Value RL, RU;
            corrcoefConfBounds(outs[0], static_cast<double>(nValid), alpha, RL, RU, mr);
            outs[2] = std::move(RL);
            if (nargout >= 4) outs[3] = std::move(RU);
        }
        return;
    }
    outs[0] = corrcoefFromCov(covNanAware(M, 0, CovNan::Omitrows, mr), mr);
    if (nargout >= 2) {
        const double n = static_cast<double>(countCompleteRows(M));
        outs[1] = corrcoefPValues(outs[0], n, mr);
        if (nargout >= 3) {
            Value RL, RU;
            corrcoefConfBounds(outs[0], n, alpha, RL, RU, mr);
            outs[2] = std::move(RL);
            if (nargout >= 4) outs[3] = std::move(RU);
        }
    }
}

// nan*_reg adapters all moved to toolboxes/stats/src/nan_aware/nan_aware.cpp.

} // namespace detail

} // namespace numkit::stats
