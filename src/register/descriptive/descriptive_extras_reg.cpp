// toolboxes/stats/src/descriptive/descriptive_extras_reg.cpp
//
// CallContext register half of descriptive/descriptive_extras.cpp (Phase 2b
// compute/register split). Engine-coupled glue over the engine-free compute.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/stats/distributions/students_t.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include "descriptive/descriptive_extras_detail.hpp"
#include "support/reduce_vecdim.hpp"
#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::stats {

namespace detail {

void prepareCurveData_reg(Span<const Value> args, size_t nargout,
                          Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "prepareCurveData");
    auto *mr = ctx.engine->resource();
    Value w_empty = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    const Value &w = (args.size() >= 3) ? args[2] : w_empty;
    auto [xo, yo, wo] = prepareCurveData(args[0], args[1], w, mr);
    outs[0] = std::move(xo);
    if (nargout > 1) outs[1] = std::move(yo);
    if (nargout > 2) outs[2] = std::move(wo);
}

void prepareSurfaceData_reg(Span<const Value> args, size_t nargout,
                            Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "prepareSurfaceData");
    auto [xo, yo, zo] = prepareSurfaceData(args[0], args[1], args[2], ctx.engine->resource());
    outs[0] = std::move(xo);
    if (nargout > 1) outs[1] = std::move(yo);
    if (nargout > 2) outs[2] = std::move(zo);
}

void ksdensity_reg(Span<const Value> args, size_t nargout,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "ksdensity");  // no max: name-value tail (shared reg also serves the kde alias: union contract)
    auto *mr = ctx.engine->resource();
    Value pts = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    double bw_user = 0.0;
    std::string kernel = "normal";
    std::string function_mode = "pdf";
    size_t numpoints = 100;
    const Value *weights = nullptr;
    auto lower = [](std::string s) {
        for (auto &c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    size_t i = 1;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()
        && !args[i].isEmpty()) {
        pts = args[i];
        ++i;
    }
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString()) break;
        const std::string name = lower(args[i].toString());
        const Value &v = args[i + 1];
        if      (name == "bandwidth" || name == "width") {
            if (v.isChar() || v.isString()) {
                // 'normal-approx' / 'plug-in' string forms — only
                // 'normal-approx' (default behavior) is supported.
                const std::string s = lower(v.toString());
                if (s != "normal-approx" && s != "plug-in")
                    throw Error("ksdensity: unknown Bandwidth string '" + s + "'",
                                0, 0, "ksdensity", "", "nk:ksdensity:bw");
                bw_user = 0.0;
            } else {
                bw_user = v.toScalar();
            }
        }
        else if (name == "kernel")    kernel = v.toString();
        else if (name == "function")  function_mode = v.toString();
        else if (name == "numpoints") numpoints = (size_t)v.toScalar();
        else if (name == "weights")   { if (!v.isEmpty()) weights = &v; }
        else if (name == "censoring" || name == "support"
                 || name == "boundarycorrection") {
            if (!v.isEmpty())
                throw Error("ksdensity: '" + name + "' is not yet supported",
                            0, 0, "ksdensity", "", "nk:ksdensity:nyi");
        }
        // 'PlotFcn' silently ignored (no-op headless).
        i += 2;
    }
    auto R = ksdensity_full(args[0], pts, bw_user, kernel, function_mode, numpoints, weights, mr);
    outs[0] = std::move(R.f);
    if (nargout > 1) outs[1] = std::move(R.xi);
    if (nargout > 2) outs[2] = std::move(R.bw);
}

void datastats_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "datastats");
    auto build = [&](const Value &v) {
        auto [num, mx, mn, me, md, rg, sd] =
            datastats(v, ctx.engine->resource());
        Value s = Value::structure(ctx.engine->resource());
        s.field("num")    = num;
        s.field("max")    = mx;
        s.field("min")    = mn;
        s.field("mean")   = me;
        s.field("median") = md;
        s.field("range")  = rg;
        s.field("std")    = sd;
        return s;
    };
    outs[0] = build(args[0]);
    // Two-arg form returns separate stats structs for x and y.
    // We only fill outs[1] if a second argument was supplied AND the
    // caller actually requested two outputs (otherwise it's a no-op).
    if (args.size() >= 2 && outs.size() > 1)
        outs[1] = build(args[1]);
}

void bounds_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "bounds");
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
            else throw Error("bounds: unknown flag '" + s + "'",
                              0, 0, "bounds", "", "nk:bounds:badFlag");
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, "bounds");
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            std::vector<int> set = vecdim::parseDimSet(a, "bounds");
            if (vecdim::coversAllDims(set, args[0]))
                flatten = true;
            else
                vecdimSet = std::move(set);
        }
    }
    auto *mr = ctx.engine->resource();
    if (!vecdimSet.empty()) {
        // Partial vecdim — same kernels as the scalar-dim path.
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        const int lastd = static_cast<int>(c.dims().ndim());
        auto [lo, hi] = bounds(c, lastd, mr);
        outs[0] = vecdim::expandReduced(std::move(lo), args[0], vecdimSet, mr);
        if (nargout > 1)
            outs[1] = vecdim::expandReduced(std::move(hi), args[0], vecdimSet, mr);
        return;
    }
    if (flatten) {
        Value flat = Value::matrix(1, args[0].numel(), ValueType::DOUBLE, mr);
        if (args[0].numel() > 0) {
            const double *src = args[0].doubleData();
            std::copy(src, src + args[0].numel(), flat.doubleDataMut());
        }
        auto [lo, hi] = bounds(flat, 2, mr);
        outs[0] = std::move(lo);
        if (nargout > 1) outs[1] = std::move(hi);
    } else {
        auto [lo, hi] = bounds(args[0], dim, mr);
        outs[0] = std::move(lo);
        if (nargout > 1) outs[1] = std::move(hi);
    }
}

void iqr_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "iqr");
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
            else throw Error("iqr: unknown flag '" + s + "'",
                              0, 0, "iqr", "", "nk:iqr:badFlag");
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, "iqr");
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            std::vector<int> set = vecdim::parseDimSet(a, "iqr");
            if (vecdim::coversAllDims(set, args[0]))
                flatten = true;
            else
                vecdimSet = std::move(set);
        }
    }
    auto *mr = ctx.engine->resource();
    if (!vecdimSet.empty()) {
        // Partial vecdim — same kernel as the scalar-dim path.
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        const int lastd = static_cast<int>(c.dims().ndim());
        outs[0] = vecdim::expandReduced(iqr(c, lastd, mr), args[0],
                                        vecdimSet, mr);
        return;
    }
    if (flatten) {
        // Flatten and compute on the 1×N row.
        Value flat = Value::matrix(1, args[0].numel(), ValueType::DOUBLE, mr);
        if (args[0].numel() > 0) {
            const double *src = args[0].doubleData();
            std::copy(src, src + args[0].numel(), flat.doubleDataMut());
        }
        outs[0] = iqr(flat, 2, mr);
    } else {
        outs[0] = iqr(args[0], dim, mr);
    }
}

void maxk_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "maxk");  // no max: name-value tail in the reference
    const int k = argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "k", "maxk"), 2, "k", "maxk");
    argcheck::expectedNonNegative(double(k), 2, "k", "maxk");
    int dim = 0;
    // Optional positional dim (numeric scalar that's not a string).
    size_t i = 2;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()
        && !args[i].isEmpty()) {
        dim = argcheck::dimArg(args[i], "maxk"); ++i;
    }
    // Remaining args may be Name-Value pairs; only 'ComparisonMethod'
    // is documented (real|abs|auto). For real input 'auto' = 'real'; 'abs'
    // ranks by magnitude |x| (returning the original signed values).
    bool byAbs = false;
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("maxk: expected Name-Value pair",
                        0, 0, "maxk", "", "nk:maxk:nv");
        std::string name = args[i].toString();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (name == "comparisonmethod") {
            std::string m = args[i + 1].toString();
            std::transform(m.begin(), m.end(), m.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            if (m != "real" && m != "abs" && m != "auto")
                throw Error("maxk: ComparisonMethod must be 'real', 'abs' or 'auto'",
                            0, 0, "maxk", "", "nk:maxk:cm");
            byAbs = (m == "abs");
        } else {
            throw Error("maxk: unknown Name-Value '" + name + "'",
                        0, 0, "maxk", "", "nk:maxk:nv");
        }
        i += 2;
    }
    auto *mr = ctx.engine->resource();
    Value idx;
    Value *idxPtr = (nargout >= 2) ? &idx : nullptr;
    outs[0] = topKAlongDim(args[0], dim, k, /*ascending=*/false, "maxk", mr, idxPtr, byAbs);
    if (nargout >= 2) outs[1] = std::move(idx);
}

void mink_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "mink");  // no max: name-value tail in the reference
    const int k = argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "k", "mink"), 2, "k", "mink");
    argcheck::expectedNonNegative(double(k), 2, "k", "mink");
    int dim = 0;
    size_t i = 2;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()
        && !args[i].isEmpty()) {
        dim = argcheck::dimArg(args[i], "mink"); ++i;
    }
    bool byAbs = false;
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("mink: expected Name-Value pair",
                        0, 0, "mink", "", "nk:mink:nv");
        std::string name = args[i].toString();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (name == "comparisonmethod") {
            std::string m = args[i + 1].toString();
            std::transform(m.begin(), m.end(), m.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            if (m != "real" && m != "abs" && m != "auto")
                throw Error("mink: ComparisonMethod must be 'real', 'abs' or 'auto'",
                            0, 0, "mink", "", "nk:mink:cm");
            byAbs = (m == "abs");
        } else {
            throw Error("mink: unknown Name-Value '" + name + "'",
                        0, 0, "mink", "", "nk:mink:nv");
        }
        i += 2;
    }
    auto *mr = ctx.engine->resource();
    Value idx;
    Value *idxPtr = (nargout >= 2) ? &idx : nullptr;
    outs[0] = topKAlongDim(args[0], dim, k, /*ascending=*/true, "mink", mr, idxPtr, byAbs);
    if (nargout >= 2) outs[1] = std::move(idx);
}

// Common parser for mape/rmse trailing args: optional dim ('all', vecdim,
// integer scalar). Returns (dim, flatten, partial vecdim set). Vector
// inputs to mape/rmse are inherently 1-D so flatten and dim=0 produce the
// same result.
namespace {
void parseDimOrAll(const Value &x, Span<const Value> args, size_t pos,
                   int &dim, bool &flatten, std::vector<int> &vecdimSet,
                   const char *fn)
{
    dim = 0; flatten = false; vecdimSet.clear();
    if (pos >= args.size() || args[pos].isEmpty()) return;
    const Value &a = args[pos];
    if (a.isChar() || a.isString()) {
        std::string s = a.toString();
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (s == "all") { flatten = true; return; }
        // Any other string belongs to the flag/Weights parser that
        // runs after this (nanflags / 'Weights'); do not consume it.
        return;
    }
    if (a.numel() == 1) { dim = argcheck::dimArg(a, fn); return; }
    // vecdim: full coverage reduces like 'all'; a partial set collapses
    // the listed dims into one trailing dim.
    std::vector<int> set = vecdim::parseDimSet(a, fn);
    if (vecdim::coversAllDims(set, x))
        flatten = true;
    else
        vecdimSet = std::move(set);
}

Value flattenToRow(const Value &x, std::pmr::memory_resource *mr)
{
    Value flat = Value::matrix(1, x.numel(), ValueType::DOUBLE, mr);
    if (x.numel() > 0) {
        const double *src = x.doubleData();
        std::copy(src, src + x.numel(), flat.doubleDataMut());
    }
    return flat;
}
} // anonymous

void mape_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "mape");  // no max: flag/NV tail in the reference
    int dim = 0; bool flatten = false; std::vector<int> vecdimSet;
    parseDimOrAll(args[0], args, 2, dim, flatten, vecdimSet, "mape");
    auto *mr = ctx.engine->resource();
    // Flags + Weights (probe71 contract; classes live in the compute
    // layer's parseErrorMetricOpts).
    const bool allConsumed =
        flatten && args.size() > 2 && (args[2].isChar() || args[2].isString());
    const std::size_t flagStart =
        (args.size() > 2 && !args[2].isEmpty() &&
         (!(args[2].isChar() || args[2].isString()) || allConsumed)) ? 3 : 2;
    const std::size_t dimLen = flatten ? std::max(args[0].numel(),
                                                  args[1].numel())
        : args[0].dims().dim(dim > 0 ? static_cast<std::size_t>(dim) - 1 : 0);
    const ErrorMetricOpts opts = parseErrorMetricOpts(
        args, flagStart, true, args[0], args[1], dimLen, "mape");
    if (!vecdimSet.empty()) {
        // Partial vecdim — collapse F and A with the same set so the
        // slices stay paired, then reduce along the last dim.
        Value f = vecdim::collapseDims(args[0], vecdimSet, mr);
        Value a = vecdim::collapseDims(args[1], vecdimSet, mr);
        const int lastd = static_cast<int>(f.dims().ndim());
        outs[0] = vecdim::expandReduced(mape(f, a, lastd, opts, mr), args[0],
                                        vecdimSet, mr);
        return;
    }
    if (flatten) {
        outs[0] = mape(flattenToRow(args[0], mr), flattenToRow(args[1], mr),
                       2, opts, mr);
    } else {
        outs[0] = mape(args[0], args[1], dim, opts, mr);
    }
}

void rmse_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "rmse");  // no max: flag/NV tail in the reference
    int dim = 0; bool flatten = false; std::vector<int> vecdimSet;
    parseDimOrAll(args[0], args, 2, dim, flatten, vecdimSet, "rmse");
    auto *mr = ctx.engine->resource();
    const bool allConsumed =
        flatten && args.size() > 2 && (args[2].isChar() || args[2].isString());
    const std::size_t flagStart =
        (args.size() > 2 && !args[2].isEmpty() &&
         (!(args[2].isChar() || args[2].isString()) || allConsumed)) ? 3 : 2;
    const std::size_t dimLen = flatten ? std::max(args[0].numel(),
                                                  args[1].numel())
        : args[0].dims().dim(dim > 0 ? static_cast<std::size_t>(dim) - 1 : 0);
    const ErrorMetricOpts opts = parseErrorMetricOpts(
        args, flagStart, false, args[0], args[1], dimLen, "rmse");
    if (!vecdimSet.empty()) {
        // Partial vecdim — collapse F and A with the same set so the
        // slices stay paired, then reduce along the last dim.
        Value f = vecdim::collapseDims(args[0], vecdimSet, mr);
        Value a = vecdim::collapseDims(args[1], vecdimSet, mr);
        const int lastd = static_cast<int>(f.dims().ndim());
        outs[0] = vecdim::expandReduced(rmse(f, a, lastd, opts, mr), args[0],
                                        vecdimSet, mr);
        return;
    }
    if (flatten) {
        outs[0] = rmse(flattenToRow(args[0], mr), flattenToRow(args[1], mr),
                       2, opts, mr);
    } else {
        outs[0] = rmse(args[0], args[1], dim, opts, mr);
    }
}

void ecdf_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "ecdf");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    std::string function_mode = "cdf";
    double alpha = 0.05;
    const Value *freq = nullptr;
    auto lower = [](std::string s) {
        for (auto &c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    for (size_t i = 1; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) break;
        const std::string key = lower(args[i].toString());
        const Value &v = args[i + 1];
        if      (key == "function")  function_mode = v.toString();
        else if (key == "frequency") {
            if (!v.isEmpty()) freq = &v;
        }
        else if (key == "alpha")     alpha = v.toScalar();
        else if (key == "censoring") {
            if (!v.isEmpty())
                throw Error("ecdf: 'Censoring' is not yet supported "
                            "(Kaplan-Meier estimator). Skip the arg or "
                            "filter censored observations beforehand.",
                            0, 0, "ecdf", "", "nk:ecdf:censoring_nyi");
        }
        else if (key == "iterationlimit" || key == "tolerance"
                 || key == "icmfrequency" || key == "bounds") {
            // Silently accepted (no-op for non-censored ecdf).
        }
    }
    const bool want_bounds = (nargout > 2);
    auto R = ecdf_full(args[0], freq, function_mode, alpha, want_bounds, mr);
    outs[0] = std::move(R.f);
    if (nargout > 1) outs[1] = std::move(R.x);
    if (nargout > 2) outs[2] = std::move(R.flo);
    if (nargout > 3) outs[3] = std::move(R.fup);
}

void ecdfhist_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "ecdfhist");  // no max: name-value tail in the reference
    int m = 10;
    if (args.size() >= 3 && !args[2].isEmpty())
        m = static_cast<int>(argcheck::roundLength(args[2], 3, "m", "ecdfhist"));
    auto [n, c] = ecdfhist(args[0], args[1], m, ctx.engine->resource());
    outs[0] = std::move(n);
    if (nargout > 1) outs[1] = std::move(c);
}

// ── partialcorr adapter ──────────────────────────────────────────────

void partialcorr_reg(Span<const Value> args, size_t /*nargout*/,
                     Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "partialcorr");  // no max: name-value tail in the reference
    // Marshalling only — the matrix/float gates, the (X) / (X, Z) /
    // (X, Y, Z) positional split and the Rows/Type/Tail name-value
    // contract live in partialcorr_full (correlation.cpp). A numeric 2nd
    // argument is Z (or Y when a numeric 3rd follows); anything else
    // starts the name-value tail.
    const Value *y = nullptr, *z = nullptr;
    std::size_t posN = 1;
    if (args.size() > 1 && args[1].isNumeric()) {
        z = &args[1];
        posN = 2;
        if (args.size() > 2 && args[2].isNumeric()) {
            y = &args[1];
            z = &args[2];
            posN = 3;
        }
    }
    outs[0] = partialcorr_full(
        args[0], y, z,
        Span<const Value>(args.data() + posN, args.size() - posN),
        ctx.engine->resource());
}

// ── corr / detrend adapters ──────────────────────────────────────────

// corr(X[, Y][, NV…]) — marshalling only; the Type/Rows/Tail/Weights
// name-value contract, the zero-weight row prep, the complex gates and
// the tail-aware p-values live in corr_full (correlation.cpp).
void corr_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "corr");  // no max: name-value tail in the reference
    // Distinguish corr(X[, NV…]) from corr(X, Y[, NV…]): a char/string
    // 2nd argument starts the name-value tail (reference ischar dispatch).
    const bool twoArg =
        (args.size() >= 2 && !args[1].isChar() && !args[1].isString());
    const std::size_t nvStart = twoArg ? 2 : 1;
    CorrOut r = corr_full(args[0], twoArg ? &args[1] : nullptr,
                          Span<const Value>(args.data() + nvStart,
                                            args.size() - nvStart),
                          nargout >= 2, ctx.engine->resource());
    outs[0] = std::move(r.rho);
    if (nargout >= 2) outs[1] = std::move(r.p);
}

// detrend(A, ...) — marshalling only; the full parse+validate+
// compute contract lives in cleaning.cpp's detrend_full (float gate,
// degree/bp/nanflag/Continuous NVs, single + N-D results).
void detrend_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "detrend");  // no max: flag/option tail in the reference
    auto *mr = ctx.engine->resource();
    outs[0] = detrend_full(args[0],
                           Span<const Value>(args.data() + 1, args.size() - 1), mr);
}

// ── missing-data adapters ────────────────────────────────────────────

// Method-aware isoutlier: per-column detection (MATLAB operates per column
// for matrices) via detect_one_column. detectTf is the value detect_one_column
// expects (median/mean: 3 == MATLAB ThresholdFactor; quartiles: 2*MATLAB-tf
// because detect_one_column scales by tf*0.5).
// isoutlier / rmoutliers — marshalling only; the parse+validate+
// compute contracts (method prefixes, windows, ThresholdFactor /
// MaxNumOutliers ranges, per-element moving thresholds, single
// contagion, row/column removal) live in cleaning.cpp.
void isoutlier_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isoutlier");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    IsOutlierOut r = isoutlier_full(args[0],
                                    Span<const Value>(args.data() + 1, args.size() - 1),
                                    nargout > 1, mr);
    outs[0] = std::move(r.tf);
    if (nargout > 1) outs[1] = std::move(r.lo);
    if (nargout > 2) outs[2] = std::move(r.hi);
    if (nargout > 3) outs[3] = std::move(r.center);
}

void rmoutliers_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "rmoutliers");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    RmOutliersOut r = rmoutliers_full(args[0],
                                      Span<const Value>(args.data() + 1, args.size() - 1), mr);
    outs[0] = std::move(r.B);
    if (nargout > 1) outs[1] = std::move(r.ind);
    if (nargout > 2) outs[2] = std::move(r.tf);
    if (nargout > 3) outs[3] = std::move(r.lo);
    if (nargout > 4) outs[4] = std::move(r.hi);
    if (nargout > 5) outs[5] = std::move(r.center);
}

// fillmissing / rmmissing / standardizeMissing — marshalling only;
// the contracts (constant-value requirements, EndValues semantics,
// movmean/movmedian fills, row removal, indicator vectors) live in
// cleaning.cpp.
void fillmissing_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "fillmissing");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    FillMissingOut r = fillmissing_full(args[0],
                                        Span<const Value>(args.data() + 1, args.size() - 1),
                                        nargout > 1, mr);
    outs[0] = std::move(r.B);
    if (nargout > 1) outs[1] = std::move(r.filled);
}

void rmmissing_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "rmmissing");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    RmMissingOut r = rmmissing_full(args[0],
                                    Span<const Value>(args.data() + 1, args.size() - 1), mr);
    outs[0] = std::move(r.B);
    if (nargout > 1) outs[1] = std::move(r.ind);
}

void standardizeMissing_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "standardizeMissing");
    outs[0] = standardizeMissing_full(args[0], args[1], ctx.engine->resource());
}

// filloutliers(A, fillmethod[, findmethod][, NV])
//   fillmethod : numeric scalar | "center" | "clip" | "previous" |
//                "next" | "nearest" | "linear"
//   findmethod : "median" (default) | "mean" | "quartiles"
//   NV         : ThresholdFactor (default per-method)
void filloutliers_reg(Span<const Value> args, size_t /*nargout*/,
                      Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "filloutliers");  // no max: name-value tail in the reference
    auto lower = [](std::string v) {
        for (char &ch : v)
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return v;
    };

    std::string detect = "median";
    double tf = 3.0;
    bool tf_set = false;
    double loP = 0.0, hiP = 0.0;
    std::size_t i = 2;
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        const std::string s = lower(args[i].toString());
        // Distinguish findmethod string vs NV name. NV names are known.
        if (s == "thresholdfactor" || s == "maxnumoutliers" ||
            s == "samplepoints"    || s == "outlierlocations") {
            // fall through — handled by NV loop below.
        } else {
            detect = s;
            if (detect == "median" || detect == "mean" || detect == "quartiles") {
                ++i;
            } else if (detect == "percentiles") {
                ++i;
                if (i >= args.size() || args[i].numel() != 2)
                    throw Error("filloutliers: 'percentiles' requires a "
                                "2-element [lower upper] vector",
                                0, 0, "filloutliers", "",
                                "nk:filloutliers:percentiles");
                loP = args[i].elemAsDouble(0);
                hiP = args[i].elemAsDouble(1);
                ++i;
            } else {
                throw Error("filloutliers: findmethod must be 'median', "
                            "'mean', 'quartiles', or 'percentiles' in this "
                            "revision (the reference engine also "
                            "supports 'grubbs', 'gesd', "
                            "'movmedian', 'movmean' — deferred)",
                            0, 0, "filloutliers", "",
                            "nk:filloutliers:findmethod");
            }
        }
    }
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("filloutliers: expected name-value pair",
                        0, 0, "filloutliers", "", "nk:filloutliers:nv");
        const std::string nm = lower(args[i].toString());
        if (nm == "thresholdfactor") {
            tf = args[i + 1].toScalar(); tf_set = true;
        } else {
            throw Error("filloutliers: unsupported name-value parameter '"
                        + args[i].toString() + "'",
                        0, 0, "filloutliers", "", "nk:filloutliers:nv");
        }
        i += 2;
    }
    // MATLAB's per-method default ThresholdFactor.
    if (!tf_set) {
        if (detect == "quartiles") tf = 3.0;     // 1.5·IQR → scaled by 0.5 internally so 3.0 here
        else                       tf = 3.0;
    } else if (detect == "quartiles") {
        // User-set tf for quartiles means "k" in [Q1 - k·IQR, Q3 + k·IQR].
        // Our internal formula uses 0.5·tf·IQR, so multiply by 2.
        tf = 2.0 * tf;
    }
    outs[0] = filloutliers_of(args[0], args[1], detect, tf, loP, hiP,
                              ctx.engine->resource());
}

// ── range / mad / geomean / harmmean / moment / trimmean adapters ────

void range_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "range");
    const int dim = (args.size() >= 2) ? argcheck::dimArg(args[1], "range") : 0;
    outs[0] = range_of(args[0], dim, ctx.engine->resource());
}

void mad_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "mad");
    int flag = 0;
    int dim  = 0;
    if (args.size() >= 2) {
        const double fv = argcheck::expectedReal(args[1], 2, "flag", "mad");
        argcheck::expectedFlag01(fv, 2, "flag", "mad");
        flag = static_cast<int>(fv);
    }
    if (args.size() >= 3)
        dim = argcheck::dimArg(args[2], "mad");
    outs[0] = mad_of(args[0], flag, dim, ctx.engine->resource());
}

// Parse a trailing 'omitnan'/'includenan' nanflag from a geomean/harmmean
// arg list. Returns the omit flag and the count of remaining numeric args.
bool parseMeanNanFlag(Span<const Value> args, const char *fn, std::size_t &nargs)
{
    nargs = args.size();
    if (nargs >= 2 && (args[nargs - 1].isChar() || args[nargs - 1].isString())) {
        std::string f = args[nargs - 1].toString();
        for (char &c : f) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (f == "omitnan")     { --nargs; return true; }
        if (f == "includenan")  { --nargs; return false; }
        throw Error(std::string(fn) + ": unknown option '" + f + "'",
                    0, 0, fn, "", std::string("nk:") + fn + ":badopt");
    }
    return false;
}

void geomean_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "geomean");  // no max: flag/option tail in the reference
    std::size_t nargs;
    const bool omitnan = parseMeanNanFlag(args, "geomean", nargs);
    const int dim = (nargs >= 2) ? argcheck::dimArg(args[1], "geomean") : 0;
    outs[0] = geomean_of(args[0], dim, omitnan, ctx.engine->resource());
}

void harmmean_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "harmmean");  // no max: flag/option tail in the reference
    std::size_t nargs;
    const bool omitnan = parseMeanNanFlag(args, "harmmean", nargs);
    const int dim = (nargs >= 2) ? argcheck::dimArg(args[1], "harmmean") : 0;
    outs[0] = harmmean_of(args[0], dim, omitnan, ctx.engine->resource());
}

void moment_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "moment");
    const int order = argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "order", "moment"), 2, "order", "moment");
    int dim = 0;
    std::vector<int> vecdimSet;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        // A vecdim (full or partial cover) always collapses — moment has
        // no separate 'all' path, and the collapse handles both.
        if (args[2].numel() > 1)
            vecdimSet = vecdim::parseDimSet(args[2], "moment");
        else
            dim = argcheck::dimArg(args[2], "moment");
    }
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet,
                                       ctx.engine->resource());
        outs[0] = vecdim::expandReduced(
            moment_of(c, order, static_cast<int>(c.dims().ndim()),
                      ctx.engine->resource()),
            args[0], vecdimSet, ctx.engine->resource());
        return;
    }
    outs[0] = moment_of(args[0], order, dim, ctx.engine->resource());
}

void trimmean_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "trimmean");
    const double pct = args[1].toScalar();

    // trimmean(x, percent [, flag] [, dim]). The 3rd arg is EITHER a string
    // flag ('round' default, or 'floor') OR a numeric dim; if a flag is
    // present the dim may follow it. Distinguish by type before toScalar.
    bool useFloor = false;
    int dim = 0;
    std::vector<int> vecdimSet;
    std::size_t i = 2;
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        std::string f = args[i].toString();
        for (char &c : f) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (f == "floor")      useFloor = true;
        else if (f == "round") useFloor = false;
        else
            throw Error("trimmean: flag must be 'round' or 'floor'",
                        0, 0, "trimmean", "", "nk:trimmean:flag");
        ++i;
    }
    if (i < args.size()) {
        // A vecdim (full or partial cover) always collapses — trimmean
        // has no separate 'all' path, and the collapse handles both.
        if (args[i].numel() > 1)
            vecdimSet = vecdim::parseDimSet(args[i], "trimmean");
        else
            dim = argcheck::dimArg(args[i], "trimmean");
    }
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet,
                                       ctx.engine->resource());
        outs[0] = vecdim::expandReduced(
            trimmean_of(c, pct, static_cast<int>(c.dims().ndim()), useFloor,
                        ctx.engine->resource()),
            args[0], vecdimSet, ctx.engine->resource());
        return;
    }
    outs[0] = trimmean_of(args[0], pct, dim, useFloor, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::stats
