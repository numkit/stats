// toolboxes/stats/src/descriptive/descriptive_extras_reg.cpp
//
// CallContext register half of descriptive/descriptive_extras.cpp (Phase 2b
// compute/register split). Engine-coupled glue over the engine-free compute.
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/stats/distributions/students_t.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "descriptive/descriptive_extras_detail.hpp"
#include "support/reduce_vecdim.hpp"
#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void prepareCurveData_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                          keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "prepareCurveData");
    auto *mr = ctx.engine->allocator();
    Value w_empty = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    const Value &w = (args.size() >= 3) ? args[2] : w_empty;
    auto [xo, yo, wo] = prepareCurveData(args[0], args[1], w, mr);
    outs[0] = std::move(xo);
    if (nargout > 1) outs[1] = std::move(yo);
    if (nargout > 2) outs[2] = std::move(wo);
}

void prepareSurfaceData_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                            keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "prepareSurfaceData");
    auto [xo, yo, zo] = prepareSurfaceData(args[0], args[1], args[2], ctx.engine->allocator());
    outs[0] = std::move(xo);
    if (nargout > 1) outs[1] = std::move(yo);
    if (nargout > 2) outs[2] = std::move(zo);
}

void ksdensity_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "ksdensity");  // no max: name-value tail (shared reg also serves the kde alias: union contract)
    auto *mr = ctx.engine->allocator();
    Value pts = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    double bw_user = 0.0;
    keel::String kernel = keel::String::from("normal");
    keel::String function_mode = keel::String::from("pdf");
    keel::usize numpoints = 100;
    const Value *weights = nullptr;
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        for (auto &c : s) c = keel::to_ascii_lower(c);
        return s;
    };
    keel::usize i = 1;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()
        && !args[i].isEmpty()) {
        pts = args[i];
        ++i;
    }
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString()) break;
        const keel::String nameStr = lower(args[i].toString().as_ref());
        const keel::StringRef name = nameStr.as_ref();
        const Value &v = args[i + 1];
        if      (name == "bandwidth" || name == "width") {
            if (v.isChar() || v.isString()) {
                // 'normal-approx' / 'plug-in' string forms — only
                // 'normal-approx' (default behavior) is supported.
                const keel::String sStr = lower(v.toString().as_ref());
                const keel::StringRef s = sStr.as_ref();
                if (s != "normal-approx" && s != "plug-in")
                    throw Error(keel::format("ksdensity: unknown Bandwidth string '{}'", s),
                                0, 0, "ksdensity", "", "numkit:ksdensity:bw");
                bw_user = 0.0;
            } else {
                bw_user = v.toScalar();
            }
        }
        else if (name == "kernel")    kernel = v.toString();
        else if (name == "function")  function_mode = v.toString();
        else if (name == "numpoints") numpoints = (keel::usize)v.toScalar();
        else if (name == "weights")   { if (!v.isEmpty()) weights = &v; }
        else if (name == "censoring" || name == "support"
                 || name == "boundarycorrection") {
            if (!v.isEmpty())
                throw Error(keel::format("ksdensity: '{}' is not yet supported", name),
                            0, 0, "ksdensity", "", "numkit:ksdensity:nyi");
        }
        // 'PlotFcn' silently ignored (no-op headless).
        i += 2;
    }
    auto R = ksdensity_full(args[0], pts, bw_user, kernel.as_ref(), function_mode.as_ref(), numpoints, weights, mr);
    outs[0] = std::move(R.f);
    if (nargout > 1) outs[1] = std::move(R.xi);
    if (nargout > 2) outs[2] = std::move(R.bw);
}

void datastats_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "datastats");
    auto build = [&](const Value &v) {
        auto [num, mx, mn, me, md, rg, sd] =
            datastats(v, ctx.engine->allocator());
        Value s = Value::structure(ctx.engine->allocator());
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

void bounds_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "bounds");
    int dim = 0;
    bool flatten = false;
    keel::Array<int> vecdimSet;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &a = args[1];
        if (a.isChar() || a.isString()) {
            keel::String s = a.toString();
            std::transform(s.begin(), s.end(), s.begin(),
                [](unsigned char c){ return keel::to_ascii_lower(c); });
            if (s.as_ref() == "all") flatten = true;
            else throw Error(keel::format("bounds: unknown flag '{}'", s.as_ref()),
                              0, 0, "bounds", "", "numkit:bounds:badFlag");
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, "bounds");
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            keel::Array<int> set = vecdim::parseDimSet(a, "bounds");
            if (vecdim::coversAllDims(set, args[0]))
                flatten = true;
            else
                vecdimSet = std::move(set);
        }
    }
    auto *mr = ctx.engine->allocator();
    if (!vecdimSet.empty()) {
        // Partial vecdim — same kernels as the scalar-dim path.
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        const int lastd = keel::as<int>(c.dims().ndim());
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

void iqr_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "iqr");
    int dim = 0;
    bool flatten = false;
    keel::Array<int> vecdimSet;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &a = args[1];
        if (a.isChar() || a.isString()) {
            keel::String s = a.toString();
            std::transform(s.begin(), s.end(), s.begin(),
                [](unsigned char c){ return keel::to_ascii_lower(c); });
            if (s.as_ref() == "all") flatten = true;
            else throw Error(keel::format("iqr: unknown flag '{}'", s.as_ref()),
                              0, 0, "iqr", "", "numkit:iqr:badFlag");
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, "iqr");
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            keel::Array<int> set = vecdim::parseDimSet(a, "iqr");
            if (vecdim::coversAllDims(set, args[0]))
                flatten = true;
            else
                vecdimSet = std::move(set);
        }
    }
    auto *mr = ctx.engine->allocator();
    if (!vecdimSet.empty()) {
        // Partial vecdim — same kernel as the scalar-dim path.
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        const int lastd = keel::as<int>(c.dims().ndim());
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

void maxk_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "maxk");  // no max: name-value tail in the reference
    const int k = argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "k", "maxk"), 2, "k", "maxk");
    argcheck::expectedNonNegative(double(k), 2, "k", "maxk");
    int dim = 0;
    // Optional positional dim (numeric scalar that's not a string).
    keel::usize i = 2;
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
                        0, 0, "maxk", "", "numkit:maxk:nv");
        keel::String name = args[i].toString();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c){ return keel::to_ascii_lower(c); });
        if (name.as_ref() == "comparisonmethod") {
            keel::String m = args[i + 1].toString();
            std::transform(m.begin(), m.end(), m.begin(),
                           [](unsigned char c){ return keel::to_ascii_lower(c); });
            if (m.as_ref() != "real" && m.as_ref() != "abs" && m.as_ref() != "auto")
                throw Error("maxk: ComparisonMethod must be 'real', 'abs' or 'auto'",
                            0, 0, "maxk", "", "numkit:maxk:cm");
            byAbs = (m.as_ref() == "abs");
        } else {
            throw Error(keel::format("maxk: unknown Name-Value '{}'", name.as_ref()),
                        0, 0, "maxk", "", "numkit:maxk:nv");
        }
        i += 2;
    }
    auto *mr = ctx.engine->allocator();
    Value idx;
    Value *idxPtr = (nargout >= 2) ? &idx : nullptr;
    outs[0] = topKAlongDim(args[0], dim, k, /*ascending=*/false, "maxk", mr, idxPtr, byAbs);
    if (nargout >= 2) outs[1] = std::move(idx);
}

void mink_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "mink");  // no max: name-value tail in the reference
    const int k = argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "k", "mink"), 2, "k", "mink");
    argcheck::expectedNonNegative(double(k), 2, "k", "mink");
    int dim = 0;
    keel::usize i = 2;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()
        && !args[i].isEmpty()) {
        dim = argcheck::dimArg(args[i], "mink"); ++i;
    }
    bool byAbs = false;
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("mink: expected Name-Value pair",
                        0, 0, "mink", "", "numkit:mink:nv");
        keel::String name = args[i].toString();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c){ return keel::to_ascii_lower(c); });
        if (name.as_ref() == "comparisonmethod") {
            keel::String m = args[i + 1].toString();
            std::transform(m.begin(), m.end(), m.begin(),
                           [](unsigned char c){ return keel::to_ascii_lower(c); });
            if (m.as_ref() != "real" && m.as_ref() != "abs" && m.as_ref() != "auto")
                throw Error("mink: ComparisonMethod must be 'real', 'abs' or 'auto'",
                            0, 0, "mink", "", "numkit:mink:cm");
            byAbs = (m.as_ref() == "abs");
        } else {
            throw Error(keel::format("mink: unknown Name-Value '{}'", name.as_ref()),
                        0, 0, "mink", "", "numkit:mink:nv");
        }
        i += 2;
    }
    auto *mr = ctx.engine->allocator();
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
void parseDimOrAll(const Value &x, keel::ArrayRef<Value> args, keel::usize pos,
                   int &dim, bool &flatten, keel::Array<int> &vecdimSet,
                   const char *fn)
{
    dim = 0; flatten = false; vecdimSet.clear();
    if (pos >= args.size() || args[pos].isEmpty()) return;
    const Value &a = args[pos];
    if (a.isChar() || a.isString()) {
        keel::String s = a.toString();
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return keel::to_ascii_lower(c); });
        if (s.as_ref() == "all") { flatten = true; return; }
        // Any other string belongs to the flag/Weights parser that
        // runs after this (nanflags / 'Weights'); do not consume it.
        return;
    }
    if (a.numel() == 1) { dim = argcheck::dimArg(a, fn); return; }
    // vecdim: full coverage reduces like 'all'; a partial set collapses
    // the listed dims into one trailing dim.
    keel::Array<int> set = vecdim::parseDimSet(a, fn);
    if (vecdim::coversAllDims(set, x))
        flatten = true;
    else
        vecdimSet = std::move(set);
}

Value flattenToRow(const Value &x, keel::Allocator *mr)
{
    Value flat = Value::matrix(1, x.numel(), ValueType::DOUBLE, mr);
    if (x.numel() > 0) {
        const double *src = x.doubleData();
        std::copy(src, src + x.numel(), flat.doubleDataMut());
    }
    return flat;
}
} // anonymous

void mape_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "mape");  // no max: flag/NV tail in the reference
    int dim = 0; bool flatten = false; keel::Array<int> vecdimSet;
    parseDimOrAll(args[0], args, 2, dim, flatten, vecdimSet, "mape");
    auto *mr = ctx.engine->allocator();
    // Flags + Weights (probe71 contract; classes live in the compute
    // layer's parseErrorMetricOpts).
    const bool allConsumed =
        flatten && args.size() > 2 && (args[2].isChar() || args[2].isString());
    const keel::usize flagStart =
        (args.size() > 2 && !args[2].isEmpty() &&
         (!(args[2].isChar() || args[2].isString()) || allConsumed)) ? 3 : 2;
    const keel::usize dimLen = flatten ? std::max(args[0].numel(),
                                                  args[1].numel())
        : args[0].dims().dim(dim > 0 ? keel::as<keel::usize>(dim) - 1 : 0);
    const ErrorMetricOpts opts = parseErrorMetricOpts(
        args, flagStart, true, args[0], args[1], dimLen, "mape");
    if (!vecdimSet.empty()) {
        // Partial vecdim — collapse F and A with the same set so the
        // slices stay paired, then reduce along the last dim.
        Value f = vecdim::collapseDims(args[0], vecdimSet, mr);
        Value a = vecdim::collapseDims(args[1], vecdimSet, mr);
        const int lastd = keel::as<int>(f.dims().ndim());
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

void rmse_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "rmse");  // no max: flag/NV tail in the reference
    int dim = 0; bool flatten = false; keel::Array<int> vecdimSet;
    parseDimOrAll(args[0], args, 2, dim, flatten, vecdimSet, "rmse");
    auto *mr = ctx.engine->allocator();
    const bool allConsumed =
        flatten && args.size() > 2 && (args[2].isChar() || args[2].isString());
    const keel::usize flagStart =
        (args.size() > 2 && !args[2].isEmpty() &&
         (!(args[2].isChar() || args[2].isString()) || allConsumed)) ? 3 : 2;
    const keel::usize dimLen = flatten ? std::max(args[0].numel(),
                                                  args[1].numel())
        : args[0].dims().dim(dim > 0 ? keel::as<keel::usize>(dim) - 1 : 0);
    const ErrorMetricOpts opts = parseErrorMetricOpts(
        args, flagStart, false, args[0], args[1], dimLen, "rmse");
    if (!vecdimSet.empty()) {
        // Partial vecdim — collapse F and A with the same set so the
        // slices stay paired, then reduce along the last dim.
        Value f = vecdim::collapseDims(args[0], vecdimSet, mr);
        Value a = vecdim::collapseDims(args[1], vecdimSet, mr);
        const int lastd = keel::as<int>(f.dims().ndim());
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

void ecdf_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "ecdf");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    keel::String function_mode = keel::String::from("cdf");
    double alpha = 0.05;
    const Value *freq = nullptr;
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        for (auto &c : s) c = keel::to_ascii_lower(c);
        return s;
    };
    for (keel::usize i = 1; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) break;
        const keel::String keyStr = lower(args[i].toString().as_ref());
        const keel::StringRef key = keyStr.as_ref();
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
                            0, 0, "ecdf", "", "numkit:ecdf:censoring_nyi");
        }
        else if (key == "iterationlimit" || key == "tolerance"
                 || key == "icmfrequency" || key == "bounds") {
            // Silently accepted (no-op for non-censored ecdf).
        }
    }
    const bool want_bounds = (nargout > 2);
    auto R = ecdf_full(args[0], freq, function_mode.as_ref(), alpha, want_bounds, mr);
    outs[0] = std::move(R.f);
    if (nargout > 1) outs[1] = std::move(R.x);
    if (nargout > 2) outs[2] = std::move(R.flo);
    if (nargout > 3) outs[3] = std::move(R.fup);
}

void ecdfhist_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "ecdfhist");  // no max: name-value tail in the reference
    int m = 10;
    if (args.size() >= 3 && !args[2].isEmpty())
        m = keel::as<int>(argcheck::roundLength(args[2], 3, "m", "ecdfhist"));
    auto [n, c] = ecdfhist(args[0], args[1], m, ctx.engine->allocator());
    outs[0] = std::move(n);
    if (nargout > 1) outs[1] = std::move(c);
}

// ── partialcorr adapter ──────────────────────────────────────────────

void partialcorr_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "partialcorr");  // no max: name-value tail in the reference
    // Marshalling only — the matrix/float gates, the (X) / (X, Z) /
    // (X, Y, Z) positional split and the Rows/Type/Tail name-value
    // contract live in partialcorr_full (correlation.cpp). A numeric 2nd
    // argument is Z (or Y when a numeric 3rd follows); anything else
    // starts the name-value tail.
    const Value *y = nullptr, *z = nullptr;
    keel::usize posN = 1;
    if (args.size() > 1 && args[1].isNumeric()) {
        z = &args[1];
        posN = 2;
        if (args.size() > 2 && args[2].isNumeric()) {
            y = &args[1];
            z = &args[2];
            posN = 3;
        }
    }
    CorrOut r = partialcorr_full(
        args[0], y, z,
        keel::ArrayRef<Value>(args.data() + posN, args.size() - posN),
        nargout >= 2, ctx.engine->allocator());
    outs[0] = std::move(r.rho);
    if (nargout >= 2) outs[1] = std::move(r.p);
}

// ── corr / detrend adapters ──────────────────────────────────────────

// corr(X[, Y][, NV…]) — marshalling only; the Type/Rows/Tail/Weights
// name-value contract, the zero-weight row prep, the complex gates and
// the tail-aware p-values live in corr_full (correlation.cpp).
void corr_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "corr");  // no max: name-value tail in the reference
    // Distinguish corr(X[, NV…]) from corr(X, Y[, NV…]): a char/string
    // 2nd argument starts the name-value tail (reference ischar dispatch).
    const bool twoArg =
        (args.size() >= 2 && !args[1].isChar() && !args[1].isString());
    const keel::usize nvStart = twoArg ? 2 : 1;
    CorrOut r = corr_full(args[0], twoArg ? &args[1] : nullptr,
                          keel::ArrayRef<Value>(args.data() + nvStart,
                                            args.size() - nvStart),
                          nargout >= 2, ctx.engine->allocator());
    outs[0] = std::move(r.rho);
    if (nargout >= 2) outs[1] = std::move(r.p);
}

// detrend(A, ...) — marshalling only; the full parse+validate+
// compute contract lives in cleaning.cpp's detrend_full (float gate,
// degree/bp/nanflag/Continuous NVs, single + N-D results).
void detrend_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "detrend");  // no max: flag/option tail in the reference
    auto *mr = ctx.engine->allocator();
    outs[0] = detrend_full(args[0],
                           keel::ArrayRef<Value>(args.data() + 1, args.size() - 1), mr);
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
void isoutlier_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isoutlier");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    IsOutlierOut r = isoutlier_full(args[0],
                                    keel::ArrayRef<Value>(args.data() + 1, args.size() - 1),
                                    nargout > 1, mr);
    outs[0] = std::move(r.tf);
    if (nargout > 1) outs[1] = std::move(r.lo);
    if (nargout > 2) outs[2] = std::move(r.hi);
    if (nargout > 3) outs[3] = std::move(r.center);
}

void rmoutliers_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "rmoutliers");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    RmOutliersOut r = rmoutliers_full(args[0],
                                      keel::ArrayRef<Value>(args.data() + 1, args.size() - 1), mr);
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
void fillmissing_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "fillmissing");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    auto callFn = [&ctx](const Value &fh, keel::ArrayRef<Value> fargs) -> Value {
        return ctx.engine->callFunctionHandle(fh, fargs, ctx.env);
    };
    FillMissingOut r = fillmissing_full(args[0],
                                        keel::ArrayRef<Value>(args.data() + 1, args.size() - 1),
                                        nargout > 1, mr, callFn);
    outs[0] = std::move(r.B);
    if (nargout > 1) outs[1] = std::move(r.filled);
}

void rmmissing_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "rmmissing");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    RmMissingOut r = rmmissing_full(args[0],
                                    keel::ArrayRef<Value>(args.data() + 1, args.size() - 1), mr);
    outs[0] = std::move(r.B);
    if (nargout > 1) outs[1] = std::move(r.ind);
}

void standardizeMissing_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "standardizeMissing");
    outs[0] = standardizeMissing_full(args[0], args[1], ctx.engine->allocator());
}

// filloutliers(A, fillmethod[, findmethod][, NV])
//   fillmethod : numeric scalar | "center" | "clip" | "previous" |
//                "next" | "nearest" | "linear"
//   findmethod : "median" (default) | "mean" | "quartiles"
//   NV         : ThresholdFactor (default per-method)
void filloutliers_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "filloutliers");  // no max: name-value tail in the reference
    auto lower = [](keel::StringRef sv) {
        keel::String v = keel::String::from(sv);
        for (char &ch : v)
            ch = keel::to_ascii_lower(ch);
        return v;
    };

    keel::String detect = keel::String::from("median");
    double tf = 3.0;
    bool tf_set = false;
    double loP = 0.0, hiP = 0.0;
    keel::usize i = 2;
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        keel::String s = lower(args[i].toString().as_ref());
        const keel::StringRef sr = s.as_ref();
        // Distinguish findmethod string vs NV name. NV names are known.
        if (sr == "thresholdfactor" || sr == "maxnumoutliers" ||
            sr == "samplepoints"    || sr == "outlierlocations") {
            // fall through — handled by NV loop below.
        } else {
            detect = std::move(s);
            const keel::StringRef detRef = detect.as_ref();
            if (detRef == "median" || detRef == "mean" || detRef == "quartiles") {
                ++i;
            } else if (detRef == "percentiles") {
                ++i;
                if (i >= args.size() || args[i].numel() != 2)
                    throw Error("filloutliers: 'percentiles' requires a "
                                "2-element [lower upper] vector",
                                0, 0, "filloutliers", "",
                                "numkit:filloutliers:percentiles");
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
                            "numkit:filloutliers:findmethod");
            }
        }
    }
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("filloutliers: expected name-value pair",
                        0, 0, "filloutliers", "", "numkit:filloutliers:nv");
        const keel::String rawNm = args[i].toString();
        const keel::String nm = lower(rawNm.as_ref());
        if (nm.as_ref() == "thresholdfactor") {
            tf = args[i + 1].toScalar(); tf_set = true;
        } else {
            throw Error(keel::format("filloutliers: unsupported name-value parameter '{}'",
                                     rawNm.as_ref()),
                        0, 0, "filloutliers", "", "numkit:filloutliers:nv");
        }
        i += 2;
    }
    // MATLAB's per-method default ThresholdFactor.
    if (!tf_set) {
        if (detect.as_ref() == "quartiles") tf = 3.0;     // 1.5·IQR → scaled by 0.5 internally so 3.0 here
        else                                tf = 3.0;
    } else if (detect.as_ref() == "quartiles") {
        // User-set tf for quartiles means "k" in [Q1 - k·IQR, Q3 + k·IQR].
        // Our internal formula uses 0.5·tf·IQR, so multiply by 2.
        tf = 2.0 * tf;
    }
    outs[0] = filloutliers_of(args[0], args[1], detect.as_ref(), tf, loP, hiP,
                              ctx.engine->allocator());
}

// ── range / mad / geomean / harmmean / moment / trimmean adapters ────

void range_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "range");
    int dim = 0;
    keel::Array<int> vecdimSet;
    if (args.size() >= 2)
        vecdim::parseDimSlot(args[0], args[1], "range", dim, vecdimSet);
    auto *mr = ctx.engine->allocator();
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        outs[0] = vecdim::expandReduced(
            range_of(c, keel::as<int>(c.dims().ndim()), mr),
            args[0], vecdimSet, mr);
        return;
    }
    outs[0] = range_of(args[0], dim, mr);
}

void mad_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "mad");
    int flag = 0;
    int dim  = 0;
    keel::Array<int> vecdimSet;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const double fv = argcheck::expectedReal(args[1], 2, "flag", "mad");
        argcheck::expectedFlag01(fv, 2, "flag", "mad");
        flag = keel::as<int>(fv);
    }
    if (args.size() >= 3)
        vecdim::parseDimSlot(args[0], args[2], "mad", dim, vecdimSet);
    auto *mr = ctx.engine->allocator();
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        outs[0] = vecdim::expandReduced(
            mad_of(c, flag, keel::as<int>(c.dims().ndim()), mr),
            args[0], vecdimSet, mr);
        return;
    }
    outs[0] = mad_of(args[0], flag, dim, mr);
}

// Parse a trailing 'omitnan'/'includenan' nanflag from a geomean/harmmean
// arg list. Returns the omit flag and the count of remaining numeric args.
bool parseMeanNanFlag(keel::ArrayRef<Value> args, const char *fn, keel::usize &nargs)
{
    nargs = args.size();
    if (nargs >= 2 && (args[nargs - 1].isChar() || args[nargs - 1].isString())) {
        keel::String f = args[nargs - 1].toString();
        for (char &c : f) c = keel::to_ascii_lower(c);
        if (f.as_ref() == "omitnan")     { --nargs; return true; }
        if (f.as_ref() == "includenan")  { --nargs; return false; }
        if (f.as_ref() == "all" && nargs == 2) return false;  // 'all' is the dim slot, not a nanflag
        throw Error(keel::format("{}: unknown option '{}'", fn, f.as_ref()),
                    0, 0, fn, "", keel::format("numkit:{}:badopt", fn));
    }
    return false;
}

void geomean_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "geomean");  // no max: flag/option tail in the reference
    keel::usize nargs;
    const bool omitnan = parseMeanNanFlag(args, "geomean", nargs);
    int dim = 0;
    keel::Array<int> vecdimSet;
    if (nargs >= 2)
        vecdim::parseDimSlot(args[0], args[1], "geomean", dim, vecdimSet);
    auto *mr = ctx.engine->allocator();
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        outs[0] = vecdim::expandReduced(
            geomean_of(c, keel::as<int>(c.dims().ndim()), omitnan, mr),
            args[0], vecdimSet, mr);
        return;
    }
    outs[0] = geomean_of(args[0], dim, omitnan, mr);
}

void harmmean_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "harmmean");  // no max: flag/option tail in the reference
    keel::usize nargs;
    const bool omitnan = parseMeanNanFlag(args, "harmmean", nargs);
    int dim = 0;
    keel::Array<int> vecdimSet;
    if (nargs >= 2)
        vecdim::parseDimSlot(args[0], args[1], "harmmean", dim, vecdimSet);
    auto *mr = ctx.engine->allocator();
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        outs[0] = vecdim::expandReduced(
            harmmean_of(c, keel::as<int>(c.dims().ndim()), omitnan, mr),
            args[0], vecdimSet, mr);
        return;
    }
    outs[0] = harmmean_of(args[0], dim, omitnan, mr);
}

void moment_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "moment");
    const int order = argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "order", "moment"), 2, "order", "moment");
    int dim = 0;
    keel::Array<int> vecdimSet;
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
                                       ctx.engine->allocator());
        outs[0] = vecdim::expandReduced(
            moment_of(c, order, keel::as<int>(c.dims().ndim()),
                      ctx.engine->allocator()),
            args[0], vecdimSet, ctx.engine->allocator());
        return;
    }
    outs[0] = moment_of(args[0], order, dim, ctx.engine->allocator());
}

void trimmean_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "trimmean");
    const double pct = args[1].toScalar();

    // trimmean(x, percent [, flag] [, dim]). The 3rd arg is EITHER a string
    // flag ('round' default, or 'floor') OR a numeric dim; if a flag is
    // present the dim may follow it. Distinguish by type before toScalar.
    bool useFloor = false;
    int dim = 0;
    keel::Array<int> vecdimSet;
    keel::usize i = 2;
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        keel::String f = args[i].toString();
        for (char &c : f) c = keel::to_ascii_lower(c);
        if (f.as_ref() == "floor")      useFloor = true;
        else if (f.as_ref() == "round") useFloor = false;
        else
            throw Error("trimmean: flag must be 'round' or 'floor'",
                        0, 0, "trimmean", "", "numkit:trimmean:flag");
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
                                       ctx.engine->allocator());
        outs[0] = vecdim::expandReduced(
            trimmean_of(c, pct, keel::as<int>(c.dims().ndim()), useFloor,
                        ctx.engine->allocator()),
            args[0], vecdimSet, ctx.engine->allocator());
        return;
    }
    outs[0] = trimmean_of(args[0], pct, dim, useFloor, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::stats
