// toolboxes/signal/src/descriptive/descriptive_reg.cpp
//
// CallContext register half of descriptive/descriptive.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/format.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/stats/distributions/normal.hpp>     // norminv for corrcoef conf bounds
#include <numkit/stats/distributions/students_t.hpp> // tcdf for corrcoef p-values
#include <numkit/stats/nan_aware/nan_aware.hpp>  // var_reg / std_reg / median_reg dispatch into stats:: when 'omitnan' is given
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include "descriptive/descriptive_detail.hpp"
#include "support/reduce_vecdim.hpp"
#include <numkit/ops/helpers.hpp>
#include <numkit/builtin/detail/datafun/var_reduction.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/value/error.hpp>


namespace numkit::stats {

namespace detail {

// Helper: weighted variance / std on a flat 1-D array.
//   normFlag = 0  â†’ divide by Î£w - 0 (default for scalar w=0; sample)
//   normFlag = 1  â†’ divide by Î£w     (population; ML estimate)
//
// MATLAB's documented denominator for weighted variance is Î£w (the
// "default" for vector weights â€” equivalent to normFlag=1 semantics).
// var(A, W, ...) with vector W therefore implies normFlag=1.
keel::f64 weightedVarFlat(const keel::f64 *x, const keel::f64 *w, keel::usize n,
                       bool sqrtIt, bool omitNan)
{
    if (n == 0) return keel::Limits<keel::f64>::quiet_nan();
    keel::f64 sw = 0.0, sxw = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 xi = x[i], wi = w[i];
        if (omitNan && keel::is_nan(xi)) continue;
        if (wi < 0.0)
            throw Error("var/std: weights must be non-negative",
                        0, 0, "var/std", "", "numkit:varstd:negWeight");
        sw  += wi;
        sxw += wi * xi;
    }
    if (sw == 0.0) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 mean = sxw / sw;
    keel::f64 ss = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 xi = x[i], wi = w[i];
        if (omitNan && keel::is_nan(xi)) continue;
        const keel::f64 d = xi - mean;
        ss += wi * d * d;
    }
    const keel::f64 v = ss / sw;
    return sqrtIt ? keel::sqrt(v) : v;
}

// Helper: scalar-normFlag variance/std on a flat 1-D array.
keel::f64 scalarVarFlat(const keel::f64 *x, keel::usize n, keel::i32 normFlag,
                     bool sqrtIt, bool omitNan)
{
    if (n == 0) return keel::Limits<keel::f64>::quiet_nan();
    keel::f64 s = 0.0;
    keel::usize cnt = 0;
    for (keel::usize i = 0; i < n; ++i) {
        if (omitNan && keel::is_nan(x[i])) continue;
        s += x[i]; ++cnt;
    }
    if (cnt == 0) return keel::Limits<keel::f64>::quiet_nan();
    if (cnt == 1) return (normFlag == 1) ? 0.0
                                         : keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 mean = s / keel::as<keel::f64>(cnt);
    keel::f64 ss = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        if (omitNan && keel::is_nan(x[i])) continue;
        const keel::f64 d = x[i] - mean;
        ss += d * d;
    }
    const keel::f64 denom = (normFlag == 1) ? keel::as<keel::f64>(cnt)
                                         : keel::as<keel::f64>(cnt - 1);
    const keel::f64 v = ss / denom;
    return sqrtIt ? keel::sqrt(v) : v;
}

// Convert a Value to a flat row (used for 'all' / vecdim full-flatten /
// weight-vector inputs). Skips no elements.
keel::Array<keel::f64> flatten(const Value &x)
{
    keel::Array<keel::f64> out(x.numel());
    for (keel::usize i = 0; i < x.numel(); ++i) out[i] = x.elemAsDouble(i);
    return out;
}

// Common var/std driver â€” handles every variant (scalar w + scalar dim,
// 'all', vecdim full-flatten, weight vector).
//
// `sqrtIt = true` makes this `std`.
Value varStdDispatch(keel::ArrayRef<Value> args, bool sqrtIt, const char *fn,
                     keel::Allocator *mr, Value *wOut = nullptr)
{
    bool omitNan = false;
    keel::usize n = stripNanFlag(args, omitNan, fn);

    const Value &x = args[0];
    // Probed R2025b: the first input must be single or double
    // (logical, complex and even char pass â€” integers reject).
    if (isIntegerType(x.type()))
        throw Error(keel::format("{}: invalid data type. First input "
                                 "argument must be single or double (got {}).",
                                 fn, mtypeName(x.type())),
                    0, 0, fn, "", keel::format("numkit:{}:integerClass", fn));

    // Parse args[1] (w) and args[2] (dim/'all'/vecdim) into a normalised
    // shape: scalar normFlag (0 or 1), or weight vector + flatten flag.
    keel::i32 normFlag = 0;
    bool isWeightVec = false;
    const Value *wVec = nullptr;

    if (n >= 2 && !args[1].isEmpty()) {
        if (args[1].isChar() || args[1].isString())
            // var(X,'all') is NOT a form â€” flags only live in the dim
            // position (probed R2025b: unknownFlag).
            throw Error(keel::format("{}: invalid option. Option must be 'omitnan', "
                                     "'includenan', or a weight vector.", fn),
                        0, 0, fn, "",
                        keel::format("numkit:{}:unknownFlag", fn));
        if (args[1].numel() == 1 && args[1].isNumeric()) {
            const keel::f64 wf = argcheck::expectedReal(args[1], 2, "w", fn);
            if (wf != 0.0 && wf != 1.0)
                throw Error(keel::format("{}: W must be a double or "
                                         "single vector of nonnegative weights.", fn),
                            0, 0, fn, "",
                            keel::format("numkit:{}:invalidWgts", fn));
            normFlag = keel::as<keel::i32>(wf);
        } else {
            isWeightVec = true;
            wVec = &args[1];
        }
    }

    keel::i32 dim = 0;
    bool flattenAll = false;
    keel::Array<keel::i32> vecdimSet;
    if (n >= 3 && !args[2].isEmpty()) {
        const Value &a = args[2];
        if (a.isChar() || a.isString()) {
            keel::String s = a.toString();
            s.to_ascii_lower_in_place();
            if (s.as_ref() == "all") flattenAll = true;
            else throw Error(keel::format("{}: unknown dim flag '{}'", fn, s.as_ref()),
                              0, 0, fn, "", keel::format("numkit:{}:dim", fn));
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, fn);
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            keel::Array<keel::i32> set = vecdim::parseDimSet(a, fn);
            if (vecdim::coversAllDims(set, x))
                flattenAll = true;
            else {
                if (isWeightVec)
                    throw Error(keel::format("{}: weight vector along a "
                                             "dimension set is supported for 2-D inputs "
                                             "only (N-D deferred)", fn),
                                0, 0, fn, "", keel::format("numkit:{}:wND", fn));
                vecdimSet = keel::move(set);
            }
        }
    }

    if (wOut) {
        if (isWeightVec) {
            *wOut = *wVec;
        } else {
            keel::usize wLen = 1;
            if (flattenAll) {
                wLen = x.numel();
            } else if (!vecdimSet.empty()) {
                for (keel::i32 d : vecdimSet) {
                    if (d >= 1 && d <= x.dims().ndim())
                        wLen *= x.dims().dim(d - 1);
                }
            } else {
                const keel::i32 effDim = (dim == 0) ? firstNonSingletonDim(x) : dim;
                if (effDim >= 1 && effDim <= x.dims().ndim())
                    wLen = x.dims().dim(effDim - 1);
            }
            Value wv = Value::matrix(1, wLen, ValueType::DOUBLE, mr);
            keel::f64 *wd = wv.doubleDataMut();
            for (keel::usize i = 0; i < wLen; ++i) wd[i] = 1.0;
            *wOut = keel::move(wv);
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
                throw Error(keel::format("{}: weight vector length must "
                                         "match number of elements", fn),
                            0, 0, fn, "", keel::format("numkit:{}:invalidSizeWgts", fn));
            const keel::f64 v = weightedVarFlat(xv.data(), wv.data(),
                                             xv.size(), sqrtIt, omitNan);
            return Value::scalar(v, mr);
        }
        // Matrix (or explicit dim): MATLAB applies the weight vector along
        // the operating dimension, computing one weighted variance per
        // slice. Weight length must equal size(x, dim). 2-D only (N-D
        // weighted reduction is deferred).
        if (x.dims().ndim() > 2)
            throw Error(keel::format("{}: weight vector along a dimension "
                                     "is supported for 2-D inputs only (N-D deferred)", fn),
                        0, 0, fn, "", keel::format("numkit:{}:wND", fn));
        const keel::i32 d = (dim == 0) ? firstNonSingletonDim(x) : dim;
        if (d != 1 && d != 2)
            throw Error(keel::format("{}: dim out of range", fn),
                        0, 0, fn, "", keel::format("numkit:{}:dim", fn));
        const keel::usize r = keel::as<keel::usize>(x.dims().dim(0));
        const keel::usize c = keel::as<keel::usize>(x.dims().dim(1));
        auto wv = flatten(*wVec);
        const keel::usize need = (d == 1) ? r : c;
        if (wv.size() != need)
            throw Error(keel::format("{}: weight vector length must match "
                                     "the length of the operating dimension", fn),
                        0, 0, fn, "", keel::format("numkit:{}:invalidSizeWgts", fn));
        if (d == 1) {
            auto out = Value::matrix(1, c, ValueType::DOUBLE, mr);
            keel::f64 *od = out.doubleDataMut();
            keel::Array<keel::f64> col(r);
            for (keel::usize j = 0; j < c; ++j) {
                for (keel::usize i = 0; i < r; ++i) col[i] = x.elemAsDouble(j * r + i);
                od[j] = weightedVarFlat(col.data(), wv.data(), r, sqrtIt, omitNan);
            }
            return out;
        }
        auto out = Value::matrix(r, 1, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        keel::Array<keel::f64> row(c);
        for (keel::usize i = 0; i < r; ++i) {
            for (keel::usize j = 0; j < c; ++j) row[j] = x.elemAsDouble(j * r + i);
            od[i] = weightedVarFlat(row.data(), wv.data(), c, sqrtIt, omitNan);
        }
        return out;
    }

    // â”€â”€ Partial vecdim: collapse the listed dims and reduce along the
    // last dim of the collapsed view (same kernels as the scalar-dim
    // path, so every flag combination carries over).
    if (!vecdimSet.empty()) {
        Value c = vecdim::collapseDims(x, vecdimSet, mr);
        const keel::i32 lastd = keel::as<keel::i32>(c.dims().ndim());
        if (omitNan) {
            if (x.type() == ValueType::COMPLEX)
                return vecdim::expandReduced(
                    varianceComplex(c, normFlag, lastd, mr, sqrtIt, true),
                    x, vecdimSet, mr);
            Value r = sqrtIt
                        ? ::numkit::stats::nanstdev(c, normFlag, lastd, mr)
                        : ::numkit::stats::nanvar  (c, normFlag, lastd, mr);
            if (x.type() == ValueType::SINGLE)
                r = narrowToSingle(keel::move(r), mr);
            return vecdim::expandReduced(keel::move(r), x, vecdimSet, mr);
        }
        Value r = sqrtIt ? stdev(c, normFlag, lastd, mr)
                         : var(c, normFlag, lastd, mr);
        return vecdim::expandReduced(keel::move(r), x, vecdimSet, mr);
    }

    // â”€â”€ Flatten 'all' / vecdim path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (flattenAll) {
        auto xv = flatten(x);
        const keel::f64 v = scalarVarFlat(xv.data(), xv.size(), normFlag,
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
            r = narrowToSingle(keel::move(r), mr);
        return r;
    }
    return sqrtIt ? stdev(x, normFlag, dim, mr) : var(x, normFlag, dim, mr);
}

void var_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
             CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "var");
    Value wOut;
    outs[0] = varStdDispatch(args, /*sqrtIt=*/false, "var", ctx.engine->allocator(),
                             (nargout >= 2) ? &wOut : nullptr);
    if (nargout >= 2) outs[1] = keel::move(wOut);
}

void std_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
             CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "std");
    Value wOut;
    outs[0] = varStdDispatch(args, /*sqrtIt=*/true, "std", ctx.engine->allocator(),
                             (nargout >= 2) ? &wOut : nullptr);
    if (nargout >= 2) outs[1] = keel::move(wOut);
}

void median_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "median");  // no max: flag/option tail in the reference
    // MATLAB narrows an all-zero-imaginary complex median back to a real double.
    // The scalar / dim / 'all' / omitnan branches each assign outs[0] and return
    // separately, so a scope guard narrows whichever one fires (no-op unless the
    // result is left complex with all-zero imag).
    struct NarrowGuard {
        Value *v;
        keel::Allocator *mr;
        ~NarrowGuard()
        {
            if (v->isComplex())
                *v = numkit::narrowComplex(keel::move(*v), mr);
        }
    } narrowGuard{&outs[0], ctx.engine->allocator()};
    // Probed R2025b: median preserves the LOGICAL class (result is the
    // majority/rounded median re-narrowed to logical); integers already
    // round-and-saturate in the kernel. Runs on every exit path like
    // narrowGuard above.
    struct LogicalGuard {
        Value *v;
        bool active;
        ~LogicalGuard()
        {
            if (!active) return;
            if (v->type() == ValueType::DOUBLE) {
                Value lg = Value::matrix(v->dims().rows(), v->dims().cols(),
                                         ValueType::LOGICAL);
                for (keel::usize i = 0; i < v->numel(); ++i)
                    lg.logicalDataMut()[i] =
                        v->doubleData()[i] != 0.0 ? 1 : 0;
                *v = keel::move(lg);
            }
        }
    } logicalGuard{&outs[0], args[0].type() == ValueType::LOGICAL};
    bool omitNan = false;
    keel::usize n = stripNanFlag(args, omitNan, "median");
    keel::i32 dim = 0;
    bool isAll = false;
    keel::Array<keel::i32> vecdimSet;
    if (n >= 2 && !args[1].isEmpty()) {
        const Value &a = args[1];
        if (a.isChar() || a.isString()) {
            keel::String s = a.toString();
            s.to_ascii_lower_in_place();
            if (s.as_ref() == "all") isAll = true;
            else throw Error(keel::format("median: unknown option '{}'", s.as_ref()),
                              0, 0, "median", "", "numkit:median:unknownOption");
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, "median");
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            keel::Array<keel::i32> set = vecdim::parseDimSet(a, "median");
            if (vecdim::coversAllDims(set, args[0]))
                isAll = true;
            else
                vecdimSet = keel::move(set);
        }
    }
    if (!vecdimSet.empty()) {
        // Partial vecdim â€” same kernels as the scalar-dim path (the
        // NarrowGuard above still narrows an all-zero-imaginary result).
        Value c = vecdim::collapseDims(args[0], vecdimSet,
                                       ctx.engine->allocator());
        const keel::i32 lastd = keel::as<keel::i32>(c.dims().ndim());
        if (omitNan) {
            rejectComplexOmitNan(args[0], "median");
            Value r = ::numkit::stats::nanmedian(c, lastd, ctx.engine->allocator());
            if (args[0].type() == ValueType::SINGLE)
                r = narrowToSingle(keel::move(r), ctx.engine->allocator());
            outs[0] = vecdim::expandReduced(keel::move(r), args[0], vecdimSet,
                                            ctx.engine->allocator());
            return;
        }
        outs[0] = vecdim::expandReduced(median(c, lastd, ctx.engine->allocator()),
                                        args[0], vecdimSet,
                                        ctx.engine->allocator());
        return;
    }
    if (isAll) {
        // 'all' â†’ flatten + median over all elements (skipping NaN if omitnan).
        if (args[0].type() == ValueType::COMPLEX) {
            const keel::usize total = args[0].numel();
            keel::ArenaTemp scratch = keel::get_scratch(ctx.engine->allocator());
            auto buf = keel::Array<keel::c64>(total, scratch.allocator());
            const keel::c64 *src = args[0].complexData();
            keel::usize k = 0;
            for (keel::usize i = 0; i < total; ++i) {
                if (omitNan && isComplexNaNStats(src[i])) continue;
                buf[k++] = src[i];
            }
            outs[0] = Value::complexScalar(complexMedianFromSlice(buf.data(), k),
                                           ctx.engine->allocator());
            return;
        }
        const keel::usize total = args[0].numel();
        keel::ArenaTemp scratch = keel::get_scratch(ctx.engine->allocator());
        auto buf = keel::Array<keel::f64>(total, scratch.allocator());
        const bool fastDouble = (args[0].type() == ValueType::DOUBLE);
        if (fastDouble)
            keel::copy(args[0].doubleData(), args[0].doubleData() + total, buf.data());
        else
            for (keel::usize i = 0; i < total; ++i) buf[i] = args[0].elemAsDouble(i);
        keel::usize k = total;
        if (omitNan) k = compactNonNan(buf.data(), total);
        keel::f64 v = medianFromSlice(buf.data(), k);
        Value r = Value::scalar(v, ctx.engine->allocator());
        if (args[0].type() == ValueType::SINGLE)
            r = narrowToSingle(keel::move(r), ctx.engine->allocator());
        outs[0] = keel::move(r);
        return;
    }
    if (omitNan) {
        rejectComplexOmitNan(args[0], "median");
        Value r = ::numkit::stats::nanmedian(args[0], dim, ctx.engine->allocator());
        if (args[0].type() == ValueType::SINGLE)
            r = narrowToSingle(keel::move(r), ctx.engine->allocator());
        outs[0] = keel::move(r);
        return;
    }
    outs[0] = median(args[0], dim, ctx.engine->allocator());
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
    keel::i32 dim = 0;
    bool flatten = false;
    QMethod method = QMethod::Midpoint;
};

QArgs parseQArgs(keel::ArrayRef<Value> args, keel::usize start, const Value &x,
                 const char *fn)
{
    QArgs q;
    keel::usize i = start;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()
        && !args[i].isEmpty()) {
        if (args[i].numel() == 1) {
            q.dim = argcheck::dimArg(args[i], "median");
        } else {
            // Vecdim: covers-all-dims â†’ flatten; otherwise error.
            keel::Array<keel::i32> dims;
            for (keel::usize j = 0; j < args[i].numel(); ++j)
                dims.push_back(argcheck::dimArg(args[i].elemAsDouble(j), "median"));
            const keel::i32 rank = x.dims().is3D() ? 3
                              : (x.dims().isVector() || x.isScalar() ? 1 : 2);
            keel::Array<keel::u8> seen(keel::as<keel::usize>(rank) + 1, 0);
            for (keel::i32 d : dims) {
                if (d < 1 || d > rank)
                    throw Error(keel::format("{}: vecdim entries out of range", fn),
                                0, 0, fn, "", keel::format("numkit:{}:vecdim", fn));
                seen[keel::as<keel::usize>(d)] = 1;
            }
            bool allCovered = true;
            for (keel::i32 d = 1; d <= rank; ++d) if (!seen[keel::as<keel::usize>(d)]) allCovered = false;
            if (!allCovered)
                throw Error(keel::format("{}: partial vecdim reduction is "
                                         "not yet supported in nk (only full-flatten "
                                         "vecdim like [1 2] or 'all')", fn),
                            0, 0, fn, "", keel::format("numkit:{}:vecdim", fn));
            q.flatten = true;
        }
        ++i;
    } else if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        keel::String s = args[i].toString();
        s.to_ascii_lower_in_place();
        if (s.as_ref() == "all") {
            q.flatten = true;
            ++i;
        }
        // else: leave for the Name-Value loop below ("Method=...")
    }

    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error(keel::format("{}: expected Name-Value pair", fn),
                        0, 0, fn, "", keel::format("numkit:{}:nv", fn));
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        if (name.as_ref() == "method") {
            if (!args[i + 1].isChar() && !args[i + 1].isString())
                throw Error(keel::format("{}: Method must be a string", fn),
                            0, 0, fn, "", keel::format("numkit:{}:method", fn));
            keel::String m = args[i + 1].toString();
            m.to_ascii_lower_in_place();
            // MATLAB's documented values are 'exact' (default) and
            // 'approximate'. 'exact' is the linear-interpolation order-
            // statistic method, which is nk's Midpoint. The
            // midpoint/inclusive/exclusive names are kept for compatibility.
            if      (m.as_ref() == "exact")       q.method = QMethod::Midpoint;
            else if (m.as_ref() == "midpoint")    q.method = QMethod::Midpoint;
            else if (m.as_ref() == "inclusive")   q.method = QMethod::Inclusive;
            else if (m.as_ref() == "exclusive")   q.method = QMethod::Exclusive;
            else if (m.as_ref() == "approximate") q.method = QMethod::Approximate;
            else
                throw Error(keel::format("{}: Method must be 'exact' or "
                                         "'approximate'", fn),
                            0, 0, fn, "", keel::format("numkit:{}:method", fn));
        } else {
            throw Error(keel::format("{}: unknown Name-Value '{}'", fn, name.as_ref()),
                        0, 0, fn, "", keel::format("numkit:{}:nv", fn));
        }
        i += 2;
    }
    return q;
}

void quantile_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "quantile");  // no max: name-value tail in the reference
    auto q = parseQArgs(args, 2, args[0], "quantile");
    outs[0] = quantileWithOpts(args[0], args[1], q.dim, q.flatten, q.method, 1.0, "quantile", ctx.engine->allocator());
}

void prctile_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "prctile");  // no max: name-value tail in the reference
    auto q = parseQArgs(args, 2, args[0], "prctile");
    outs[0] = quantileWithOpts(args[0], args[1], q.dim, q.flatten, q.method, 0.01, "prctile", ctx.engine->allocator());
}

// 3rd mode output C: a cell array of the modal values. Each cell holds a
// column vector of the values that attain the modal frequency in that
// slice, sorted ascending; MATLAB ignores NaN. Supported for a real DOUBLE
// vector, 2-D matrix, or 'all'-flattened input.
Value computeModeCell(const Value &x, keel::i32 dim, bool flatten,
                      keel::Allocator *mr)
{
    if (x.type() != ValueType::DOUBLE || x.dims().is3D() || x.dims().ndim() > 2)
        throw Error("mode: the 3rd output C is currently supported only for a "
                    "real double vector or 2-D matrix input",
                    0, 0, "mode", "", "numkit:mode:cellNd");
    auto tiedCol = [&](const keel::f64 *s, keel::usize stride, keel::usize n) -> Value {
        keel::Array<keel::f64> v;
        v.reserve(n);
        for (keel::usize k = 0; k < n; ++k) {
            const keel::f64 d = s[k * stride];
            if (!keel::is_nan(d)) v.push_back(d);
        }
        keel::sort(v.begin(), v.end());
        const keel::usize N = v.size();
        keel::usize maxc = 0;
        for (keel::usize i = 0; i < N;) {
            keel::usize j = i;
            while (j < N && v[j] == v[i]) ++j;
            if (j - i > maxc) maxc = j - i;
            i = j;
        }
        keel::Array<keel::f64> tied;
        for (keel::usize i = 0; i < N;) {
            keel::usize j = i;
            while (j < N && v[j] == v[i]) ++j;
            if (j - i == maxc) tied.push_back(v[i]);
            i = j;
        }
        Value col = Value::matrix(tied.size(), 1, ValueType::DOUBLE, mr);
        for (keel::usize k = 0; k < tied.size(); ++k) col.doubleDataMut()[k] = tied[k];
        return col;
    };
    const keel::f64 *data = x.numel() ? x.doubleData() : nullptr;
    if (flatten || x.isScalar() || x.dims().isVector()) {
        Value cell = Value::cell(1, 1, mr);
        cell.cellAt(0) = tiedCol(data, 1, x.numel());
        return cell;
    }
    const keel::usize R = x.dims().rows(), Cn = x.dims().cols();
    const keel::i32 rdim = (dim == 0) ? 1 : dim;
    if (rdim == 1) {
        Value cell = Value::cell(1, Cn, mr);
        for (keel::usize c = 0; c < Cn; ++c) cell.cellAt(c) = tiedCol(data + c * R, 1, R);
        return cell;
    }
    Value cell = Value::cell(R, 1, mr);
    for (keel::usize r = 0; r < R; ++r) cell.cellAt(r) = tiedCol(data + r, R, Cn);
    return cell;
}

void mode_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "mode");
    keel::i32 dim = 0;
    bool flatten = false;
    keel::Array<keel::i32> vecdimSet;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &a = args[1];
        if (a.isChar() || a.isString()) {
            keel::String s = a.toString();
            s.to_ascii_lower_in_place();
            if (s.as_ref() == "all") flatten = true;
            else throw Error(keel::format("mode: unknown flag '{}'", s.as_ref()),
                              0, 0, "mode", "", "numkit:mode:badFlag");
        } else if (a.numel() == 1) {
            dim = argcheck::dimArg(a, "mode");
        } else {
            // vecdim: full coverage reduces like 'all'; a partial set
            // collapses the listed dims into one trailing dim below.
            keel::Array<keel::i32> set = vecdim::parseDimSet(a, "mode");
            if (vecdim::coversAllDims(set, args[0]))
                flatten = true;
            else
                vecdimSet = keel::move(set);
        }
    }
    auto *mr = ctx.engine->allocator();
    if (!vecdimSet.empty()) {
        // Partial vecdim â€” same kernels as the scalar-dim path; the
        // 3rd output's cell array reshapes through the same mechanism
        // (permute/reshape support cell values).
        Value c = vecdim::collapseDims(args[0], vecdimSet, mr);
        const keel::i32 lastd = keel::as<keel::i32>(c.dims().ndim());
        auto [v, cnt] = mode(c, lastd, mr);
        outs[0] = vecdim::expandReduced(keel::move(v), args[0], vecdimSet, mr);
        if (nargout > 1)
            outs[1] = vecdim::expandReduced(keel::move(cnt), args[0], vecdimSet, mr);
        if (nargout > 2)
            outs[2] = vecdim::expandReduced(computeModeCell(c, lastd, false, mr),
                                            args[0], vecdimSet, mr);
        return;
    }
    if (flatten) {
        Value flat = Value::matrix(1, args[0].numel(), ValueType::DOUBLE, mr);
        if (args[0].numel() > 0) {
            const keel::f64 *src = args[0].doubleData();
            keel::copy(src, src + args[0].numel(), flat.doubleDataMut());
        }
        auto [v, c] = mode(flat, 2, mr);
        outs[0] = keel::move(v);
        if (nargout > 1) outs[1] = keel::move(c);
        if (nargout > 2) outs[2] = computeModeCell(args[0], 0, true, mr);
        return;
    }
    auto [v, c] = mode(args[0], dim, mr);
    outs[0] = keel::move(v);
    if (nargout > 1)
        outs[1] = keel::move(c);
    if (nargout > 2)
        outs[2] = computeModeCell(args[0], dim, false, mr);
}

// skewness_reg / kurtosis_reg moved to toolboxes/stats/src/moments/moments.cpp

void cov_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
             CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "cov");
    keel::Allocator *mr = ctx.engine->allocator();

    // A trailing NaN-policy string ('includenan'|'omitrows'|'partialrows')
    // may follow any signature. Strip it; the remaining args are numeric.
    CovNan nanMode = CovNan::Include;
    keel::usize nargs = args.size();
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
                const keel::f64 v = args[1].toScalar();
                if (v == 0.0 || v == 1.0) {
                    outs[0] = covNanAware(args[0], keel::as<keel::i32>(v),
                                          nanMode, mr);
                    return;
                }
            }
            outs[0] = covNanAware(assembleXY(args[0], args[1], mr), 0,
                                  nanMode, mr);
            return;
        }
        const keel::f64 wv = argcheck::expectedReal(args[2], 3, "w", "cov");
        argcheck::expectedFlag01(wv, 3, "w", "cov");
        outs[0] = covNanAware(assembleXY(args[0], args[1], mr),
                              keel::as<keel::i32>(wv), nanMode, mr);
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
            const keel::f64 v = args[1].toScalar();
            if (v == 0.0 || v == 1.0) {
                outs[0] = cov(args[0], keel::as<keel::i32>(v), mr);
                return;
            }
        }
        outs[0] = cov(args[0], args[1], 0, mr);
        return;
    }
    // 3-arg form: cov(x, y, normFlag).
    const keel::f64 wv = argcheck::expectedReal(args[2], 3, "w", "cov");
    argcheck::expectedFlag01(wv, 3, "w", "cov");
    outs[0] = cov(args[0], args[1], keel::as<keel::i32>(wv), mr);
}

// Widen a single matrix to double (stats kernels are double-based; the
// result narrows back â€” parity predicates cover the fp difference).
static Value widenSingleToDouble(const Value &v, keel::Allocator *mr)
{
    Value d = Value::matrix(v.dims().rows(), v.dims().cols(), ValueType::DOUBLE, mr);
    const keel::f32 *sv = v.singleData();
    keel::f64 *dv = d.doubleDataMut();
    for (keel::usize i = 0; i < v.numel(); ++i)
        dv[i] = keel::as<keel::f64>(sv[i]);
    return d;
}
void corrcoef_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "corrcoef");  // no max: name-value tail in the reference
    keel::Allocator *mr = ctx.engine->allocator();
    // Probed R2025b: integer inputs reject (mixedClasses); single
    // contagion; complex input is fine for R alone but rejects with a
    // ComplexInputs-class error once p-values are requested.
    for (keel::usize i = 0; i < args.size(); ++i) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) break;  // NV tail begins
        if (isIntegerType(a.type()))
            throw Error("corrcoef: integers can only be combined with "
                        "integers of the same class, or scalar doubles.",
                        0, 0, "corrcoef", "", "numkit:corrcoef:mixedClasses");
    }

    const bool twoArg =
        (args.size() >= 2 && !args[1].isChar() && !args[1].isString());
    const keel::usize nvStart = twoArg ? 2 : 1;
    // Single inputs widen to double for the kernels; the result narrows
    // back to single (the reference computes single end-to-end â€” parity
    // uses closeness predicates).
    const CorrcoefRows rows = parseCorrcoefRows(args, nvStart);
    const keel::f64 alpha = parseCorrcoefAlpha(args, nvStart);
    const bool hasComplex = args[0].isComplex() || (twoArg && args[1].isComplex());
    if (hasComplex && nargout >= 2) {
        throw Error("corrcoef: cannot compute p-values for complex inputs.",
                    0, 0, "corrcoef", "", "numkit:corrcoef:ComplexInputs");
    }

    if (rows == CorrcoefRows::All) {
        keel::f64 n;
        if (!twoArg) {
            const Value &x = args[0];
            n = (x.dims().isVector() || x.isScalar())
                    ? keel::as<keel::f64>(x.numel())          // single variable
                    : keel::as<keel::f64>(x.dims().rows());   // n obs × p vars
            Value xd = (!x.isComplex() && x.type() == ValueType::SINGLE)
                ? widenSingleToDouble(x, mr) : Value(x);
            Value R = corrcoef(xd, mr);
            if (nargout >= 2) {
                outs[1] = corrcoefPValues(R, n, mr);
            }
            outs[0] = keel::move(R);
            if (!outs[0].isComplex() && x.type() == ValueType::SINGLE)
                outs[0] = narrowToSingle(keel::move(outs[0]), mr);
        } else {
            n = keel::as<keel::f64>(args[0].numel());         // corrcoef(x, y)
            Value x0 = (!args[0].isComplex() && args[0].type() == ValueType::SINGLE)
                ? widenSingleToDouble(args[0], mr) : Value(args[0]);
            Value x1 = (!args[1].isComplex() && args[1].type() == ValueType::SINGLE)
                ? widenSingleToDouble(args[1], mr) : Value(args[1]);
            Value R = corrcoef(x0, x1, mr);
            if (nargout >= 2) {
                outs[1] = corrcoefPValues(R, n, mr);
            }
            outs[0] = keel::move(R);
            if (!outs[0].isComplex() &&
                (args[0].type() == ValueType::SINGLE || args[1].type() == ValueType::SINGLE))
                outs[0] = narrowToSingle(keel::move(outs[0]), mr);
        }
        if (nargout >= 3) {
            Value RL, RU;
            corrcoefConfBounds(outs[0], n, alpha, RL, RU, mr);
            outs[2] = keel::move(RL);
            if (nargout >= 4) outs[3] = keel::move(RU);
        }
        return;
    }

    // NaN-aware paths. Assemble the working matrix ([x(:) y(:)] for the
    // two-vector form) and apply listwise/pairwise deletion.
    const Value M = twoArg ? assembleXY(args[0], args[1], mr) : args[0];
    const bool isVec = M.dims().isVector() || M.isScalar();

    if (rows == CorrcoefRows::Pairwise) {
        if (isVec) {
            const keel::usize nValid = countCompleteRows(M);
            Value rMat = Value::matrix(1, 1, ValueType::DOUBLE, mr);
            rMat.doubleDataMut()[0] = (nValid <= 1) ? keel::kNaN : 1.0;
            outs[0] = rMat;
            if (nargout >= 2) outs[1] = rMat;
            if (nargout >= 3) {
                Value RL, RU;
                corrcoefConfBounds(outs[0], keel::as<keel::f64>(nValid), alpha, RL, RU, mr);
                outs[2] = keel::move(RL);
                if (nargout >= 4) outs[3] = keel::move(RU);
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
                outs[2] = keel::move(RL);
                if (nargout >= 4) outs[3] = keel::move(RU);
            }
        }
        return;
    }

    // Complete (listwise deletion).
    if (isVec) {
        const keel::usize nValid = countCompleteRows(M);
        Value rMat = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        rMat.doubleDataMut()[0] = (nValid <= 1) ? keel::kNaN : 1.0;
        outs[0] = rMat;
        if (nargout >= 2) outs[1] = rMat;
        if (nargout >= 3) {
            Value RL, RU;
            corrcoefConfBounds(outs[0], keel::as<keel::f64>(nValid), alpha, RL, RU, mr);
            outs[2] = keel::move(RL);
            if (nargout >= 4) outs[3] = keel::move(RU);
        }
        return;
    }
    outs[0] = corrcoefFromCov(covNanAware(M, 0, CovNan::Omitrows, mr), mr);
    if (nargout >= 2) {
        const keel::f64 n = keel::as<keel::f64>(countCompleteRows(M));
        outs[1] = corrcoefPValues(outs[0], n, mr);
        if (nargout >= 3) {
            Value RL, RU;
            corrcoefConfBounds(outs[0], n, alpha, RL, RU, mr);
            outs[2] = keel::move(RL);
            if (nargout >= 4) outs[3] = keel::move(RU);
        }
    }
}

// nan*_reg adapters all moved to toolboxes/stats/src/nan_aware/nan_aware.cpp.

} // namespace detail

} // namespace numkit::stats
