// toolboxes/stats/src/resample/resample_reg.cpp
//
// CallContext register half of resample/resample.cpp (Phase 2b split): the
// callback-driven adapters (bootstrp/bootci/crossval/datasample/jackknife/
// randsample/combnk) + BootstrpCallbackBuiltin + registerBootstrpCallbackBuiltin.
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/stats/resample/resample.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "resample/resample_detail.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <keel/random.h>
#include <numeric>

namespace numkit::stats {

namespace detail {

void randsample_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "randsample");  // no max: positional tail in the reference
    // Marshalling only — the count/population forms, REPLACE and the
    // positional weights contract live in randsample_full.
    outs[0] = randsample_full(ctx.engine->rng(), args,
                              ctx.engine->allocator());
}

void datasample_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "datasample");  // no max: name-value tail in the reference
    // Marshalling only — K/dim/Replace/Weights contracts and the
    // per-dim gather live in datasample_full.
    outs[0] = datasample_full(ctx.engine->rng(), args,
                              ctx.engine->allocator());
}

// Helper: draw N indices in [0, N-1] with replacement, write into idx_out.
static void drawBootstrapIndices(::numkit::ops::RngContext &rng, keel::i32 N, keel::i32 *idx_out)
{
    for (keel::i32 i = 0; i < N; ++i)
        idx_out[i] = keel::as<keel::i32>(keel::random::uniform_i64(rng, 0, N - 1));
}

// Resample row indices of X into a same-shape Value.
static Value resampleRows(const Value &X, const keel::i32 *idx, keel::i32 N, keel::Allocator *mr)
{
    if (X.dims().ndim() <= 1 || X.dims().dim(1) == 1) {
        // Vector input: return same-orientation vector of N samples.
        const bool col = X.dims().ndim() >= 2 && X.dims().dim(1) == 1;
        auto out = col
            ? Value::matrix(keel::as<keel::usize>(N), 1, ValueType::DOUBLE, mr)
            : Value::matrix(1, keel::as<keel::usize>(N), ValueType::DOUBLE, mr);
        const keel::f64 *xd = X.doubleData();
        keel::f64 *od = out.doubleDataMut();
        for (keel::i32 i = 0; i < N; ++i) od[i] = xd[idx[i]];
        return out;
    }
    // Matrix input: resample rows.
    const keel::usize m = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize cols = keel::as<keel::usize>(X.dims().dim(1));
    auto out = Value::matrix(keel::as<keel::usize>(N), cols, ValueType::DOUBLE, mr);
    const keel::f64 *xd = X.doubleData();
    keel::f64 *od = out.doubleDataMut();
    for (keel::i32 i = 0; i < N; ++i)
        for (keel::usize j = 0; j < cols; ++j)
            od[i + j * N] = xd[idx[i] + j * m];
    return out;
}

void bootstrp_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "bootstrp");  // no max: name-value tail in the reference
    if (!args[1].isFuncHandle())
        throw Error("bootstrp: 2nd argument must be a function handle",
                    0, 0, "bootstrp", "", "numkit:bootstrp:notFuncHandle");
    const keel::i32 nboot = (keel::i32)args[0].toScalar();
    if (nboot < 1)
        throw Error("bootstrp: nboot must be >= 1",
                    0, 0, "bootstrp", "", "numkit:bootstrp:badN");
    auto *mr = ctx.engine->allocator();
    const Value &X = args[2];
    const keel::i32 N = keel::as<keel::i32>(X.dims().dim(0));
    if (N == 0)
        throw Error("bootstrp: empty data", 0, 0, "bootstrp", "", "numkit:bootstrp:empty");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> idx(N, scratch.allocator());

    // First call to determine output dimensionality of fn(sample).
    drawBootstrapIndices(ctx.engine->rng(), N, idx.data());
    auto sample0 = resampleRows(X, idx.data(), N, mr);
    Value callArgs0[1] = { sample0 };
    auto stat0 = ctx.engine->callFunctionHandle(
        args[1], keel::ArrayRef<Value>(callArgs0, 1), ctx.env);
    const keel::usize K = stat0.numel();
    if (K == 0)
        throw Error("bootstrp: bootfun returned empty", 0, 0, "bootstrp", "",
                    "numkit:bootstrp:emptyStat");

    // Output is nboot × K (each row = one bootstrap statistic).
    auto out = Value::matrix(keel::as<keel::usize>(nboot), K,
                             ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    // Row 0: stat0.
    for (keel::usize j = 0; j < K; ++j)
        od[0 + j * nboot] = stat0.elemAsDouble(j);

    // Rows 1..nboot-1.
    for (keel::i32 b = 1; b < nboot; ++b) {
        drawBootstrapIndices(ctx.engine->rng(), N, idx.data());
        auto sample = resampleRows(X, idx.data(), N, mr);
        Value callArgs[1] = { sample };
        auto stat = ctx.engine->callFunctionHandle(
            args[1], keel::ArrayRef<Value>(callArgs, 1), ctx.env);
        if (stat.numel() != K)
            throw Error("bootstrp: bootfun returned varying-size output",
                        0, 0, "bootstrp", "", "numkit:bootstrp:varyingStat");
        for (keel::usize j = 0; j < K; ++j)
            od[b + j * nboot] = stat.elemAsDouble(j);
    }
    outs[0] = std::move(out);
}

// crossval(predfun, X, Y[, 'kfold', K]) — k-fold cross-validation.
// predfun(Xtrain, Ytrain, Xtest, Ytest) returns a scalar (typically
// MSE or accuracy). Default K = 10. Returns K × 1 column of fold values.
//
// Fold split: contiguous blocks of rows in original order. Random
// shuffling deferred (MATLAB has 'cvpartition' for that).
void crossval_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "crossval");  // no max: name-value tail in the reference
    if (!args[0].isFuncHandle())
        throw Error("crossval: 1st argument must be a function handle "
                    "predfun(Xtrain, Ytrain, Xtest, Ytest)",
                    0, 0, "crossval", "", "numkit:crossval:notFuncHandle");

    auto *mr = ctx.engine->allocator();
    const Value &X = args[1];
    const Value &Y = args[2];

    const keel::usize m = X.dims().dim(0);
    if (Y.dims().dim(0) != m)
        throw Error("crossval: X and Y must have the same number of rows",
                    0, 0, "crossval", "", "numkit:crossval:dimMismatch");
    if (m < 2)
        throw Error("crossval: need at least 2 observations",
                    0, 0, "crossval", "", "numkit:crossval:tooSmall");

    keel::i32 K = 10;
    for (keel::usize i = 3; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("crossval: option name must be a string",
                        0, 0, "crossval", "", "numkit:crossval:badOption");
        const auto opt = args[i].toString();
        if (opt.equals_ignore_ascii_case("kfold"))
            K = argcheck::expectedInteger(
                argcheck::expectedReal(args[i + 1], keel::i32(i) + 2, "KFold", "crossval"),
                keel::i32(i) + 2, "KFold", "crossval");
        else
            throw Error(keel::format("crossval: unknown option '{}' (only "
                                     "'kfold' supported in this revision)", opt),
                        0, 0, "crossval", "", "numkit:crossval:badOption");
    }
    if (K < 2 || K > keel::as<keel::i32>(m))
        K = keel::as<keel::i32>(std::min(m, keel::usize{10}));

    auto out = Value::matrix(keel::as<keel::usize>(K), 1,
                             ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    const keel::usize pX = (X.dims().ndim() >= 2)
        ? keel::as<keel::usize>(X.dims().dim(1)) : 1;
    const keel::usize pY = (Y.dims().ndim() >= 2)
        ? keel::as<keel::usize>(Y.dims().dim(1)) : 1;
    const keel::f64 *Xd = X.doubleData();
    const keel::f64 *Yd = Y.doubleData();

    auto sliceRows = [&](const keel::f64 *src, keel::usize rows_src,
                         keel::usize cols, keel::ArrayRef<keel::usize> rows) {
        Value v = Value::matrix(rows.size(), cols, ValueType::DOUBLE, mr);
        keel::f64 *vd = v.doubleDataMut();
        const keel::usize nr = rows.size();
        for (keel::usize j = 0; j < cols; ++j)
            for (keel::usize i = 0; i < nr; ++i)
                vd[i + j * nr] = src[rows[i] + j * rows_src];
        return v;
    };

    for (keel::i32 k = 0; k < K; ++k) {
        const keel::usize lo = (keel::as<keel::usize>(k) * m) / keel::as<keel::usize>(K);
        const keel::usize hi = (keel::as<keel::usize>(k + 1) * m) / keel::as<keel::usize>(K);
        keel::Array<keel::usize> testRows, trainRows;
        testRows.reserve(hi - lo);
        trainRows.reserve(m - (hi - lo));
        for (keel::usize i = 0; i < m; ++i) {
            if (i >= lo && i < hi) testRows.push_back(i);
            else                   trainRows.push_back(i);
        }
        Value Xtr = sliceRows(Xd, m, pX, trainRows.as_ref());
        Value Ytr = sliceRows(Yd, m, pY, trainRows.as_ref());
        Value Xte = sliceRows(Xd, m, pX, testRows.as_ref());
        Value Yte = sliceRows(Yd, m, pY, testRows.as_ref());
        Value callArgs[4] = { Xtr, Ytr, Xte, Yte };
        Value res = ctx.engine->callFunctionHandle(
            args[0], keel::ArrayRef<Value>(callArgs, 4), ctx.env);
        if (res.numel() != 1)
            throw Error("crossval: predfun must return a scalar",
                        0, 0, "crossval", "", "numkit:crossval:badPredfun");
        od[k] = res.toScalar();
    }
    outs[0] = std::move(out);
}

// bootci(nboot, bootfun, X[, alpha]) — percentile bootstrap CI.
// Returns 2×K matrix: row 1 = lower bound, row 2 = upper bound.
void bootci_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "bootci");  // no max: name-value tail in the reference
    if (!args[1].isFuncHandle())
        throw Error("bootci: 2nd argument must be a function handle",
                    0, 0, "bootci", "", "numkit:bootci:notFuncHandle");
    const keel::i32 nboot = (keel::i32)args[0].toScalar();
    if (nboot < 10)
        throw Error("bootci: nboot must be >= 10 for meaningful CI",
                    0, 0, "bootci", "", "numkit:bootci:badN");
    keel::f64 alpha = 0.05;
    if (args.size() >= 4 && !args[3].isEmpty()) {
        alpha = args[3].toScalar();
        if (alpha <= 0.0 || alpha >= 1.0)
            throw Error("bootci: alpha must be in (0, 1)",
                        0, 0, "bootci", "", "numkit:bootci:alpha");
    }
    auto *mr = ctx.engine->allocator();

    // Reuse bootstrp to get the nboot × K matrix.
    Value pseudo_args[3] = { args[0], args[1], args[2] };
    Value boot_stats;
    {
        Value local_outs[1];
        bootstrp_reg(keel::ArrayRef<Value>(pseudo_args, 3), 1,
                     keel::MutArrayRef<Value>(local_outs, 1), ctx);
        boot_stats = std::move(local_outs[0]);
    }

    const keel::usize K = keel::as<keel::usize>(boot_stats.dims().dim(1));
    const keel::f64 *bd = boot_stats.doubleData();

    auto ci = Value::matrix(2, K, ValueType::DOUBLE, mr);
    keel::f64 *cd = ci.doubleDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> col(nboot, scratch.allocator());
    for (keel::usize j = 0; j < K; ++j) {
        for (keel::i32 i = 0; i < nboot; ++i) col[i] = bd[i + j * nboot];
        std::sort(col.begin(), col.end());
        // Linear-interpolation percentile (matches MATLAB prctile default).
        auto qFn = [&](keel::f64 q) {
            const keel::f64 pos = q * (keel::as<keel::f64>(nboot) - 1.0);
            const keel::usize lo = keel::as<keel::usize>(std::floor(pos));
            const keel::usize hi = std::min(keel::as<keel::usize>(lo + 1),
                                              keel::as<keel::usize>(nboot - 1));
            const keel::f64 frac = pos - keel::as<keel::f64>(lo);
            return col[lo] * (1.0 - frac) + col[hi] * frac;
        };
        cd[0 + j * 2] = qFn(alpha / 2.0);
        cd[1 + j * 2] = qFn(1.0 - alpha / 2.0);
    }
    outs[0] = std::move(ci);
}

// jackknife(jackfun, X): leave-one-out resampling. Row i of the n x K output is
// jackfun applied to X with the i-th observation deleted. Mirrors bootstrp_reg
// (inline, via callFunctionHandle) rather than the old C++ stub. Observations
// are rows of a matrix, or elements of a vector (either orientation).
void jackknife_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "jackknife");  // no max: name-value tail in the reference
    if (!args[0].isFuncHandle())
        throw Error("jackknife: 1st argument must be a function handle",
                    0, 0, "jackknife", "", "numkit:jackknife:notFuncHandle");
    auto *mr = ctx.engine->allocator();
    const Value &X = args[1];
    const auto &d = X.dims();
    const bool isVec = (d.ndim() <= 1) || (d.dim(0) == 1) || (d.dim(1) == 1);
    const keel::i32 n = isVec ? keel::as<keel::i32>(X.numel()) : keel::as<keel::i32>(d.dim(0));
    if (n < 1)
        throw Error("jackknife: empty data", 0, 0, "jackknife", "",
                    "numkit:jackknife:empty");

    // Observations are rows; treat a vector (either orientation) as a column so
    // resampleRows selects one element per observation (not one row of many).
    const Value Xeff = isVec ? X.reshape(keel::as<keel::usize>(n), 1, mr) : X;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> idx(n > 1 ? n - 1 : 0, scratch.allocator());
    auto leaveOneOut = [&](keel::i32 skip) {
        keel::i32 w = 0;
        for (keel::i32 i = 0; i < n; ++i)
            if (i != skip) idx[w++] = i;
        return resampleRows(Xeff, idx.data(), n - 1, mr);
    };

    // First leave-one-out replicate fixes the statistic width K.
    Value sample0 = leaveOneOut(0);
    Value callArgs0[1] = { sample0 };
    auto stat0 = ctx.engine->callFunctionHandle(
        args[0], keel::ArrayRef<Value>(callArgs0, 1), ctx.env);
    const keel::usize K = stat0.numel();
    if (K == 0)
        throw Error("jackknife: jackfun returned empty", 0, 0, "jackknife", "",
                    "numkit:jackknife:emptyStat");

    auto out = Value::matrix(keel::as<keel::usize>(n), K, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize j = 0; j < K; ++j)
        od[0 + j * keel::as<keel::usize>(n)] = stat0.elemAsDouble(j);

    for (keel::i32 s = 1; s < n; ++s) {
        Value sample = leaveOneOut(s);
        Value callArgs[1] = { sample };
        auto stat = ctx.engine->callFunctionHandle(
            args[0], keel::ArrayRef<Value>(callArgs, 1), ctx.env);
        if (stat.numel() != K)
            throw Error("jackknife: jackfun returned varying-size output",
                        0, 0, "jackknife", "", "numkit:jackknife:varyingStat");
        for (keel::usize j = 0; j < K; ++j)
            od[keel::as<keel::usize>(s) + j * keel::as<keel::usize>(n)] =
                stat.elemAsDouble(j);
    }
    outs[0] = std::move(out);
}

void combnk_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "combnk");
    const keel::i32 K = (keel::i32)args[1].toScalar();
    auto *mr = ctx.engine->allocator();
    const Value &v = args[0];
    // MATLAB treats a SCALAR v as the 1-element set {v} (NOT 1:v): combnk(5,2)
    // chooses 2 from {5} -> empty 0x2. So always route through the set
    // (vector) path; the int->1:N overload is a separate convenience entry.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> buf(v.numel(), scratch.allocator());
    for (keel::usize i = 0; i < v.numel(); ++i) buf[i] = v.elemAsDouble(i);
    outs[0] = combnk(keel::ArrayRef<keel::f64>(buf.data(), buf.size()), K, mr);
}

} // namespace detail

} // namespace numkit::stats
