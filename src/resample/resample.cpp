// toolboxes/stats/src/resample/resample.cpp

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/resample/resample.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <numeric>

#include "resample_detail.hpp"

namespace numkit::stats {

namespace {
void randsample_argcheck(keel::ArrayRef<keel::f64> w, keel::i32 N)
{
    if (!w.empty() && keel::as<keel::i32>(w.size()) != N)
        throw Error("randsample: weights length must equal N",
                    0, 0, "randsample", "", "numkit:randsample:size");
}
} // namespace

Value randsample(::numkit::ops::RngContext &rng, keel::i32 N, keel::i32 K, bool with_replacement, const Value &weights, keel::Allocator *mr)
{
    if (N <= 0 || K <= 0)
        return Value::matrix(K > 0 ? K : 0, 1, ValueType::DOUBLE, mr);

    keel::Array<keel::f64> w;
    if (weights.numel() > 0) w = read_vec(weights);
    randsample_argcheck(w.as_ref(), N);

    keel::Array<keel::i32> idx;
    {
        idx = sample_indices(N, K, with_replacement, w.as_ref(), rng);
    }
    Value out = Value::matrix((keel::usize)idx.size(), 1, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < idx.size(); ++i) od[i] = keel::f64(idx[i] + 1);  // 1-based
    return out;
}

Value datasample(::numkit::ops::RngContext &rng, const Value &X, keel::i32 K, keel::i32 dim, bool with_replacement, const Value &weights, keel::Allocator *mr)
{
    const keel::usize M = X.dims().rows();
    const keel::usize D = X.dims().cols();
    const keel::i32 N = (dim == 2) ? (keel::i32)D : (keel::i32)M;
    if (N <= 0)
        throw Error("datasample: empty input", 0, 0, "datasample", "",
                    "numkit:datasample:empty");

    keel::Array<keel::f64> w;
    if (weights.numel() > 0) w = read_vec(weights);
    if (!w.empty() && (keel::i32)w.size() != N)
        throw Error("datasample: weights length must equal sample-axis size",
                    0, 0, "datasample", "", "numkit:datasample:size");

    keel::Array<keel::i32> idx;
    {
        idx = sample_indices(N, K, with_replacement, w.as_ref(), rng);
    }

    if (dim == 2) {
        // Sample columns.
        Value out = Value::matrix(M, idx.size(), ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize k = 0; k < idx.size(); ++k)
            for (keel::usize r = 0; r < M; ++r)
                od[k * M + r] = X.elemAsDouble((keel::usize)idx[k] * M + r);
        return out;
    }
    // Sample rows.
    Value out = Value::matrix(idx.size(), D, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize k = 0; k < idx.size(); ++k)
        for (keel::usize c = 0; c < D; ++c)
            od[c * idx.size() + k] = X.elemAsDouble(c * M + (keel::usize)idx[k]);
    return out;
}

// NOTE: bootstrp / jackknife are implemented inline in their register half
// (bundle/.../resample/resample_reg.cpp) via callFunctionHandle — they need the
// engine to invoke the user statistic, so there is no Engine-free C++ core here.


Value combnk(keel::i32 N, keel::i32 K, keel::Allocator *mr)
{
    if (N < 0)
        throw Error("combnk: N must be non-negative", 0, 0, "combnk", "",
                    "numkit:combnk:badN");
    keel::Array<keel::f64> items((keel::usize)N);
    for (keel::i32 i = 0; i < N; ++i) items[(keel::usize)i] = keel::f64(i + 1);
    return combnkImpl(items.as_ref(), K, mr);
}

Value combnk(keel::ArrayRef<keel::f64> v, keel::i32 K, keel::Allocator *mr)
{
    return combnkImpl(v, K, mr);
}

// ════════════════════════════════════════════════════════════════════
// Engine adapters
// ════════════════════════════════════════════════════════════════════

// ── full entries: the probed randsample / datasample contracts ────────
// (rejection classes frozen by build/probe75.m S01-S12 / D01-D09).

namespace {

// REPLACE accepts true/false/1/0/'true'/'false' (randsample); anything
// else is BadReplaceValue.
bool parseReplaceFlag(const Value &v, const char *fn, const char *tag)
{
    if (v.isLogical() || (v.isScalar() && v.isNumeric())) {
        const keel::f64 b = v.elemAsDouble(0);
        if (b == 0.0 || b == 1.0) return b != 0.0;
    } else if (v.isChar() || v.isString()) {
        keel::String t = v.toString();
        t.to_ascii_lower_in_place();
        if (t == "true") return true;
        if (t == "false") return false;
    }
    throw Error(keel::format("{}: REPLACE must be either true or false.", fn),
                0, 0, fn, "", tag);
}

// Weight vector: length-n, non-negative, positive sum (NaN caught by
// the >= test).
keel::Array<keel::f64> parseWeightVector(const Value &w, keel::usize n,
                                      const char *fn, const char *sizeTag,
                                      const char *badTag)
{
    keel::Array<keel::f64> out;
    if (w.isEmpty()) return out;
    if (!w.dims().isVector() || keel::as<keel::usize>(w.numel()) != n)
        throw Error(keel::format("{}: W must have length {}.", fn, n),
                    0, 0, fn, "", sizeTag);
    out.resize(n);
    keel::f64 sumw = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        out[i] = w.elemAsDouble(i);
        if (!(out[i] >= 0.0))
            throw Error(keel::format("{}: W must contain non-negative "
                                     "values with at least one positive value.", fn),
                        0, 0, fn, "", badTag);
        sumw += out[i];
    }
    if (!(sumw > 0.0))
        throw Error(keel::format("{}: W must contain non-negative "
                                 "values with at least one positive value.", fn),
                    0, 0, fn, "", badTag);
    return out;
}

} // namespace

Value randsample_full(::numkit::ops::RngContext &rng, keel::ArrayRef<Value> args,
                      keel::Allocator *mr)
{
    if (args.size() < 2)
        throw Error("randsample: requires the input arguments N/POPULATION "
                    "and K.",
                    0, 0, "randsample", "", "numkit:randsample:tooFewInputs");

    // Count form vs population form (probe S05: population must be a
    // vector).
    bool havePopulation = false;
    Value population;
    keel::usize N = 0;
    if (args[0].isScalar() && args[0].isNumeric()
            && args[0].elemAsDouble(0) == std::floor(args[0].elemAsDouble(0))
            && args[0].elemAsDouble(0) >= 0.0) {
        N = keel::as<keel::usize>(args[0].elemAsDouble(0));
    } else {
        havePopulation = true;
        population = args[0];
        if (!population.dims().isVector() || population.numel() == 0)
            throw Error("randsample: POPULATION must be a vector.",
                        0, 0, "randsample", "", "numkit:randsample:badPopulation");
        N = keel::as<keel::usize>(population.numel());
    }
    const keel::f64 kD = args[1].elemAsDouble(0);
    const keel::usize K = keel::as<keel::usize>(std::max(kD, 0.0));

    const bool replace = (args.size() >= 3 && !args[2].isEmpty())
        ? parseReplaceFlag(args[2], "randsample", "numkit:randsample:badReplaceValue")
        : false;
    keel::Array<keel::f64> w;
    if (args.size() >= 4 && !args[3].isEmpty())
        w = parseWeightVector(args[3], N, "randsample",
                              "numkit:randsample:inputSizeMismatch",
                              "numkit:randsample:invalidWeights");

    if (!replace) {
        if (K > N)
            throw Error(keel::format("randsample: K must be less than or equal "
                                     "{} for sampling without replacement.", N),
                        0, 0, "randsample", "", "numkit:randsample:sampleTooLarge");
        if (!w.empty())
            throw Error("randsample: weighted sampling without replacement "
                        "is not supported.",
                        0, 0, "randsample", "", "numkit:randsample:noWeighting");
    }

    if (replace && N == 0) {
        if (K == 0) return Value::matrix(0, 1, ValueType::DOUBLE, mr);
        throw Error("randsample: K must be 0 when N=0 or the input "
                    "population is empty.",
                    0, 0, "randsample", "", "numkit:randsample:emptyPopulation");
    }

    keel::Array<keel::i32> idx = sample_indices(
        keel::as<keel::i32>(N), keel::as<keel::i32>(K), replace, w.as_ref(), rng);

    // Population form: index the population values (orientation kept —
    // a row population returns a row, probe S12); count form returns a
    // column of 1-based indices.
    if (havePopulation) {
        const bool row = population.dims().dim(0) == 1
                         && population.dims().dim(1) > 1;
        const keel::usize m = idx.size();
        if (population.isCell()) {
            Value out = row ? Value::cell(1, m, mr) : Value::cell(m, 1, mr);
            for (keel::usize i = 0; i < m; ++i)
                out.cellAt(i) = population.cellAt(keel::as<keel::usize>(idx[i]));
            return out;
        }
        if (population.isString()) {
            Value out = row ? Value::stringArray(1, m, mr) : Value::stringArray(m, 1, mr);
            for (keel::usize i = 0; i < m; ++i)
                out.stringElemSet(i, population.stringElem(keel::as<keel::usize>(idx[i])));
            return out;
        }
        const bool single = (population.type() == ValueType::SINGLE);
        Value out = row ? Value::matrix(1, m, single ? ValueType::SINGLE
                                                     : ValueType::DOUBLE, mr)
                        : Value::matrix(m, 1, single ? ValueType::SINGLE
                                                     : ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < m; ++i) {
            const keel::f64 v = population.elemAsDouble(
                keel::as<keel::usize>(idx[i]));
            if (single) out.singleDataMut()[i] = keel::as<keel::f32>(v);
            else out.doubleDataMut()[i] = v;
        }
        return out;
    }
    Value out = Value::matrix(idx.size(), 1, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < idx.size(); ++i)
        od[i] = keel::as<keel::f64>(idx[i] + 1);
    return out;
}

Value datasample_full(::numkit::ops::RngContext &rng, keel::ArrayRef<Value> args,
                      keel::Allocator *mr)
{
    if (args.size() < 2)
        throw Error("datasample: requires the input arguments DATA and K.",
                    0, 0, "datasample", "", "numkit:datasample:tooFewInputs");
    const Value &X = args[0];
    const keel::f64 kD = args[1].elemAsDouble(0);
    if (!(args[1].isScalar() && args[1].isNumeric() && kD >= 0.0
            && kD == std::floor(kD)))
        throw Error("datasample: sample size K must be a non-negative "
                    "integer.",
                    0, 0, "datasample", "", "numkit:datasample:invalidK");
    const keel::usize K = keel::as<keel::usize>(kD);

    // dim: a numeric positional 3rd argument, else the first
    // non-singleton dimension.
    keel::usize dim = 0;
    keel::usize nvStart = 2;
    if (args.size() > 2 && !args[2].isChar() && !args[2].isString()
            && !args[2].isObject()) {
        dim = keel::as<keel::usize>(std::max(args[2].elemAsDouble(0), 1.0));
        nvStart = 3;
    } else {
        for (keel::i32 d = 0; d < X.dims().ndim(); ++d)
            if (X.dims().dim(d) != 1) {
                dim = keel::as<keel::usize>(d + 1);
                break;
            }
        if (dim == 0) dim = 1;
    }
    const keel::usize nd = keel::as<keel::usize>(X.dims().ndim());
    const keel::usize n = dim <= nd
        ? keel::as<keel::usize>(X.dims().dim(keel::as<keel::i32>(dim - 1))) : 1;

    // NV: Replace (LOGICAL scalar only — a numeric 1 rejects, probe
    // D03) / Weights.
    bool replace = true;   // datasample default
    Value wRaw;
    for (keel::usize i = nvStart; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        if (name == "replace") {
            if (!args[i + 1].isLogical() || !args[i + 1].isScalar())
                throw Error("datasample: REPLACE must be either true or "
                            "false.",
                            0, 0, "datasample", "",
                            "numkit:datasample:invalidReplace");
            replace = args[i + 1].elemAsDouble(0) != 0.0;
        } else if (name == "weights") {
            wRaw = args[i + 1];
        }
    }
    keel::Array<keel::f64> w = parseWeightVector(
        wRaw, n, "datasample", "numkit:datasample:inputSizeMismatch",
        "numkit:datasample:invalidWeights");

    if (replace) {
        if (n == 0) {
            if (K == 0) return Value::matrix(0, 1, ValueType::DOUBLE, mr);
            throw Error("datasample: sample size K must be 0 when sampling "
                        "from empty data.",
                        0, 0, "datasample", "", "numkit:datasample:emptyData");
        }
    } else {
        if (K > n)
            throw Error(keel::format("datasample: K must be less than or equal to the "
                                     "number of sampled elements ({}).", n),
                        0, 0, "datasample", "",
                        "numkit:datasample:sampleTooLarge");
        if (!w.empty()) {
            keel::usize posCount = 0;
            for (keel::f64 v : w) if (v > 0.0) ++posCount;
            if (posCount < K)
                throw Error("datasample: there must be at least K positive "
                            "weights when sampling without replacement.",
                            0, 0, "datasample", "",
                            "numkit:datasample:tooFewPosWeights");
        }
    }

    keel::Array<keel::i32> idx = sample_indices(
        keel::as<keel::i32>(n), keel::as<keel::i32>(K), replace, w.as_ref(), rng);

    // Gather along dim (1 → rows, 2 → columns).
    if (dim == 1) {
        const keel::usize cols = nd >= 2
            ? keel::as<keel::usize>(X.dims().dim(1)) : 1;
        if (X.isCell()) {
            Value out = Value::cell(idx.size(), cols, mr);
            for (keel::usize c = 0; c < cols; ++c)
                for (keel::usize k = 0; k < idx.size(); ++k)
                    out.cellAt(c * idx.size() + k) =
                        X.cellAt(c * n + keel::as<keel::usize>(idx[k]));
            return out;
        }
        if (X.isString()) {
            Value out = Value::stringArray(idx.size(), cols, mr);
            for (keel::usize c = 0; c < cols; ++c)
                for (keel::usize k = 0; k < idx.size(); ++k)
                    out.stringElemSet(c * idx.size() + k,
                        X.stringElem(c * n + keel::as<keel::usize>(idx[k])));
            return out;
        }
        Value out = Value::matrix(idx.size(), cols, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize c = 0; c < cols; ++c)
            for (keel::usize k = 0; k < idx.size(); ++k)
                od[c * idx.size() + k] =
                    X.elemAsDouble(c * n + keel::as<keel::usize>(idx[k]));
        return out;
    }
    if (dim == 2) {
        const keel::usize rows = keel::as<keel::usize>(X.dims().dim(0));
        if (X.isCell()) {
            Value out = Value::cell(rows, idx.size(), mr);
            for (keel::usize k = 0; k < idx.size(); ++k)
                for (keel::usize r = 0; r < rows; ++r)
                    out.cellAt(k * rows + r) =
                        X.cellAt(keel::as<keel::usize>(idx[k]) * rows + r);
            return out;
        }
        if (X.isString()) {
            Value out = Value::stringArray(rows, idx.size(), mr);
            for (keel::usize k = 0; k < idx.size(); ++k)
                for (keel::usize r = 0; r < rows; ++r)
                    out.stringElemSet(k * rows + r,
                        X.stringElem(keel::as<keel::usize>(idx[k]) * rows + r));
            return out;
        }
        Value out = Value::matrix(rows, idx.size(), ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize k = 0; k < idx.size(); ++k)
            for (keel::usize r = 0; r < rows; ++r)
                od[k * rows + r] =
                    X.elemAsDouble(keel::as<keel::usize>(idx[k]) * rows + r);
        return out;
    }
    // dim ≥ 3: each output slice along dim copies the source slice at
    // the sampled index (reference: subs{dim} = i).
    keel::Array<keel::usize> od_(nd, 1);
    for (keel::usize d = 0; d < nd; ++d)
        od_[d] = keel::as<keel::usize>(X.dims().dim(keel::as<keel::i32>(d)));
    od_[dim - 1] = idx.size();
    Value out = Value::matrixND(od_.data(), keel::as<keel::i32>(nd),
                                ValueType::DOUBLE, mr);
    keel::Array<keel::usize> sstride(nd, 1), ostride(nd, 1);
    for (keel::usize d = 1; d < nd; ++d) {
        sstride[d] = sstride[d - 1]
            * keel::as<keel::usize>(X.dims().dim(keel::as<keel::i32>(d - 1)));
        ostride[d] = ostride[d - 1] * od_[d - 1];
    }
    const keel::usize totalOther = [&] {
        keel::usize t = 1;
        for (keel::usize d = 0; d < nd; ++d)
            if (d + 1 != dim)
                t *= keel::as<keel::usize>(X.dims().dim(keel::as<keel::i32>(d)));
        return t;
    }();
    keel::Array<keel::usize> coord(nd, 0);
    for (keel::usize t = 0; t < totalOther; ++t) {
        // mixed-radix increment over the dims except `dim`.
        for (keel::usize k = 0; k < idx.size(); ++k) {
            coord[dim - 1] = keel::as<keel::usize>(idx[k]);
            keel::usize srcLin = 0, outLin = 0;
            for (keel::usize d = 0; d < nd; ++d) {
                srcLin += coord[d] * sstride[d];
                outLin += (d + 1 == dim ? k : coord[d]) * ostride[d];
            }
            out.doubleDataMut()[outLin] = X.elemAsDouble(srcLin);
        }
        for (keel::usize d = nd; d-- > 0;) {
            if (d + 1 == dim) continue;
            if (++coord[d] < keel::as<keel::usize>(X.dims().dim(keel::as<keel::i32>(d))))
                break;
            coord[d] = 0;
        }
    }
    return out;
}

} // namespace numkit::stats
