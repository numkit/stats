// toolboxes/stats/src/resample/resample.cpp

#include <numkit/stats/resample/resample.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

#include "resample_detail.hpp"

namespace numkit::stats {


Value randsample(::numkit::ops::RngContext &rng, int N, int K, bool with_replacement, const Value &weights, std::pmr::memory_resource *mr)
{
    if (N <= 0 || K <= 0)
        return Value::matrix(K > 0 ? K : 0, 1, ValueType::DOUBLE, mr);

    std::vector<double> w;
    if (weights.numel() > 0) w = read_vec(weights);
    if (!w.empty() && (int)w.size() != N)
        throw Error("randsample: weights length must equal N",
                    0, 0, "randsample", "", "numkit:randsample:size");

    auto &gen = rng;
    std::vector<int> idx;
    {
        idx = sample_indices(N, K, with_replacement, w, gen.engine());
    }
    Value out = Value::matrix((size_t)idx.size(), 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (size_t i = 0; i < idx.size(); ++i) od[i] = double(idx[i] + 1);  // 1-based
    return out;
}

Value datasample(::numkit::ops::RngContext &rng, const Value &X, int K, int dim, bool with_replacement, const Value &weights, std::pmr::memory_resource *mr)
{
    const size_t M = X.dims().rows();
    const size_t D = X.dims().cols();
    const int N = (dim == 2) ? (int)D : (int)M;
    if (N <= 0)
        throw Error("datasample: empty input", 0, 0, "datasample", "",
                    "numkit:datasample:empty");

    std::vector<double> w;
    if (weights.numel() > 0) w = read_vec(weights);
    if (!w.empty() && (int)w.size() != N)
        throw Error("datasample: weights length must equal sample-axis size",
                    0, 0, "datasample", "", "numkit:datasample:size");

    auto &gen = rng;
    std::vector<int> idx;
    {
        idx = sample_indices(N, K, with_replacement, w, gen.engine());
    }

    if (dim == 2) {
        // Sample columns.
        Value out = Value::matrix(M, idx.size(), ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (size_t k = 0; k < idx.size(); ++k)
            for (size_t r = 0; r < M; ++r)
                od[k * M + r] = X.elemAsDouble((size_t)idx[k] * M + r);
        return out;
    }
    // Sample rows.
    Value out = Value::matrix(idx.size(), D, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (size_t k = 0; k < idx.size(); ++k)
        for (size_t c = 0; c < D; ++c)
            od[c * idx.size() + k] = X.elemAsDouble(c * M + (size_t)idx[k]);
    return out;
}

// NOTE: bootstrp / jackknife are implemented inline in their register half
// (bundle/.../resample/resample_reg.cpp) via callFunctionHandle — they need the
// engine to invoke the user statistic, so there is no Engine-free C++ core here.


Value combnk(int N, int K, std::pmr::memory_resource *mr)
{
    if (N < 0)
        throw Error("combnk: N must be non-negative", 0, 0, "combnk", "",
                    "numkit:combnk:badN");
    std::vector<double> items((size_t)N);
    for (int i = 0; i < N; ++i) items[(size_t)i] = double(i + 1);
    return combnkImpl(items, K, mr);
}

Value combnk(Span<const double> v, int K, std::pmr::memory_resource *mr)
{
    std::vector<double> items(v.begin(), v.end());
    return combnkImpl(items, K, mr);
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
        const double b = v.elemAsDouble(0);
        if (b == 0.0 || b == 1.0) return b != 0.0;
    } else if (v.isChar() || v.isString()) {
        std::string t = v.toString();
        for (char &c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (t == "true") return true;
        if (t == "false") return false;
    }
    throw Error(std::string(fn) + ": REPLACE must be either true or false.",
                0, 0, fn, "", tag);
}

// Weight vector: length-n, non-negative, positive sum (NaN caught by
// the >= test).
std::vector<double> parseWeightVector(const Value &w, std::size_t n,
                                      const char *fn, const char *sizeTag,
                                      const char *badTag)
{
    std::vector<double> out;
    if (w.isEmpty()) return out;
    if (!w.dims().isVector() || static_cast<std::size_t>(w.numel()) != n)
        throw Error(std::string(fn) + ": W must have length "
                    + std::to_string(n) + ".",
                    0, 0, fn, "", sizeTag);
    out.resize(n);
    double sumw = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = w.elemAsDouble(i);
        if (!(out[i] >= 0.0))
            throw Error(std::string(fn) + ": W must contain non-negative "
                        "values with at least one positive value.",
                        0, 0, fn, "", badTag);
        sumw += out[i];
    }
    if (!(sumw > 0.0))
        throw Error(std::string(fn) + ": W must contain non-negative "
                    "values with at least one positive value.",
                    0, 0, fn, "", badTag);
    return out;
}

} // namespace

Value randsample_full(::numkit::ops::RngContext &rng, Span<const Value> args,
                      std::pmr::memory_resource *mr)
{
    if (args.size() < 2)
        throw Error("randsample: requires the input arguments N/POPULATION "
                    "and K.",
                    0, 0, "randsample", "", "numkit:randsample:tooFewInputs");

    // Count form vs population form (probe S05: population must be a
    // vector).
    bool havePopulation = false;
    Value population;
    std::size_t N = 0;
    if (args[0].isScalar() && args[0].isNumeric()
            && args[0].elemAsDouble(0) == std::floor(args[0].elemAsDouble(0))
            && args[0].elemAsDouble(0) >= 0.0) {
        N = static_cast<std::size_t>(args[0].elemAsDouble(0));
    } else {
        havePopulation = true;
        population = args[0];
        if (!population.dims().isVector() || population.numel() == 0)
            throw Error("randsample: POPULATION must be a vector.",
                        0, 0, "randsample", "", "numkit:randsample:badPopulation");
        N = static_cast<std::size_t>(population.numel());
    }
    const double kD = args[1].elemAsDouble(0);
    const std::size_t K = static_cast<std::size_t>(std::max(kD, 0.0));

    const bool replace = (args.size() >= 3 && !args[2].isEmpty())
        ? parseReplaceFlag(args[2], "randsample", "numkit:randsample:badReplaceValue")
        : false;
    std::vector<double> w;
    if (args.size() >= 4 && !args[3].isEmpty())
        w = parseWeightVector(args[3], N, "randsample",
                              "numkit:randsample:inputSizeMismatch",
                              "numkit:randsample:invalidWeights");

    if (!replace) {
        if (K > N)
            throw Error("randsample: K must be less than or equal "
                        + std::to_string(N) + " for sampling without "
                        "replacement.",
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

    auto &gen = rng;
    std::vector<int> idx = sample_indices(
        static_cast<int>(N), static_cast<int>(K), replace, w, gen.engine());

    // Population form: index the population values (orientation kept —
    // a row population returns a row, probe S12); count form returns a
    // column of 1-based indices.
    if (havePopulation) {
        const bool row = population.dims().dim(0) == 1
                         && population.dims().dim(1) > 1;
        const bool single = (population.type() == ValueType::SINGLE);
        const std::size_t m = idx.size();
        Value out = row ? Value::matrix(1, m, single ? ValueType::SINGLE
                                                     : ValueType::DOUBLE, mr)
                        : Value::matrix(m, 1, single ? ValueType::SINGLE
                                                     : ValueType::DOUBLE, mr);
        for (std::size_t i = 0; i < m; ++i) {
            const double v = population.elemAsDouble(
                static_cast<std::size_t>(idx[i]));
            if (single) out.singleDataMut()[i] = static_cast<float>(v);
            else out.doubleDataMut()[i] = v;
        }
        return out;
    }
    Value out = Value::matrix(idx.size(), 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t i = 0; i < idx.size(); ++i)
        od[i] = static_cast<double>(idx[i] + 1);
    return out;
}

Value datasample_full(::numkit::ops::RngContext &rng, Span<const Value> args,
                      std::pmr::memory_resource *mr)
{
    if (args.size() < 2)
        throw Error("datasample: requires the input arguments DATA and K.",
                    0, 0, "datasample", "", "numkit:datasample:tooFewInputs");
    const Value &X = args[0];
    const double kD = args[1].elemAsDouble(0);
    if (!(args[1].isScalar() && args[1].isNumeric() && kD >= 0.0
            && kD == std::floor(kD)))
        throw Error("datasample: sample size K must be a non-negative "
                    "integer.",
                    0, 0, "datasample", "", "numkit:datasample:invalidK");
    const std::size_t K = static_cast<std::size_t>(kD);

    // dim: a numeric positional 3rd argument, else the first
    // non-singleton dimension.
    std::size_t dim = 0;
    std::size_t nvStart = 2;
    if (args.size() > 2 && !args[2].isChar() && !args[2].isString()
            && !args[2].isObject()) {
        dim = static_cast<std::size_t>(std::max(args[2].elemAsDouble(0), 1.0));
        nvStart = 3;
    } else {
        for (int d = 0; d < X.dims().ndim(); ++d)
            if (X.dims().dim(d) != 1) {
                dim = static_cast<std::size_t>(d + 1);
                break;
            }
        if (dim == 0) dim = 1;
    }
    const std::size_t nd = static_cast<std::size_t>(X.dims().ndim());
    const std::size_t n = dim <= nd
        ? static_cast<std::size_t>(X.dims().dim(dim - 1)) : 1;

    // NV: Replace (LOGICAL scalar only — a numeric 1 rejects, probe
    // D03) / Weights.
    bool replace = true;   // datasample default
    Value wRaw;
    for (std::size_t i = nvStart; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) continue;
        std::string name = args[i].toString();
        for (char &c : name)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
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
    std::vector<double> w = parseWeightVector(
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
            throw Error("datasample: K must be less than or equal to the "
                        "number of sampled elements (" + std::to_string(n)
                        + ").",
                        0, 0, "datasample", "",
                        "numkit:datasample:sampleTooLarge");
        if (!w.empty()) {
            std::size_t posCount = 0;
            for (double v : w) if (v > 0.0) ++posCount;
            if (posCount < K)
                throw Error("datasample: there must be at least K positive "
                            "weights when sampling without replacement.",
                            0, 0, "datasample", "",
                            "numkit:datasample:tooFewPosWeights");
        }
    }

    auto &gen = rng;
    std::vector<int> idx = sample_indices(
        static_cast<int>(n), static_cast<int>(K), replace, w, gen.engine());

    // Gather along dim (1 → rows, 2 → columns).
    if (dim == 1) {
        const std::size_t cols = nd >= 2
            ? static_cast<std::size_t>(X.dims().dim(1)) : 1;
        Value out = Value::matrix(idx.size(), cols, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (std::size_t c = 0; c < cols; ++c)
            for (std::size_t k = 0; k < idx.size(); ++k)
                od[c * idx.size() + k] =
                    X.elemAsDouble(c * n + static_cast<std::size_t>(idx[k]));
        return out;
    }
    if (dim == 2) {
        const std::size_t rows = static_cast<std::size_t>(X.dims().dim(0));
        Value out = Value::matrix(rows, idx.size(), ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (std::size_t k = 0; k < idx.size(); ++k)
            for (std::size_t r = 0; r < rows; ++r)
                od[k * rows + r] =
                    X.elemAsDouble(static_cast<std::size_t>(idx[k]) * rows + r);
        return out;
    }
    // dim ≥ 3: each output slice along dim copies the source slice at
    // the sampled index (reference: subs{dim} = i).
    std::vector<std::size_t> od_(nd, 1);
    for (std::size_t d = 0; d < nd; ++d)
        od_[d] = static_cast<std::size_t>(X.dims().dim(d));
    od_[dim - 1] = idx.size();
    Value out = Value::matrixND(od_.data(), static_cast<int>(nd),
                                ValueType::DOUBLE, mr);
    std::vector<std::size_t> sstride(nd, 1), ostride(nd, 1);
    for (std::size_t d = 1; d < nd; ++d) {
        sstride[d] = sstride[d - 1]
            * static_cast<std::size_t>(X.dims().dim(d - 1));
        ostride[d] = ostride[d - 1] * od_[d - 1];
    }
    const std::size_t totalOther = [&] {
        std::size_t t = 1;
        for (std::size_t d = 0; d < nd; ++d)
            if (d + 1 != dim)
                t *= static_cast<std::size_t>(X.dims().dim(d));
        return t;
    }();
    std::vector<std::size_t> coord(nd, 0);
    for (std::size_t t = 0; t < totalOther; ++t) {
        // mixed-radix increment over the dims except `dim`.
        for (std::size_t k = 0; k < idx.size(); ++k) {
            coord[dim - 1] = static_cast<std::size_t>(idx[k]);
            std::size_t srcLin = 0, outLin = 0;
            for (std::size_t d = 0; d < nd; ++d) {
                srcLin += coord[d] * sstride[d];
                outLin += (d + 1 == dim ? k : coord[d]) * ostride[d];
            }
            out.doubleDataMut()[outLin] = X.elemAsDouble(srcLin);
        }
        for (std::size_t d = nd; d-- > 0;) {
            if (d + 1 == dim) continue;
            if (++coord[d] < static_cast<std::size_t>(X.dims().dim(d)))
                break;
            coord[d] = 0;
        }
    }
    return out;
}

} // namespace numkit::stats
