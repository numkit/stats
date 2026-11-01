// toolboxes/stats/src/descriptive/tabulate.cpp
//
// MATLAB tabulate(x): frequency table.
//
//   T = tabulate(x)
//
// Output is a 3-column double matrix. Row layout:
//   T(i, 1) = value
//   T(i, 2) = count
//   T(i, 3) = 100 * count / N    (percent of non-NaN total)
//
// Two output shapes (matching MATLAB R2025b):
//   - All non-NaN values are positive integers:
//     dense rows for k = 1 .. max(x)  (zeros for missing k)
//   - Otherwise:
//     one row per unique non-NaN value, sorted ascending
//
// NaN values are excluded both from the row set and from the
// percentage denominator.
//
// Empty input -> 0-by-1 matrix (matches MATLAB).

#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace numkit::stats {

namespace {

void tabulate_argcheck(const Value &x)
{
    if (x.isStruct() || x.isFuncHandle() || x.isObject() || x.isComplex())
        throw Error("tabulate: unsupported input type",
                    0, 0, "tabulate", "", "numkit:tabulate:InvalidInput");
    if (x.isCell()) {
        for (size_t i = 0; i < x.numel(); ++i) {
            const Value &el = x.cellAt(i);
            if (!el.isChar() && !el.isString())
                throw Error("tabulate: cell input must be a cell array of character vectors",
                            0, 0, "tabulate", "", "numkit:tabulate:InvalidInput");
        }
    }
}

} // namespace

Value tabulate(const Value &x, std::pmr::memory_resource *mr)
{
    tabulate_argcheck(x);

    if (x.isCell() || x.isString() || x.isChar()) {
        std::vector<std::string> items;
        if (x.isCell()) {
            const size_t N0 = x.numel();
            items.reserve(N0);
            for (size_t i = 0; i < N0; ++i)
                items.push_back(x.cellAt(i).toString());
        } else if (x.isString()) {
            const size_t N0 = x.numel();
            items.reserve(N0);
            for (size_t i = 0; i < N0; ++i)
                items.push_back(x.stringElem(i));
        } else {
            const size_t R = x.dims().rows();
            if (x.numel() > 0) {
                items.reserve(R);
                for (size_t r = 0; r < R; ++r) {
                    std::string s = x.charRow(r);
                    while (!s.empty() && s.back() == ' ')
                        s.pop_back();
                    items.push_back(std::move(s));
                }
            }
        }
        const size_t N = items.size();
        if (N == 0)
            return Value::cell(0, 3, mr);

        std::vector<std::string> uniq;
        std::vector<size_t> cnt;
        uniq.reserve(N);
        cnt.reserve(N);
        for (const auto &s : items) {
            auto it = std::find(uniq.begin(), uniq.end(), s);
            if (it == uniq.end()) {
                uniq.push_back(s);
                cnt.push_back(1);
            } else {
                cnt[static_cast<size_t>(it - uniq.begin())]++;
            }
        }
        const size_t K = uniq.size();
        Value T = Value::cell(K, 3, mr);
        for (size_t k = 0; k < K; ++k) {
            T.cellAt(k)         = Value::fromString(uniq[k], mr);
            T.cellAt(k + K)     = Value::scalar(static_cast<double>(cnt[k]), mr);
            T.cellAt(k + 2 * K) = Value::scalar(static_cast<double>(cnt[k]) * 100.0
                                                / static_cast<double>(N), mr);
        }
        return T;
    }

    const size_t N0 = x.numel();
    if (N0 == 0)
        return Value::matrix(0, 1, ValueType::DOUBLE, mr);

    // Collect non-NaN values; check positive-integer property.
    std::vector<double> v;
    v.reserve(N0);
    bool all_pos_int = true;
    double max_val = 0.0;
    for (size_t i = 0; i < N0; ++i) {
        const double xi = x.elemAsDouble(i);
        if (std::isnan(xi)) continue;
        if (xi <= 0.0 || std::floor(xi) != xi)
            all_pos_int = false;
        else if (xi > max_val)
            max_val = xi;
        v.push_back(xi);
    }
    const size_t N = v.size();
    if (N == 0)
        return Value::matrix(0, 1, ValueType::DOUBLE, mr);

    if (all_pos_int) {
        const size_t K = static_cast<size_t>(max_val);
        std::vector<size_t> count(K, 0);
        for (double xi : v) {
            count[static_cast<size_t>(xi) - 1]++;
        }
        Value T = Value::matrix(K, 3, ValueType::DOUBLE, mr);
        double *o = T.doubleDataMut();
        for (size_t k = 0; k < K; ++k) {
            o[k]            = static_cast<double>(k + 1);   // value
            o[k + K]        = static_cast<double>(count[k]); // count
            // count * (100/N) precomputed reciprocal is 1 ulp off MATLAB's
            // count*100/N; arithmetic-coding-style loops amplify that ulp
            // into 1e-7 output divergence (bugs/closed/math/
            // numeric-divergence-chap10-7.md) — multiply-then-divide, no
            // hoisted reciprocal.
            o[k + 2 * K]    = static_cast<double>(count[k]) * 100.0
                              / static_cast<double>(N);
        }
        return T;
    }

    // Sparse layout: unique sorted values.
    std::sort(v.begin(), v.end());
    // Build unique list with counts.
    std::vector<double>  uniq;
    std::vector<size_t>  cnt;
    uniq.reserve(N);
    cnt.reserve(N);
    size_t i = 0;
    while (i < N) {
        size_t j = i + 1;
        while (j < N && v[j] == v[i]) ++j;
        uniq.push_back(v[i]);
        cnt.push_back(j - i);
        i = j;
    }
    const size_t K = uniq.size();
    Value T = Value::matrix(K, 3, ValueType::DOUBLE, mr);
    double *o = T.doubleDataMut();
    for (size_t k = 0; k < K; ++k) {
        o[k]            = uniq[k];
        o[k + K]        = static_cast<double>(cnt[k]);
        o[k + 2 * K]    = static_cast<double>(cnt[k]) * 100.0
                          / static_cast<double>(N);
    }
    return T;
}

} // namespace numkit::stats
