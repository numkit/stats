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

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::stats {

namespace {

void tabulate_argcheck(const Value &x)
{
    if (x.isStruct() || x.isFuncHandle() || x.isObject() || x.isComplex())
        throw Error("tabulate: unsupported input type",
                    0, 0, "tabulate", "", "numkit:tabulate:InvalidInput");
    if (x.isCell()) {
        for (keel::usize i = 0; i < x.numel(); ++i) {
            const Value &el = x.cellAt(i);
            if (!el.isChar() && !el.isString())
                throw Error("tabulate: cell input must be a cell array of character vectors",
                            0, 0, "tabulate", "", "numkit:tabulate:InvalidInput");
        }
    }
}

} // namespace

Value tabulate(const Value &x, keel::Allocator *mr)
{
    tabulate_argcheck(x);

    if (x.isCell() || x.isString() || x.isChar()) {
        keel::Array<keel::String> items;
        if (x.isCell()) {
            const keel::usize N0 = x.numel();
            items.reserve(N0);
            for (keel::usize i = 0; i < N0; ++i)
                items.push_back(x.cellAt(i).toString());
        } else if (x.isString()) {
            const keel::usize N0 = x.numel();
            items.reserve(N0);
            for (keel::usize i = 0; i < N0; ++i)
                items.push_back(x.stringElem(i).clone());
        } else {
            const keel::usize R = x.dims().rows();
            if (x.numel() > 0) {
                items.reserve(R);
                for (keel::usize r = 0; r < R; ++r) {
                    keel::String s = x.charRow(r);
                    while (!s.empty() && s.back() == ' ')
                        s.pop_back();
                    items.push_back(keel::move(s));
                }
            }
        }
        const keel::usize N = items.size();
        if (N == 0)
            return Value::cell(0, 3, mr);

        keel::Array<keel::String> uniq;
        keel::Array<keel::usize> cnt;
        uniq.reserve(N);
        cnt.reserve(N);
        for (const auto &s : items) {
            auto it = keel::find(uniq.begin(), uniq.end(), s);
            if (it == uniq.end()) {
                uniq.push_back(s.clone());
                cnt.push_back(1);
            } else {
                cnt[keel::as<keel::usize>(it - uniq.begin())]++;
            }
        }
        const keel::usize K = uniq.size();
        Value T = Value::cell(K, 3, mr);
        for (keel::usize k = 0; k < K; ++k) {
            T.cellAt(k)         = Value::fromString(uniq[k], mr);
            T.cellAt(k + K)     = Value::scalar(keel::as<keel::f64>(cnt[k]), mr);
            T.cellAt(k + 2 * K) = Value::scalar(keel::as<keel::f64>(cnt[k]) * 100.0
                                                / keel::as<keel::f64>(N), mr);
        }
        return T;
    }

    const keel::usize N0 = x.numel();
    if (N0 == 0)
        return Value::matrix(0, 1, ValueType::DOUBLE, mr);

    // Collect non-NaN values; check positive-integer property.
    keel::Array<keel::f64> v;
    v.reserve(N0);
    bool all_pos_int = true;
    keel::f64 max_val = 0.0;
    for (keel::usize i = 0; i < N0; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (keel::is_nan(xi)) continue;
        if (xi <= 0.0 || keel::floor(xi) != xi)
            all_pos_int = false;
        else if (xi > max_val)
            max_val = xi;
        v.push_back(xi);
    }
    const keel::usize N = v.size();
    if (N == 0)
        return Value::matrix(0, 1, ValueType::DOUBLE, mr);

    if (all_pos_int) {
        const keel::usize K = keel::as<keel::usize>(max_val);
        keel::Array<keel::usize> count(K, 0);
        for (keel::f64 xi : v) {
            count[keel::as<keel::usize>(xi) - 1]++;
        }
        Value T = Value::matrix(K, 3, ValueType::DOUBLE, mr);
        keel::f64 *o = T.doubleDataMut();
        for (keel::usize k = 0; k < K; ++k) {
            o[k]            = keel::as<keel::f64>(k + 1);   // value
            o[k + K]        = keel::as<keel::f64>(count[k]); // count
            // count * (100/N) precomputed reciprocal is 1 ulp off MATLAB's
            // count*100/N; arithmetic-coding-style loops amplify that ulp
            // into 1e-7 output divergence (bugs/closed/math/
            // numeric-divergence-chap10-7.md) — multiply-then-divide, no
            // hoisted reciprocal.
            o[k + 2 * K]    = keel::as<keel::f64>(count[k]) * 100.0
                              / keel::as<keel::f64>(N);
        }
        return T;
    }

    // Sparse layout: unique sorted values.
    keel::sort(v.begin(), v.end());
    // Build unique list with counts.
    keel::Array<keel::f64>  uniq;
    keel::Array<keel::usize>  cnt;
    uniq.reserve(N);
    cnt.reserve(N);
    keel::usize i = 0;
    while (i < N) {
        keel::usize j = i + 1;
        while (j < N && v[j] == v[i]) ++j;
        uniq.push_back(v[i]);
        cnt.push_back(j - i);
        i = j;
    }
    const keel::usize K = uniq.size();
    Value T = Value::matrix(K, 3, ValueType::DOUBLE, mr);
    keel::f64 *o = T.doubleDataMut();
    for (keel::usize k = 0; k < K; ++k) {
        o[k]            = uniq[k];
        o[k + K]        = keel::as<keel::f64>(cnt[k]);
        o[k + 2 * K]    = keel::as<keel::f64>(cnt[k]) * 100.0
                          / keel::as<keel::f64>(N);
    }
    return T;
}

} // namespace numkit::stats
