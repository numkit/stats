// toolboxes/stats/src/descriptive/grpstats.cpp
//
// MATLAB grpstats: per-group statistics.
//
//   M = grpstats(X, group)               default: mean per group
//   M = grpstats(X, group, fn)           fn = 'mean' | 'std' | 'sum' |
//                                              'numel' | 'min' | 'max' |
//                                              'var' | 'sem'
//   [M, S, ...] = grpstats(X, group, {'mean','std',...})
//                                       one output per fn name
//
// X may be a vector or matrix (groups apply along rows).
// Output rows = unique non-NaN groups (sorted ascending).
// NaN values in X are excluded per (column, group) cell from the
// reduction (using nan-aware accumulators).
//
// Cell-of-string fn argument deferred until our cell-string handling
// is uniform across libs.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::stats {

namespace {

// Aggregator IDs for the supported fn names.
enum class Agg { Mean, Std, Sum, Numel, Min, Max, Var, Sem };

Agg parseFnName(keel::StringRef name)
{
    if (name == "mean")  return Agg::Mean;
    if (name == "std")   return Agg::Std;
    if (name == "sum")   return Agg::Sum;
    if (name == "numel") return Agg::Numel;
    if (name == "min")   return Agg::Min;
    if (name == "max")   return Agg::Max;
    if (name == "var")   return Agg::Var;
    if (name == "sem")   return Agg::Sem;
    throw Error(keel::format("grpstats: unsupported fn name (got '{}'); expected one of {{mean,std,sum,numel,min,max,var,sem}}", name),
                0, 0, "grpstats", "", "numkit:grpstats:UnsupportedFn");
}

// Apply aggregator to a vector of values (NaN excluded).
keel::f64 aggregate(Agg fn, const keel::Array<keel::f64> &vals)
{
    // Filter NaN.
    keel::Array<keel::f64> v;
    v.reserve(vals.size());
    for (keel::f64 x : vals) if (!std::isnan(x)) v.push_back(x);
    const keel::usize n = v.size();

    switch (fn) {
        case Agg::Numel:
            return keel::as<keel::f64>(n);
        case Agg::Sum: {
            keel::f64 s = 0.0;
            for (keel::f64 x : v) s += x;
            return s;
        }
        case Agg::Min: {
            if (n == 0) return std::numeric_limits<keel::f64>::quiet_NaN();
            keel::f64 m = v[0];
            for (keel::usize i = 1; i < n; ++i) m = std::min(m, v[i]);
            return m;
        }
        case Agg::Max: {
            if (n == 0) return std::numeric_limits<keel::f64>::quiet_NaN();
            keel::f64 m = v[0];
            for (keel::usize i = 1; i < n; ++i) m = std::max(m, v[i]);
            return m;
        }
        case Agg::Mean: {
            if (n == 0) return std::numeric_limits<keel::f64>::quiet_NaN();
            keel::f64 s = 0.0;
            for (keel::f64 x : v) s += x;
            return s / keel::as<keel::f64>(n);
        }
        case Agg::Std:
        case Agg::Var:
        case Agg::Sem: {
            if (n < 2) {
                // MATLAB std/var of <2 elements: 0 by convention with N
                // normalisation, NaN with N-1. grpstats uses N-1
                // convention -> NaN.
                return std::numeric_limits<keel::f64>::quiet_NaN();
            }
            keel::f64 s = 0.0;
            for (keel::f64 x : v) s += x;
            const keel::f64 m = s / keel::as<keel::f64>(n);
            keel::f64 ss = 0.0;
            for (keel::f64 x : v) {
                const keel::f64 d = x - m;
                ss += d * d;
            }
            const keel::f64 var = ss / keel::as<keel::f64>(n - 1);
            if (fn == Agg::Var) return var;
            const keel::f64 sd = std::sqrt(var);
            if (fn == Agg::Std) return sd;
            return sd / std::sqrt(keel::as<keel::f64>(n));   // Sem
        }
    }
    return std::numeric_limits<keel::f64>::quiet_NaN();
}

void grpstats_argcheck(const Value &X, const Value &group)
{
    const keel::usize Nrows = X.dims().rows();
    const keel::usize Ncols = X.dims().cols();
    if (group.numel() != Nrows && !(Nrows == 1 && group.numel() == Ncols))
        throw Error("grpstats: group length must match number of rows of X",
                    0, 0, "grpstats", "", "numkit:grpstats:LenMismatch");
}

} // namespace

keel::Array<Value>
grpstats(const Value &X, const Value &group, keel::ArrayRef<keel::String> fn_names, keel::Allocator *mr)
{
    grpstats_argcheck(X, group);
    const keel::usize Nrows = X.dims().rows();
    const keel::usize Ncols = X.dims().cols();

    // For row-vector input, MATLAB applies per-element groups (treating
    // X as length(X) observations of 1 column).
    bool row_vec = (Nrows == 1 && Ncols >= 1);
    const keel::usize N = row_vec ? Ncols : Nrows;
    const keel::usize C = row_vec ? 1     : Ncols;

    // Find unique non-NaN groups (sorted ascending).
    keel::Array<keel::f64> g(N);
    for (keel::usize i = 0; i < N; ++i) g[i] = group.elemAsDouble(i);
    keel::Array<keel::f64> u_g;
    u_g.reserve(N);
    for (keel::f64 v : g) if (!std::isnan(v)) u_g.push_back(v);
    std::sort(u_g.begin(), u_g.end());
    u_g.resize(std::unique(u_g.begin(), u_g.end()) - u_g.begin());
    const keel::usize Ng = u_g.size();

    // Collect fns; default mean.
    keel::Array<Agg> fns;
    if (fn_names.empty()) {
        fns.push_back(Agg::Mean);
    } else {
        fns.reserve(fn_names.size());
        for (const auto &n : fn_names) fns.push_back(parseFnName(n));
    }
    const keel::usize F = fns.size();

    // Build outputs: F values, each (Ng × C).
    keel::Array<Value> outs;
    outs.reserve(F);
    for (keel::usize f = 0; f < F; ++f) {
        outs.push_back(Value::matrix(Ng, C, ValueType::DOUBLE, mr));
    }

    // Aggregate per (group, column).
    keel::Array<keel::f64> bucket;
    bucket.reserve(N);
    for (keel::usize k = 0; k < Ng; ++k) {
        const keel::f64 gk = u_g[k];
        for (keel::usize c = 0; c < C; ++c) {
            bucket.clear();
            for (keel::usize i = 0; i < N; ++i) {
                if (g[i] != gk) continue;
                const keel::f64 x = row_vec
                                     ? X.elemAsDouble(i)
                                     : X.elemAsDouble(i + c * Nrows);
                bucket.push_back(x);
            }
            for (keel::usize f = 0; f < F; ++f) {
                const keel::f64 v = aggregate(fns[f], bucket);
                outs[f].doubleDataMut()[k + c * Ng] = v;
            }
        }
    }
    return outs;
}

} // namespace numkit::stats
