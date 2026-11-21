// toolboxes/.../crosstab_detail.hpp — private compute substrate (anon-in-header,
// internal linkage per TU) shared by crosstab.cpp compute and crosstab_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/array.h>
#include <keel/core.h>
#include <keel/string.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::stats {

namespace {

// Sorted-unique values from x (NaN excluded).
keel::Array<keel::f64> sortedUniqueNoNaN(const Value &x)
{
    const keel::usize N = x.numel();
    keel::Array<keel::f64> v;
    v.reserve(N);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (std::isnan(xi)) continue;
        v.push_back(xi);
    }
    std::sort(v.begin(), v.end());
    v.resize(std::unique(v.begin(), v.end()) - v.begin());
    return v;
}

// Find index of value in sorted unique list (binary search).
[[maybe_unused]] keel::usize indexOf(const keel::Array<keel::f64> &u, keel::f64 val)
{
    auto it = std::lower_bound(u.begin(), u.end(), val);
    return keel::as<keel::usize>(it - u.begin());
}

// num2str-style label for a (typically integer) group value, matching
// MATLAB crosstab's `labels` output. Integer values print without a
// decimal point; non-integers fall back to %.5g (num2str's default
// significant-digit count for the simple decimals seen in group ids).
[[maybe_unused]] keel::String labelString(keel::f64 v)
{
    char buf[64];
    if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15)
        std::snprintf(buf, sizeof(buf), "%lld", keel::as<long long>(v));
    else
        std::snprintf(buf, sizeof(buf), "%.5g", v);
    return keel::String::from(buf);
}

} // namespace

} // namespace numkit::stats
