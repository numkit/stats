// toolboxes/.../anova_detail.hpp — private compute substrate (anon-in-header,
// internal linkage per TU) shared by anova.cpp compute and anova_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/array.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <numeric>
#include <keel/tuple.h>


namespace numkit::stats {

namespace {

// Bucket observations by group label (preserve first-seen order to match
// MATLAB-style ascending numeric order most of the time; for strict
// matches, callers can sort labels first).
struct Group {
    keel::f64 label;
    keel::Array<keel::f64> values;
};

keel::Array<Group> bucket(const Value &y, const Value &group)
{
    const keel::usize N = y.numel();
    if (group.numel() != N)
        throw Error("anova1: y and group must be the same length",
                    0, 0, "anova1", "", "numkit:anova1:size");
    keel::Array<Group> g;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 yi = y.elemAsDouble(i);
        const keel::f64 li = group.elemAsDouble(i);
        if (keel::is_nan(yi) || keel::is_nan(li)) continue;
        bool found = false;
        for (auto &gg : g) if (gg.label == li) { gg.values.push_back(yi); found = true; break; }
        if (!found) {
            Group ng{li, keel::Array<keel::f64>{}};
            ng.values.push_back(yi);
            g.push_back(keel::move(ng));
        }
    }
    // Sort by label ascending — matches MATLAB's anova1 default ordering.
    std::sort(g.begin(), g.end(),
              [](const Group &a, const Group &b) { return a.label < b.label; });
    return g;
}

} // anonymous

// anova2: two-way ANOVA core (10-tuple of doubles: pCols,pRows,Fc,Fr,dfC,dfR,
// dfE,ssC,ssR,ssE). Def in anova.cpp (external).
keel::Tuple<keel::f64, keel::f64, keel::f64, keel::f64, keel::f64,
           keel::f64, keel::f64, keel::f64, keel::f64, keel::f64>
anova2(const Value &Y, keel::Allocator *mr);

} // namespace numkit::stats
