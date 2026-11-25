// toolboxes/stats/src/descriptive/grp2idx.cpp
//
// MATLAB grp2idx(S): turn a grouping variable into a 1-based index vector.
//
//   [G, GN, GL] = grp2idx(S)
//
//   G  = column vector of group indices, one per element of S.
//   GN = column cell of the group names (as char rows).
//   GL = group levels — for the non-categorical inputs handled here, GL == GN.
//
// Ordering rules (matching MATLAB R2025b):
//   - Numeric / logical S: groups are the sorted-ascending unique values;
//     each group name is num2str of the value. NaN maps to a NaN index and is
//     excluded from the group set.
//   - cellstr / string S: groups are the unique strings in FIRST-APPEARANCE
//     order; the name is the string itself.
//   - A single char row vector is treated as one group label.
//
// (categorical input and column-aligned char matrices are not yet handled.)

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/number.h>
#include <keel/string.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::stats {

namespace {

// Format a numeric group value the way the reference engine's grp2idx labels it:
// integers print with no decimals, other values with ~5 significant digits.
keel::String fmtGroupNum(keel::f64 v)
{
    char buf[64];
    if (keel::is_finite(v) && keel::floor(v) == v && keel::abs(v) < 1e15)
        return keel::String::from(keel::format_i64_to(buf, keel::as<keel::i64>(v)));
    return keel::String::from(keel::format_f64_general_to(buf, v, 5));
}

void grp2idx_argcheck(const Value &s)
{
    if (s.isStruct() || s.isComplex())
        throw Error("grp2idx: grouping variable must be real numeric, logical, char, string, or cellstr",
                    0, 0, "grp2idx", "", "numkit:grp2idx:badType");
}

} // namespace

// ── Public C++ API (see descriptive.hpp) ──────────────────────────────

Grp2idxResult grp2idx(const Value &s, keel::Allocator *mr)
{
    grp2idx_argcheck(s);
    const keel::usize n = s.numel();

    keel::Array<keel::f64>       g;        // index per element (NaN allowed)
    keel::Array<keel::String> names;    // group names in group order

    if (s.isCell() || (s.isString() && n != 1)) {
        // cellstr / multi-element string array: first-appearance order.
        const auto &vec = s.cellDataVec();
        g.reserve(vec.size());
        for (const auto &e : vec) {
            keel::String str = e.toString();
            keel::i32 idx = -1;
            for (keel::usize k = 0; k < names.size(); ++k)
                if (names[k] == str) { idx = keel::as<keel::i32>(k); break; }
            if (idx < 0) { names.push_back(keel::move(str)); idx = keel::as<keel::i32>(names.size()) - 1; }
            g.push_back(keel::as<keel::f64>(idx + 1));
        }
    } else if (s.isChar() || s.isString()) {
        // A single char row / string scalar is one group label.
        names.push_back(s.toString());
        g.push_back(1.0);
    } else {
        // Numeric / logical: sorted-ascending unique values; NaN -> NaN index.
        keel::Array<keel::f64> uniq;
        uniq.reserve(n);
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 xi = s.elemAsDouble(i);
            if (!keel::is_nan(xi)) uniq.push_back(xi);
        }
        keel::sort(uniq.begin(), uniq.end());
        uniq.resize(keel::unique(uniq.begin(), uniq.end()) - uniq.begin());
        names.reserve(uniq.size());
        for (keel::f64 u : uniq) names.push_back(fmtGroupNum(u));
        g.reserve(n);
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 xi = s.elemAsDouble(i);
            if (keel::is_nan(xi)) {
                g.push_back(keel::Limits<keel::f64>::quiet_nan());
                continue;
            }
            const auto it = keel::lower_bound(uniq.begin(), uniq.end(), xi);
            g.push_back(keel::as<keel::f64>((it - uniq.begin()) + 1));
        }
    }

    Grp2idxResult r;
    // G: column vector.
    r.G = Value::matrix(g.size(), 1, ValueType::DOUBLE, mr);
    keel::copy(g.begin(), g.end(), r.G.doubleDataMut());
    // GN / GL: column cell of char rows (GL == GN for these input types).
    r.GN = Value::cell(names.size(), 1, mr);
    r.GL = Value::cell(names.size(), 1, mr);
    for (keel::usize k = 0; k < names.size(); ++k) {
        r.GN.cellAt(k) = Value::fromString(names[k], mr);
        r.GL.cellAt(k) = Value::fromString(names[k], mr);
    }
    return r;
}

} // namespace numkit::stats
