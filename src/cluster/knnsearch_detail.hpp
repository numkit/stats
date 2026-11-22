// toolboxes/.../knnsearch_detail.hpp — private compute substrate (anon-in-header,
// internal linkage per TU) shared by knnsearch.cpp compute and knnsearch_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>


namespace numkit::stats {

namespace {

// Parse positional/name-value args. Forms accepted:
//   knnsearch(X, Y)                     → K=1
//   knnsearch(X, Y, K)                  → K-NN, default metric
//   knnsearch(X, Y, 'K', K, ...)        → name-value
//   knnsearch(X, Y, 'Distance', metric, ...)
struct KnnOpts { keel::i32 K = 1; keel::String metric = keel::String::from("euclidean"); keel::f64 p = 2.0; };

[[maybe_unused]] KnnOpts parse_knn_args(keel::ArrayRef<Value> args, const char *fn)
{
    KnnOpts o;
    if (args.size() == 3 && args[2].isNumeric() && args[2].isScalar()) {
        // legacy positional K
        o.K = keel::as<keel::i32>(args[2].toScalar());
        return o;
    }
    // name-value pairs
    keel::usize i = 2;
    while (i < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error(keel::format("{}: expected name-value pair", fn),
                        0, 0, fn, "", "numkit:knn:nvpair");
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        if (i + 1 >= args.size())
            throw Error(keel::format("{}: '{}' missing value", fn, name),
                        0, 0, fn, "", "numkit:knn:nvval");
        const Value &v = args[i + 1];
        if (name == "k") {
            o.K = keel::as<keel::i32>(v.toScalar());
        } else if (name == "distance") {
            if (!v.isChar() && !v.isString())
                throw Error(keel::format("{}: Distance must be a string", fn),
                            0, 0, fn, "", "numkit:knn:dist");
            o.metric = v.toString();
        } else if (name == "p" || name == "minkowskiexponent") {
            o.p = v.toScalar();
        } else {
            // ignore unknown names (NSMethod, BucketSize, IncludeTies, …)
        }
        i += 2;
    }
    return o;
}

} // anonymous

} // namespace numkit::stats
