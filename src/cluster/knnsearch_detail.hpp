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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::stats {

namespace {

// Parse positional/name-value args. Forms accepted:
//   knnsearch(X, Y)                     → K=1
//   knnsearch(X, Y, K)                  → K-NN, default metric
//   knnsearch(X, Y, 'K', K, ...)        → name-value
//   knnsearch(X, Y, 'Distance', metric, ...)
struct KnnOpts { int K = 1; keel::String metric = keel::String::from("euclidean"); double p = 2.0; };

KnnOpts parse_knn_args(keel::ArrayRef<Value> args, const char *fn)
{
    KnnOpts o;
    if (args.size() == 3 && args[2].isNumeric() && args[2].isScalar()) {
        // legacy positional K
        o.K = static_cast<int>(args[2].toScalar());
        return o;
    }
    // name-value pairs
    keel::usize i = 2;
    while (i < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error(keel::format("{}: expected name-value pair", fn),
                        0, 0, fn, "", "numkit:knn:nvpair");
        keel::String name = args[i].toString();
        for (auto &c : name) c = keel::to_ascii_lower(c);
        if (i + 1 >= args.size())
            throw Error(keel::format("{}: '{}' missing value", fn, name),
                        0, 0, fn, "", "numkit:knn:nvval");
        const Value &v = args[i + 1];
        if (name == "k") {
            o.K = static_cast<int>(v.toScalar());
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
