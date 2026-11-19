// toolboxes/.../nan_aware_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by nan_aware.cpp + nan_aware_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

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

[[maybe_unused]] void validateNormFlag(int w, const char *fn)
{
    if (w != 0 && w != 1)
        throw Error(keel::format("{}: normalization flag must be 0 or 1", fn),
                     0, 0, fn, "", keel::format("numkit:{}:badFlag", fn));
}

[[maybe_unused]] double medianFromSlice(double *data, keel::usize n)
{
    if (n == 0) return std::nan("");
    if (n == 1) return data[0];
    const keel::usize mid = n / 2;
    std::nth_element(data, data + mid, data + n);
    if (n % 2 == 1)
        return data[mid];
    const double upper = data[mid];
    const double lower = *std::max_element(data, data + mid);
    return 0.5 * (lower + upper);
}

} // namespace
namespace {

// Read x as a column-major n×p matrix; vector → n×1.
[[maybe_unused]] void readObsMatrix(const Value &x, keel::Array<double> &out,
                    keel::usize &n, keel::usize &p)
{
    if (x.isEmpty()) { n = p = 0; return; }
    if (x.dims().isVector()) {
        n = x.numel();
        p = 1;
    } else {
        n = x.dims().rows();
        p = x.dims().cols();
    }
    out.resize(n * p);
    if (x.type() == ValueType::DOUBLE) {
        // Column-major copy — same layout as Value storage.
        const double *src = x.doubleData();
        std::memcpy(out.data(), src, n * p * sizeof(double));
    } else {
        for (keel::usize k = 0; k < n * p; ++k)
            out[k] = x.elemAsDouble(k);
    }
}

// Drop rows where any column has NaN. Compacts in place. Returns the
// new row count.
[[maybe_unused]] keel::usize dropNanRows(double *data, keel::usize n, keel::usize p)
{
    keel::usize w = 0;
    for (keel::usize r = 0; r < n; ++r) {
        bool keep = true;
        for (keel::usize c = 0; c < p; ++c) {
            if (std::isnan(data[c * n + r])) { keep = false; break; }
        }
        if (keep) {
            // Copy row r to row w (column by column).
            if (w != r)
                for (keel::usize c = 0; c < p; ++c)
                    data[c * n + w] = data[c * n + r];
            ++w;
        }
    }
    // Tail rows [w..n) on each column are now garbage — but the kept
    // count is what we use downstream. Caller treats `w` as new n.
    return w;
}

// In-place column centering for an n_kept × p block stored with stride
// n_orig (NOT compacted). To keep things simple we instead PACK the
// kept rows into a fresh contiguous n_kept × p layout before centering.
[[maybe_unused]] void packKeptRows(const double *src, keel::usize n_orig,
                   keel::usize n_kept, keel::usize p, double *dst)
{
    for (keel::usize c = 0; c < p; ++c)
        for (keel::usize r = 0; r < n_kept; ++r)
            dst[c * n_kept + r] = src[c * n_orig + r];
}

[[maybe_unused]] void centerColumnsLocal(double *data, keel::usize n, keel::usize p)
{
    for (keel::usize c = 0; c < p; ++c) {
        double s = 0.0;
        for (keel::usize r = 0; r < n; ++r) s += data[c * n + r];
        const double m = s / static_cast<double>(n);
        for (keel::usize r = 0; r < n; ++r) data[c * n + r] -= m;
    }
}

[[maybe_unused]] Value covFromCenteredLocal(const double *X, keel::usize n,
                            keel::usize p, double divisor,
                            keel::Allocator *mr)
{
    auto out = Value::matrix(p, p, ValueType::DOUBLE, mr);
    double *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j) {
            double s = 0.0;
            for (keel::usize r = 0; r < n; ++r)
                s += X[i * n + r] * X[j * n + r];
            dst[j * p + i] = s / divisor;
        }
    return out;
}

} // namespace

} // namespace numkit::stats
