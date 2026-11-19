// toolboxes/.../resample_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by resample.cpp + resample_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/ops/matlab_mt19937.hpp>
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

// Sample K indices in [0..N-1] given (optional) weights.
[[maybe_unused]] keel::Array<int> sample_indices(int N, int K, bool with_replacement,
                                keel::ArrayRef<double> weights,
                                numkit::ops::RngContext &rng)
{
    keel::Array<int> out;
    out.reserve((keel::usize)K);

    if (weights.empty()) {
        if (with_replacement) {
            for (int i = 0; i < K; ++i) {
                const double u = rng.nextUniform();
                int idx = static_cast<int>(std::floor(u * static_cast<double>(N)));
                if (idx >= N) idx = N - 1;
                out.push_back(idx);
            }
        } else {
            const int take = std::min(K, N);
            Value rp = numkit::ops::randperm(rng, static_cast<keel::usize>(N),
                                             static_cast<keel::usize>(take), nullptr);
            const double *rpd = rp.doubleData();
            for (int i = 0; i < take; ++i)
                out.push_back(static_cast<int>(rpd[i]) - 1);
        }
    } else {
        auto &gen = rng.engine();
        // Weighted sampling.
        if (with_replacement) {
            std::discrete_distribution<int> dd(weights.begin(), weights.end());
            for (int i = 0; i < K; ++i) out.push_back(dd(gen));
        } else {
            // Without replacement: Efraimidis-Spirakis (2006) wswor algorithm.
            // Each positive-weight item i receives key k_i = u_i^(1/w_i), or
            // equivalently log(u_i)/w_i = -Exp(1)/w_i; the top K keys are drawn
            // in descending key order. Exact O(N log N) with zero rejection risk.
            std::exponential_distribution<double> expDist(1.0);
            keel::Array<std::pair<double, int>> keyed;
            keyed.reserve(N);
            for (int i = 0; i < N; ++i) {
                const double w = weights[static_cast<keel::usize>(i)];
                if (w > 0.0) {
                    const double e = std::max(expDist(gen), 1e-300);
                    keyed.emplace_back(-e / w, i);
                }
            }
            const keel::usize take = std::min(static_cast<keel::usize>(K), keyed.size());
            std::partial_sort(keyed.begin(), keyed.begin() + take, keyed.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });
            for (keel::usize k = 0; k < take; ++k)
                out.push_back(keyed[k].second);
        }
    }
    return out;
}

[[maybe_unused]] keel::Array<double> read_vec(const Value &v) {
    const keel::usize n = v.numel();
    keel::Array<double> out(n);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

} // anonymous
namespace {
// Shared enumeration: combinations of `items` taken `K` at a time.
[[maybe_unused]] Value combnkImpl(keel::ArrayRef<double> items, int K,
                 keel::Allocator *mr)
{
    const int N = static_cast<int>(items.size());
    if (K < 0)
        throw Error("combnk: K must be non-negative", 0, 0, "combnk", "",
                    "numkit:combnk:badK");
    // MATLAB: choosing K > N elements yields an empty 0xK result (not an
    // error). E.g. combnk(1:4, 5) -> 0x5, combnk(5, 2) -> 0x2 (scalar 5 is
    // the 1-element set {5}, so K=2 > N=1).
    if (K > N)
        return Value::matrix(0, static_cast<keel::usize>(K), ValueType::DOUBLE, mr);

    // Number of combinations.
    long long C = 1;
    for (int i = 0; i < K; ++i) C = C * (N - i) / (i + 1);

    Value out = Value::matrix((keel::usize)C, (keel::usize)K, ValueType::DOUBLE, mr);
    if (C == 0) return out;
    double *od = out.doubleDataMut();
    if (K == 0) return out;

    // Enumerate combinations in lex order by index recursion.
    keel::Array<int> idx((keel::usize)K);
    for (int i = 0; i < K; ++i) idx[(keel::usize)i] = i;
    long long row = 0;
    while (true) {
        for (int j = 0; j < K; ++j)
            od[(keel::usize)j * (keel::usize)C + (keel::usize)row] =
                items[(keel::usize)idx[(keel::usize)j]];
        ++row;
        int j = K - 1;
        while (j >= 0 && idx[(keel::usize)j] == N - K + j) --j;
        if (j < 0) break;
        ++idx[(keel::usize)j];
        for (int k = j + 1; k < K; ++k) idx[(keel::usize)k] = idx[(keel::usize)k - 1] + 1;
    }
    return out;
}
} // anon

} // namespace numkit::stats
