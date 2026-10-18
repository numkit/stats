// toolboxes/.../resample_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by resample.cpp + resample_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <numkit/value/value.hpp>
#include <numkit/ops/matlab_mt19937.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory_resource>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::stats {

namespace {

// Sample K indices in [0..N-1] given (optional) weights.
std::vector<int> sample_indices(int N, int K, bool with_replacement,
                                const std::vector<double> &weights,
                                numkit::ops::MatlabMT19937 &gen)
{
    std::vector<int> out;
    out.reserve((size_t)K);

    if (weights.empty()) {
        if (with_replacement) {
            std::uniform_int_distribution<int> ud(0, N - 1);
            for (int i = 0; i < K; ++i) out.push_back(ud(gen));
        } else {
            // Reservoir-style: shuffle 0..N-1 and take first K.
            std::vector<int> all(N);
            std::iota(all.begin(), all.end(), 0);
            std::shuffle(all.begin(), all.end(), gen);
            out.assign(all.begin(), all.begin() + std::min(K, N));
        }
    } else {
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
            std::vector<std::pair<double, int>> keyed;
            keyed.reserve(static_cast<std::size_t>(N));
            for (int i = 0; i < N; ++i) {
                const double w = weights[static_cast<std::size_t>(i)];
                if (w > 0.0) {
                    const double e = std::max(expDist(gen), 1e-300);
                    keyed.emplace_back(-e / w, i);
                }
            }
            const std::size_t take = std::min(static_cast<std::size_t>(K), keyed.size());
            std::partial_sort(keyed.begin(), keyed.begin() + take, keyed.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });
            for (std::size_t k = 0; k < take; ++k)
                out.push_back(keyed[k].second);
        }
    }
    return out;
}

std::vector<double> read_vec(const Value &v) {
    const size_t n = v.numel();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

} // anonymous
namespace {
// Shared enumeration: combinations of `items` taken `K` at a time.
Value combnkImpl(const std::vector<double> &items, int K,
                 std::pmr::memory_resource *mr)
{
    const int N = static_cast<int>(items.size());
    if (K < 0)
        throw Error("combnk: K must be non-negative", 0, 0, "combnk", "",
                    "numkit:combnk:badK");
    // MATLAB: choosing K > N elements yields an empty 0xK result (not an
    // error). E.g. combnk(1:4, 5) -> 0x5, combnk(5, 2) -> 0x2 (scalar 5 is
    // the 1-element set {5}, so K=2 > N=1).
    if (K > N)
        return Value::matrix(0, static_cast<size_t>(K), ValueType::DOUBLE, mr);

    // Number of combinations.
    long long C = 1;
    for (int i = 0; i < K; ++i) C = C * (N - i) / (i + 1);

    Value out = Value::matrix((size_t)C, (size_t)K, ValueType::DOUBLE, mr);
    if (C == 0) return out;
    double *od = out.doubleDataMut();
    if (K == 0) return out;

    // Enumerate combinations in lex order by index recursion.
    std::vector<int> idx((size_t)K);
    for (int i = 0; i < K; ++i) idx[(size_t)i] = i;
    long long row = 0;
    while (true) {
        for (int j = 0; j < K; ++j)
            od[(size_t)j * (size_t)C + (size_t)row] =
                items[(size_t)idx[(size_t)j]];
        ++row;
        int j = K - 1;
        while (j >= 0 && idx[(size_t)j] == N - K + j) --j;
        if (j < 0) break;
        ++idx[(size_t)j];
        for (int k = j + 1; k < K; ++k) idx[(size_t)k] = idx[(size_t)k - 1] + 1;
    }
    return out;
}
} // anon

} // namespace numkit::stats
