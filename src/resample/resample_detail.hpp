// toolboxes/.../resample_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by resample.cpp + resample_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/random.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/ops/matlab_mt19937.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)



namespace numkit::stats {

namespace {

// Sample K indices in [0..N-1] given (optional) weights.
[[maybe_unused]] keel::Array<keel::i32> sample_indices(keel::i32 N, keel::i32 K, bool with_replacement,
                                keel::ArrayRef<keel::f64> weights,
                                numkit::ops::RngContext &rng)
{
    keel::Array<keel::i32> out;
    out.reserve((keel::usize)K);

    if (weights.empty()) {
        if (with_replacement) {
            for (keel::i32 i = 0; i < K; ++i) {
                const keel::f64 u = rng.nextUniform();
                keel::i32 idx = keel::as<keel::i32>(keel::floor(u * keel::as<keel::f64>(N)));
                if (idx >= N) idx = N - 1;
                out.push_back(idx);
            }
        } else {
            const keel::i32 take = keel::min(K, N);
            Value rp = numkit::ops::randperm(rng, keel::as<keel::usize>(N),
                                             keel::as<keel::usize>(take), nullptr);
            const keel::f64 *rpd = rp.doubleData();
            for (keel::i32 i = 0; i < take; ++i)
                out.push_back(keel::as<keel::i32>(rpd[i]) - 1);
        }
    } else {
        auto &gen = rng.engine();
        // Weighted sampling.
        if (with_replacement) {
            const keel::ArrayRef<keel::f64> wref(weights.data(), weights.size());
            for (keel::i32 i = 0; i < K; ++i)
                out.push_back(keel::as<keel::i32>(keel::random::discrete_index(gen, wref)));
        } else {
            // Without replacement: Efraimidis-Spirakis (2006) wswor algorithm.
            // Each positive-weight item i receives key k_i = u_i^(1/w_i), or
            // equivalently log(u_i)/w_i = -Exp(1)/w_i; the top K keys are drawn
            // in descending key order. Exact O(N log N) with zero rejection risk.
            keel::Array<keel::Pair<keel::f64, keel::i32>> keyed;
            keyed.reserve(N);
            for (keel::i32 i = 0; i < N; ++i) {
                const keel::f64 w = weights[keel::as<keel::usize>(i)];
                if (w > 0.0) {
                    const keel::f64 e = keel::max(keel::random::exponential_f64(gen, 1.0), 1e-300);
                    keyed.emplace_back(-e / w, i);
                }
            }
            const keel::usize take = keel::min(keel::as<keel::usize>(K), keyed.size());
            keel::partial_sort(keyed.begin(), keyed.begin() + take, keyed.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });
            for (keel::usize k = 0; k < take; ++k)
                out.push_back(keyed[k].second);
        }
    }
    return out;
}

[[maybe_unused]] keel::Array<keel::f64> read_vec(const Value &v) {
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

} // anonymous
namespace {
// Shared enumeration: combinations of `items` taken `K` at a time.
[[maybe_unused]] Value combnkImpl(keel::ArrayRef<keel::f64> items, keel::i32 K,
                 keel::Allocator *mr)
{
    const keel::i32 N = keel::as<keel::i32>(items.size());
    if (K < 0)
        throw Error("combnk: K must be non-negative", 0, 0, "combnk", "",
                    "numkit:combnk:badK");
    // MATLAB: choosing K > N elements yields an empty 0xK result (not an
    // error). E.g. combnk(1:4, 5) -> 0x5, combnk(5, 2) -> 0x2 (scalar 5 is
    // the 1-element set {5}, so K=2 > N=1).
    if (K > N)
        return Value::matrix(0, keel::as<keel::usize>(K), ValueType::DOUBLE, mr);

    // Number of combinations.
    keel::i64 C = 1;
    for (keel::i32 i = 0; i < K; ++i) C = C * (N - i) / (i + 1);

    Value out = Value::matrix((keel::usize)C, (keel::usize)K, ValueType::DOUBLE, mr);
    if (C == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    if (K == 0) return out;

    // Enumerate combinations in lex order by index recursion.
    keel::Array<keel::i32> idx((keel::usize)K);
    for (keel::i32 i = 0; i < K; ++i) idx[(keel::usize)i] = i;
    keel::i64 row = 0;
    while (true) {
        for (keel::i32 j = 0; j < K; ++j)
            od[(keel::usize)j * (keel::usize)C + (keel::usize)row] =
                items[(keel::usize)idx[(keel::usize)j]];
        ++row;
        keel::i32 j = K - 1;
        while (j >= 0 && idx[(keel::usize)j] == N - K + j) --j;
        if (j < 0) break;
        ++idx[(keel::usize)j];
        for (keel::i32 k = j + 1; k < K; ++k) idx[(keel::usize)k] = idx[(keel::usize)k - 1] + 1;
    }
    return out;
}
} // anon

Value cvpartition_compute(::numkit::ops::RngContext &rng,
                          const Value &nOrGroup,
                          keel::StringRef typeStr,
                          const Value *paramArg,
                          keel::Allocator *mr);
Value cvpartition_mask(const Value &c, bool isTest, keel::i32 foldIdx,
                       const char *fnName, keel::Allocator *mr);
Value cvpartition_repartition(::numkit::ops::RngContext &rng,
                              const Value &c,
                              keel::Allocator *mr);

} // namespace numkit::stats
