/// @file resample.hpp
/// @ingroup group_stats
// toolboxes/stats/include/numkit/stats/resample/resample.hpp
//
// Random sampling and resampling utilities.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

namespace numkit {

/// @addtogroup group_stats
/// @{
 namespace ops { class RngContext; } }

namespace numkit::stats {

/// @brief Sample integers from `1..N` (`r = randsample(N, K, replacement, weights)`).
///
/// Draws `K` indices from `{1, …, N}` with optional weighting.
///
/// @param N                 Upper bound of the sampling range.
/// @param K                 Number of draws.
/// @param with_replacement  When `true`, draw with replacement; `false` →
///                          permutation-style without replacement
///                          (requires `K <= N`).
/// @param weights           Optional probability weights of length `N`
///                          (pass empty Value for uniform). Need not sum
///                          to 1 — re-normalised internally.
/// @param mr                Memory resource (nullptr → process default).
/// @return                  `K × 1` column of 1-based indices.
/// @see datasample
Value randsample(::numkit::ops::RngContext &rng, keel::i32 N, keel::i32 K, bool with_replacement, const Value &weights,
                 keel::Allocator *mr = nullptr);

/// @brief Sample rows / columns of an array
/// (`Y = datasample(X, K, dim, replacement, weights)`).
///
/// @param X                 Source data (matrix or higher-D array).
/// @param K                 Number of draws.
/// @param dim               1-based dimension to sample along
///                          (1 = rows, 2 = columns).
/// @param with_replacement  When `true`, draw with replacement.
/// @param weights           Optional probability weights along `dim`
///                          (pass empty Value for uniform).
/// @param mr                Memory resource (nullptr → process default).
/// @return                  Subsample of `X`.
/// @see randsample
Value datasample(::numkit::ops::RngContext &rng, const Value &X, keel::i32 K, keel::i32 dim, bool with_replacement,
                 const Value &weights, keel::Allocator *mr = nullptr);

/// @brief Full `randsample` entry — the complete parse+validate+compute
/// contract (probed R2025b, build/probe75.m S-series): count vs
/// population forms (a population must be a vector, orientation kept),
/// REPLACE as true/false/1/0/'true'/'false', the positional weight
/// vector (length-N non-negative with positive sum), SampleTooLarge /
/// EmptyPopulation / NoWeighting / BadPopulation classes.
Value randsample_full(::numkit::ops::RngContext &rng, keel::ArrayRef<Value> args,
                      keel::Allocator *mr = nullptr);

/// @brief Full `datasample` entry (probed R2025b, probe75.m D-series):
/// K must be a non-negative integer, dim is the positional non-string
/// 3rd argument or the first non-singleton dimension, Replace accepts
/// a LOGICAL scalar only, Weights match the sampled dimension, plus
/// EmptyData / SampleTooLarge / TooFewPosWeights classes; gathers
/// along any dim (N-D included).
Value datasample_full(::numkit::ops::RngContext &rng, keel::ArrayRef<Value> args,
                      keel::Allocator *mr = nullptr);

// NOTE: bootstrp / jackknife are not exposed as Engine-free C++ cores — they
// must invoke a user statistic via the engine, so the logic lives inline in
// their register half (bundle/.../resample/resample_reg.cpp).

/// @brief Enumerate `K`-combinations of `1..N`
/// (`C = combnk(N, K)`).
///
/// @param N   Population size (combinations drawn from `1..N`).
/// @param K   Combination size.
/// @param mr  Memory resource (nullptr → process default).
/// @return    `C(N, K) × K` DOUBLE matrix of combinations.
/// @throws Error  `K` out of `[0, N]`.
Value combnk(keel::i32 N, keel::i32 K,
             keel::Allocator *mr = nullptr);

/// @brief Enumerate `K`-combinations of an arbitrary vector
/// (`C = combnk(v, K)`).
///
/// @param v   Input vector (length n).
/// @param K   Combination size.
/// @param mr  Memory resource (nullptr → process default).
/// @return    `C(n, K) × K` DOUBLE matrix of combinations of `v`'s
///            elements.
/// @throws Error  `K` out of `[0, v.size()]`.
Value combnk(keel::ArrayRef<keel::f64> v, keel::i32 K,
             keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::stats
