// toolboxes/stats/src/descriptive/normalize_detail.hpp — private compute/register
// shared surface: zscoreCore (the flag/dim zscore core with optional mu/sigma
// out-params; def in normalize.cpp, external). Phase 2b compute/register split.
#pragma once

#include <keel/memory.h>
#include <numkit/value/value.hpp>


namespace numkit::stats {

Value zscoreCore(const Value &A, keel::i32 flag, keel::i32 dim,
                 Value *muOut = nullptr, Value *sigmaOut = nullptr,
                 keel::Allocator *mr = nullptr);

} // namespace numkit::stats
