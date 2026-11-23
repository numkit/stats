// toolboxes/stats/src/dim/pca_detail.hpp — private compute/register shared surface:
// pcares_full (residuals+reconstruction core; def in pca.cpp, external).
#pragma once
#include <keel/memory.h>
#include <numkit/value/value.hpp>
#include <keel/tuple.h>
namespace numkit::stats {
keel::Tuple<Value, Value> pcares_full(const Value &X, keel::i32 ndim,
                                     keel::Allocator *mr);
} // namespace numkit::stats
