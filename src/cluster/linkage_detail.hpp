// toolboxes/stats/src/cluster/linkage_detail.hpp — private compute/register shared
// surface: the internal cophenet_full worker (cophenet's [c,d] core; def in
// linkage.cpp, external). Phase 2b compute/register split.
#pragma once

#include <keel/memory.h>
#include <numkit/value/value.hpp>

#include <keel/tuple.h>

namespace numkit::stats {

keel::Tuple<Value, Value> cophenet_full(const Value &Z, const Value &Y,
                                       keel::Allocator *mr);

} // namespace numkit::stats
