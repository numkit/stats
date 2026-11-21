// toolboxes/stats/src/test/ad_dw_detail.hpp — private compute/register shared surface:
// dwStatAndPLeft (Durbin-Watson stat + left p-value; def in ad_dw.cpp, external).
#pragma once
#include <keel/memory.h>
#include <numkit/value/value.hpp>
namespace numkit::stats {
keel::f64 dwStatAndPLeft(const Value &r, const Value &X, bool exact,
                      keel::f64 &dwOut, keel::Allocator *mr);
} // namespace numkit::stats
