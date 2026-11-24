// toolboxes/stats/src/descriptive/corrcov.cpp
// Thin Value wrapper over keel::stats::corrcov.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::stats {

namespace {

void corrcov_argcheck(const Value &C)
{
    if (C.dims().rows() != C.dims().cols())
        throw Error("corrcov: input must be a square matrix",
                    0, 0, "corrcov", "", "numkit:corrcov:NotSquare");
    if (C.dims().rows() == 0)
        throw Error("corrcov: input must be non-empty",
                    0, 0, "corrcov", "", "numkit:corrcov:Empty");
}

} // namespace

keel::Pair<Value, Value>
corrcov(const Value &C, keel::Allocator *mr)
{
    corrcov_argcheck(C);
    const keel::usize R = C.dims().rows();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> cv(R * R, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < R * R; ++i) cv[i] = C.elemAsDouble(i);

    Value Rv = Value::matrix(R, R, ValueType::DOUBLE, mr);
    Value Sv = Value::matrix(1, R, ValueType::DOUBLE, mr);
    const auto st = keel::stats::corrcov(
        cv.as_ref(), R,
        keel::MutArrayRef<keel::f64>(Rv.doubleDataMut(), R * R),
        keel::MutArrayRef<keel::f64>(Sv.doubleDataMut(), R));
    if (st != keel::stats::StatsStatus::kOk)
        throw Error("corrcov: negative variance on diagonal",
                    0, 0, "corrcov", "", "numkit:corrcov:NegativeVar");
    return {keel::move(Rv), keel::move(Sv)};
}

} // namespace numkit::stats
