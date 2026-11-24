// toolboxes/stats/src/descriptive/tiedrank.cpp
// Thin Value wrapper over keel::stats::tiedrank.

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

bool isVector(const Value &v)
{
    return v.dims().rows() == 1 || v.dims().cols() == 1;
}

void tiedrank_argcheck(const Value &x)
{
    if (x.dims().ndim() > 2)
        throw Error("tiedrank: X must be a vector or 2D matrix",
                    0, 0, "tiedrank", "", "numkit:tiedrank:ndim");
}

} // namespace

keel::Pair<Value, Value>
tiedrank(const Value &x, keel::Allocator *mr)
{
    tiedrank_argcheck(x);
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    if (isVector(x)) {
        const keel::usize N = x.numel();
        Value r = Value::matrix(R, C, ValueType::DOUBLE, mr);
        keel::Array<keel::f64> col(N, 0.0, scratch.allocator());
        for (keel::usize i = 0; i < N; ++i) col[i] = x.elemAsDouble(i);
        keel::f64 tieadj = 0.0;
        (void)keel::stats::tiedrank(col.as_ref(),
                                    keel::MutArrayRef<keel::f64>(r.doubleDataMut(), N),
                                    tieadj, scratch.allocator());
        return {keel::move(r), Value::scalar(tieadj, mr)};
    }

    Value r  = Value::matrix(R, C, ValueType::DOUBLE, mr);
    Value ta = Value::matrix(1, C, ValueType::DOUBLE, mr);
    keel::f64 *ro = r.doubleDataMut();
    keel::f64 *to = ta.doubleDataMut();
    keel::Array<keel::f64> col(R, 0.0, scratch.allocator());
    for (keel::usize c = 0; c < C; ++c) {
        for (keel::usize i = 0; i < R; ++i)
            col[i] = x.elemAsDouble(c * R + i);
        (void)keel::stats::tiedrank(col.as_ref(),
                                    keel::MutArrayRef<keel::f64>(ro + c * R, R),
                                    to[c], scratch.allocator());
    }
    return {keel::move(r), keel::move(ta)};
}

} // namespace numkit::stats
