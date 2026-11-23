// toolboxes/stats/src/distributions/uniform.cpp

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/uniform.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"


#include "uniform_detail.hpp"

namespace numkit::stats {

namespace {
inline bool unif_argcheck(keel::f64 a, keel::f64 b) noexcept
{
    return b > a;
}
} // namespace

Value unifpdf(const Value &x, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    (void)unif_argcheck(a, b);
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::unif_pdf(xi, a, b); }, mr);
}

Value unifcdf(const Value &x, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::unif_cdf(xi, a, b); }, mr);
}

Value unifinv(const Value &p, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::unif_inv(pi, a, b); }, mr);
}

Value unifrnd(::numkit::ops::RngContext &rng, keel::f64 a, keel::f64 b, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (b <= a || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    const keel::f64 w = b - a;
    for (keel::usize i = 0; i < n; ++i) od[i] = a + w * rng.nextUniform();
    return out;
}

keel::Tuple<keel::f64, keel::f64> unifstat(keel::f64 a, keel::f64 b)
{
    if (b <= a) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return keel::make_tuple(nan, nan);
    }
    const keel::f64 w = b - a;
    return keel::make_tuple(0.5 * (a + b), (w * w) / 12.0);
}

} // namespace numkit::stats
