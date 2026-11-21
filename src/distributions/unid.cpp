// toolboxes/stats/src/distributions/unid.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/unid.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <keel/random.h>

#include "unid_detail.hpp"

namespace numkit::stats {

namespace {
inline bool unid_argcheck(double N) noexcept
{
    return N >= 1.0 && std::floor(N) == N;
}
} // namespace

Value unidpdf(const Value &k, double N, keel::Allocator *mr)
{
    if (!unid_argcheck(N))
        return elementwise(k, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    const double inv = 1.0 / N;
    return elementwise(k, [=](double ki) {
        if (ki < 1.0 || ki > N || std::floor(ki) != ki) return 0.0;
        return inv;
    }, mr);
}

Value unidcdf(const Value &k, double N, keel::Allocator *mr)
{
    if (N < 1.0 || std::floor(N) != N)
        return elementwise(k, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    const double inv = 1.0 / N;
    return elementwise(k, [=](double ki) {
        if (ki < 1.0) return 0.0;
        if (ki >= N) return 1.0;
        return std::floor(ki) * inv;
    }, mr);
}

Value unidinv(const Value &p, double N, keel::Allocator *mr)
{
    if (!(N >= 1.0) || std::floor(N) != N)  // also catches NaN N
        return elementwise(p, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    return elementwise(p, [=](double pi) {
        // MATLAB convention: p outside (0, 1] -> NaN. p=0 has no
        // integer pre-image in the support, so it is also NaN.
        if (std::isnan(pi) || pi <= 0.0 || pi > 1.0)
            return std::numeric_limits<double>::quiet_NaN();
        const double r = std::ceil(pi * N);
        return r < 1.0 ? 1.0 : (r > N ? N : r);
    }, mr);
}

Value unidrnd(::numkit::ops::RngContext &rng, double N, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (N < 1.0 || std::floor(N) != N || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize cnt = rows * cols;
    const keel::i64 nMax = keel::as<keel::i64>(N);
    for (keel::usize i = 0; i < cnt; ++i)
        od[i] = keel::as<double>(keel::random::uniform_i64(gen, 1, nMax));
    return out;
}

std::tuple<double, double> unidstat(double N)
{
    if (N < 1.0 || std::floor(N) != N) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(0.5 * (N + 1.0), (N * N - 1.0) / 12.0);
}

} // namespace numkit::stats
