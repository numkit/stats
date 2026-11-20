// toolboxes/stats/src/distributions/binomial.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/binomial.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <keel/random.h>

#include "binomial_detail.hpp"

namespace numkit::stats {

namespace {
inline bool bino_argcheck(double n, double p) noexcept
{
    return n >= 0.0 && std::floor(n) == n && p >= 0.0 && p <= 1.0;
}
} // namespace

Value binopdf(const Value &k, double n, double p, keel::Allocator *mr)
{
    if (!bino_argcheck(n, p))
        return elementwise(k, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    return elementwise(k, [=](double ki){ return bino_pmf(ki, n, p); }, mr);
}

Value binocdf(const Value &k, double n, double p, keel::Allocator *mr)
{
    if (n < 0.0 || std::floor(n) != n || p < 0.0 || p > 1.0)
        return elementwise(k, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    return elementwise(k, [=](double ki){ return bino_cdf_scalar(ki, n, p, mr); }, mr);
}

Value binoinv(const Value &p_in, double n, double p, keel::Allocator *mr)
{
    if (!bino_params_ok(n, p))
        return elementwise(p_in, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    return elementwise(p_in, [=](double pi) { return bino_inv_scalar(pi, n, p); }, mr);
}

Value binornd(::numkit::ops::RngContext &rng, double n, double p, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (n < 0.0 || std::floor(n) != n || p < 0.0 || p > 1.0 || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize cnt = rows * cols;
    const int ntrials = static_cast<int>(n);
    if (ntrials <= 1000000) {
        std::fill(od, od + cnt, 0.0);
        for (int t = 0; t < ntrials; ++t) {
            for (keel::usize i = 0; i < cnt; ++i) {
                if (rng.nextUniform() < p)
                    od[i] += 1.0;
            }
        }
        return out;
    }
    for (keel::usize i = 0; i < cnt; ++i)
        od[i] = static_cast<double>(keel::binomial_u64(gen, static_cast<keel::u64>(ntrials), p));
    return out;
}

std::tuple<double, double> binostat(double n, double p)
{
    if (n < 0.0 || std::floor(n) != n || p < 0.0 || p > 1.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(n * p, n * p * (1.0 - p));
}

} // namespace numkit::stats
