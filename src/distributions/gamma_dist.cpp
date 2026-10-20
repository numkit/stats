// toolboxes/stats/src/distributions/gamma_dist.cpp
//
// Gamma distribution. Uses MATLAB convention: gampdf(x, a, b) interprets a
// as shape and b as scale (NOT rate). So f(x) = x^(a-1) exp(-x/b) /
// (b^a · Γ(a)). cdf composes gammainc on x/b; icdf uses gammaincinv;
// rnd uses MATLAB-exact Marsaglia-Tsang on RngContext.

#include <numkit/stats/distributions/gamma_dist.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <random>

#include "gamma_dist_detail.hpp"

namespace numkit::stats {

namespace {

double sampleGamma1(::numkit::ops::RngContext &rng, double a)
{
    if (!(a > 0.0))
        return (a == 0.0) ? 0.0 : std::numeric_limits<double>::quiet_NaN();
    const double d = (a < 1.0 ? (a + 1.0) : a) - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    double u = rng.nextUniform();
    const double u_boost = (a < 1.0) ? rng.nextUniform() : 1.0;
    double v = 0.0;
    for (;;) {
        const double z = rng.nextNormal();
        v = 1.0 + c * z;
        if (v > 0.0) {
            v = v * v * v;
            const double z2 = z * z;
            if (u < 1.0 - 0.0331 * z2 * z2 ||
                std::log(u) < 0.5 * z2 + d * (1.0 - v + std::log(v))) {
                break;
            }
        }
        u = rng.nextUniform();
    }
    return (a < 1.0) ? (d * v * std::pow(u_boost, 1.0 / a)) : (d * v);
}

} // namespace

Value gampdf(const Value &x, double a, double b, std::pmr::memory_resource *mr)
{
    // MATLAB: a<0 or b<=0 → NaN; a==0 → 0 (degenerate, all mass at 0).
    if (a < 0.0 || b <= 0.0)
        return elementwise(x, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    if (a == 0.0)
        return elementwise(x, [](double){ return 0.0; }, mr);
    // log f(x) = (a-1) log x - x/b - a log b - lgamma(a)
    const double log_b = std::log(b);
    const double lga   = std::lgamma(a);
    return elementwise(x, [=](double xi) {
        if (xi < 0.0) return 0.0;
        if (xi == 0.0) {
            if (a < 1.0) return std::numeric_limits<double>::infinity();
            if (a > 1.0) return 0.0;
            return std::exp(-lga - log_b); // a == 1 → 1/b
        }
        const double lp = (a - 1.0) * std::log(xi)
                        - xi / b
                        - a * log_b
                        - lga;
        return std::exp(lp);
    }, mr);
}

Value gamcdf(const Value &x, double a, double b, std::pmr::memory_resource *mr)
{
    if (a <= 0.0 || b <= 0.0)
        return elementwise(x, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    Value xs = elementwise(x, [=](double xi) {
        return (xi <= 0.0) ? 0.0 : xi / b;
    }, mr);
    Value av = Value::scalar(a, mr);
    return ::numkit::builtin::gammainc(xs, av, mr);
}

Value gaminv(const Value &p, double a, double b, std::pmr::memory_resource *mr)
{
    // MATLAB: a<0 / b<=0 → NaN; a==0 → 0 for p∈[0,1] (degenerate quantile = 0).
    if (a < 0.0 || b <= 0.0)
        return elementwise(p, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    if (a == 0.0)
        return elementwise(p, [](double pi) {
            return (pi >= 0.0 && pi <= 1.0) ? 0.0
                                            : std::numeric_limits<double>::quiet_NaN();
        }, mr);
    Value av = Value::scalar(a, mr);
    Value q  = ::numkit::builtin::gammaincinv(p, av, mr);
    return elementwise(q, [=](double qi){ return b * qi; }, mr);
}

Value gamrnd(::numkit::ops::RngContext &rng, double a, double b, size_t rows, size_t cols, std::pmr::memory_resource *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (a <= 0.0 || b <= 0.0 || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const size_t n = rows * cols;
    for (size_t i = 0; i < n; ++i) od[i] = b * sampleGamma1(rng, a);
    return out;
}

Value randg(::numkit::ops::RngContext &rng, double shape, size_t rows, size_t cols,
            std::pmr::memory_resource *mr)
{
    // randg(shape) ≡ gamrnd(shape, 1, rows, cols) — pure scale = 1.
    return gamrnd(rng, shape, 1.0, rows, cols, mr);
}

Value randg(::numkit::ops::RngContext &rng, const Value &shapeArray, std::pmr::memory_resource *mr)
{
    const std::size_t rows = shapeArray.dims().rows();
    const std::size_t cols = shapeArray.dims().cols();
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    const std::size_t n = rows * cols;
    if (n == 0) return out;
    double *od = out.doubleDataMut();
    for (std::size_t i = 0; i < n; ++i) {
        const double a = shapeArray.elemAsDouble(i);
        od[i] = sampleGamma1(rng, a);
    }
    return out;
}

std::tuple<double, double> gamstat(double a, double b)
{
    if (a <= 0.0 || b <= 0.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(a * b, a * b * b);
}

} // namespace numkit::stats
