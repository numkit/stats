// toolboxes/stats/src/distributions/fisher_f.cpp
//
// F-distribution. Composes betainc / betaincinv on
// (v1·x/(v1·x + v2), v1/2, v2/2) for cdf / icdf; rnd from two
// independent χ²-distributed samples.

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/fisher_f.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <keel/random.h>

#include "fisher_f_detail.hpp"

namespace numkit::stats {

namespace {
inline bool fisher_f_argcheck(double v1, double v2) noexcept
{
    return v1 > 0.0 && v2 > 0.0;
}
} // namespace

Value fpdf(const Value &x, double v1, double v2, keel::Allocator *mr)
{
    if (!fisher_f_argcheck(v1, v2))
        return elementwise(x, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    // log f(x; v1, v2) = (v1/2) log(v1) + (v2/2) log(v2)
    //                  + (v1/2 - 1) log x
    //                  - ((v1+v2)/2) log(v2 + v1 x)
    //                  - lbeta(v1/2, v2/2)
    // where lbeta(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b).
    const double a = 0.5 * v1;
    const double b = 0.5 * v2;
    const double lbeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    return elementwise(x, [=](double xi) {
        if (xi < 0.0) return 0.0;
        if (xi == 0.0) {
            // Density at 0 has three regimes (limit of x^(v1/2 - 1) as x → 0+):
            //   v1 < 2  →  +Inf
            //   v1 == 2 →  finite, value computed without the (a-1)·log x term
            //   v1 > 2  →  0
            if (v1 <  2.0) return std::numeric_limits<double>::infinity();
            if (v1 == 2.0) {
                const double lp0 = a * log_v1 + b * log_v2
                                 - (a + b) * std::log(v2)
                                 - lbeta;
                return std::exp(lp0);
            }
            return 0.0;
        }
        const double lp = a * log_v1 + b * log_v2
                        + (a - 1.0) * std::log(xi)
                        - (a + b) * std::log(v2 + v1 * xi)
                        - lbeta;
        return std::exp(lp);
    }, mr);
}

Value fcdf(const Value &x, double v1, double v2, keel::Allocator *mr)
{
    if (v1 <= 0.0 || v2 <= 0.0)
        return elementwise(x, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    Value z = elementwise(x, [=](double xi) {
        if (xi <= 0.0) return 0.0;
        return (v1 * xi) / (v1 * xi + v2);
    }, mr);
    Value a = Value::scalar(0.5 * v1, mr);
    Value b = Value::scalar(0.5 * v2, mr);
    return ::numkit::builtin::betainc(z, a, b, mr);
}

Value finv(const Value &p, double v1, double v2, keel::Allocator *mr)
{
    if (v1 <= 0.0 || v2 <= 0.0)
        return elementwise(p, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    Value a = Value::scalar(0.5 * v1, mr);
    Value b = Value::scalar(0.5 * v2, mr);
    Value z = ::numkit::builtin::betaincinv(p, a, b, mr);
    // x = (v2 / v1) · z / (1 - z)
    return elementwise(z, [=](double zi){
        if (zi <= 0.0) return 0.0;
        if (zi >= 1.0) return std::numeric_limits<double>::infinity();
        return (v2 / v1) * zi / (1.0 - zi);
    }, mr);
}

Value frnd(::numkit::ops::RngContext &rng, double v1, double v2, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (v1 <= 0.0 || v2 <= 0.0 || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        const double x1 = keel::chi_squared_f64(gen, v1);
        const double x2 = keel::chi_squared_f64(gen, v2);
        od[i] = (x1 / v1) / (x2 / v2);
    }
    return out;
}

std::tuple<double, double> fstat(double v1, double v2)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    // Invalid params ⇒ NaN/NaN (matches MATLAB R2025b).
    if (v1 <= 0.0 || v2 <= 0.0) return std::make_tuple(NaN, NaN);
    const double mean = (v2 > 2.0) ? (v2 / (v2 - 2.0)) : NaN;
    double var = NaN;
    if (v2 > 4.0) {
        const double num = 2.0 * v2 * v2 * (v1 + v2 - 2.0);
        const double den = v1 * (v2 - 2.0) * (v2 - 2.0) * (v2 - 4.0);
        var = num / den;
    }
    return std::make_tuple(mean, var);
}

// ── Noncentral F ────────────────────────────────────────────────────


Value ncfpdf(const Value &x, double nu1, double nu2, double delta,
             keel::Allocator *mr)
{
    return elementwise(x, [&](double xi) { return ncfpdf_one(xi, nu1, nu2, delta); }, mr);
}


Value ncfcdf(const Value &x, double nu1, double nu2, double delta,
             bool upper, keel::Allocator *mr)
{
    return elementwise(x, [&](double xi) {
        const double F = ncfcdf_one(xi, nu1, nu2, delta, mr);
        return upper ? 1.0 - F : F;
    }, mr);
}

Value ncfinv(const Value &p, double nu1, double nu2, double delta,
             keel::Allocator *mr)
{
    return elementwise(p, [&](double pi) { return ncfinv_one(pi, nu1, nu2, delta, mr); }, mr);
}

std::tuple<double, double> ncfstat(double nu1, double nu2, double delta)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || delta < 0.0)
        return {nan, nan};
    double m = nan, v = nan;
    if (nu2 > 2.0) {
        m = nu2 * (nu1 + delta) / (nu1 * (nu2 - 2.0));
    }
    if (nu2 > 4.0) {
        const double ratio = nu2 / nu1;
        const double num = (nu1 + delta) * (nu1 + delta)
                         + (nu1 + 2.0 * delta) * (nu2 - 2.0);
        const double den = (nu2 - 2.0) * (nu2 - 2.0) * (nu2 - 4.0);
        v = 2.0 * ratio * ratio * num / den;
    }
    return {m, v};
}

Value ncfrnd(::numkit::ops::RngContext &rng, double nu1, double nu2, double delta,
             keel::usize rows, keel::usize cols,
             keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || delta < 0.0 || rows * cols == 0)
        return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::u64 J = (delta == 0.0) ? 0ULL : keel::poisson_u64(gen, 0.5 * delta);
        const double X1 = keel::chi_squared_f64(gen, nu1 + 2.0 * static_cast<double>(J));
        const double X2 = keel::chi_squared_f64(gen, nu2);
        od[i] = (X1 / nu1) / (X2 / nu2);
    }
    return out;
}

} // namespace numkit::stats
