// toolboxes/stats/src/distributions/students_t.cpp
//
// Student's t-distribution. Composes betainc / betaincinv for cdf /
// icdf; rnd uses Z / sqrt(X/ν) with Z ~ N(0,1), X ~ χ²(ν).

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/students_t.hpp>

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "students_t_detail.hpp"

namespace numkit::stats {

namespace {
inline bool t_argcheck(double nu) noexcept
{
    return nu > 0.0;
}
} // namespace

Value tpdf(const Value &x, double nu, keel::Allocator *mr)
{
    (void)t_argcheck(nu);
    return elementwise(x, [=](double xi) { return keel::stats::t_pdf(xi, nu); }, mr);
}

Value tcdf(const Value &x, double nu, keel::Allocator *mr)
{
    return elementwise(x, [=](double xi) { return keel::stats::t_cdf(xi, nu); }, mr);
}

Value tinv(const Value &p, double nu, keel::Allocator *mr)
{
    return elementwise(p, [=](double pi) { return keel::stats::t_inv(pi, nu); }, mr);
}

Value trnd(::numkit::ops::RngContext &rng, double nu, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (nu <= 0.0 || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        od[i] = keel::random::student_t_f64(gen, nu);
    }
    return out;
}

std::tuple<double, double> tstat(double nu)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if (nu <= 0.0) return std::make_tuple(NaN, NaN);
    // mean defined only for nu > 1; variance only for nu > 2.
    const double m = (nu > 1.0) ? 0.0 : NaN;
    const double v = (nu > 2.0) ? (nu / (nu - 2.0)) : NaN;
    return std::make_tuple(m, v);
}

// ── Noncentral t ────────────────────────────────────────────────────


Value nctpdf(const Value &x, double nu, double delta,
             keel::Allocator *mr)
{
    return elementwise(x, [&](double xi) { return nctpdf_one(xi, nu, delta, mr); }, mr);
}

Value nctcdf(const Value &x, double nu, double delta, bool upper,
             keel::Allocator *mr)
{
    return elementwise(x, [&](double xi) {
        double F = nctcdf_one(xi, nu, delta, mr);
        return upper ? 1.0 - F : F;
    }, mr);
}


Value nctinv(const Value &p, double nu, double delta,
             keel::Allocator *mr)
{
    return elementwise(p, [&](double pi) { return nctinv_one(pi, nu, delta, mr); }, mr);
}

std::tuple<double, double> nctstat(double nu, double delta)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(nu > 0.0)) return {nan, nan};
    double m = nan, v = nan;
    if (nu > 1.0) {
        // m = δ · sqrt(ν/2) · Γ((ν-1)/2) / Γ(ν/2)
        const double log_ratio = std::lgamma(0.5 * (nu - 1.0)) - std::lgamma(0.5 * nu);
        m = delta * std::sqrt(0.5 * nu) * std::exp(log_ratio);
    }
    if (nu > 2.0) {
        v = nu * (1.0 + delta * delta) / (nu - 2.0) - m * m;
    }
    return {m, v};
}

Value nctrnd(::numkit::ops::RngContext &rng, double nu, double delta, keel::usize rows, keel::usize cols,
             keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!(nu > 0.0) || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        const double Z = keel::random::normal_f64(gen);
        const double X = keel::random::chi_squared_f64(gen, nu);
        od[i] = (Z + delta) / std::sqrt(X / nu);
    }
    return out;
}

} // namespace numkit::stats
