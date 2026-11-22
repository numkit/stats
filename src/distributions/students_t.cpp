// toolboxes/stats/src/distributions/students_t.cpp
//
// Student's t-distribution. Composes betainc / betaincinv for cdf /
// icdf; rnd uses Z / sqrt(X/ν) with Z ~ N(0,1), X ~ χ²(ν).

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/students_t.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <mutex>
#include <keel/random.h>


#include "students_t_detail.hpp"

namespace numkit::stats {

namespace {
inline bool t_argcheck(keel::f64 nu) noexcept
{
    return nu > 0.0;
}
} // namespace

Value tpdf(const Value &x, keel::f64 nu, keel::Allocator *mr)
{
    (void)t_argcheck(nu);
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::t_pdf(xi, nu); }, mr);
}

Value tcdf(const Value &x, keel::f64 nu, keel::Allocator *mr)
{
    return elementwise(x, [=](keel::f64 xi) { return keel::stats::t_cdf(xi, nu); }, mr);
}

Value tinv(const Value &p, keel::f64 nu, keel::Allocator *mr)
{
    return elementwise(p, [=](keel::f64 pi) { return keel::stats::t_inv(pi, nu); }, mr);
}

Value trnd(::numkit::ops::RngContext &rng, keel::f64 nu, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (nu <= 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        od[i] = keel::random::student_t_f64(gen, nu);
    }
    return out;
}

std::tuple<keel::f64, keel::f64> tstat(keel::f64 nu)
{
    const keel::f64 NaN = keel::Limits<keel::f64>::quiet_nan();
    if (nu <= 0.0) return std::make_tuple(NaN, NaN);
    // mean defined only for nu > 1; variance only for nu > 2.
    const keel::f64 m = (nu > 1.0) ? 0.0 : NaN;
    const keel::f64 v = (nu > 2.0) ? (nu / (nu - 2.0)) : NaN;
    return std::make_tuple(m, v);
}

// ── Noncentral t ────────────────────────────────────────────────────


Value nctpdf(const Value &x, keel::f64 nu, keel::f64 delta,
             keel::Allocator *mr)
{
    return elementwise(x, [&](keel::f64 xi) { return nctpdf_one(xi, nu, delta, mr); }, mr);
}

Value nctcdf(const Value &x, keel::f64 nu, keel::f64 delta, bool upper,
             keel::Allocator *mr)
{
    return elementwise(x, [&](keel::f64 xi) {
        keel::f64 F = nctcdf_one(xi, nu, delta, mr);
        return upper ? 1.0 - F : F;
    }, mr);
}


Value nctinv(const Value &p, keel::f64 nu, keel::f64 delta,
             keel::Allocator *mr)
{
    return elementwise(p, [&](keel::f64 pi) { return nctinv_one(pi, nu, delta, mr); }, mr);
}

std::tuple<keel::f64, keel::f64> nctstat(keel::f64 nu, keel::f64 delta)
{
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    if (!(nu > 0.0)) return {nan, nan};
    keel::f64 m = nan, v = nan;
    if (nu > 1.0) {
        // m = δ · sqrt(ν/2) · Γ((ν-1)/2) / Γ(ν/2)
        const keel::f64 log_ratio = keel::lgamma(0.5 * (nu - 1.0)) - keel::lgamma(0.5 * nu);
        m = delta * keel::sqrt(0.5 * nu) * keel::exp(log_ratio);
    }
    if (nu > 2.0) {
        v = nu * (1.0 + delta * delta) / (nu - 2.0) - m * m;
    }
    return {m, v};
}

Value nctrnd(::numkit::ops::RngContext &rng, keel::f64 nu, keel::f64 delta, keel::usize rows, keel::usize cols,
             keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!(nu > 0.0) || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 Z = keel::random::normal_f64(gen);
        const keel::f64 X = keel::random::chi_squared_f64(gen, nu);
        od[i] = (Z + delta) / keel::sqrt(X / nu);
    }
    return out;
}

} // namespace numkit::stats
