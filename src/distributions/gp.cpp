// toolboxes/stats/src/distributions/gp.cpp

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/gp.hpp>

#include <numkit/builtin/datafun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"


#include "gp_detail.hpp"

namespace numkit::stats {

namespace {
inline bool gp_argcheck(keel::f64 sigma) noexcept
{
    return sigma > 0.0;
}
} // namespace

Value gppdf(const Value &x, keel::f64 k, keel::f64 sigma, keel::f64 theta, keel::Allocator *mr)
{
    if (!gp_argcheck(sigma))
        return elementwise(x, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    const keel::f64 inv_s = 1.0 / sigma;
    return elementwise(x, [=](keel::f64 xi) {
        const keel::f64 z = (xi - theta) * inv_s;
        if (z < 0.0) return 0.0;
        if (k == 0.0) return inv_s * keel::exp(-z);
        const keel::f64 t = 1.0 + k * z;
        if (t <= 0.0) return 0.0;
        return inv_s * keel::pow(t, -1.0 / k - 1.0);
    }, mr);
}

Value gpcdf(const Value &x, keel::f64 k, keel::f64 sigma, keel::f64 theta, keel::Allocator *mr)
{
    if (sigma <= 0.0)
        return elementwise(x, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    const keel::f64 inv_s = 1.0 / sigma;
    return elementwise(x, [=](keel::f64 xi) {
        const keel::f64 z = (xi - theta) * inv_s;
        if (z <= 0.0) return 0.0;
        if (k == 0.0) return -keel::expm1(-z);
        const keel::f64 t = 1.0 + k * z;
        if (t <= 0.0) return (k > 0) ? 0.0 : 1.0;
        return 1.0 - keel::pow(t, -1.0 / k);
    }, mr);
}

Value gpinv(const Value &p, keel::f64 k, keel::f64 sigma, keel::f64 theta, keel::Allocator *mr)
{
    if (sigma <= 0.0)
        return elementwise(p, [](keel::f64){ return keel::Limits<keel::f64>::quiet_nan(); }, mr);
    return elementwise(p, [=](keel::f64 pi) {
        return gp_inv_one(pi, k, sigma, theta);
    }, mr);
}

Value gprnd(::numkit::ops::RngContext &rng, keel::f64 k, keel::f64 sigma, keel::f64 theta, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto &gen = rng;
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (sigma <= 0.0 || rows * cols == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) {
        // genRes53 -- MATLAB-canonical 53-bit uniform.
        // gprnd uses u DIRECTLY (not 1-u) per MATLAB convention:
        //   x = theta + sigma * (u^(-k) - 1) / k       for k != 0
        //   x = theta - sigma * log(u)                  for k == 0
        // (gpinv(p) uses 1-p which is the standard ICDF; sampling
        // from rand() with the swapped-u form gives the same
        // distribution but matches MATLAB's specific bit sequence.)
        keel::f64 u = gen.genRes53();
        if (u <= 0.0) u = keel::Limits<keel::f64>::kMin;
        if (k == 0.0) {
            od[i] = theta - sigma * keel::log(u);
        } else {
            od[i] = theta + sigma * (keel::pow(u, -k) - 1.0) / k;
        }
    }
    return out;
}

std::tuple<keel::f64, keel::f64> gpstat(keel::f64 k, keel::f64 sigma, keel::f64 theta)
{
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    if (sigma <= 0.0) return std::make_tuple(nan, nan);
    if (!(k < 1.0))
        return std::make_tuple(keel::Limits<keel::f64>::infinity(), nan);
    const keel::f64 mean = theta + sigma / (1.0 - k);
    if (!(k < 0.5))
        return std::make_tuple(mean, keel::Limits<keel::f64>::infinity());
    const keel::f64 var = sigma * sigma / ((1.0 - k) * (1.0 - k) * (1.0 - 2.0 * k));
    return std::make_tuple(mean, var);
}

} // namespace numkit::stats
