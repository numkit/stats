// toolboxes/stats/src/distributions/poisson.cpp

#include <numkit/stats/distributions/poisson.hpp>
#include <numkit/stats/distributions/gamma_dist.hpp>
#include <numkit/stats/distributions/binomial.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <random>

#include "poisson_detail.hpp"

namespace numkit::stats {

namespace {

double samplePoisson1(::numkit::ops::RngContext &rng, double lam, std::pmr::memory_resource *mr)
{
    if (!(lam > 0.0))
        return (lam == 0.0) ? 0.0 : std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(lam))
        return std::numeric_limits<double>::quiet_NaN();
    double k = 0.0;
    while (lam >= 15.0) {
        const double m = std::floor(7.0 * lam / 8.0);
        const double x = randg(rng, m, 1, 1, mr).toScalar();
        if (x <= lam) {
            k += m;
            lam -= x;
        } else {
            if (m > 1.0) {
                k += binornd(rng, static_cast<int>(m - 1.0), lam / x, 1, 1, mr).toScalar();
            }
            return k;
        }
    }
    double t = -std::log(rng.nextUniform());
    while (t <= lam) {
        k += 1.0;
        t -= std::log(rng.nextUniform());
    }
    return k;
}

} // namespace

Value poisspdf(const Value &k, double lambda, std::pmr::memory_resource *mr)
{
    if (lambda < 0.0)
        return elementwise(k, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    return elementwise(k, [=](double ki){ return poiss_pmf(ki, lambda); }, mr);
}

Value poisscdf(const Value &k, double lambda, std::pmr::memory_resource *mr)
{
    if (lambda < 0.0)
        return elementwise(k, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    if (lambda == 0.0)
        return elementwise(k, [](double ki){ return ki >= 0.0 ? 1.0 : 0.0; }, mr);
    // Build a vector x = floor(k) + 1, run gammainc(lambda, x), then F = 1 - that.
    Value xs = elementwise(k, [=](double ki) {
        if (ki < 0.0) return 0.0;        // clamp; gammainc(lambda, 0) sentinel
        return std::floor(ki) + 1.0;
    }, mr);
    Value lam = Value::scalar(lambda, mr);
    Value lower = ::numkit::builtin::gammainc(lam, xs, mr);
    Value out;
    const auto &d = k.dims();
    if (k.isScalar()) {
        const double ki = k.toScalar();
        if (ki < 0.0) return Value::scalar(0.0, mr);
        return Value::scalar(1.0 - lower.elemAsDouble(0), mr);
    }
    if (d.is3D()) out = Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    else          out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    const size_t n = k.numel();
    if (n == 0) return out;
    double *od = out.doubleDataMut();
    for (size_t i = 0; i < n; ++i) {
        const double ki = k.elemAsDouble(i);
        if (ki < 0.0) { od[i] = 0.0; continue; }
        od[i] = 1.0 - lower.elemAsDouble(i);
    }
    return out;
}


Value poissinv(const Value &p, double lambda, std::pmr::memory_resource *mr)
{
    if (lambda < 0.0)
        return elementwise(p, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    return elementwise(p, [=](double pi){ return poiss_inv_scalar(pi, lambda); }, mr);
}

Value poissrnd(::numkit::ops::RngContext &rng, double lambda, size_t rows, size_t cols, std::pmr::memory_resource *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const size_t n = rows * cols;
    if (lambda < 0.0 || std::isnan(lambda)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (size_t i = 0; i < n; ++i) od[i] = nan;
        return out;
    }
    if (lambda == 0.0) {
        for (size_t i = 0; i < n; ++i) od[i] = 0.0;
        return out;
    }
    for (size_t i = 0; i < n; ++i) od[i] = samplePoisson1(rng, lambda, mr);
    return out;
}

std::tuple<double, double> poisstat(double lambda)
{
    if (lambda <= 0.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    return std::make_tuple(lambda, lambda);
}

} // namespace numkit::stats
