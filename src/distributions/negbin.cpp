// toolboxes/stats/src/distributions/negbin.cpp

#include <numkit/stats/distributions/negbin.hpp>
#include <numkit/stats/distributions/gamma_dist.hpp>
#include <numkit/stats/distributions/poisson.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "dist_helpers.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <random>

#include "negbin_detail.hpp"

namespace numkit::stats {


Value nbinpdf(const Value &k, double r, double p, std::pmr::memory_resource *mr)
{
    if (r <= 0.0 || p <= 0.0 || p > 1.0)
        return elementwise(k, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    return elementwise(k, [=](double ki){ return nbin_pmf(ki, r, p); }, mr);
}

Value nbincdf(const Value &k, double r, double p, std::pmr::memory_resource *mr)
{
    if (r <= 0.0 || p <= 0.0 || p > 1.0)
        return elementwise(k, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    return elementwise(k, [=](double ki){ return nbin_cdf_scalar(ki, r, p, mr); }, mr);
}

Value nbininv(const Value &q, double r, double p, std::pmr::memory_resource *mr)
{
    if (!nbin_params_ok(r, p))
        return elementwise(q, [](double){ return std::numeric_limits<double>::quiet_NaN(); }, mr);
    return elementwise(q, [=](double qi) { return nbin_inv_scalar(qi, r, p); }, mr);
}

Value nbinrnd(::numkit::ops::RngContext &rng, double r, double p, size_t rows, size_t cols, std::pmr::memory_resource *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (r <= 0.0 || p <= 0.0 || p > 1.0 || rows * cols == 0) return out;
    double *od = out.doubleDataMut();
    const size_t cnt = rows * cols;
    if (p == 1.0) {
        for (size_t i = 0; i < cnt; ++i) od[i] = 0.0;
        return out;
    }
    // Sample via Gamma-Poisson mixture matching the reference engine's nbinrnd:
    // lambda ~ Gamma(r, (1-p)/p), K | lambda ~ Poisson(lambda).
    Value g = gamrnd(rng, r, (1.0 - p) / p, rows, cols, mr);
    const double *gd = g.doubleData();
    for (size_t i = 0; i < cnt; ++i) {
        const double lam = gd[i];
        od[i] = (lam <= 0.0) ? 0.0 : poissrnd(rng, lam, 1, 1, mr).toScalar();
    }
    return out;
}

std::tuple<double, double> nbinstat(double r, double p)
{
    if (r <= 0.0 || p <= 0.0 || p > 1.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(nan, nan);
    }
    const double mean = r * (1.0 - p) / p;
    const double var  = mean / p;
    return std::make_tuple(mean, var);
}

} // namespace numkit::stats
