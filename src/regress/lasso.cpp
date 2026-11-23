// toolboxes/stats/src/regress/lasso.cpp
//
// LASSO / elastic-net (lasso) + regularized GLM (lassoglm) delegating
// coordinate descent and IRLS kernels to keel::stats.

#include <keel/arena.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/regress/regress.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


#include "lasso_detail.hpp"

namespace numkit::stats {

namespace {

keel::stats::GlmFamily toKeelFamily(GlmDistribution d) noexcept
{
    switch (d) {
        case GlmDistribution::Normal:   return keel::stats::GlmFamily::kNormal;
        case GlmDistribution::Binomial: return keel::stats::GlmFamily::kBinomial;
        case GlmDistribution::Poisson:  return keel::stats::GlmFamily::kPoisson;
        default:                        return keel::stats::GlmFamily::kNormal;
    }
}

} // namespace

LassoResult lasso(const Value &X, const Value &y, const Value &lambdas,
                  keel::f64 alpha, keel::Allocator *mr)
{
    if (alpha < 0.0 || alpha > 1.0)
        throw Error("lasso: alpha must be in [0, 1]", 0, 0, "lasso", "", "numkit:lasso:badAlpha");
    const keel::usize n = X.dims().rows();
    const keel::usize p = X.dims().cols();
    if (y.numel() != n)
        throw Error("lasso: length(y) must equal rows(X)", 0, 0, "lasso", "", "numkit:lasso:shapeMismatch");
    const keel::usize nL = lambdas.numel();
    if (nL == 0)
        throw Error("lasso: lambdas must be a non-empty vector", 0, 0, "lasso", "", "numkit:lasso:emptyLambda");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(n * p, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n * p; ++i) xv[i] = X.elemAsDouble(i);
    keel::Array<keel::f64> yv(n, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) yv[i] = y.elemAsDouble(i);
    keel::Array<keel::f64> lv(nL, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < nL; ++i) lv[i] = lambdas.elemAsDouble(i);

    auto Bout = Value::matrix(p, nL, ValueType::DOUBLE, mr);
    auto intOut = Value::matrix(1, nL, ValueType::DOUBLE, mr);
    auto lambdaOut = Value::matrix(1, nL, ValueType::DOUBLE, mr);
    keel::copy_n(lambdaOut.doubleDataMut(), lv.data(), nL);

    (void)keel::stats::lasso(xv.as_ref(), n, p, yv.as_ref(), lv.as_ref(), alpha,
                             keel::MutArrayRef<keel::f64>(Bout.doubleDataMut(), p * nL),
                             keel::MutArrayRef<keel::f64>(intOut.doubleDataMut(), nL),
                             scratch.allocator());
    return {keel::move(Bout), keel::move(intOut), keel::move(lambdaOut)};
}

LassoResult lassoglm(const Value &X, const Value &y, GlmDistribution distr,
                     const Value &lambdas, keel::f64 alpha, keel::Allocator *mr)
{
    if (alpha < 0.0 || alpha > 1.0)
        throw Error("lassoglm: alpha must be in [0, 1]", 0, 0, "lassoglm", "", "numkit:lassoglm:badAlpha");
    const keel::usize n = X.dims().rows();
    const keel::usize p = X.dims().cols();
    if (y.numel() != n)
        throw Error("lassoglm: length(y) must equal rows(X)", 0, 0, "lassoglm", "", "numkit:lassoglm:shapeMismatch");
    const keel::usize nL = lambdas.numel();
    if (nL == 0)
        throw Error("lassoglm: lambdas must be a non-empty vector", 0, 0, "lassoglm", "", "numkit:lassoglm:emptyLambda");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(n * p, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n * p; ++i) xv[i] = X.elemAsDouble(i);
    keel::Array<keel::f64> yv(n, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) yv[i] = y.elemAsDouble(i);
    keel::Array<keel::f64> lv(nL, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < nL; ++i) lv[i] = lambdas.elemAsDouble(i);

    auto Bout = Value::matrix(p, nL, ValueType::DOUBLE, mr);
    auto intOut = Value::matrix(1, nL, ValueType::DOUBLE, mr);
    auto lambdaOut = Value::matrix(1, nL, ValueType::DOUBLE, mr);
    keel::copy_n(lambdaOut.doubleDataMut(), lv.data(), nL);

    (void)keel::stats::lassoglm(xv.as_ref(), n, p, yv.as_ref(), toKeelFamily(distr), lv.as_ref(),
                                alpha,
                                keel::MutArrayRef<keel::f64>(Bout.doubleDataMut(), p * nL),
                                keel::MutArrayRef<keel::f64>(intOut.doubleDataMut(), nL),
                                scratch.allocator());
    return {keel::move(Bout), keel::move(intOut), keel::move(lambdaOut)};
}

} // namespace numkit::stats
