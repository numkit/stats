// toolboxes/stats/src/regress/glm.cpp
//
// Generalized linear model fit + prediction (glmfit / glmval),
// delegating IRLS numerical kernels to keel::stats.

#include <keel/arena.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>

#include <numkit/stats/regress/regress.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::stats {

namespace {

keel::stats::GlmDist toKeelDist(GlmDistribution d) noexcept
{
    switch (d) {
        case GlmDistribution::Normal:          return keel::stats::GlmDist::kNormal;
        case GlmDistribution::Binomial:        return keel::stats::GlmDist::kBinomial;
        case GlmDistribution::Poisson:         return keel::stats::GlmDist::kPoisson;
        case GlmDistribution::Gamma:           return keel::stats::GlmDist::kGamma;
        case GlmDistribution::InverseGaussian: return keel::stats::GlmDist::kInverseGaussian;
    }
    return keel::stats::GlmDist::kNormal;
}

keel::stats::GlmLink toKeelLink(GlmLink l) noexcept
{
    switch (l) {
        case GlmLink::Identity:   return keel::stats::GlmLink::kIdentity;
        case GlmLink::Logit:      return keel::stats::GlmLink::kLogit;
        case GlmLink::Log:        return keel::stats::GlmLink::kLog;
        case GlmLink::Reciprocal: return keel::stats::GlmLink::kReciprocal;
        case GlmLink::Probit:     return keel::stats::GlmLink::kProbit;
    }
    return keel::stats::GlmLink::kIdentity;
}

void glmfit_argcheck(const Value &X, keel::usize n, keel::usize p)
{
    if (X.dims().rows() != n)
        throw Error("glmfit: rows(X) must equal length(y)",
                    0, 0, "glmfit", "", "numkit:glmfit:shapeMismatch");
    if (n <= p)
        throw Error("glmfit: need rows(X) > cols(X) + 1",
                    0, 0, "glmfit", "", "numkit:glmfit:noDOF");
}

} // namespace

GlmfitResult glmfit(const Value &X, const Value &y,
                    GlmDistribution distr, GlmLink link,
                    keel::Allocator *mr)
{
    const keel::usize n = y.numel();
    const keel::usize p_in = X.dims().cols();
    const keel::usize p = p_in + 1;
    glmfit_argcheck(X, n, p);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(n * p_in, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n * p_in; ++i) xv[i] = X.elemAsDouble(i);
    keel::Array<keel::f64> yv(n, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) yv[i] = y.elemAsDouble(i);

    keel::stats::GlmFitOut fit(scratch.allocator());
    (void)keel::stats::glmfit(xv.as_ref(), yv.as_ref(), n, p_in,
                              toKeelDist(distr), toKeelLink(link),
                              fit, scratch.allocator());

    Value bv        = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value covb      = Value::matrix(p, p, ValueType::DOUBLE, mr);
    Value se        = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value coeffcorr = Value::matrix(p, p, ValueType::DOUBLE, mr);
    Value tStat     = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value resid     = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    Value wts       = Value::matrix(n, 1, ValueType::DOUBLE, mr);

    keel::copy_n(bv.doubleDataMut(), fit.beta.data(), p);
    keel::copy_n(covb.doubleDataMut(), fit.covb.data(), p * p);
    keel::copy_n(se.doubleDataMut(), fit.se.data(), p);
    keel::copy_n(coeffcorr.doubleDataMut(), fit.coeffcorr.data(), p * p);
    keel::copy_n(tStat.doubleDataMut(), fit.t_stat.data(), p);
    keel::copy_n(resid.doubleDataMut(), fit.resid.data(), n);
    keel::copy_n(wts.doubleDataMut(), fit.wts.data(), n);

    Value stats = Value::structure(mr);
    stats.field("beta")      = bv;
    stats.field("dfe")       = Value::scalar(fit.dfe, mr);
    stats.field("sfit")      = Value::scalar(fit.sfit, mr);
    stats.field("s")         = Value::scalar(fit.s, mr);
    stats.field("estdisp")   = Value::scalar(fit.estdisp ? 1.0 : 0.0, mr);
    stats.field("covb")      = keel::move(covb);
    stats.field("se")        = keel::move(se);
    stats.field("coeffcorr") = keel::move(coeffcorr);
    stats.field("t")         = keel::move(tStat);
    stats.field("resid")     = keel::move(resid);
    stats.field("wts")       = keel::move(wts);

    return {keel::move(bv), Value::scalar(fit.deviance, mr), keel::move(stats)};
}

Value glmval(const Value &b, const Value &X, GlmLink link, keel::Allocator *mr)
{
    const keel::usize p = b.numel();
    const keel::usize m = X.dims().rows();
    const keel::usize p_in = X.dims().cols();
    if (p != p_in + 1)
        throw Error("glmval: length(b) must equal cols(X) + 1",
                    0, 0, "glmval", "", "numkit:glmval:shapeMismatch");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> bv(p, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < p; ++i) bv[i] = b.elemAsDouble(i);
    keel::Array<keel::f64> xv(m * p_in, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < m * p_in; ++i) xv[i] = X.elemAsDouble(i);

    Value out = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    (void)keel::stats::glmval(bv.as_ref(), xv.as_ref(), m, p_in, toKeelLink(link),
                              keel::MutArrayRef<keel::f64>(out.doubleDataMut(), m));
    return out;
}

} // namespace numkit::stats
