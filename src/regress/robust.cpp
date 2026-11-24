// toolboxes/stats/src/regress/robust.cpp
// Thin Value wrappers over keel::stats::robustfit and keel::stats::robustcov.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/regress/regress.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::stats {

namespace {

keel::stats::RobustWeight toKeelWeight(RobustWeight wt) noexcept
{
    switch (wt) {
        case RobustWeight::Andrews:  return keel::stats::RobustWeight::kAndrews;
        case RobustWeight::Bisquare: return keel::stats::RobustWeight::kBisquare;
        case RobustWeight::Cauchy:   return keel::stats::RobustWeight::kCauchy;
        case RobustWeight::Fair:     return keel::stats::RobustWeight::kFair;
        case RobustWeight::Huber:    return keel::stats::RobustWeight::kHuber;
        case RobustWeight::Logistic: return keel::stats::RobustWeight::kLogistic;
        case RobustWeight::Ols:      return keel::stats::RobustWeight::kOls;
        case RobustWeight::Talwar:   return keel::stats::RobustWeight::kTalwar;
        case RobustWeight::Welsch:   return keel::stats::RobustWeight::kWelsch;
    }
    return keel::stats::RobustWeight::kBisquare;
}

void robustfit_argcheck(const Value &X, const Value &y, keel::usize n, keel::usize p)
{
    (void)X;
    if (y.numel() != n)
        throw Error("robustfit: length(y) must equal rows(X)",
                    0, 0, "robustfit", "", "numkit:robustfit:shapeMismatch");
    if (p == 0 || n <= p)
        throw Error("robustfit: need rows(X) > cols(X)",
                    0, 0, "robustfit", "", "numkit:robustfit:noDOF");
}

} // namespace

RobustfitResult robustfit(const Value &X, const Value &y,
                           RobustWeight weight, keel::f64 tune,
                           keel::Allocator *mr)
{
    const keel::usize n = X.dims().rows();
    const keel::usize p = X.dims().cols();
    robustfit_argcheck(X, y, n, p);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> Xv(n * p, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n * p; ++i) Xv[i] = X.elemAsDouble(i);
    keel::Array<keel::f64> yv(n, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) yv[i] = y.elemAsDouble(i);

    auto bv = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    keel::f64 s = 1.0;
    (void)keel::stats::robustfit(Xv.as_ref(), n, p, yv.as_ref(),
                                 toKeelWeight(weight), tune,
                                 keel::MutArrayRef<keel::f64>(bv.doubleDataMut(), p), s,
                                 scratch.allocator());
    return { keel::move(bv), Value::scalar(s, mr) };
}

RobustcovResult robustcov(const Value &X, keel::Allocator *mr)
{
    const keel::usize n = X.dims().rows();
    const keel::usize d = X.dims().cols();
    if (n <= d + 1)
        throw Error("robustcov: need n > d + 1 observations",
                    0, 0, "robustcov", "", "numkit:robustcov:noDOF");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> Xv(n * d, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n * d; ++i) Xv[i] = X.elemAsDouble(i);

    auto sigma = Value::matrix(d, d, ValueType::DOUBLE, mr);
    auto muV   = Value::matrix(1, d, ValueType::DOUBLE, mr);
    (void)keel::stats::robustcov(Xv.as_ref(), n, d,
                                 keel::MutArrayRef<keel::f64>(sigma.doubleDataMut(), d * d),
                                 keel::MutArrayRef<keel::f64>(muV.doubleDataMut(), d),
                                 scratch.allocator());
    return { keel::move(sigma), keel::move(muV) };
}

} // namespace numkit::stats
