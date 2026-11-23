// toolboxes/stats/src/test/ad_dw.cpp
//
// Anderson-Darling normality test (adtest) and Durbin-Watson autocorrelation
// test (dwtest), delegating numerical kernels to keel::stats.

#include <keel/arena.h>
#include <keel/tuple.h>
#include <keel/math.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>

#include <numkit/stats/test/hypothesis.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


#include "ad_dw_detail.hpp"

namespace numkit::stats {

namespace {

void adtest_argcheck(keel::f64 alpha)
{
    if (alpha <= 0.0 || alpha >= 1.0)
        throw Error("adtest: alpha must be in (0, 1)",
                    0, 0, "adtest", "", "numkit:adtest:badAlpha");
}

void dwtest_argcheck(const Value &r, const Value &X)
{
    const keel::usize n = r.numel();
    if (n < 3)
        throw Error("dwtest: need at least 3 residuals",
                    0, 0, "dwtest", "", "numkit:dwtest:tooFewObs");
    const keel::usize k = X.dims().cols();
    if (X.dims().rows() != n)
        throw Error("dwtest: rows(X) must equal length(r)",
                    0, 0, "dwtest", "", "numkit:dwtest:shapeMismatch");
    if (k >= n)
        throw Error("dwtest: design matrix has no degrees of freedom",
                    0, 0, "dwtest", "", "numkit:dwtest:noDOF");
}

} // namespace

keel::Tuple<Value, Value, Value, Value>
adtest(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    adtest_argcheck(alpha);
    const keel::usize N = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(N, 0.0, scratch.allocator());
    keel::usize finite_count = 0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        xv[i] = xi;
        if (!keel::is_nan(xi)) ++finite_count;
    }
    if (finite_count < 4)
        throw Error("adtest: need at least 4 non-NaN observations",
                    0, 0, "adtest", "", "numkit:adtest:tooFewObs");

    keel::stats::AdTestResult res{};
    const auto st = keel::stats::adtest(xv.as_ref(), alpha, res, scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("adtest: sample has zero variance",
                    0, 0, "adtest", "", "numkit:adtest:zeroVar");

    return {
        Value::scalar(keel::as<keel::f64>(res.h), mr),
        Value::scalar(res.p, mr),
        Value::scalar(res.adstat, mr),
        Value::scalar(res.cv, mr),
    };
}

keel::f64 dwStatAndPLeft(const Value &r, const Value &X, bool exact,
                         keel::f64 &dwOut, keel::Allocator *mr)
{
    dwtest_argcheck(r, X);
    const keel::usize n = r.numel();
    const keel::usize k = X.dims().cols();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> rv(n, 0.0, scratch.allocator());
    keel::f64 den = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        rv[i] = r.elemAsDouble(i);
        den += rv[i] * rv[i];
    }
    if (!(den > 0.0))
        throw Error("dwtest: residuals are all zero",
                    0, 0, "dwtest", "", "numkit:dwtest:zeroResid");

    keel::Array<keel::f64> xv(n * k, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n * k; ++i) xv[i] = X.elemAsDouble(i);

    keel::stats::DwTestResult res{};
    const auto st = keel::stats::dwtest(rv.as_ref(), xv.as_ref(), n, k, exact, res,
                                        scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("dwtest: design matrix X is rank-deficient (X''X singular)",
                    0, 0, "dwtest", "", "numkit:dwtest:singular");

    dwOut = res.dw;
    return res.p_left;
}

keel::Tuple<Value, Value>
dwtest(const Value &r, const Value &X, keel::Allocator *mr)
{
    keel::f64 dw = 0.0;
    const keel::f64 pLeft = dwStatAndPLeft(r, X, true, dw, mr);
    const keel::f64 pBoth = keel::min(1.0, 2.0 * keel::min(pLeft, 1.0 - pLeft));
    return {Value::scalar(pBoth, mr), Value::scalar(dw, mr)};
}

} // namespace numkit::stats
