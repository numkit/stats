// toolboxes/stats/src/fit/fit.cpp
// Distribution MLE fitters and negative log-likelihoods (thin Value wrappers over keel::stats).

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/stats.h>
#include <numkit/stats/fit/fit.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "fit_detail.hpp"

namespace numkit::stats {

namespace {

keel::Array<keel::f64> extractDoubles(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, 0.0, alloc);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

void binofit_argcheck(keel::usize Nx, keel::usize Nn, bool scalarN)
{
    if (!scalarN && Nn != Nx)
        throw Error("binofit: x and n must be the same length",
                    0, 0, "binofit", "", "numkit:binofit:size");
}

void checkCensFreqLen(const Value &cens, const Value &freq, keel::usize n, const char *fn)
{
    if (cens.numel() > 0 && cens.numel() != n)
        throw Error(keel::format("{}: censoring must match the data length", fn),
                    0, 0, fn, "", keel::format("numkit:{}:cens", fn));
    if (freq.numel() > 0 && freq.numel() != n)
        throw Error(keel::format("{}: freq must match the data length", fn),
                    0, 0, fn, "", keel::format("numkit:{}:freq", fn));
}

} // namespace

keel::Tuple<Value, Value, Value, Value>
normfit(const Value &x, keel::f64 alpha, const Value &cens, const Value &freq, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    auto fail = [&]() {
        return keel::make_tuple(Value::scalar(nan, mr),
                                Value::scalar(nan, mr),
                                rowCI(nan, nan, mr),
                                rowCI(nan, nan, mr));
    };
    if (N == 0) return fail();
    if (!cens.isEmpty() && cens.numel() != N) return fail();
    if (!freq.isEmpty() && freq.numel() != N) return fail();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    keel::Array<keel::u8> cv(scratch.allocator());
    if (!cens.isEmpty()) {
        cv.resize(N);
        for (keel::usize i = 0; i < N; ++i) cv[i] = cens.elemAsDouble(i) > 0.5 ? 1 : 0;
    }
    keel::Array<keel::f64> fv(scratch.allocator());
    if (!freq.isEmpty()) fv = extractDoubles(freq, scratch.allocator());

    keel::stats::Fit2Result R{};
    if (keel::stats::normfit(xv.as_ref(), cv.as_ref(), fv.as_ref(), alpha, R) != keel::stats::StatsStatus::kOk)
        return fail();
    return keel::make_tuple(Value::scalar(R.p1, mr),
                            Value::scalar(R.p2, mr),
                            rowCI(R.ci1_lo, R.ci1_hi, mr),
                            rowCI(R.ci2_lo, R.ci2_hi, mr));
}

keel::Tuple<Value, Value, Value, Value>
normfit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return normfit(x, alpha, Value::Empty, Value::Empty, mr);
}

keel::Tuple<Value, Value>
poissfit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    keel::stats::Fit1Result R{};
    if (keel::stats::poissfit(xv.as_ref(), alpha, R) != keel::stats::StatsStatus::kOk)
        return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    return {Value::scalar(R.p1, mr), rowCI(R.ci_lo, R.ci_hi, mr)};
}

keel::Tuple<Value, Value>
expfit(const Value &x, keel::f64 alpha, const Value &cens, const Value &freq, keel::Allocator *mr)
{
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    const keel::usize N = x.numel();
    if (N == 0 || (!cens.isEmpty() && cens.numel() != N) || (!freq.isEmpty() && freq.numel() != N))
        return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    keel::Array<keel::f64> cv(scratch.allocator()), fv(scratch.allocator());
    if (!cens.isEmpty()) cv = extractDoubles(cens, scratch.allocator());
    if (!freq.isEmpty()) fv = extractDoubles(freq, scratch.allocator());
    keel::stats::Fit1Result R{};
    if (keel::stats::expfit(xv.as_ref(), cv.as_ref(), fv.as_ref(), alpha, R) != keel::stats::StatsStatus::kOk)
        return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    return {Value::scalar(R.p1, mr), rowCI(R.ci_lo, R.ci_hi, mr)};
}

keel::Tuple<Value, Value, Value, Value>
unifit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    keel::stats::Fit2Result R{};
    if (keel::stats::unifit(xv.as_ref(), alpha, R) != keel::stats::StatsStatus::kOk)
        return {Value::scalar(nan, mr), Value::scalar(nan, mr),
                rowCI(nan, nan, mr), rowCI(nan, nan, mr)};
    return {Value::scalar(R.p1, mr), Value::scalar(R.p2, mr),
            rowCI(R.ci1_lo, R.ci1_hi, mr), rowCI(R.ci2_lo, R.ci2_hi, mr)};
}

keel::Tuple<Value, Value>
lognfit(const Value &x, keel::f64 alpha, const Value &cens, const Value &freq, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    Value parm = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    Value pci  = Value::matrix(2, 2, ValueType::DOUBLE, mr);
    keel::f64 *pd = parm.doubleDataMut();
    keel::f64 *cd = pci.doubleDataMut();
    auto fail = [&]() {
        for (keel::i32 i = 0; i < 2; ++i) pd[i] = nan;
        for (keel::i32 i = 0; i < 4; ++i) cd[i] = nan;
        return keel::make_tuple(keel::move(parm), keel::move(pci));
    };
    if (N < 2) return fail();
    if (!cens.isEmpty() && cens.numel() != N) return fail();
    if (!freq.isEmpty() && freq.numel() != N) return fail();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> y(N, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (!(xi > 0.0)) return fail();
        y[i] = keel::log(xi);
    }
    keel::Array<keel::u8> cv(scratch.allocator());
    if (!cens.isEmpty()) {
        cv.resize(N);
        for (keel::usize i = 0; i < N; ++i) cv[i] = cens.elemAsDouble(i) > 0.5 ? 1 : 0;
    }
    keel::Array<keel::f64> fv(scratch.allocator());
    if (!freq.isEmpty()) fv = extractDoubles(freq, scratch.allocator());

    keel::stats::Fit2Result R{};
    if (keel::stats::normfit(y.as_ref(), cv.as_ref(), fv.as_ref(), alpha, R) != keel::stats::StatsStatus::kOk)
        return fail();
    pd[0] = R.p1;     pd[1] = R.p2;
    cd[0] = R.ci1_lo; cd[1] = R.ci1_hi;
    cd[2] = R.ci2_lo; cd[3] = R.ci2_hi;
    return {keel::move(parm), keel::move(pci)};
}

keel::Tuple<Value, Value>
lognfit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return lognfit(x, alpha, Value::Empty, Value::Empty, mr);
}

keel::Tuple<Value, Value>
binofit(const Value &x, const Value &n, keel::f64 alpha, keel::Allocator *mr)
{
    const keel::usize Nx = x.numel();
    const keel::usize Nn = n.numel();
    const bool scalarN = (Nn == 1);
    binofit_argcheck(Nx, Nn, scalarN);

    Value phat = Value::matrix(Nx, 1, ValueType::DOUBLE, mr);
    Value pci  = Value::matrix(Nx, 2, ValueType::DOUBLE, mr);
    keel::f64 *pd = phat.doubleDataMut();
    keel::f64 *cd = pci.doubleDataMut();
    for (keel::usize i = 0; i < Nx; ++i) {
        const keel::f64 k = x.elemAsDouble(i);
        const keel::f64 N = scalarN ? n.elemAsDouble(0) : n.elemAsDouble(i);
        keel::stats::Fit1Result R{};
        (void)keel::stats::binofit_scalar(k, N, alpha, R);
        pd[i] = R.p1;
        cd[i] = R.ci_lo;
        cd[i + Nx] = R.ci_hi;
    }
    if (Nx == 1) phat = Value::scalar(pd[0], mr);
    return {keel::move(phat), keel::move(pci)};
}

keel::Tuple<Value, Value>
raylfit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    keel::stats::Fit1Result R{};
    if (keel::stats::raylfit(xv.as_ref(), alpha, R) != keel::stats::StatsStatus::kOk)
        return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    return {Value::scalar(R.p1, mr), rowCI(R.ci_lo, R.ci_hi, mr)};
}

// ── Negative log-likelihoods ──────────────────────────────────────────

keel::f64 normlike(keel::f64 mu, keel::f64 sigma, const Value &x, const Value &cens, const Value &freq, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    if (N == 0) return 0.0;
    if (!(sigma > 0.0)) return keel::Limits<keel::f64>::quiet_nan();
    checkCensFreqLen(cens, freq, N, "normlike");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    auto cv = extractDoubles(cens, scratch.allocator());
    auto fv = extractDoubles(freq, scratch.allocator());
    return keel::stats::normlike(mu, sigma, xv.as_ref(), cv.as_ref(), fv.as_ref());
}

keel::f64 explike(keel::f64 mu, const Value &x, keel::Allocator * /*mr*/)
{
    return explike_full(mu, x, Value::Empty, Value::Empty, nullptr);
}

keel::f64 explike_full(keel::f64 mu, const Value &x, const Value &cens, const Value &freq, keel::f64 *avarOut)
{
    const keel::usize N = x.numel();
    if (N > 0 && mu > 0.0) checkCensFreqLen(cens, freq, N, "explike");
    keel::ArenaTemp scratch = keel::get_scratch(nullptr);
    auto xv = extractDoubles(x, scratch.allocator());
    auto cv = extractDoubles(cens, scratch.allocator());
    auto fv = extractDoubles(freq, scratch.allocator());
    return keel::stats::explike(mu, xv.as_ref(), cv.as_ref(), fv.as_ref(), avarOut);
}

keel::f64 lognlike(keel::f64 mu, keel::f64 sigma, const Value &x, keel::Allocator * /*mr*/)
{
    return lognlike_full(mu, sigma, x, Value::Empty, Value::Empty, nullptr);
}

keel::f64 lognlike_full(keel::f64 mu, keel::f64 sigma, const Value &x, const Value &cens, const Value &freq, keel::f64 *avarOut)
{
    const keel::usize N = x.numel();
    if (N > 0 && sigma > 0.0) checkCensFreqLen(cens, freq, N, "lognlike");
    keel::ArenaTemp scratch = keel::get_scratch(nullptr);
    auto xv = extractDoubles(x, scratch.allocator());
    auto cv = extractDoubles(cens, scratch.allocator());
    auto fv = extractDoubles(freq, scratch.allocator());
    return keel::stats::lognlike(mu, sigma, xv.as_ref(), cv.as_ref(), fv.as_ref(), avarOut);
}

keel::f64 gamlike(keel::f64 a, keel::f64 b, const Value &x, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    return keel::stats::gamlike(a, b, xv.as_ref());
}

keel::f64 betalike(keel::f64 a, keel::f64 b, const Value &x, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    return keel::stats::betalike(a, b, xv.as_ref());
}

keel::f64 wbllike(keel::f64 scale, keel::f64 shape, const Value &x, keel::Allocator *mr)
{
    return wbllike_full(scale, shape, x, Value::Empty, Value::Empty, mr);
}

keel::f64 wbllike_full(keel::f64 scale, keel::f64 shape, const Value &x, const Value &cens, const Value &freq, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    if (N > 0 && scale > 0.0 && shape > 0.0) checkCensFreqLen(cens, freq, N, "wbllike");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    auto cv = extractDoubles(cens, scratch.allocator());
    auto fv = extractDoubles(freq, scratch.allocator());
    return keel::stats::wbllike(scale, shape, xv.as_ref(), cv.as_ref(), fv.as_ref());
}

keel::f64 evlike(keel::f64 mu, keel::f64 sigma, const Value &x, keel::Allocator *mr)
{
    return evlike_full(mu, sigma, x, Value::Empty, Value::Empty, mr);
}

keel::f64 evlike_full(keel::f64 mu, keel::f64 sigma, const Value &x, const Value &cens, const Value &freq, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    if (N > 0 && sigma > 0.0) checkCensFreqLen(cens, freq, N, "evlike");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    auto cv = extractDoubles(cens, scratch.allocator());
    auto fv = extractDoubles(freq, scratch.allocator());
    return keel::stats::evlike(mu, sigma, xv.as_ref(), cv.as_ref(), fv.as_ref());
}

keel::f64 gevlike(keel::f64 k, keel::f64 sigma, keel::f64 mu, const Value &x, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    return keel::stats::gevlike(k, sigma, mu, xv.as_ref());
}

keel::f64 gplike(keel::f64 k, keel::f64 sigma, const Value &x, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = extractDoubles(x, scratch.allocator());
    return keel::stats::gplike(k, sigma, xv.as_ref());
}

} // namespace numkit::stats
