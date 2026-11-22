// toolboxes/stats/src/test/hypothesis.cpp — delegated to keel::stats

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/stats.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/test/hypothesis.hpp>

#include <numkit/stats/distributions/binomial.hpp>
#include <numkit/stats/distributions/chi2.hpp>
#include <numkit/stats/distributions/fisher_f.hpp>
#include <numkit/stats/distributions/normal.hpp>
#include <numkit/stats/distributions/students_t.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <limits>
#include <tuple>
#include <utility>

#include "hypothesis_detail.hpp"

namespace numkit::stats {

namespace {

keel::stats::TestTail to_keel_tail(TestTail tail) noexcept
{
    switch (tail) {
        case TestTail::Both:  return keel::stats::TestTail::kBoth;
        case TestTail::Right: return keel::stats::TestTail::kRight;
        case TestTail::Left:  return keel::stats::TestTail::kLeft;
    }
    return keel::stats::TestTail::kBoth;
}

keel::Array<keel::f64> to_flat(const Value &x, keel::Allocator *alloc)
{
    const keel::usize n = x.numel();
    keel::Array<keel::f64> out(n, alloc);
    for (keel::usize i = 0; i < n; ++i)
        out[i] = x.elemAsDouble(i);
    return out;
}

void ttest_argcheck(const Value &x)
{
    if (x.numel() < 2)
        throw Error("ttest: need at least 2 samples", 0, 0, "ttest", "",
                    "numkit:ttest:nsamples");
}

void ttest2_argcheck(const Value &x, const Value &y)
{
    if (x.numel() < 2 || y.numel() < 2)
        throw Error("ttest2: need at least 2 samples per group",
                    0, 0, "ttest2", "", "numkit:ttest2:nsamples");
}

void ztest_argcheck(keel::f64 sigma)
{
    if (sigma <= 0.0)
        throw Error("ztest: sigma must be positive", 0, 0, "ztest", "",
                    "numkit:ztest:badsigma");
}

void vartest_argcheck(keel::f64 v)
{
    if (v <= 0.0)
        throw Error("vartest: v must be positive", 0, 0, "vartest", "",
                    "numkit:vartest:badv");
}

void vartest2_argcheck(const Value &x, const Value &y)
{
    if (x.numel() < 2 || y.numel() < 2)
        throw Error("vartest2: need >= 2 samples per group and var(y) > 0",
                    0, 0, "vartest2", "", "numkit:vartest2:nsamples");
}

void kstest_argcheck(const Value &x)
{
    if (x.numel() < 1)
        throw Error("kstest: empty sample", 0, 0, "kstest", "",
                    "numkit:kstest:nsamples");
}

Value extractCol(const Value &x, keel::usize c, keel::Allocator *mr)
{
    const keel::usize R = x.dims().rows();
    Value out = Value::matrix(R, 1, ValueType::DOUBLE, mr);
    keel::f64 *d = out.doubleDataMut();
    for (keel::usize r = 0; r < R; ++r)
        d[r] = x.elemAsDouble(r + c * R);
    return out;
}

Value extractRow(const Value &x, keel::usize r, keel::Allocator *mr)
{
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    Value out = Value::matrix(1, C, ValueType::DOUBLE, mr);
    keel::f64 *d = out.doubleDataMut();
    for (keel::usize c = 0; c < C; ++c)
        d[c] = x.elemAsDouble(r + c * R);
    return out;
}

std::tuple<Value, Value, Value, Value>
ttest_1d(const Value &x, keel::f64 m, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    ttest_argcheck(x);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    keel::stats::TTestResult res{};
    if (keel::stats::ttest_1sample(xv.as_ref(), m, alpha, to_keel_tail(tail), res) !=
        keel::stats::StatsStatus::kOk)
        throw Error("ttest: need at least 2 samples", 0, 0, "ttest", "",
                    "numkit:ttest:nsamples");

    Value ci = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    ci.doubleDataMut()[0] = res.ci_lo;
    ci.doubleDataMut()[1] = res.ci_hi;

    Value stats = Value::structure(mr);
    stats.field("tstat") = Value::scalar(res.tstat, mr);
    stats.field("df")    = Value::scalar(res.df, mr);
    stats.field("sd")    = Value::scalar(res.sd, mr);

    return std::make_tuple(Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.p, mr),
                           std::move(ci),
                           std::move(stats));
}

std::tuple<Value, Value, Value, Value>
ttest2_1d(const Value &x, const Value &y, keel::f64 alpha, TestTail tail, keel::StringRef vartype,
          keel::Allocator *mr)
{
    ttest2_argcheck(x, y);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    auto yv = to_flat(y, scratch.allocator());
    const bool equal_var = (vartype == "equal");
    keel::stats::TTest2Result res{};
    if (keel::stats::ttest_2sample(xv.as_ref(), yv.as_ref(), alpha, to_keel_tail(tail),
                                   equal_var, res) != keel::stats::StatsStatus::kOk)
        throw Error("ttest2: need at least 2 samples per group",
                    0, 0, "ttest2", "", "numkit:ttest2:nsamples");

    Value ci = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    ci.doubleDataMut()[0] = res.ci_lo;
    ci.doubleDataMut()[1] = res.ci_hi;

    Value sdVal;
    if (equal_var) {
        sdVal = Value::scalar(res.sd_pooled, mr);
    } else {
        sdVal = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        sdVal.doubleDataMut()[0] = res.sd_x;
        sdVal.doubleDataMut()[1] = res.sd_y;
    }

    Value stats = Value::structure(mr);
    stats.field("tstat") = Value::scalar(res.tstat, mr);
    stats.field("df")    = Value::scalar(res.df, mr);
    stats.field("sd")    = std::move(sdVal);

    return std::make_tuple(Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.p, mr),
                           std::move(ci),
                           std::move(stats));
}

std::tuple<Value, Value, Value, Value>
ztest_1d(const Value &x, keel::f64 m, keel::f64 sigma, keel::f64 alpha, TestTail tail,
         keel::Allocator *mr)
{
    ztest_argcheck(sigma);
    if (x.numel() < 1)
        throw Error("ztest: need at least 1 sample", 0, 0, "ztest", "",
                    "numkit:ztest:nsamples");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    keel::stats::ZTestResult res{};
    (void)keel::stats::ztest_1sample(xv.as_ref(), m, sigma, alpha, to_keel_tail(tail), res);

    Value ci = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    ci.doubleDataMut()[0] = res.ci_lo;
    ci.doubleDataMut()[1] = res.ci_hi;

    return std::make_tuple(Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.p, mr),
                           std::move(ci),
                           Value::scalar(res.zval, mr));
}

std::tuple<Value, Value, Value, Value>
vartest_1d(const Value &x, keel::f64 v, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    vartest_argcheck(v);
    if (x.numel() < 2)
        throw Error("vartest: need at least 2 samples", 0, 0, "vartest", "",
                    "numkit:vartest:nsamples");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    keel::stats::VarTestResult res{};
    (void)keel::stats::vartest_1sample(xv.as_ref(), v, alpha, to_keel_tail(tail), res);

    Value ci = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    ci.doubleDataMut()[0] = res.ci_lo;
    ci.doubleDataMut()[1] = res.ci_hi;

    Value stats = Value::structure(mr);
    stats.field("chisqstat") = Value::scalar(res.chisqstat, mr);
    stats.field("df")        = Value::scalar(res.df, mr);

    return std::make_tuple(Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.p, mr),
                           std::move(ci),
                           std::move(stats));
}

std::tuple<Value, Value, Value, Value>
vartest2_1d(const Value &x, const Value &y, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    vartest2_argcheck(x, y);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    auto yv = to_flat(y, scratch.allocator());
    keel::stats::VarTest2Result res{};
    if (keel::stats::vartest_2sample(xv.as_ref(), yv.as_ref(), alpha, to_keel_tail(tail), res) !=
        keel::stats::StatsStatus::kOk)
        throw Error("vartest2: need >= 2 samples per group and var(y) > 0",
                    0, 0, "vartest2", "", "numkit:vartest2:nsamples");

    Value ci = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    ci.doubleDataMut()[0] = res.ci_lo;
    ci.doubleDataMut()[1] = res.ci_hi;

    Value stats = Value::structure(mr);
    stats.field("fstat") = Value::scalar(res.fstat, mr);
    stats.field("df1")   = Value::scalar(res.df1, mr);
    stats.field("df2")   = Value::scalar(res.df2, mr);

    return std::make_tuple(Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.p, mr),
                           std::move(ci),
                           std::move(stats));
}

} // namespace

// ════════════════════════════════════════════════════════════════════
// ttest — one-sample
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
ttest_dim(const Value &x, keel::f64 m, keel::f64 alpha, TestTail tail, keel::i32 dim,
          keel::Allocator *mr)
{
    ttest_argcheck(x);
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    const bool isVec = (R <= 1 || C <= 1);
    if (isVec) {
        const keel::i32 nsDim = (R == 1 && C > 1) ? 2 : 1;
        if (dim == 0 || dim == nsDim)
            return ttest_1d(x, m, alpha, tail, mr);
    }
    const keel::i32 effDim = (dim == 0) ? 1 : dim;
    if (effDim == 1) {
        Value hOut     = Value::matrix(1, C, ValueType::DOUBLE, mr);
        Value pOut     = Value::matrix(1, C, ValueType::DOUBLE, mr);
        Value ciOut    = Value::matrix(2, C, ValueType::DOUBLE, mr);
        Value tstatOut = Value::matrix(1, C, ValueType::DOUBLE, mr);
        Value dfOut    = Value::matrix(1, C, ValueType::DOUBLE, mr);
        Value sdOut    = Value::matrix(1, C, ValueType::DOUBLE, mr);
        for (keel::usize c = 0; c < C; ++c) {
            auto [hc, pc, cic, stc] = ttest_1d(extractCol(x, c, mr), m, alpha, tail, mr);
            hOut.doubleDataMut()[c]        = hc.toScalar();
            pOut.doubleDataMut()[c]        = pc.toScalar();
            ciOut.doubleDataMut()[0 + 2*c] = cic.elemAsDouble(0);
            ciOut.doubleDataMut()[1 + 2*c] = cic.elemAsDouble(1);
            tstatOut.doubleDataMut()[c]    = stc.field("tstat").toScalar();
            dfOut.doubleDataMut()[c]       = stc.field("df").toScalar();
            sdOut.doubleDataMut()[c]       = stc.field("sd").toScalar();
        }
        Value stats = Value::structure(mr);
        stats.field("tstat") = std::move(tstatOut);
        stats.field("df")    = std::move(dfOut);
        stats.field("sd")    = std::move(sdOut);
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(stats));
    }
    if (effDim == 2) {
        Value hOut     = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        Value pOut     = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        Value ciOut    = Value::matrix(R, 2, ValueType::DOUBLE, mr);
        Value tstatOut = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        Value dfOut    = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        Value sdOut    = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        for (keel::usize r = 0; r < R; ++r) {
            auto [hr, pr, cir, str] = ttest_1d(extractRow(x, r, mr), m, alpha, tail, mr);
            hOut.doubleDataMut()[r]     = hr.toScalar();
            pOut.doubleDataMut()[r]     = pr.toScalar();
            ciOut.doubleDataMut()[r]    = cir.elemAsDouble(0);
            ciOut.doubleDataMut()[r+R]  = cir.elemAsDouble(1);
            tstatOut.doubleDataMut()[r] = str.field("tstat").toScalar();
            dfOut.doubleDataMut()[r]    = str.field("df").toScalar();
            sdOut.doubleDataMut()[r]    = str.field("sd").toScalar();
        }
        Value stats = Value::structure(mr);
        stats.field("tstat") = std::move(tstatOut);
        stats.field("df")    = std::move(dfOut);
        stats.field("sd")    = std::move(sdOut);
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(stats));
    }
    throw Error("ttest: need at least 2 samples along operating dimension",
                0, 0, "ttest", "", "numkit:ttest:nsamples");
}

std::tuple<Value, Value, Value, Value>
ttest(const Value &x, keel::f64 m, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    ttest_argcheck(x);
    return ttest_dim(x, m, alpha, tail, 0, mr);
}

// ════════════════════════════════════════════════════════════════════
// ttest2 — two-sample
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
ttest2_dim(const Value &x, const Value &y, keel::f64 alpha, TestTail tail,
           keel::StringRef vartype, keel::i32 dim, keel::Allocator *mr)
{
    ttest2_argcheck(x, y);
    const keel::usize Rx = x.dims().rows(), Cx = x.dims().cols();
    const keel::usize Ry = y.dims().rows(), Cy = y.dims().cols();
    const bool isVec = (Rx <= 1 || Cx <= 1) && (Ry <= 1 || Cy <= 1);
    if (isVec) {
        const keel::i32 nsDim = (Rx == 1 && Cx > 1) ? 2 : 1;
        if (dim == 0 || dim == nsDim)
            return ttest2_1d(x, y, alpha, tail, vartype, mr);
    }
    const keel::i32 effDim = (dim == 0) ? 1 : dim;
    const bool equalVar = (vartype == "equal");
    if (effDim == 1) {
        if (Cx != Cy)
            throw Error("ttest2: X and Y must have the same number of columns along Dim=1",
                        0, 0, "ttest2", "", "numkit:ttest2:size");
        Value hOut     = Value::matrix(1, Cx, ValueType::DOUBLE, mr);
        Value pOut     = Value::matrix(1, Cx, ValueType::DOUBLE, mr);
        Value ciOut    = Value::matrix(2, Cx, ValueType::DOUBLE, mr);
        Value tstatOut = Value::matrix(1, Cx, ValueType::DOUBLE, mr);
        Value dfOut    = Value::matrix(1, Cx, ValueType::DOUBLE, mr);
        Value sdOut    = Value::matrix(equalVar ? 1 : 2, Cx, ValueType::DOUBLE, mr);
        for (keel::usize c = 0; c < Cx; ++c) {
            auto [hc, pc, cic, stc] =
                ttest2_1d(extractCol(x, c, mr), extractCol(y, c, mr), alpha, tail, vartype, mr);
            hOut.doubleDataMut()[c]        = hc.toScalar();
            pOut.doubleDataMut()[c]        = pc.toScalar();
            ciOut.doubleDataMut()[0 + 2*c] = cic.elemAsDouble(0);
            ciOut.doubleDataMut()[1 + 2*c] = cic.elemAsDouble(1);
            tstatOut.doubleDataMut()[c]    = stc.field("tstat").toScalar();
            dfOut.doubleDataMut()[c]       = stc.field("df").toScalar();
            const Value &sdc = stc.field("sd");
            if (equalVar) {
                sdOut.doubleDataMut()[c] = sdc.toScalar();
            } else {
                sdOut.doubleDataMut()[0 + 2*c] = sdc.elemAsDouble(0);
                sdOut.doubleDataMut()[1 + 2*c] = sdc.elemAsDouble(1);
            }
        }
        Value stats = Value::structure(mr);
        stats.field("tstat") = std::move(tstatOut);
        stats.field("df")    = std::move(dfOut);
        stats.field("sd")    = std::move(sdOut);
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(stats));
    }
    if (effDim == 2) {
        if (Rx != Ry)
            throw Error("ttest2: X and Y must have the same number of rows along Dim=2",
                        0, 0, "ttest2", "", "numkit:ttest2:size");
        Value hOut     = Value::matrix(Rx, 1, ValueType::DOUBLE, mr);
        Value pOut     = Value::matrix(Rx, 1, ValueType::DOUBLE, mr);
        Value ciOut    = Value::matrix(Rx, 2, ValueType::DOUBLE, mr);
        Value tstatOut = Value::matrix(Rx, 1, ValueType::DOUBLE, mr);
        Value dfOut    = Value::matrix(Rx, 1, ValueType::DOUBLE, mr);
        Value sdOut    = Value::matrix(Rx, equalVar ? 1 : 2, ValueType::DOUBLE, mr);
        for (keel::usize r = 0; r < Rx; ++r) {
            auto [hr, pr, cir, str] =
                ttest2_1d(extractRow(x, r, mr), extractRow(y, r, mr), alpha, tail, vartype, mr);
            hOut.doubleDataMut()[r]       = hr.toScalar();
            pOut.doubleDataMut()[r]       = pr.toScalar();
            ciOut.doubleDataMut()[r]      = cir.elemAsDouble(0);
            ciOut.doubleDataMut()[r + Rx] = cir.elemAsDouble(1);
            tstatOut.doubleDataMut()[r]   = str.field("tstat").toScalar();
            dfOut.doubleDataMut()[r]      = str.field("df").toScalar();
            const Value &sdr = str.field("sd");
            if (equalVar) {
                sdOut.doubleDataMut()[r] = sdr.toScalar();
            } else {
                sdOut.doubleDataMut()[r]      = sdr.elemAsDouble(0);
                sdOut.doubleDataMut()[r + Rx] = sdr.elemAsDouble(1);
            }
        }
        Value stats = Value::structure(mr);
        stats.field("tstat") = std::move(tstatOut);
        stats.field("df")    = std::move(dfOut);
        stats.field("sd")    = std::move(sdOut);
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(stats));
    }
    throw Error("ttest2: need at least 2 samples per group",
                0, 0, "ttest2", "", "numkit:ttest2:nsamples");
}

std::tuple<Value, Value, Value, Value>
ttest2(const Value &x, const Value &y, keel::f64 alpha, TestTail tail,
       keel::StringRef vartype, keel::Allocator *mr)
{
    ttest2_argcheck(x, y);
    return ttest2_dim(x, y, alpha, tail, vartype, 0, mr);
}

// ════════════════════════════════════════════════════════════════════
// ztest — known-σ
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
ztest_dim(const Value &x, keel::f64 m, keel::f64 sigma, keel::f64 alpha, TestTail tail,
          keel::i32 dim, keel::Allocator *mr)
{
    ztest_argcheck(sigma);
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    const bool isVec = (R <= 1 || C <= 1);
    if (isVec) {
        const keel::i32 nsDim = (R == 1 && C > 1) ? 2 : 1;
        if (dim == 0 || dim == nsDim)
            return ztest_1d(x, m, sigma, alpha, tail, mr);
    }
    const keel::i32 effDim = (dim == 0) ? 1 : dim;
    if (effDim == 1) {
        Value hOut  = Value::matrix(1, C, ValueType::DOUBLE, mr);
        Value pOut  = Value::matrix(1, C, ValueType::DOUBLE, mr);
        Value ciOut = Value::matrix(2, C, ValueType::DOUBLE, mr);
        Value zOut  = Value::matrix(1, C, ValueType::DOUBLE, mr);
        for (keel::usize c = 0; c < C; ++c) {
            auto [hc, pc, cic, zc] = ztest_1d(extractCol(x, c, mr), m, sigma, alpha, tail, mr);
            hOut.doubleDataMut()[c]        = hc.toScalar();
            pOut.doubleDataMut()[c]        = pc.toScalar();
            ciOut.doubleDataMut()[0 + 2*c] = cic.elemAsDouble(0);
            ciOut.doubleDataMut()[1 + 2*c] = cic.elemAsDouble(1);
            zOut.doubleDataMut()[c]        = zc.toScalar();
        }
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(zOut));
    }
    if (effDim == 2) {
        Value hOut  = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        Value pOut  = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        Value ciOut = Value::matrix(R, 2, ValueType::DOUBLE, mr);
        Value zOut  = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        for (keel::usize r = 0; r < R; ++r) {
            auto [hr, pr, cir, zr] = ztest_1d(extractRow(x, r, mr), m, sigma, alpha, tail, mr);
            hOut.doubleDataMut()[r]      = hr.toScalar();
            pOut.doubleDataMut()[r]      = pr.toScalar();
            ciOut.doubleDataMut()[r]     = cir.elemAsDouble(0);
            ciOut.doubleDataMut()[r + R] = cir.elemAsDouble(1);
            zOut.doubleDataMut()[r]      = zr.toScalar();
        }
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(zOut));
    }
    throw Error("ztest: unsupported dimension", 0, 0, "ztest", "", "numkit:ztest:dim");
}

std::tuple<Value, Value, Value, Value>
ztest(const Value &x, keel::f64 m, keel::f64 sigma, keel::f64 alpha, TestTail tail,
      keel::Allocator *mr)
{
    ztest_argcheck(sigma);
    return ztest_dim(x, m, sigma, alpha, tail, 0, mr);
}

// ════════════════════════════════════════════════════════════════════
// vartest — chi-squared one-sample
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
vartest_dim(const Value &x, keel::f64 v, keel::f64 alpha, TestTail tail, keel::i32 dim,
            keel::Allocator *mr)
{
    vartest_argcheck(v);
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    const bool isVec = (R <= 1 || C <= 1);
    if (isVec) {
        const keel::i32 nsDim = (R == 1 && C > 1) ? 2 : 1;
        if (dim == 0 || dim == nsDim)
            return vartest_1d(x, v, alpha, tail, mr);
    }
    const keel::i32 effDim = (dim == 0) ? 1 : dim;
    if (effDim == 1) {
        Value hOut     = Value::matrix(1, C, ValueType::DOUBLE, mr);
        Value pOut     = Value::matrix(1, C, ValueType::DOUBLE, mr);
        Value ciOut    = Value::matrix(2, C, ValueType::DOUBLE, mr);
        Value chisqOut = Value::matrix(1, C, ValueType::DOUBLE, mr);
        Value dfOut    = Value::matrix(1, C, ValueType::DOUBLE, mr);
        for (keel::usize c = 0; c < C; ++c) {
            auto [hc, pc, cic, stc] = vartest_1d(extractCol(x, c, mr), v, alpha, tail, mr);
            hOut.doubleDataMut()[c]        = hc.toScalar();
            pOut.doubleDataMut()[c]        = pc.toScalar();
            ciOut.doubleDataMut()[0 + 2*c] = cic.elemAsDouble(0);
            ciOut.doubleDataMut()[1 + 2*c] = cic.elemAsDouble(1);
            chisqOut.doubleDataMut()[c]    = stc.field("chisqstat").toScalar();
            dfOut.doubleDataMut()[c]       = stc.field("df").toScalar();
        }
        Value stats = Value::structure(mr);
        stats.field("chisqstat") = std::move(chisqOut);
        stats.field("df")        = std::move(dfOut);
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(stats));
    }
    if (effDim == 2) {
        Value hOut     = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        Value pOut     = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        Value ciOut    = Value::matrix(R, 2, ValueType::DOUBLE, mr);
        Value chisqOut = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        Value dfOut    = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        for (keel::usize r = 0; r < R; ++r) {
            auto [hr, pr, cir, str] = vartest_1d(extractRow(x, r, mr), v, alpha, tail, mr);
            hOut.doubleDataMut()[r]      = hr.toScalar();
            pOut.doubleDataMut()[r]      = pr.toScalar();
            ciOut.doubleDataMut()[r]     = cir.elemAsDouble(0);
            ciOut.doubleDataMut()[r + R] = cir.elemAsDouble(1);
            chisqOut.doubleDataMut()[r]  = str.field("chisqstat").toScalar();
            dfOut.doubleDataMut()[r]     = str.field("df").toScalar();
        }
        Value stats = Value::structure(mr);
        stats.field("chisqstat") = std::move(chisqOut);
        stats.field("df")        = std::move(dfOut);
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(stats));
    }
    throw Error("vartest: need at least 2 samples", 0, 0, "vartest", "",
                "numkit:vartest:nsamples");
}

std::tuple<Value, Value, Value, Value>
vartest(const Value &x, keel::f64 v, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    vartest_argcheck(v);
    return vartest_dim(x, v, alpha, tail, 0, mr);
}

// ════════════════════════════════════════════════════════════════════
// vartest2 — F-test for equal variances
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
vartest2_dim(const Value &x, const Value &y, keel::f64 alpha, TestTail tail, keel::i32 dim,
             keel::Allocator *mr)
{
    vartest2_argcheck(x, y);
    const keel::usize Rx = x.dims().rows(), Cx = x.dims().cols();
    const keel::usize Ry = y.dims().rows(), Cy = y.dims().cols();
    const bool isVec = (Rx <= 1 || Cx <= 1) && (Ry <= 1 || Cy <= 1);
    if (isVec) {
        const keel::i32 nsDim = (Rx == 1 && Cx > 1) ? 2 : 1;
        if (dim == 0 || dim == nsDim)
            return vartest2_1d(x, y, alpha, tail, mr);
    }
    const keel::i32 effDim = (dim == 0) ? 1 : dim;
    if (effDim == 1) {
        if (Cx != Cy)
            throw Error("vartest2: X and Y must have the same number of columns along Dim=1",
                        0, 0, "vartest2", "", "numkit:vartest2:size");
        Value hOut   = Value::matrix(1, Cx, ValueType::DOUBLE, mr);
        Value pOut   = Value::matrix(1, Cx, ValueType::DOUBLE, mr);
        Value ciOut  = Value::matrix(2, Cx, ValueType::DOUBLE, mr);
        Value fOut   = Value::matrix(1, Cx, ValueType::DOUBLE, mr);
        Value df1Out = Value::matrix(1, Cx, ValueType::DOUBLE, mr);
        Value df2Out = Value::matrix(1, Cx, ValueType::DOUBLE, mr);
        for (keel::usize c = 0; c < Cx; ++c) {
            auto [hc, pc, cic, stc] =
                vartest2_1d(extractCol(x, c, mr), extractCol(y, c, mr), alpha, tail, mr);
            hOut.doubleDataMut()[c]        = hc.toScalar();
            pOut.doubleDataMut()[c]        = pc.toScalar();
            ciOut.doubleDataMut()[0 + 2*c] = cic.elemAsDouble(0);
            ciOut.doubleDataMut()[1 + 2*c] = cic.elemAsDouble(1);
            fOut.doubleDataMut()[c]        = stc.field("fstat").toScalar();
            df1Out.doubleDataMut()[c]      = stc.field("df1").toScalar();
            df2Out.doubleDataMut()[c]      = stc.field("df2").toScalar();
        }
        Value stats = Value::structure(mr);
        stats.field("fstat") = std::move(fOut);
        stats.field("df1")   = std::move(df1Out);
        stats.field("df2")   = std::move(df2Out);
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(stats));
    }
    if (effDim == 2) {
        if (Rx != Ry)
            throw Error("vartest2: X and Y must have the same number of rows along Dim=2",
                        0, 0, "vartest2", "", "numkit:vartest2:size");
        Value hOut   = Value::matrix(Rx, 1, ValueType::DOUBLE, mr);
        Value pOut   = Value::matrix(Rx, 1, ValueType::DOUBLE, mr);
        Value ciOut  = Value::matrix(Rx, 2, ValueType::DOUBLE, mr);
        Value fOut   = Value::matrix(Rx, 1, ValueType::DOUBLE, mr);
        Value df1Out = Value::matrix(Rx, 1, ValueType::DOUBLE, mr);
        Value df2Out = Value::matrix(Rx, 1, ValueType::DOUBLE, mr);
        for (keel::usize r = 0; r < Rx; ++r) {
            auto [hr, pr, cir, str] =
                vartest2_1d(extractRow(x, r, mr), extractRow(y, r, mr), alpha, tail, mr);
            hOut.doubleDataMut()[r]       = hr.toScalar();
            pOut.doubleDataMut()[r]       = pr.toScalar();
            ciOut.doubleDataMut()[r]      = cir.elemAsDouble(0);
            ciOut.doubleDataMut()[r + Rx] = cir.elemAsDouble(1);
            fOut.doubleDataMut()[r]       = str.field("fstat").toScalar();
            df1Out.doubleDataMut()[r]     = str.field("df1").toScalar();
            df2Out.doubleDataMut()[r]     = str.field("df2").toScalar();
        }
        Value stats = Value::structure(mr);
        stats.field("fstat") = std::move(fOut);
        stats.field("df1")   = std::move(df1Out);
        stats.field("df2")   = std::move(df2Out);
        return std::make_tuple(std::move(hOut), std::move(pOut), std::move(ciOut),
                               std::move(stats));
    }
    throw Error("vartest2: need >= 2 samples per group and var(y) > 0",
                0, 0, "vartest2", "", "numkit:vartest2:nsamples");
}

std::tuple<Value, Value, Value, Value>
vartest2(const Value &x, const Value &y, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    vartest2_argcheck(x, y);
    return vartest2_dim(x, y, alpha, tail, 0, mr);
}

// ════════════════════════════════════════════════════════════════════
// kstest / kstest2 — Kolmogorov-Smirnov
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
kstest(const Value &x, const Value &cdf, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    kstest_argcheck(x);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());

    keel::Array<keel::f64> xg(scratch.allocator());
    keel::Array<keel::f64> fg(scratch.allocator());
    if (cdf.numel() >= 2) {
        const keel::usize rows = cdf.dims().rows();
        xg.resize(rows);
        fg.resize(rows);
        for (keel::usize i = 0; i < rows; ++i) {
            xg[i] = cdf.elemAsDouble(0 * rows + i);
            fg[i] = cdf.elemAsDouble(1 * rows + i);
        }
    }

    keel::stats::KsTestResult res{};
    (void)keel::stats::kstest_1sample(xv.as_ref(), xg.as_ref(), fg.as_ref(), alpha,
                                      to_keel_tail(tail), res, scratch.allocator());

    return std::make_tuple(Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.p, mr),
                           Value::scalar(res.ksstat, mr),
                           Value::scalar(res.cv, mr));
}

std::tuple<Value, Value, Value, Value>
kstest2(const Value &x, const Value &y, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    if (x.numel() < 1 || y.numel() < 1)
        throw Error("kstest2: empty sample", 0, 0, "kstest2", "",
                    "numkit:kstest2:nsamples");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    auto yv = to_flat(y, scratch.allocator());
    keel::stats::KsTestResult res{};
    (void)keel::stats::kstest_2sample(xv.as_ref(), yv.as_ref(), alpha, to_keel_tail(tail),
                                      res, scratch.allocator());

    return std::make_tuple(Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.p, mr),
                           Value::scalar(res.ksstat, mr),
                           Value::scalar(res.cv, mr));
}

// ════════════════════════════════════════════════════════════════════
// jbtest — Jarque-Bera normality
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
jbtest(const Value &x, keel::f64 alpha, keel::f64 mctol, keel::Allocator *mr)
{
    if (x.numel() < 4)
        throw Error("jbtest: need at least 4 samples", 0, 0, "jbtest", "",
                    "numkit:jbtest:nsamples");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    keel::stats::JbTestResult res{};
    (void)keel::stats::jbtest(xv.as_ref(), alpha, mctol, res, scratch.allocator());

    return std::make_tuple(Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.p, mr),
                           Value::scalar(res.jbstat, mr),
                           Value::scalar(res.cv, mr));
}

std::tuple<Value, Value, Value, Value>
jbtest(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return jbtest(x, alpha, std::numeric_limits<keel::f64>::quiet_NaN(), mr);
}

// ════════════════════════════════════════════════════════════════════
// signtest — non-parametric sign test
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value>
signtest(const Value &x, const Value &y_or_m, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    const keel::usize Nx = x.numel();
    const bool paired = (!y_or_m.isEmpty() && !y_or_m.isScalar());
    if (paired && y_or_m.numel() != Nx)
        throw Error("signtest: X and Y must have the same length",
                    0, 0, "signtest", "", "numkit:signtest:size");
    const keel::f64 m0 = (paired || y_or_m.isEmpty()) ? 0.0 : y_or_m.toScalar();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    keel::Array<keel::f64> yv(scratch.allocator());
    if (paired) yv = to_flat(y_or_m, scratch.allocator());

    keel::stats::SignTestResult res{};
    (void)keel::stats::signtest(xv.as_ref(), yv.as_ref(), m0, alpha, to_keel_tail(tail), res);

    return std::make_tuple(Value::scalar(res.p, mr),
                           Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(keel::f64(res.sign), mr));
}

// ════════════════════════════════════════════════════════════════════
// fishertest — Fisher's exact 2×2 contingency
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value, Value>
fishertest(const Value &T, keel::f64 alpha, TestTail tail, keel::Allocator *mr)
{
    if (T.dims().rows() != 2 || T.dims().cols() != 2)
        throw Error("fishertest: T must be a 2×2 matrix",
                    0, 0, "fishertest", "", "numkit:fishertest:size");
    const keel::f64 a = T.elemAsDouble(0);  // (1,1)
    const keel::f64 c = T.elemAsDouble(1);  // (2,1)
    const keel::f64 b = T.elemAsDouble(2);  // (1,2)
    const keel::f64 d = T.elemAsDouble(3);  // (2,2)

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::stats::FisherTestResult res{};
    (void)keel::stats::fishertest(a, b, c, d, alpha, to_keel_tail(tail), res, scratch.allocator());

    return std::make_tuple(Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.p, mr),
                           Value::scalar(res.odds_ratio, mr),
                           Value::scalar(res.ci_lo, mr),
                           Value::scalar(res.ci_hi, mr));
}

// ════════════════════════════════════════════════════════════════════
// chi2gof — chi-squared goodness-of-fit (frequency form)
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
chi2gof(const Value &observed, const Value &expected, keel::i32 nparams, keel::f64 alpha,
        keel::Allocator *mr)
{
    if (expected.numel() != observed.numel())
        throw Error("chi2gof: observed and expected must have same length",
                    0, 0, "chi2gof", "", "numkit:chi2gof:size");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto ov = to_flat(observed, scratch.allocator());
    auto ev = to_flat(expected, scratch.allocator());
    keel::stats::Chi2GofResult res{};
    (void)keel::stats::chi2gof_freq(ov.as_ref(), ev.as_ref(), nparams, alpha, res);

    return std::make_tuple(Value::scalar(res.p, mr),
                           Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.chi2stat, mr),
                           Value::scalar(res.df, mr));
}

// ════════════════════════════════════════════════════════════════════
// vartestn — k-sample variance equality test (5 variants)
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
vartestn_full(const Value &x, const Value &group, keel::i32 test, keel::Allocator *mr)
{
    const keel::usize Nx = x.numel();
    if (Nx == 0 || group.numel() != Nx)
        throw Error("vartestn: x and group must be same length",
                    0, 0, "vartestn", "", "numkit:vartestn:size");

    keel::stats::VarTestNMethod method = keel::stats::VarTestNMethod::kBartlett;
    switch (test) {
        case 1: method = keel::stats::VarTestNMethod::kLeveneQuadratic; break;
        case 2: method = keel::stats::VarTestNMethod::kLeveneAbsolute;  break;
        case 3: method = keel::stats::VarTestNMethod::kBrownForsythe;   break;
        case 4: method = keel::stats::VarTestNMethod::kOBrien;          break;
        default: method = keel::stats::VarTestNMethod::kBartlett;       break;
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    auto gv = to_flat(group, scratch.allocator());
    keel::stats::VarTestNResult res{};
    (void)keel::stats::vartestn(xv.as_ref(), gv.as_ref(), method, res, scratch.allocator());

    return std::make_tuple(Value::scalar(res.p, mr),
                           Value::scalar(res.stat, mr),
                           Value::scalar(res.df1, mr),
                           Value::scalar(res.df2, mr));
}

std::tuple<Value, Value, Value>
vartestn(const Value &x, const Value &group, keel::f64 /*alpha*/, keel::Allocator *mr)
{
    auto [p, stat, df1, df2] = vartestn_full(x, group, /*Bartlett*/ 0, mr);
    (void)df2;
    return std::make_tuple(std::move(p), std::move(stat), std::move(df1));
}

// ════════════════════════════════════════════════════════════════════
// runstest — Wald-Wolfowitz runs test for randomness
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value, Value, Value>
runstest(const Value &x, keel::f64 v_in, keel::f64 alpha, TestTail tail,
         keel::StringRef method_in, keel::Allocator *mr)
{
    keel::String method = keel::String::from(method_in);
    for (auto &c : method) c = keel::to_ascii_lower(c);
    const bool exact = (method != "approximate");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    keel::stats::RunsTestResult res{};
    (void)keel::stats::runstest(xv.as_ref(), v_in, alpha, to_keel_tail(tail), exact, res,
                                scratch.allocator());

    return std::make_tuple(Value::scalar(res.p, mr),
                           Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(keel::f64(res.nruns), mr),
                           Value::scalar(keel::f64(res.n1), mr),
                           Value::scalar(keel::f64(res.n0), mr),
                           Value::scalar(res.zval, mr));
}

// ════════════════════════════════════════════════════════════════════
// ranksum — Wilcoxon rank-sum (Mann-Whitney U)
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
ranksum(const Value &x, const Value &y, keel::f64 alpha, TestTail tail,
        keel::StringRef method_in, keel::Allocator *mr)
{
    keel::String method = keel::String::from(method_in);
    for (auto &c : method) c = keel::to_ascii_lower(c);
    keel::i32 method_mode = 0;
    if (method == "exact")            method_mode = 1;
    else if (method == "approximate") method_mode = 2;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    auto yv = to_flat(y, scratch.allocator());
    keel::stats::RankTestResult res{};
    (void)keel::stats::ranksum(xv.as_ref(), yv.as_ref(), alpha, to_keel_tail(tail),
                               method_mode, res, scratch.allocator());

    return std::make_tuple(Value::scalar(res.p, mr),
                           Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.stat, mr),
                           Value::scalar(res.zval, mr));
}

// ════════════════════════════════════════════════════════════════════
// signrank — Wilcoxon signed-rank
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
signrank(const Value &x, const Value &y_or_m, keel::f64 alpha, TestTail tail,
         keel::StringRef method_in, keel::Allocator *mr)
{
    const keel::usize Nx = x.numel();
    const bool paired = (!y_or_m.isEmpty() && !y_or_m.isScalar());
    if (paired && y_or_m.numel() != Nx)
        throw Error("signrank: X and Y must have the same length",
                    0, 0, "signrank", "", "numkit:signrank:size");
    const keel::f64 m0 = (paired || y_or_m.isEmpty()) ? 0.0 : y_or_m.toScalar();

    keel::String method = keel::String::from(method_in);
    for (auto &c : method) c = keel::to_ascii_lower(c);
    keel::i32 method_mode = 0;
    if (method == "exact")            method_mode = 1;
    else if (method == "approximate") method_mode = 2;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    keel::Array<keel::f64> yv(scratch.allocator());
    if (paired) yv = to_flat(y_or_m, scratch.allocator());

    keel::stats::RankTestResult res{};
    (void)keel::stats::signrank(xv.as_ref(), yv.as_ref(), m0, alpha, to_keel_tail(tail),
                                method_mode, res, scratch.allocator());

    return std::make_tuple(Value::scalar(res.p, mr),
                           Value::scalar(keel::f64(res.h), mr),
                           Value::scalar(res.stat, mr),
                           Value::scalar(res.zval, mr));
}

// ════════════════════════════════════════════════════════════════════
// ansaribradley — Ansari-Bradley two-sample scale (dispersion) test
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
ansaribradley(const Value &x, const Value &y, keel::f64 alpha, TestTail tail,
              keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(x, scratch.allocator());
    auto yv = to_flat(y, scratch.allocator());
    keel::stats::RankTestResult res{};
    (void)keel::stats::ansaribradley(xv.as_ref(), yv.as_ref(), alpha, to_keel_tail(tail),
                                     res, scratch.allocator());

    return std::make_tuple(Value::scalar(keel::as<keel::f64>(res.h), mr),
                           Value::scalar(res.p, mr),
                           Value::scalar(res.stat, mr),
                           Value::scalar(res.zval, mr));
}

// ════════════════════════════════════════════════════════════════════
// lillietest — Lilliefors normality test
// ════════════════════════════════════════════════════════════════════

std::tuple<Value, Value, Value, Value>
lillietest(const Value &X, keel::f64 alpha, keel::Allocator *mr)
{
    if (X.numel() < 4)
        throw Error("lillietest: sample size must be >= 4",
                    0, 0, "lillietest", "", "numkit:lillietest:nsamples");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw Error("lillietest: alpha must be in (0, 1)",
                    0, 0, "lillietest", "", "numkit:lillietest:alpha");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = to_flat(X, scratch.allocator());
    keel::stats::LillieTestResult res{};
    const auto st = keel::stats::lillietest(xv.as_ref(), alpha, res, scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("lillietest: sample has zero variance",
                    0, 0, "lillietest", "", "numkit:lillietest:zeroVar");

    return std::make_tuple(Value::scalar(keel::as<keel::f64>(res.h), mr),
                           Value::scalar(res.p, mr),
                           Value::scalar(res.kstat, mr),
                           Value::scalar(res.cv, mr));
}

} // namespace numkit::stats
