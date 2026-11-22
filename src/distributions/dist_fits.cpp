// toolboxes/stats/src/distributions/dist_fits.cpp
//
// MLE / moment fitters for continuous + count distributions (delegated to keel::stats):
//   gamfit  — Gamma(a, b) shape + scale
//   wblfit  — Weibull(scale, shape)
//   betafit — Beta(a, b)
//   nbinfit — NegBin(r, p)
//   evfit   — Type-I EV (Gumbel-min)
//   gpfit   — Generalised Pareto
//   gevfit  — Generalised Extreme Value

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/distributions/beta.hpp>
#include <numkit/stats/distributions/extreme_value.hpp>
#include <numkit/stats/distributions/gamma_dist.hpp>
#include <numkit/stats/distributions/gev.hpp>
#include <numkit/stats/distributions/gp.hpp>
#include <numkit/stats/distributions/negbin.hpp>
#include <numkit/stats/distributions/weibull.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <cmath>

namespace numkit::stats {

namespace {

keel::Array<keel::f64> to_flat(const Value &x)
{
    keel::Array<keel::f64> v(x.numel());
    for (keel::usize i = 0; i < v.size(); ++i) v[i] = x.elemAsDouble(i);
    return v;
}

Value make_fit2_params(const keel::stats::Fit2Result &r, keel::Allocator *mr)
{
    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = r.p1;
    out.doubleDataMut()[1] = r.p2;
    return out;
}

Value make_fit2_ci(const keel::stats::Fit2Result &r, keel::Allocator *mr)
{
    auto out = Value::matrix(2, 2, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    od[0] = r.ci1_lo;
    od[1] = r.ci1_hi;
    od[2] = r.ci2_lo;
    od[3] = r.ci2_hi;
    return out;
}

keel::stats::Fit2Result run_gamfit(const Value &x, const Value &censoring, const Value &freq,
                                   keel::f64 alpha, bool compute_ci)
{
    auto xv = to_flat(x);
    const keel::usize n = xv.size();
    if (n < 2)
        throw Error("gamfit: need at least 2 observations",
                    0, 0, "gamfit", "", "numkit:gamfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!(v > 0.0))
            throw Error("gamfit: all observations must be positive",
                        0, 0, "gamfit", "", "numkit:gamfit:notPositive");
    }
    keel::Array<keel::f64> cens, wf;
    if (!censoring.isEmpty()) {
        if (censoring.numel() != n)
            throw Error("gamfit: censoring length mismatch",
                        0, 0, "gamfit", "", "numkit:gamfit:censLen");
        cens.resize(n);
        for (keel::usize i = 0; i < n; ++i)
            cens[i] = censoring.elemAsDouble(i) ? 1.0 : 0.0;
    }
    if (!freq.isEmpty()) {
        if (freq.numel() != n)
            throw Error("gamfit: freq length mismatch",
                        0, 0, "gamfit", "", "numkit:gamfit:freqLen");
        wf.resize(n);
        for (keel::usize i = 0; i < n; ++i) {
            wf[i] = freq.elemAsDouble(i);
            if (!(wf[i] >= 0.0))
                throw Error("gamfit: freq must be non-negative",
                            0, 0, "gamfit", "", "numkit:gamfit:freqNeg");
        }
    }
    keel::stats::Fit2Result res{};
    const auto st = keel::stats::gamfit(xv.as_ref(), cens.as_ref(), wf.as_ref(), alpha,
                                        compute_ci, res);
    if (st != keel::stats::StatsStatus::kOk)
        throw Error("gamfit: need at least 2 uncensored observations",
                    0, 0, "gamfit", "", "numkit:gamfit:tooFewUncens");
    return res;
}

keel::stats::Fit2Result run_wblfit(const Value &x, const Value &censoring, const Value &freq,
                                   keel::f64 alpha, bool compute_ci)
{
    auto xv = to_flat(x);
    const keel::usize n = xv.size();
    if (n < 2)
        throw Error("wblfit: need at least 2 observations",
                    0, 0, "wblfit", "", "numkit:wblfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!(v > 0.0))
            throw Error("wblfit: all observations must be positive",
                        0, 0, "wblfit", "", "numkit:wblfit:notPositive");
    }
    keel::Array<keel::f64> cens, wf;
    if (!censoring.isEmpty()) {
        if (censoring.numel() != n)
            throw Error("wblfit: censoring length mismatch",
                        0, 0, "wblfit", "", "numkit:wblfit:censLen");
        cens.resize(n);
        for (keel::usize i = 0; i < n; ++i)
            cens[i] = censoring.elemAsDouble(i) ? 1.0 : 0.0;
    }
    if (!freq.isEmpty()) {
        if (freq.numel() != n)
            throw Error("wblfit: freq length mismatch",
                        0, 0, "wblfit", "", "numkit:wblfit:freqLen");
        wf.resize(n);
        for (keel::usize i = 0; i < n; ++i) {
            wf[i] = freq.elemAsDouble(i);
            if (!(wf[i] >= 0.0))
                throw Error("wblfit: freq must be non-negative",
                            0, 0, "wblfit", "", "numkit:wblfit:freqNeg");
        }
    }
    keel::stats::Fit2Result res{};
    (void)keel::stats::wblfit(xv.as_ref(), cens.as_ref(), wf.as_ref(), alpha, compute_ci, res);
    return res;
}

keel::stats::Fit2Result run_betafit(const Value &x, keel::f64 alpha, bool compute_ci)
{
    auto xv = to_flat(x);
    if (xv.size() < 2)
        throw Error("betafit: need at least 2 observations",
                    0, 0, "betafit", "", "numkit:betafit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!(v > 0.0 && v < 1.0))
            throw Error("betafit: all observations must be in (0, 1)",
                        0, 0, "betafit", "", "numkit:betafit:outOfRange");
    }
    keel::stats::Fit2Result res{};
    (void)keel::stats::betafit(xv.as_ref(), alpha, compute_ci, res);
    return res;
}

keel::stats::Fit2Result run_nbinfit(const Value &x, keel::f64 alpha, bool compute_ci)
{
    auto xv = to_flat(x);
    if (xv.size() < 2)
        throw Error("nbinfit: need at least 2 observations",
                    0, 0, "nbinfit", "", "numkit:nbinfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (v < 0.0 || std::fabs(v - std::round(v)) > 1e-9)
            throw Error("nbinfit: observations must be non-negative integers",
                        0, 0, "nbinfit", "", "numkit:nbinfit:notInteger");
    }
    keel::stats::Fit2Result res{};
    const auto st = keel::stats::nbinfit(xv.as_ref(), alpha, compute_ci, res);
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("nbinfit: sample is under-dispersed (var ≤ mean); "
                    "MLE undefined for negative binomial",
                    0, 0, "nbinfit", "", "numkit:nbinfit:underDispersed");
    return res;
}

keel::stats::Fit2Result run_evfit(const Value &x, const Value &censoring, const Value &freq,
                                  keel::f64 alpha, bool compute_ci)
{
    auto xv = to_flat(x);
    const keel::usize n = xv.size();
    if (n < 2)
        throw Error("evfit: need at least 2 observations",
                    0, 0, "evfit", "", "numkit:evfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!std::isfinite(v))
            throw Error("evfit: observations must be finite",
                        0, 0, "evfit", "", "numkit:evfit:notFinite");
    }
    keel::Array<keel::f64> cens, wf;
    if (!censoring.isEmpty()) {
        if (censoring.numel() != n)
            throw Error("evfit: censoring must match data length",
                        0, 0, "evfit", "", "numkit:evfit:censLen");
        cens.resize(n);
        for (keel::usize i = 0; i < n; ++i)
            cens[i] = censoring.elemAsDouble(i) ? 1.0 : 0.0;
    }
    if (!freq.isEmpty()) {
        if (freq.numel() != n)
            throw Error("evfit: freq must match data length",
                        0, 0, "evfit", "", "numkit:evfit:freqLen");
        wf.resize(n);
        for (keel::usize i = 0; i < n; ++i) {
            wf[i] = freq.elemAsDouble(i);
            if (!(wf[i] >= 0.0))
                throw Error("evfit: freq must be non-negative",
                            0, 0, "evfit", "", "numkit:evfit:freqNeg");
        }
    }
    keel::stats::Fit2Result res{};
    const auto st = keel::stats::evfit(xv.as_ref(), cens.as_ref(), wf.as_ref(), alpha,
                                       compute_ci, res);
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("evfit: zero variance (data constant)",
                    0, 0, "evfit", "", "numkit:evfit:zeroVariance");
    if (st != keel::stats::StatsStatus::kOk)
        throw Error("evfit: need at least 2 uncensored observations",
                    0, 0, "evfit", "", "numkit:evfit:tooFewUncens");
    return res;
}

keel::stats::Fit2Result run_gpfit(const Value &x, keel::f64 alpha, bool compute_ci)
{
    auto xv = to_flat(x);
    if (xv.size() < 2)
        throw Error("gpfit: need at least 2 observations",
                    0, 0, "gpfit", "", "numkit:gpfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!std::isfinite(v) || v < 0.0)
            throw Error("gpfit: observations must be finite and ≥ 0 "
                        "(threshold θ = 0 assumed)",
                        0, 0, "gpfit", "", "numkit:gpfit:invalidData");
    }
    keel::stats::Fit2Result res{};
    (void)keel::stats::gpfit(xv.as_ref(), alpha, compute_ci, res);
    return res;
}

keel::stats::Fit3Result run_gevfit(const Value &x, keel::f64 alpha, bool compute_ci)
{
    auto xv = to_flat(x);
    if (xv.size() < 3)
        throw Error("gevfit: need at least 3 observations",
                    0, 0, "gevfit", "", "numkit:gevfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!std::isfinite(v))
            throw Error("gevfit: observations must be finite",
                        0, 0, "gevfit", "", "numkit:gevfit:notFinite");
    }
    keel::stats::Fit3Result res{};
    (void)keel::stats::gevfit(xv.as_ref(), alpha, compute_ci, res);
    return res;
}

} // namespace

Value gamfit(const Value &x, keel::Allocator *mr)
{
    return gamfit(x, Value::Empty, Value::Empty, mr);
}

Value gamfit(const Value &x, const Value &censoring, const Value &freq, keel::Allocator *mr)
{
    return make_fit2_params(run_gamfit(x, censoring, freq, 0.05, false), mr);
}

Value wblfit(const Value &x, const Value &censoring, const Value &freq, keel::Allocator *mr)
{
    return make_fit2_params(run_wblfit(x, censoring, freq, 0.05, false), mr);
}

Value wblfit(const Value &x, keel::Allocator *mr)
{
    return wblfit(x, Value::Empty, Value::Empty, mr);
}

Value betafit(const Value &x, keel::Allocator *mr)
{
    return make_fit2_params(run_betafit(x, 0.05, false), mr);
}

Value nbinfit(const Value &x, keel::Allocator *mr)
{
    return make_fit2_params(run_nbinfit(x, 0.05, false), mr);
}

Value evfit(const Value &x, keel::Allocator *mr)
{
    return evfit(x, Value::Empty, Value::Empty, mr);
}

Value evfit(const Value &x, const Value &censoring, const Value &freq, keel::Allocator *mr)
{
    return make_fit2_params(run_evfit(x, censoring, freq, 0.05, false), mr);
}

Value gpfit(const Value &x, keel::Allocator *mr)
{
    return make_fit2_params(run_gpfit(x, 0.05, false), mr);
}

Value gamfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return gamfit_ci(x, alpha, Value::Empty, Value::Empty, mr);
}

Value gamfit_ci(const Value &x, keel::f64 alpha,
                const Value &censoring, const Value &freq,
                keel::Allocator *mr)
{
    return make_fit2_ci(run_gamfit(x, censoring, freq, alpha, true), mr);
}

Value wblfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return wblfit_ci(x, alpha, Value::Empty, Value::Empty, mr);
}

Value wblfit_ci(const Value &x, keel::f64 alpha,
                const Value &censoring, const Value &freq,
                keel::Allocator *mr)
{
    return make_fit2_ci(run_wblfit(x, censoring, freq, alpha, true), mr);
}

Value betafit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return make_fit2_ci(run_betafit(x, alpha, true), mr);
}

Value nbinfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return make_fit2_ci(run_nbinfit(x, alpha, true), mr);
}

Value evfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return evfit_ci(x, alpha, Value::Empty, Value::Empty, mr);
}

Value evfit_ci(const Value &x, keel::f64 alpha,
               const Value &censoring, const Value &freq,
               keel::Allocator *mr)
{
    return make_fit2_ci(run_evfit(x, censoring, freq, alpha, true), mr);
}

Value gpfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return make_fit2_ci(run_gpfit(x, alpha, true), mr);
}

Value gevfit(const Value &x, keel::Allocator *mr)
{
    const auto r = run_gevfit(x, 0.05, false);
    auto out = Value::matrix(1, 3, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = r.p1;
    out.doubleDataMut()[1] = r.p2;
    out.doubleDataMut()[2] = r.p3;
    return out;
}

Value gevfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    const auto r = run_gevfit(x, alpha, true);
    auto out = Value::matrix(2, 3, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    od[0] = r.ci1_lo;
    od[1] = r.ci1_hi;
    od[2] = r.ci2_lo;
    od[3] = r.ci2_hi;
    od[4] = r.ci3_lo;
    od[5] = r.ci3_hi;
    return out;
}

} // namespace numkit::stats
