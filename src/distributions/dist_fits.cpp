// toolboxes/stats/src/distributions/dist_fits.cpp
//
// MLE / moment fitters for continuous + count distributions:
//   gamfit  — Gamma(a, b) shape + scale  (Newton on profile)
//   wblfit  — Weibull(scale, shape)       (Newton on shape)
//   betafit — Beta(a, b)                  (2-D Newton on digamma system)
//   nbinfit — NegBin(r, p)                (Newton on profile)
//   evfit   — Type-I EV (Gumbel-min)      (1-D Newton, μ profiled)
//   gpfit   — Generalised Pareto          (PWM / Hosking-Wallis)

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/stats/distributions/gamma_dist.hpp>
#include <numkit/stats/distributions/weibull.hpp>
#include <numkit/stats/distributions/beta.hpp>
#include <numkit/stats/distributions/negbin.hpp>
#include <numkit/stats/distributions/extreme_value.hpp>
#include <numkit/stats/distributions/gp.hpp>
#include <numkit/stats/distributions/gev.hpp>

#include <numkit/builtin/specfun.hpp>   // gammainc (used by gamfit_full)

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "dist_fits_detail.hpp"

namespace numkit::stats {


// Generalised Gamma NLL with right-censoring and frequency weights.
//   log f(x; a, b) = (a-1) log x - x/b - a log b - log Γ(a)
//   log S(x; a, b) = log(1 - gammainc(x/b, a))
// (gammainc(x, a) is the regularised LOWER incomplete gamma; survival
//  is the upper tail = 1 - lower.)
static keel::f64 nll_gam_full(keel::f64 a, keel::f64 b,
                           const keel::Array<keel::f64> &xv,
                           const keel::Array<keel::f64> &cens,
                           const keel::Array<keel::f64> &freq,
                           keel::Allocator *mr)
{
    if (!(a > 0.0) || !(b > 0.0)) return std::numeric_limits<keel::f64>::infinity();
    const keel::usize n = xv.size();
    keel::f64 nuw = 0.0, lxw = 0.0, sxw = 0.0, S_cens = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 f = freq[i];
        const keel::f64 u = 1.0 - cens[i];
        nuw += f * u;
        lxw += f * u * std::log(xv[i]);
        sxw += f * u * xv[i];
        if (cens[i] > 0.5 && f > 0.0) {
            Value xv_arg = Value::scalar(xv[i] / b, mr);
            Value av_arg = Value::scalar(a, mr);
            const keel::f64 P = ::numkit::builtin::gammainc(xv_arg, av_arg, mr).toScalar();
            const keel::f64 Q = 1.0 - P;
            if (!(Q > 0.0)) return std::numeric_limits<keel::f64>::infinity();
            S_cens -= f * std::log(Q);
        }
    }
    return nuw * (std::lgamma(a) + a * std::log(b))
         - (a - 1.0) * lxw + sxw / b + S_cens;
}

Value gamfit(const Value &x, keel::Allocator *mr)
{
    return gamfit(x, Value::Empty, Value::Empty, mr);
}

namespace {
void gamfit_argcheck(const keel::Array<keel::f64> &xv, const Value &censoring, const Value &freq,
                     keel::Array<keel::f64> &cens, keel::Array<keel::f64> &wf)
{
    const keel::usize n = xv.size();
    if (n < 2)
        throw Error("gamfit: need at least 2 observations",
                    0, 0, "gamfit", "", "numkit:gamfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!(v > 0.0))
            throw Error("gamfit: all observations must be positive",
                        0, 0, "gamfit", "", "numkit:gamfit:notPositive");
    }
    cens.assign(n, 0.0);
    wf.assign(n, 1.0);
    if (!censoring.isEmpty()) {
        if (censoring.numel() != n)
            throw Error("gamfit: censoring length mismatch",
                        0, 0, "gamfit", "", "numkit:gamfit:censLen");
        for (keel::usize i = 0; i < n; ++i)
            cens[i] = censoring.elemAsDouble(i) ? 1.0 : 0.0;
    }
    if (!freq.isEmpty()) {
        if (freq.numel() != n)
            throw Error("gamfit: freq length mismatch",
                        0, 0, "gamfit", "", "numkit:gamfit:freqLen");
        for (keel::usize i = 0; i < n; ++i) {
            wf[i] = freq.elemAsDouble(i);
            if (!(wf[i] >= 0.0))
                throw Error("gamfit: freq must be non-negative",
                            0, 0, "gamfit", "", "numkit:gamfit:freqNeg");
        }
    }
}
} // namespace

Value gamfit(const Value &x, const Value &censoring, const Value &freq,
             keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    keel::Array<keel::f64> cens, wf;
    gamfit_argcheck(xv, censoring, freq, cens, wf);

    bool freq_trivial = true;
    for (keel::usize i = 0; i < n; ++i)
        if (wf[i] != 1.0) { freq_trivial = false; break; }
    bool has_cens = false;
    for (keel::usize i = 0; i < n; ++i)
        if (cens[i] != 0.0) { has_cens = true; break; }

    if (!has_cens && freq_trivial) {
        // Existing closed-form path.
        keel::f64 sum = 0.0, sumLog = 0.0;
        for (keel::f64 v : xv) { sum += v; sumLog += std::log(v); }
        const keel::f64 mean = sum / keel::as<keel::f64>(n);
        const keel::f64 meanLog = sumLog / keel::as<keel::f64>(n);
        const keel::f64 s = std::log(mean) - meanLog;
        if (!(s > 0.0)) {
            auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
            out.doubleDataMut()[0] = std::numeric_limits<keel::f64>::infinity();
            out.doubleDataMut()[1] = mean;
            return out;
        }
        keel::f64 a = (3.0 - s + std::sqrt((s - 3.0) * (s - 3.0) + 24.0 * s))
                   / (12.0 * s);
        for (keel::i32 it = 0; it < 50; ++it) {
            const keel::f64 fa = std::log(a) - digamma(a) - s;
            const keel::f64 fpa = 1.0 / a - trigamma(a);
            const keel::f64 step = fa / fpa;
            const keel::f64 newA = a - step;
            if (newA > 0.0) a = newA; else a *= 0.5;
            if (std::fabs(step) < 1e-10 * std::max(a, 1.0)) break;
        }
        const keel::f64 b = mean / a;
        auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        out.doubleDataMut()[0] = a;
        out.doubleDataMut()[1] = b;
        return out;
    }

    // Weighted/censored path: 2-D Newton on (a, b) via FD on
    // nll_gam_full. Initial guess from weighted uncensored moments.
    keel::f64 nuw = 0.0, sx = 0.0, slx = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 u = 1.0 - cens[i];
        nuw += wf[i] * u;
        sx  += wf[i] * u * xv[i];
        slx += wf[i] * u * std::log(xv[i]);
    }
    if (!(nuw >= 2.0))
        throw Error("gamfit: need at least 2 uncensored observations",
                    0, 0, "gamfit", "", "numkit:gamfit:tooFewUncens");
    const keel::f64 mean_uw = sx / nuw;
    const keel::f64 meanLog_uw = slx / nuw;
    const keel::f64 s_init = std::log(mean_uw) - meanLog_uw;
    keel::f64 a, b;
    if (s_init > 0.0) {
        a = (3.0 - s_init + std::sqrt((s_init - 3.0) * (s_init - 3.0) + 24.0 * s_init))
            / (12.0 * s_init);
        b = mean_uw / a;
    } else {
        a = 1.0;
        b = mean_uw;
    }

    auto nll = [&](keel::f64 aa, keel::f64 bb) { return nll_gam_full(aa, bb, xv, cens, wf, mr); };
    keel::f64 f_cur = nll(a, b);
    for (keel::i32 it = 0; it < 100; ++it) {
        const keel::f64 ha = std::max(std::fabs(a), 1.0) * 1e-5;
        const keel::f64 hb = std::max(std::fabs(b), 1.0) * 1e-5;
        const keel::f64 ga = (nll(a + ha, b) - nll(a - ha, b)) / (2.0 * ha);
        const keel::f64 gb = (nll(a, b + hb) - nll(a, b - hb)) / (2.0 * hb);
        if (std::sqrt(ga * ga + gb * gb) < 1e-12) break;
        const keel::f64 Haa = (nll(a + ha, b) - 2.0 * f_cur + nll(a - ha, b)) / (ha * ha);
        const keel::f64 Hbb = (nll(a, b + hb) - 2.0 * f_cur + nll(a, b - hb)) / (hb * hb);
        const keel::f64 Hab = (nll(a + ha, b + hb) - nll(a + ha, b - hb)
                         - nll(a - ha, b + hb) + nll(a - ha, b - hb))
                          / (4.0 * ha * hb);
        const keel::f64 det = Haa * Hbb - Hab * Hab;
        if (std::fabs(det) < 1e-300) break;
        const keel::f64 da = (-ga * Hbb + gb * Hab) / det;
        const keel::f64 db = (-gb * Haa + ga * Hab) / det;
        keel::f64 sc = 1.0;
        bool ok = false;
        for (keel::i32 bt = 0; bt < 30; ++bt) {
            const keel::f64 an = a + sc * da;
            const keel::f64 bn = b + sc * db;
            if (an > 0.0 && bn > 0.0) {
                const keel::f64 f_new = nll(an, bn);
                if (std::isfinite(f_new) && f_new < f_cur - 1e-15) {
                    a = an; b = bn; f_cur = f_new; ok = true;
                    break;
                }
            }
            sc *= 0.5;
        }
        if (!ok) break;
        if (std::fabs(sc * da) + std::fabs(sc * db)
            < 1e-12 * (std::fabs(a) + std::fabs(b) + 1.0)) break;
    }

    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = a;
    out.doubleDataMut()[1] = b;
    return out;
}

// Generalised Weibull NLL with right-censoring and frequency weights.
//   log f(x; a, b) = log b - b log a + (b - 1) log x - (x/a)^b
//   log S(x; a, b) = -(x/a)^b
// Weighted-censored NLL:
//   Σ f_i u_i · [-log b + b log a - (b-1) log x_i] + Σ f_i (x_i/a)^b
static keel::f64 nll_wbl_full(keel::f64 a, keel::f64 b,
                           const keel::Array<keel::f64> &xv,
                           const keel::Array<keel::f64> &cens,
                           const keel::Array<keel::f64> &freq)
{
    if (!(a > 0.0) || !(b > 0.0)) return std::numeric_limits<keel::f64>::infinity();
    const keel::usize n = xv.size();
    keel::f64 nuw = 0.0, lxw = 0.0, sumW = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 f = freq[i];
        const keel::f64 u = 1.0 - cens[i];
        nuw  += f * u;
        lxw  += f * u * std::log(xv[i]);
        sumW += f * std::pow(xv[i] / a, b);
    }
    return nuw * (-std::log(b) + b * std::log(a)) - (b - 1.0) * lxw + sumW;
}

Value wblfit(const Value &x, const Value &censoring, const Value &freq,
             keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    if (n < 2)
        throw Error("wblfit: need at least 2 observations",
                    0, 0, "wblfit", "", "numkit:wblfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!(v > 0.0))
            throw Error("wblfit: all observations must be positive",
                        0, 0, "wblfit", "", "numkit:wblfit:notPositive");
    }
    keel::Array<keel::f64> cens(n, 0.0), wf(n, 1.0);
    if (!censoring.isEmpty()) {
        if (censoring.numel() != n)
            throw Error("wblfit: censoring length mismatch",
                        0, 0, "wblfit", "", "numkit:wblfit:censLen");
        for (keel::usize i = 0; i < n; ++i)
            cens[i] = censoring.elemAsDouble(i) ? 1.0 : 0.0;
    }
    if (!freq.isEmpty()) {
        if (freq.numel() != n)
            throw Error("wblfit: freq length mismatch",
                        0, 0, "wblfit", "", "numkit:wblfit:freqLen");
        for (keel::usize i = 0; i < n; ++i) {
            wf[i] = freq.elemAsDouble(i);
            if (!(wf[i] >= 0.0))
                throw Error("wblfit: freq must be non-negative",
                            0, 0, "wblfit", "", "numkit:wblfit:freqNeg");
        }
    }

    // Detect trivial case → existing closed-form path.
    bool freq_trivial = true;
    for (keel::usize i = 0; i < n; ++i)
        if (wf[i] != 1.0) { freq_trivial = false; break; }
    bool has_cens = false;
    for (keel::usize i = 0; i < n; ++i)
        if (cens[i] != 0.0) { has_cens = true; break; }

    if (!has_cens && freq_trivial) {
        // Fast path: existing Newton on shape via profile σ.
        keel::Array<keel::f64> lx(n);
        keel::f64 sumLog = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            lx[i] = std::log(xv[i]);
            sumLog += lx[i];
        }
        const keel::f64 meanLog = sumLog / keel::as<keel::f64>(n);
        keel::f64 v2 = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 d = lx[i] - meanLog;
            v2 += d * d;
        }
        const keel::f64 sdLog = std::sqrt(v2 / keel::as<keel::f64>(n));
        keel::f64 b = 3.141592653589793 / (sdLog * std::sqrt(6.0) + 1e-12);
        if (!(b > 0.1) || !std::isfinite(b)) b = 1.0;

        for (keel::i32 it = 0; it < 100; ++it) {
            keel::f64 S0 = 0.0, S1 = 0.0, S2 = 0.0;
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 w = std::pow(xv[i], b);
                S0 += w;
                S1 += w * lx[i];
                S2 += w * lx[i] * lx[i];
            }
            const keel::f64 R = S1 / S0;
            const keel::f64 g = 1.0 / b + meanLog - R;
            const keel::f64 dR = (S2 * S0 - S1 * S1) / (S0 * S0);
            const keel::f64 dg = -1.0 / (b * b) - dR;
            const keel::f64 step = g / dg;
            const keel::f64 newB = b - step;
            if (newB > 0.0) b = newB; else b *= 0.5;
            if (std::fabs(step) < 1e-10 * std::max(b, 1.0)) break;
        }
        keel::f64 S = 0.0;
        for (keel::f64 v : xv) S += std::pow(v, b);
        const keel::f64 a = std::pow(S / keel::as<keel::f64>(n), 1.0 / b);
        auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        out.doubleDataMut()[0] = a;
        out.doubleDataMut()[1] = b;
        return out;
    }

    // Weighted/censored path: profile a given b.
    // For each b: a(b) = (Σ f x^b / Σ f u)^{1/b}.
    // Then 1-D Newton on b via score d(NLL)/db = 0 evaluated with a(b).
    // Initial b from naïve closed form on uncensored subset.
    keel::f64 nuw_init = 0.0;
    keel::f64 sum_lx_uncens_w = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 u = 1.0 - cens[i];
        nuw_init += wf[i] * u;
        sum_lx_uncens_w += wf[i] * u * std::log(xv[i]);
    }
    const keel::f64 meanLog_w = (nuw_init > 0.0) ? sum_lx_uncens_w / nuw_init : 0.0;
    keel::f64 v2 = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 u = 1.0 - cens[i];
        const keel::f64 d = std::log(xv[i]) - meanLog_w;
        v2 += wf[i] * u * d * d;
    }
    const keel::f64 sdLog_w = (nuw_init > 0.0) ? std::sqrt(v2 / nuw_init) : 1.0;
    keel::f64 bcur = 3.141592653589793 / (sdLog_w * std::sqrt(6.0) + 1e-12);
    if (!(bcur > 0.05) || !std::isfinite(bcur)) bcur = 1.0;

    auto profile_a = [&](keel::f64 b) -> keel::f64 {
        keel::f64 S = 0.0, U = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            S += wf[i] * std::pow(xv[i], b);
            U += wf[i] * (1.0 - cens[i]);
        }
        if (!(U > 0.0)) return std::numeric_limits<keel::f64>::quiet_NaN();
        return std::pow(S / U, 1.0 / b);
    };

    // Score for b under cens+freq: solve
    //   ∂NLL/∂b = -nuw/b + Σ f u log x_i - log a · nuw
    //           + Σ f (x/a)^b · log(x/a) - log a · nuw_uncens correction
    // Easier: 1-D Newton via FD on `nll_wbl_full(profile_a(b), b)`.
    auto nll_at_b = [&](keel::f64 b) -> keel::f64 {
        const keel::f64 a = profile_a(b);
        return nll_wbl_full(a, b, xv, cens, wf);
    };
    keel::f64 f_cur = nll_at_b(bcur);
    for (keel::i32 it = 0; it < 100; ++it) {
        const keel::f64 h = std::max(std::fabs(bcur), 1.0) * 1e-6;
        const keel::f64 g = (nll_at_b(bcur + h) - nll_at_b(bcur - h)) / (2.0 * h);
        const keel::f64 H = (nll_at_b(bcur + h) - 2.0 * f_cur + nll_at_b(bcur - h))
                          / (h * h);
        if (std::fabs(H) < 1e-300) break;
        const keel::f64 step = g / H;
        keel::f64 bn = bcur - step;
        // Backtracking.
        keel::f64 sc = 1.0;
        bool ok = false;
        for (keel::i32 bt = 0; bt < 30; ++bt) {
            bn = bcur - sc * step;
            if (bn > 0.0) {
                const keel::f64 f_new = nll_at_b(bn);
                if (std::isfinite(f_new) && f_new < f_cur - 1e-15) {
                    bcur = bn; f_cur = f_new; ok = true; break;
                }
            }
            sc *= 0.5;
        }
        if (!ok) break;
        if (std::fabs(sc * step) < 1e-12 * std::max(bcur, 1.0)) break;
    }
    const keel::f64 a_final = profile_a(bcur);

    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = a_final;
    out.doubleDataMut()[1] = bcur;
    return out;
}

Value wblfit(const Value &x, keel::Allocator *mr)
{
    return wblfit(x, Value::Empty, Value::Empty, mr);
}

Value betafit(const Value &x, keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    if (n < 2)
        throw Error("betafit: need at least 2 observations",
                    0, 0, "betafit", "", "numkit:betafit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!(v > 0.0 && v < 1.0))
            throw Error("betafit: all observations must be in (0, 1)",
                        0, 0, "betafit", "", "numkit:betafit:outOfRange");
    }
    keel::f64 sumX = 0.0, sumLogX = 0.0, sumLog1mX = 0.0;
    for (keel::f64 v : xv) {
        sumX     += v;
        sumLogX  += std::log(v);
        sumLog1mX += std::log(1.0 - v);
    }
    const keel::f64 meanX = sumX / keel::as<keel::f64>(n);
    // MoM initial guess: a = μ·(μ(1-μ)/σ² - 1), b = (1-μ)·(...)
    keel::f64 v0 = 0.0;
    for (keel::f64 v : xv) {
        const keel::f64 d = v - meanX;
        v0 += d * d;
    }
    v0 /= keel::as<keel::f64>(n);
    if (!(v0 > 0.0) || v0 >= meanX * (1.0 - meanX))
        v0 = meanX * (1.0 - meanX) * 0.5;   // safeguard for MoM
    const keel::f64 common = meanX * (1.0 - meanX) / v0 - 1.0;
    keel::f64 a = meanX * common;
    keel::f64 b = (1.0 - meanX) * common;
    if (!(a > 0.0)) a = 1.0;
    if (!(b > 0.0)) b = 1.0;

    const keel::f64 sLogX  = sumLogX  / keel::as<keel::f64>(n);
    const keel::f64 sLog1X = sumLog1mX / keel::as<keel::f64>(n);

    // Newton on (a, b). 2x2 system with diagonal trigamma minus
    // ψ'(a+b) on both off-diagonal entries.
    for (keel::i32 it = 0; it < 100; ++it) {
        const keel::f64 psiAB = digamma(a + b);
        const keel::f64 f1 = digamma(a) - psiAB - sLogX;
        const keel::f64 f2 = digamma(b) - psiAB - sLog1X;
        const keel::f64 tAB = trigamma(a + b);
        const keel::f64 J11 = trigamma(a) - tAB;
        const keel::f64 J22 = trigamma(b) - tAB;
        const keel::f64 J12 = -tAB;
        const keel::f64 det = J11 * J22 - J12 * J12;
        if (std::fabs(det) < 1e-300) break;
        const keel::f64 da = (J22 * f1 - J12 * f2) / det;
        const keel::f64 db = (J11 * f2 - J12 * f1) / det;
        const keel::f64 newA = a - da;
        const keel::f64 newB = b - db;
        if (newA > 0.0) a = newA; else a *= 0.5;
        if (newB > 0.0) b = newB; else b *= 0.5;
        if (std::fabs(da) + std::fabs(db)
            < 1e-10 * (std::fabs(a) + std::fabs(b) + 1.0)) break;
    }

    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = a;
    out.doubleDataMut()[1] = b;
    return out;
}

Value nbinfit(const Value &x, keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    if (n < 2)
        throw Error("nbinfit: need at least 2 observations",
                    0, 0, "nbinfit", "", "numkit:nbinfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (v < 0.0 || std::fabs(v - std::round(v)) > 1e-9)
            throw Error("nbinfit: observations must be non-negative integers",
                        0, 0, "nbinfit", "", "numkit:nbinfit:notInteger");
    }
    keel::f64 sum = 0.0;
    for (keel::f64 v : xv) sum += v;
    const keel::f64 meanX = sum / keel::as<keel::f64>(n);
    keel::f64 v2 = 0.0;
    for (keel::f64 v : xv) {
        const keel::f64 d = v - meanX;
        v2 += d * d;
    }
    const keel::f64 varX = v2 / keel::as<keel::f64>(n - 1);
    if (varX <= meanX)
        throw Error("nbinfit: sample is under-dispersed (var ≤ mean); "
                    "MLE undefined for negative binomial",
                    0, 0, "nbinfit", "", "numkit:nbinfit:underDispersed");
    // MoM: r = μ² / (σ² - μ), p = μ / σ².
    keel::f64 r = (meanX * meanX) / (varX - meanX);
    if (!(r > 0.0)) r = 1.0;

    // Newton on the profile log-likelihood in r (p = r/(r + meanX)).
    //   d/dr log L = sum [ψ(x_i + r) - ψ(r)] + n [log(r/(r + meanX))]
    for (keel::i32 it = 0; it < 100; ++it) {
        const keel::f64 frac = r / (r + meanX);
        keel::f64 S1 = 0.0, S2 = 0.0;
        for (keel::f64 v : xv) {
            S1 += digamma(v + r) - digamma(r);
            S2 += trigamma(v + r) - trigamma(r);
        }
        const keel::f64 f = S1 + keel::as<keel::f64>(n) * std::log(frac);
        const keel::f64 fp = S2 + keel::as<keel::f64>(n) * meanX / (r * (r + meanX));
        if (std::fabs(fp) < 1e-300) break;
        const keel::f64 step = f / fp;
        const keel::f64 newR = r - step;
        if (newR > 0.0) r = newR;
        else            r *= 0.5;
        if (std::fabs(step) < 1e-10 * std::max(r, 1.0)) break;
    }
    const keel::f64 p = r / (r + meanX);

    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = r;
    out.doubleDataMut()[1] = p;
    return out;
}

// Generalised EV-min NLL with right-censoring and frequency weights.
// Censoring indicator: cens_i = 1 if x_i is right-censored.
// Weighted contribution per observation:
//   uncens: f_i · [log σ − t_i + exp(t_i)]
//   cens  : f_i · [exp(t_i)]
// where t_i = (x_i − μ)/σ. Combined:
//   NLL = (Σf_i (1−c_i)) · log σ − Σf_i (1−c_i) t_i + Σf_i exp(t_i).
static keel::f64 nll_ev_full(keel::f64 mu, keel::f64 sigma,
                          const keel::Array<keel::f64> &xv,
                          const keel::Array<keel::f64> &cens,
                          const keel::Array<keel::f64> &freq)
{
    if (!(sigma > 0.0)) return std::numeric_limits<keel::f64>::infinity();
    const keel::usize n = xv.size();
    // Shift by max(t_i) to keep exp from overflowing.
    keel::f64 t_max = -std::numeric_limits<keel::f64>::infinity();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 t = (xv[i] - mu) / sigma;
        if (t > t_max) t_max = t;
    }
    keel::f64 nuw = 0.0;        // Σ f_i (1 - c_i)
    keel::f64 Suw = 0.0;        // Σ f_i (1 - c_i) t_i
    keel::f64 Wexp = 0.0;       // Σ f_i · exp(t_i - t_max)
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 t = (xv[i] - mu) / sigma;
        const keel::f64 f = freq[i];
        const keel::f64 u = 1.0 - cens[i];
        nuw  += f * u;
        Suw  += f * u * t;
        Wexp += f * std::exp(t - t_max);
    }
    // Adding back the shift: Σ f exp(t) = exp(t_max) · Wexp.
    // log σ · nuw - Suw + exp(t_max) · Wexp.
    return nuw * std::log(sigma) - Suw + std::exp(t_max) * Wexp;
}

Value evfit(const Value &x, keel::Allocator *mr)
{
    return evfit(x, Value::Empty, Value::Empty, mr);
}

Value evfit(const Value &x, const Value &censoring, const Value &freq,
            keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    if (n < 2)
        throw Error("evfit: need at least 2 observations",
                    0, 0, "evfit", "", "numkit:evfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!std::isfinite(v))
            throw Error("evfit: observations must be finite",
                        0, 0, "evfit", "", "numkit:evfit:notFinite");
    }
    // Parse censoring (zeros if empty) and freq (ones if empty).
    keel::Array<keel::f64> cens(n, 0.0), wf(n, 1.0);
    if (!censoring.isEmpty()) {
        if (censoring.numel() != n)
            throw Error("evfit: censoring must match data length",
                        0, 0, "evfit", "", "numkit:evfit:censLen");
        for (keel::usize i = 0; i < n; ++i)
            cens[i] = censoring.elemAsDouble(i) ? 1.0 : 0.0;
    }
    if (!freq.isEmpty()) {
        if (freq.numel() != n)
            throw Error("evfit: freq must match data length",
                        0, 0, "evfit", "", "numkit:evfit:freqLen");
        for (keel::usize i = 0; i < n; ++i) {
            wf[i] = freq.elemAsDouble(i);
            if (!(wf[i] >= 0.0))
                throw Error("evfit: freq must be non-negative",
                            0, 0, "evfit", "", "numkit:evfit:freqNeg");
        }
    }

    // Effective weights / counts.
    keel::f64 sum_fw = 0.0, sum_uw = 0.0;
    keel::f64 sum_uwx = 0.0;
    keel::f64 sum_uwxx = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 f = wf[i];
        const keel::f64 u = 1.0 - cens[i];
        sum_fw   += f;
        sum_uw   += f * u;
        sum_uwx  += f * u * xv[i];
        sum_uwxx += f * u * xv[i] * xv[i];
    }
    if (sum_uw < 2.0)
        throw Error("evfit: need at least 2 uncensored observations",
                    0, 0, "evfit", "", "numkit:evfit:tooFewUncens");
    const keel::f64 mean_uw = sum_uwx / sum_uw;
    keel::f64 var_uw = sum_uwxx / sum_uw - mean_uw * mean_uw;
    // Constant uncensored data is degenerate (no info about σ).
    if (!(var_uw > 0.0)) {
        // Treat truly constant data (no freq weighting or all-equal
        // uncensored values) as an error to match MATLAB's behaviour
        // on degenerate input.
        bool freq_trivial = true;
        for (keel::usize i = 0; i < n; ++i)
            if (wf[i] != 1.0) { freq_trivial = false; break; }
        const bool has_cens_local = (sum_uw < sum_fw);
        if (!has_cens_local && freq_trivial)
            throw Error("evfit: zero variance (data constant)",
                        0, 0, "evfit", "", "numkit:evfit:zeroVariance");
        var_uw = 1.0;
    }

    // Initial guess: σ from weighted/uncensored variance moment.
    // For Gumbel-min, E[X] = μ - γ_E·σ ⇒ μ = E[X] + γ_E·σ.
    keel::f64 sigma = std::sqrt(6.0 * var_uw) / 3.141592653589793;
    if (!(sigma > 0.0) || !std::isfinite(sigma)) sigma = 1.0;
    keel::f64 mu = mean_uw + 0.57721566490153286 * sigma;

    // Check if censoring or non-trivial freq is present — if not,
    // dispatch to the existing fast closed-form path.
    const bool has_cens = (sum_uw < sum_fw);
    bool freq_trivial = true;
    for (keel::usize i = 0; i < n; ++i)
        if (wf[i] != 1.0) { freq_trivial = false; break; }

    if (!has_cens && freq_trivial) {
        // Closed-form (existing path): profile μ via log-sum-exp.
        keel::f64 sum = 0.0;
        for (keel::f64 v : xv) sum += v;
        const keel::f64 mean = sum / keel::as<keel::f64>(n);
        keel::f64 x_max = xv[0];
        for (keel::f64 v : xv) if (v > x_max) x_max = v;
        keel::f64 T = 0.0;
        for (keel::i32 it = 0; it < 100; ++it) {
            T = 0.0; keel::f64 U = 0.0, V = 0.0;
            for (keel::f64 v : xv) {
                const keel::f64 e = std::exp((v - x_max) / sigma);
                T += e;  U += v * e;  V += v * v * e;
            }
            const keel::f64 meanW = U / T;
            const keel::f64 varW  = V / T - meanW * meanW;
            const keel::f64 f  = meanW - mean - sigma;
            const keel::f64 fp = -varW / (sigma * sigma) - 1.0;
            if (!(std::fabs(fp) > 1e-300)) break;
            const keel::f64 step = f / fp;
            const keel::f64 newS = sigma - step;
            if (newS > 0.0) sigma = newS;
            else            sigma *= 0.5;
            if (std::fabs(step) < 1e-12 * std::max(sigma, 1.0)) break;
        }
        T = 0.0;
        for (keel::f64 v : xv) T += std::exp((v - x_max) / sigma);
        mu = x_max + sigma * (std::log(T) - std::log(keel::as<keel::f64>(n)));
    } else {
        // 2-D Newton with FD gradient/Hessian.
        auto grad = [&](keel::f64 m, keel::f64 s, keel::f64 &gm, keel::f64 &gs) {
            const keel::f64 hm = std::max(std::fabs(m), 1.0) * 1e-6;
            const keel::f64 hs = std::max(std::fabs(s), 1.0) * 1e-6;
            gm = (nll_ev_full(m + hm, s, xv, cens, wf)
                - nll_ev_full(m - hm, s, xv, cens, wf)) / (2 * hm);
            gs = (nll_ev_full(m, s + hs, xv, cens, wf)
                - nll_ev_full(m, s - hs, xv, cens, wf)) / (2 * hs);
        };
        keel::f64 f_cur = nll_ev_full(mu, sigma, xv, cens, wf);
        for (keel::i32 it = 0; it < 100; ++it) {
            keel::f64 gm, gs;
            grad(mu, sigma, gm, gs);
            if (std::sqrt(gm * gm + gs * gs) < 1e-12) break;
            // FD-Hessian on gradient.
            const keel::f64 hm = std::max(std::fabs(mu), 1.0) * 1e-5;
            const keel::f64 hs = std::max(std::fabs(sigma), 1.0) * 1e-5;
            keel::f64 gm_mp, gs_mp, gm_mm, gs_mm, gm_sp, gs_sp, gm_sm, gs_sm;
            grad(mu + hm, sigma, gm_mp, gs_mp);
            grad(mu - hm, sigma, gm_mm, gs_mm);
            grad(mu, sigma + hs, gm_sp, gs_sp);
            grad(mu, sigma - hs, gm_sm, gs_sm);
            const keel::f64 Hmm = (gm_mp - gm_mm) / (2 * hm);
            const keel::f64 Hss = (gs_sp - gs_sm) / (2 * hs);
            const keel::f64 Hms = 0.5 * ((gm_sp - gm_sm) / (2 * hs)
                                    + (gs_mp - gs_mm) / (2 * hm));
            const keel::f64 det = Hmm * Hss - Hms * Hms;
            if (std::fabs(det) < 1e-300) break;
            const keel::f64 dm = (-gm * Hss + gs * Hms) / det;
            const keel::f64 ds = (-gs * Hmm + gm * Hms) / det;
            keel::f64 step = 1.0;
            bool ok = false;
            for (keel::i32 bt = 0; bt < 30; ++bt) {
                const keel::f64 mu_new = mu + step * dm;
                const keel::f64 s_new  = sigma + step * ds;
                if (s_new > 0.0) {
                    const keel::f64 f_new = nll_ev_full(mu_new, s_new, xv, cens, wf);
                    if (std::isfinite(f_new) && f_new < f_cur - 1e-15) {
                        mu = mu_new;  sigma = s_new;  f_cur = f_new;
                        ok = true;
                        break;
                    }
                }
                step *= 0.5;
            }
            if (!ok) break;
            if (std::fabs(step * dm) + std::fabs(step * ds)
                < 1e-12 * (std::fabs(mu) + std::fabs(sigma) + 1.0)) break;
        }
    }

    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = mu;
    out.doubleDataMut()[1] = sigma;
    return out;
}

// Generalised Pareto NLL helper. Returns +Inf if the support
// constraint (1 + k·x_i/σ > 0 for all i) is violated, allowing the
// optimiser to reject infeasible steps via the line-search.
static keel::f64 gp_nll(keel::f64 k, keel::f64 sigma,
                     const keel::Array<keel::f64> &xv)
{
    if (!(sigma > 0.0)) return std::numeric_limits<keel::f64>::infinity();
    const keel::usize n = xv.size();
    keel::f64 nll = keel::as<keel::f64>(n) * std::log(sigma);
    if (std::fabs(k) < 1e-10) {
        // Exponential limit.
        keel::f64 s = 0.0;
        for (keel::f64 v : xv) s += v;
        return nll + s / sigma;
    }
    const keel::f64 inv_sigma = 1.0 / sigma;
    for (keel::f64 v : xv) {
        const keel::f64 u = k * v * inv_sigma;
        if (u <= -1.0) return std::numeric_limits<keel::f64>::infinity();
    }
    keel::f64 s = 0.0;
    for (keel::f64 v : xv) s += std::log1p(k * v * inv_sigma);
    nll += (1.0 + 1.0 / k) * s;
    return nll;
}

// Central-finite-difference gradient of NLL w.r.t. (k, σ). Robust
// at k = 0 (where the analytical formula is removable but the limit
// requires a Taylor expansion); central FD captures the smooth surface
// uniformly across the parameter space.
static void gp_nll_grad(keel::f64 k, keel::f64 sigma,
                        const keel::Array<keel::f64> &xv,
                        keel::f64 &gk, keel::f64 &gs)
{
    const keel::f64 hk = std::max(std::fabs(k), 1.0) * 1e-7;
    const keel::f64 hs = std::max(std::fabs(sigma), 1.0) * 1e-7;
    gk = (gp_nll(k + hk, sigma, xv) - gp_nll(k - hk, sigma, xv)) / (2.0 * hk);
    gs = (gp_nll(k, sigma + hs, xv) - gp_nll(k, sigma - hs, xv)) / (2.0 * hs);
}

Value gpfit(const Value &x, keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    if (n < 2)
        throw Error("gpfit: need at least 2 observations",
                    0, 0, "gpfit", "", "numkit:gpfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!std::isfinite(v) || v < 0.0)
            throw Error("gpfit: observations must be finite and ≥ 0 "
                        "(threshold θ = 0 assumed)",
                        0, 0, "gpfit", "", "numkit:gpfit:invalidData");
    }

    // ── Initial guess: PWM (Hosking-Wallis 1987, α-form) ──────────
    keel::Array<keel::f64> xs = xv.clone();
    std::sort(xs.begin(), xs.end());
    keel::f64 sumX = 0.0;
    for (keel::f64 v : xs) sumX += v;
    const keel::f64 beta0 = sumX / keel::as<keel::f64>(n);
    keel::f64 beta1 = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 F = (keel::as<keel::f64>(i + 1) - 0.35) / keel::as<keel::f64>(n);
        beta1 += (1.0 - F) * xs[i];
    }
    beta1 /= keel::as<keel::f64>(n);

    const keel::f64 denom = beta0 - 2.0 * beta1;
    keel::f64 k_hat, sigma_hat;
    if (std::fabs(denom) < 1e-300 || !std::isfinite(denom)) {
        k_hat = 0.0;
        sigma_hat = beta0;
    } else {
        k_hat = 2.0 - beta0 / denom;
        sigma_hat = 2.0 * beta0 * beta1 / denom;
        if (!(sigma_hat > 0.0) || !std::isfinite(sigma_hat)) {
            k_hat = 0.0;
            sigma_hat = beta0;
        }
    }

    // ── Newton-Raphson refinement (Grimshaw-style) ────────────────
    // 2-D Newton on (k, σ) with analytical gradient and central-FD
    // Hessian on the gradient. Line search guards against infeasible
    // or NLL-worsening steps.
    const keel::f64 x_max = xs.back();
    keel::f64 f0 = gp_nll(k_hat, sigma_hat, xv);
    if (!std::isfinite(f0)) {
        // PWM produced infeasible point — fall back to safe init.
        sigma_hat = beta0;
        k_hat = 0.0;
        f0 = gp_nll(k_hat, sigma_hat, xv);
    }

    for (keel::i32 it = 0; it < 100; ++it) {
        keel::f64 gk, gs;
        gp_nll_grad(k_hat, sigma_hat, xv, gk, gs);
        const keel::f64 gnorm = std::sqrt(gk * gk + gs * gs);
        if (gnorm < 1e-12) break;

        // Numerical Hessian via central FD on gradient.
        const keel::f64 hk = std::max(std::fabs(k_hat), 1.0) * 1e-6;
        const keel::f64 hs = std::max(std::fabs(sigma_hat), 1.0) * 1e-6;
        keel::f64 gk_kp, gs_kp, gk_km, gs_km, gk_sp, gs_sp, gk_sm, gs_sm;
        gp_nll_grad(k_hat + hk, sigma_hat, xv, gk_kp, gs_kp);
        gp_nll_grad(k_hat - hk, sigma_hat, xv, gk_km, gs_km);
        gp_nll_grad(k_hat, sigma_hat + hs, xv, gk_sp, gs_sp);
        gp_nll_grad(k_hat, sigma_hat - hs, xv, gk_sm, gs_sm);
        const keel::f64 Hkk = (gk_kp - gk_km) / (2.0 * hk);
        const keel::f64 Hss = (gs_sp - gs_sm) / (2.0 * hs);
        const keel::f64 Hks = 0.5 * ((gk_sp - gk_sm) / (2.0 * hs)
                                + (gs_kp - gs_km) / (2.0 * hk));

        // Solve H · [dk; dσ] = -[gk; gs].
        const keel::f64 det = Hkk * Hss - Hks * Hks;
        if (std::fabs(det) < 1e-300) break;
        keel::f64 dk = (-gk * Hss + gs * Hks) / det;
        keel::f64 ds = (-gs * Hkk + gk * Hks) / det;

        // Backtracking line search with feasibility guard.
        keel::f64 step = 1.0;
        bool improved = false;
        for (keel::i32 bt = 0; bt < 30; ++bt) {
            const keel::f64 k_new = k_hat + step * dk;
            const keel::f64 s_new = sigma_hat + step * ds;
            if (s_new > 0.0 && (k_new >= 0.0
                || -k_new * x_max / s_new < 1.0 - 1e-12)) {
                const keel::f64 f_new = gp_nll(k_new, s_new, xv);
                if (std::isfinite(f_new) && f_new < f0 - 1e-15) {
                    k_hat = k_new;
                    sigma_hat = s_new;
                    f0 = f_new;
                    improved = true;
                    break;
                }
            }
            step *= 0.5;
        }
        if (!improved) break;
        if (std::fabs(step * dk) + std::fabs(step * ds)
            < 1e-12 * (std::fabs(k_hat) + std::fabs(sigma_hat) + 1.0)) break;
    }

    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = k_hat;
    out.doubleDataMut()[1] = sigma_hat;
    return out;
}

// ════════════════════════════════════════════════════════════════════
// Confidence intervals (Wald, observed Fisher information)
// ════════════════════════════════════════════════════════════════════


Value gamfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return gamfit_ci(x, alpha, Value::Empty, Value::Empty, mr);
}

Value gamfit_ci(const Value &x, keel::f64 alpha,
                const Value &censoring, const Value &freq,
                keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    keel::Array<keel::f64> cens(n, 0.0), wf(n, 1.0);
    if (!censoring.isEmpty()) {
        for (keel::usize i = 0; i < n; ++i)
            cens[i] = censoring.elemAsDouble(i) ? 1.0 : 0.0;
    }
    if (!freq.isEmpty()) {
        for (keel::usize i = 0; i < n; ++i) wf[i] = freq.elemAsDouble(i);
    }
    Value parm = gamfit(x, censoring, freq, mr);
    const keel::f64 a_hat = parm.elemAsDouble(0);
    const keel::f64 b_hat = parm.elemAsDouble(1);
    bool trivial = censoring.isEmpty() && freq.isEmpty();
    if (trivial) {
        return wald_ci_2d([&](keel::f64 a, keel::f64 b) { return nll_gam(a, b, xv); },
                          a_hat, b_hat, CITransform::LOG, CITransform::LOG,
                          alpha, mr);
    }
    return wald_ci_2d([&](keel::f64 a, keel::f64 b) {
                          return nll_gam_full(a, b, xv, cens, wf, mr);
                      },
                      a_hat, b_hat, CITransform::LOG, CITransform::LOG,
                      alpha, mr);
}

Value wblfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return wblfit_ci(x, alpha, Value::Empty, Value::Empty, mr);
}

Value wblfit_ci(const Value &x, keel::f64 alpha,
                const Value &censoring, const Value &freq,
                keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    keel::Array<keel::f64> cens(n, 0.0), wf(n, 1.0);
    if (!censoring.isEmpty()) {
        for (keel::usize i = 0; i < n; ++i)
            cens[i] = censoring.elemAsDouble(i) ? 1.0 : 0.0;
    }
    if (!freq.isEmpty()) {
        for (keel::usize i = 0; i < n; ++i) wf[i] = freq.elemAsDouble(i);
    }
    Value parm = wblfit(x, censoring, freq, mr);
    const keel::f64 a_hat = parm.elemAsDouble(0);
    const keel::f64 b_hat = parm.elemAsDouble(1);
    // For trivial case use the simple NLL; for cens/freq use the
    // generalised NLL.
    bool trivial = censoring.isEmpty() && freq.isEmpty();
    if (trivial) {
        return wald_ci_2d([&](keel::f64 a, keel::f64 b) { return nll_wbl(a, b, xv); },
                          a_hat, b_hat, CITransform::LOG, CITransform::LOG,
                          alpha, mr);
    }
    return wald_ci_2d([&](keel::f64 a, keel::f64 b) {
                          return nll_wbl_full(a, b, xv, cens, wf);
                      },
                      a_hat, b_hat, CITransform::LOG, CITransform::LOG,
                      alpha, mr);
}

// Specialised CI using analytical Hessian (digamma-based NLLs).
// Generic FD-Hessian introduces ~1% noise on these likelihoods, so we
// drop down to closed-form to keep CI bit-equal with MATLAB.


Value betafit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    auto xv = toFlat(x);
    Value parm = betafit(x, mr);
    const keel::f64 a = parm.elemAsDouble(0);
    const keel::f64 b = parm.elemAsDouble(1);
    // Analytical observed Fisher information:
    //   I_aa = n[ψ'(a) − ψ'(a+b)]
    //   I_bb = n[ψ'(b) − ψ'(a+b)]
    //   I_ab = −n·ψ'(a+b)
    const keel::f64 N = keel::as<keel::f64>(xv.size());
    const keel::f64 tab = trigamma(a + b);
    const keel::f64 H11 = N * (trigamma(a) - tab);
    const keel::f64 H22 = N * (trigamma(b) - tab);
    const keel::f64 H12 = -N * tab;
    return wald_ci_from_hessian(a, b, H11, H22, H12,
                                CITransform::LOG, CITransform::LOG,
                                alpha, mr);
}

Value nbinfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    auto xv = toFlat(x);
    Value parm = nbinfit(x, mr);
    const keel::f64 r = parm.elemAsDouble(0);
    const keel::f64 p = parm.elemAsDouble(1);
    // Analytical observed Fisher info:
    //   I_rr = -Σ ψ'(x_i + r) + n·ψ'(r)
    //   I_pp = n·r/p² + (Σx)/(1−p)²
    //   I_rp = −n/p
    const keel::f64 N = keel::as<keel::f64>(xv.size());
    keel::f64 sxh = 0.0;
    keel::f64 sx = 0.0;
    for (keel::f64 v : xv) {
        sxh += trigamma(v + r);
        sx  += v;
    }
    const keel::f64 H11 = -sxh + N * trigamma(r);
    const keel::f64 H22 = N * r / (p * p) + sx / ((1.0 - p) * (1.0 - p));
    const keel::f64 H12 = -N / p;
    return wald_ci_from_hessian(r, p, H11, H22, H12,
                                CITransform::LOG, CITransform::LOGIT,
                                alpha, mr);
}

Value evfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return evfit_ci(x, alpha, Value::Empty, Value::Empty, mr);
}

Value evfit_ci(const Value &x, keel::f64 alpha,
               const Value &censoring, const Value &freq,
               keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    keel::Array<keel::f64> cens(n, 0.0), wf(n, 1.0);
    if (!censoring.isEmpty()) {
        for (keel::usize i = 0; i < n; ++i)
            cens[i] = censoring.elemAsDouble(i) ? 1.0 : 0.0;
    }
    if (!freq.isEmpty()) {
        for (keel::usize i = 0; i < n; ++i)
            wf[i] = freq.elemAsDouble(i);
    }
    Value parm = evfit(x, censoring, freq, mr);
    const keel::f64 mu_hat    = parm.elemAsDouble(0);
    const keel::f64 sigma_hat = parm.elemAsDouble(1);
    return wald_ci_2d([&](keel::f64 mu, keel::f64 sigma) {
                          return nll_ev_full(mu, sigma, xv, cens, wf);
                      },
                      mu_hat, sigma_hat, CITransform::LINEAR, CITransform::LOG,
                      alpha, mr);
}

Value gpfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    auto xv = toFlat(x);
    Value parm = gpfit(x, mr);
    const keel::f64 k_hat     = parm.elemAsDouble(0);
    const keel::f64 sigma_hat = parm.elemAsDouble(1);
    return wald_ci_2d([&](keel::f64 k, keel::f64 sigma) { return gp_nll(k, sigma, xv); },
                      k_hat, sigma_hat, CITransform::LINEAR, CITransform::LOG,
                      alpha, mr);
}

// ── gevfit (3-param GEV MLE) ─────────────────────────────────────────

// GEV NLL: log f(x; k, σ, μ).
//   For k ≠ 0: t = 1 + k(x-μ)/σ;  f = (1/σ) t^{-1/k - 1} exp(-t^{-1/k})
//     log f = -log σ + (-1 - 1/k) log t - t^{-1/k}
//   For k → 0 (Gumbel-max limit): z = (x-μ)/σ
//     log f = -log σ - z - exp(-z)
// Returns +Inf if any support is violated.
static keel::f64 nll_gev(keel::f64 k, keel::f64 sigma, keel::f64 mu,
                      const keel::Array<keel::f64> &xv)
{
    if (!(sigma > 0.0)) return std::numeric_limits<keel::f64>::infinity();
    const keel::usize n = xv.size();
    keel::f64 nll = keel::as<keel::f64>(n) * std::log(sigma);
    if (std::fabs(k) < 1e-10) {
        // Gumbel-max limit.
        for (keel::f64 x : xv) {
            const keel::f64 z = (x - mu) / sigma;
            nll += z + std::exp(-z);
        }
    } else {
        for (keel::f64 x : xv) {
            const keel::f64 t = 1.0 + k * (x - mu) / sigma;
            if (!(t > 0.0)) return std::numeric_limits<keel::f64>::infinity();
            const keel::f64 log_t = std::log(t);
            // t^{-1/k} = exp(-log_t/k)
            const keel::f64 t_pow = std::exp(-log_t / k);
            nll += (1.0 + 1.0 / k) * log_t + t_pow;
        }
    }
    return nll;
}

Value gevfit(const Value &x, keel::Allocator *mr)
{
    auto xv = toFlat(x);
    const keel::usize n = xv.size();
    if (n < 3)
        throw Error("gevfit: need at least 3 observations",
                    0, 0, "gevfit", "", "numkit:gevfit:tooFewObs");
    for (keel::f64 v : xv) {
        if (!std::isfinite(v))
            throw Error("gevfit: observations must be finite",
                        0, 0, "gevfit", "", "numkit:gevfit:notFinite");
    }

    // PWM initial guess (Hosking-Wallis-Wood 1985).
    keel::Array<keel::f64> xs = xv.clone();
    std::sort(xs.begin(), xs.end());
    keel::f64 b0 = 0.0, b1 = 0.0, b2 = 0.0;
    const keel::f64 dN = keel::as<keel::f64>(n);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 dI = keel::as<keel::f64>(i + 1);
        // PWM using plotting positions p_i = (i - 0.35) / n.
        const keel::f64 Fhat = (dI - 0.35) / dN;
        b0 += xs[i];
        b1 += xs[i] * Fhat;
        b2 += xs[i] * Fhat * Fhat;
    }
    b0 /= dN; b1 /= dN; b2 /= dN;
    // Hosking et al. 1985 approximation for k:
    //   c = (2 b1 - b0) / (3 b2 - b0) - log(2) / log(3)
    //   k ≈ 7.8590·c + 2.9554·c²
    const keel::f64 LN2_LN3 = 0.6309297535714574;  // ln(2)/ln(3)
    const keel::f64 denom = 3.0 * b2 - b0;
    keel::f64 k_init = 0.0;
    if (std::fabs(denom) > 1e-300) {
        const keel::f64 c = (2.0 * b1 - b0) / denom - LN2_LN3;
        k_init = 7.8590 * c + 2.9554 * c * c;
    }
    // σ via:
    //   σ = k·(2 b1 - b0) / (Γ(1+k) · (1 - 2^{-k})), with k → 0 limit.
    keel::f64 sigma_init;
    if (std::fabs(k_init) < 1e-6) {
        sigma_init = (2.0 * b1 - b0) / std::log(2.0);
    } else {
        const keel::f64 gam_1pk = std::tgamma(1.0 + k_init);
        const keel::f64 one_m_2nk = 1.0 - std::pow(2.0, -k_init);
        sigma_init = k_init * (2.0 * b1 - b0) / (gam_1pk * one_m_2nk);
    }
    if (!(sigma_init > 0.0) || !std::isfinite(sigma_init))
        sigma_init = std::max(b0 * 0.5, 1.0);
    // μ via:
    //   μ = b0 - σ · (1 - Γ(1+k)) / k, k → 0 limit: μ = b0 - σ · γ_E
    keel::f64 mu_init;
    if (std::fabs(k_init) < 1e-6) {
        mu_init = b0 - sigma_init * 0.57721566490153286;
    } else {
        const keel::f64 gam_1pk = std::tgamma(1.0 + k_init);
        mu_init = b0 - sigma_init * (1.0 - gam_1pk) / k_init;
    }

    // Feasibility check on initial guess; if it violates support, shift.
    auto feasible = [&](keel::f64 k, keel::f64 sigma, keel::f64 mu) -> bool {
        if (!(sigma > 0.0)) return false;
        if (std::fabs(k) < 1e-10) return true;
        for (keel::f64 v : xv) {
            if (1.0 + k * (v - mu) / sigma <= 0.0) return false;
        }
        return true;
    };
    if (!feasible(k_init, sigma_init, mu_init)) {
        // Fall back to safe Gumbel init.
        k_init = 0.0;
        sigma_init = std::sqrt(6.0 * 1.0) / 3.141592653589793;
        keel::f64 mean = 0.0;
        for (keel::f64 v : xv) mean += v;
        mean /= dN;
        mu_init = mean - 0.57721566490153286 * sigma_init;
    }

    // 3-D Newton on NLL with FD gradient + FD Hessian.
    keel::f64 k = k_init, sigma = sigma_init, mu = mu_init;
    keel::f64 f_cur = nll_gev(k, sigma, mu, xv);
    if (!std::isfinite(f_cur)) {
        k = 0.0; sigma = 1.0; mu = 0.0;
        f_cur = nll_gev(k, sigma, mu, xv);
    }

    auto nll = [&](keel::f64 K, keel::f64 S, keel::f64 M) { return nll_gev(K, S, M, xv); };
    for (keel::i32 it = 0; it < 200; ++it) {
        const keel::f64 hk = std::max(std::fabs(k), 0.5) * 1e-5;
        const keel::f64 hs = std::max(std::fabs(sigma), 1.0) * 1e-5;
        const keel::f64 hm = std::max(std::fabs(mu), 1.0) * 1e-5;
        // Gradient.
        const keel::f64 gk = (nll(k + hk, sigma, mu) - nll(k - hk, sigma, mu)) / (2.0 * hk);
        const keel::f64 gs = (nll(k, sigma + hs, mu) - nll(k, sigma - hs, mu)) / (2.0 * hs);
        const keel::f64 gm = (nll(k, sigma, mu + hm) - nll(k, sigma, mu - hm)) / (2.0 * hm);
        const keel::f64 gn = std::sqrt(gk * gk + gs * gs + gm * gm);
        if (gn < 1e-12) break;
        // Hessian (symmetric).
        const keel::f64 Hkk = (nll(k + hk, sigma, mu) - 2.0 * f_cur + nll(k - hk, sigma, mu)) / (hk * hk);
        const keel::f64 Hss = (nll(k, sigma + hs, mu) - 2.0 * f_cur + nll(k, sigma - hs, mu)) / (hs * hs);
        const keel::f64 Hmm = (nll(k, sigma, mu + hm) - 2.0 * f_cur + nll(k, sigma, mu - hm)) / (hm * hm);
        const keel::f64 Hks = (nll(k + hk, sigma + hs, mu) - nll(k + hk, sigma - hs, mu)
                         -  nll(k - hk, sigma + hs, mu) + nll(k - hk, sigma - hs, mu))
                          / (4.0 * hk * hs);
        const keel::f64 Hkm = (nll(k + hk, sigma, mu + hm) - nll(k + hk, sigma, mu - hm)
                         -  nll(k - hk, sigma, mu + hm) + nll(k - hk, sigma, mu - hm))
                          / (4.0 * hk * hm);
        const keel::f64 Hsm = (nll(k, sigma + hs, mu + hm) - nll(k, sigma + hs, mu - hm)
                         -  nll(k, sigma - hs, mu + hm) + nll(k, sigma - hs, mu - hm))
                          / (4.0 * hs * hm);
        // Solve 3x3 H · d = -g (Cramer's rule for small system).
        const keel::f64 H11 = Hkk, H12 = Hks, H13 = Hkm;
        const keel::f64 H21 = Hks, H22 = Hss, H23 = Hsm;
        const keel::f64 H31 = Hkm, H32 = Hsm, H33 = Hmm;
        const keel::f64 det = H11 * (H22 * H33 - H23 * H32)
                         - H12 * (H21 * H33 - H23 * H31)
                         + H13 * (H21 * H32 - H22 * H31);
        if (std::fabs(det) < 1e-300) break;
        // -g = b
        const keel::f64 b1n = -gk, b2n = -gs, b3n = -gm;
        const keel::f64 dk = (b1n * (H22 * H33 - H23 * H32)
                         - H12 * (b2n * H33 - H23 * b3n)
                         + H13 * (b2n * H32 - H22 * b3n)) / det;
        const keel::f64 ds = (H11 * (b2n * H33 - H23 * b3n)
                         - b1n * (H21 * H33 - H23 * H31)
                         + H13 * (H21 * b3n - b2n * H31)) / det;
        const keel::f64 dm = (H11 * (H22 * b3n - b2n * H32)
                         - H12 * (H21 * b3n - b2n * H31)
                         + b1n * (H21 * H32 - H22 * H31)) / det;
        // Backtracking line search.
        keel::f64 sc = 1.0;
        bool ok = false;
        for (keel::i32 bt = 0; bt < 30; ++bt) {
            const keel::f64 kn = k + sc * dk;
            const keel::f64 sn = sigma + sc * ds;
            const keel::f64 mn = mu + sc * dm;
            if (sn > 0.0 && feasible(kn, sn, mn)) {
                const keel::f64 f_new = nll(kn, sn, mn);
                if (std::isfinite(f_new) && f_new < f_cur - 1e-15) {
                    k = kn; sigma = sn; mu = mn; f_cur = f_new;
                    ok = true; break;
                }
            }
            sc *= 0.5;
        }
        if (!ok) break;
        if (std::fabs(sc * dk) + std::fabs(sc * ds) + std::fabs(sc * dm)
            < 1e-12 * (std::fabs(k) + std::fabs(sigma) + std::fabs(mu) + 1.0)) break;
    }

    auto out = Value::matrix(1, 3, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = k;
    out.doubleDataMut()[1] = sigma;
    out.doubleDataMut()[2] = mu;
    return out;
}

Value gevfit_ci(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    auto xv = toFlat(x);
    Value parm = gevfit(x, mr);
    const keel::f64 k_hat     = parm.elemAsDouble(0);
    const keel::f64 sigma_hat = parm.elemAsDouble(1);
    const keel::f64 mu_hat    = parm.elemAsDouble(2);

    // Compute 3-D FD Hessian of NLL at parmhat → V = H^{-1} → SE_j.
    auto nll = [&](keel::f64 K, keel::f64 S, keel::f64 M) { return nll_gev(K, S, M, xv); };
    const keel::f64 hk = std::max(std::fabs(k_hat), 0.5) * 1e-5;
    const keel::f64 hs = std::max(std::fabs(sigma_hat), 1.0) * 1e-5;
    const keel::f64 hm = std::max(std::fabs(mu_hat), 1.0) * 1e-5;
    const keel::f64 f00 = nll(k_hat, sigma_hat, mu_hat);
    const keel::f64 Hkk = (nll(k_hat + hk, sigma_hat, mu_hat) - 2 * f00
                     + nll(k_hat - hk, sigma_hat, mu_hat)) / (hk * hk);
    const keel::f64 Hss = (nll(k_hat, sigma_hat + hs, mu_hat) - 2 * f00
                     + nll(k_hat, sigma_hat - hs, mu_hat)) / (hs * hs);
    const keel::f64 Hmm = (nll(k_hat, sigma_hat, mu_hat + hm) - 2 * f00
                     + nll(k_hat, sigma_hat, mu_hat - hm)) / (hm * hm);
    const keel::f64 Hks = (nll(k_hat + hk, sigma_hat + hs, mu_hat)
                     -  nll(k_hat + hk, sigma_hat - hs, mu_hat)
                     -  nll(k_hat - hk, sigma_hat + hs, mu_hat)
                     +  nll(k_hat - hk, sigma_hat - hs, mu_hat)) / (4 * hk * hs);
    const keel::f64 Hkm = (nll(k_hat + hk, sigma_hat, mu_hat + hm)
                     -  nll(k_hat + hk, sigma_hat, mu_hat - hm)
                     -  nll(k_hat - hk, sigma_hat, mu_hat + hm)
                     +  nll(k_hat - hk, sigma_hat, mu_hat - hm)) / (4 * hk * hm);
    const keel::f64 Hsm = (nll(k_hat, sigma_hat + hs, mu_hat + hm)
                     -  nll(k_hat, sigma_hat + hs, mu_hat - hm)
                     -  nll(k_hat, sigma_hat - hs, mu_hat + hm)
                     +  nll(k_hat, sigma_hat - hs, mu_hat - hm)) / (4 * hs * hm);
    // Invert 3×3 to get cov diagonals.
    const keel::f64 H[9] = {Hkk, Hks, Hkm, Hks, Hss, Hsm, Hkm, Hsm, Hmm};
    const keel::f64 det = H[0]*(H[4]*H[8]-H[5]*H[7]) - H[1]*(H[3]*H[8]-H[5]*H[6])
                     + H[2]*(H[3]*H[7]-H[4]*H[6]);
    Value out = Value::matrix(2, 3, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    if (std::fabs(det) < 1e-300) {
        for (keel::i32 i = 0; i < 6; ++i) od[i] = std::numeric_limits<keel::f64>::quiet_NaN();
        return out;
    }
    // V_jj = cofactor_jj / det.
    const keel::f64 V_kk = (H[4]*H[8] - H[5]*H[7]) / det;
    const keel::f64 V_ss = (H[0]*H[8] - H[2]*H[6]) / det;
    const keel::f64 V_mm = (H[0]*H[4] - H[1]*H[3]) / det;
    const keel::f64 se_k = std::sqrt(std::max(V_kk, 0.0));
    const keel::f64 se_s = std::sqrt(std::max(V_ss, 0.0));
    const keel::f64 se_m = std::sqrt(std::max(V_mm, 0.0));

    // z_{α/2} via full Wichura AS-241 (handles tail beyond |q| > 0.425).
    auto z_alpha = [](keel::f64 a) -> keel::f64 {
        const keel::f64 p = 1.0 - 0.5 * a;
        const keel::f64 q = p - 0.5;
        if (std::fabs(q) <= 0.425) {
            const keel::f64 r = 0.180625 - q * q;
            return q * (((((((2.509080928730122727e3 * r
                + 3.343057558358812877e4) * r + 6.726577092700870006e4) * r
                + 4.592195393154987305e4) * r + 1.373169376550946215e4) * r
                + 1.971590950306227941e3) * r + 1.331416678917643998e2) * r
                + 3.387132872796366608) /
                (((((((5.226495278852854561e3 * r + 2.872908573572194738e4) * r
                + 3.930789580009271061e4) * r + 2.121379430158659746e4) * r
                + 5.394196021424751670e3) * r + 6.871159355626076150e2) * r
                + 4.231333070160091088e1) * r + 1.0);
        }
        keel::f64 r = (q < 0.0) ? p : 1.0 - p;
        r = std::sqrt(-std::log(r));
        keel::f64 v;
        if (r <= 5.0) {
            r -= 1.6;
            v = (((((((7.74545014278341407640e-4 * r
                + 2.27238449892691845833e-2) * r + 2.41780725177450611770e-1) * r
                + 1.27045825245236838258e0) * r + 3.64784832476320460504e0) * r
                + 5.76949722146069140550e0) * r + 4.63033784615654529590e0) * r
                + 1.42343711074968357734e0) /
                (((((((1.05075007164441684324e-9 * r
                + 5.47593808499534494600e-4) * r + 1.51986665636164571966e-2) * r
                + 1.48103976427480074590e-1) * r + 6.89767334985100004550e-1) * r
                + 1.67638483018380384940e0) * r + 2.05319162663775882187e0) * r
                + 1.0);
        } else {
            r -= 5.0;
            v = (((((((2.01033439929228813265e-7 * r
                + 2.71155556874348757815e-5) * r + 1.24266094738807843860e-3) * r
                + 2.65321895265761230930e-2) * r + 2.96560571828504891230e-1) * r
                + 1.78482653991729133580e0) * r + 5.46378491116411436990e0) * r
                + 6.65790464350110377720e0) /
                (((((((2.04426310338993978564e-15 * r
                + 1.42151175831644588870e-7) * r + 1.84631831751005468180e-5) * r
                + 7.86869131145613259100e-4) * r + 1.48753612908506148525e-2) * r
                + 1.36929880988350204800e-1) * r + 5.99832206555887937690e-1) * r
                + 1.0);
        }
        return (q < 0.0) ? -v : v;
    };
    const keel::f64 z = z_alpha(alpha);
    // k linear CI, σ log CI, μ linear CI.
    const keel::f64 se_log_sigma = (sigma_hat > 0.0) ? se_s / sigma_hat : 0.0;
    od[0] = k_hat - z * se_k;
    od[1] = k_hat + z * se_k;
    od[2] = sigma_hat * std::exp(-z * se_log_sigma);
    od[3] = sigma_hat * std::exp( z * se_log_sigma);
    od[4] = mu_hat - z * se_m;
    od[5] = mu_hat + z * se_m;
    return out;
}

} // namespace numkit::stats
