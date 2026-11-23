// toolboxes/.../fit_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by fit.cpp + fit_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <numeric>


namespace numkit::stats {

namespace {

[[maybe_unused]] void mean_var(const Value &x, keel::f64 &mean, keel::f64 &var, keel::usize &N) {
    N = x.numel();
    if (N == 0) { mean = 0.0; var = 0.0; return; }
    keel::f64 s = 0.0;
    for (keel::usize i = 0; i < N; ++i) s += x.elemAsDouble(i);
    mean = s / keel::f64(N);
    if (N < 2) { var = 0.0; return; }
    keel::f64 sq = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 d = x.elemAsDouble(i) - mean;
        sq += d * d;
    }
    var = sq / keel::f64(N - 1);
}

[[maybe_unused]] Value rowCI(keel::f64 lo, keel::f64 hi, keel::Allocator *mr) {
    Value v = Value::matrix(2, 1, ValueType::DOUBLE, mr);
    keel::f64 *d = v.doubleDataMut();
    d[0] = lo;
    d[1] = hi;
    return v;
}

keel::f64 tinv_scalar(keel::f64 p, keel::f64 nu, keel::Allocator *mr) {
    Value pv = Value::scalar(p, mr);
    return tinv(pv, nu, mr).toScalar();
}

keel::f64 chi2inv_scalar(keel::f64 p, keel::f64 k, keel::Allocator *mr) {
    Value pv = Value::scalar(p, mr);
    return chi2inv(pv, k, mr).toScalar();
}

// ── Normal MLE helpers (used by normfit / lognfit cens+freq paths) ────

constexpr keel::f64 kPi = keel::kPi;

// Inverse-Mills ratio λ(α) = φ(α)/Φ(-α). For right-censored normal MLE.
inline keel::f64 inv_mills(keel::f64 a) {
    const keel::f64 pdf  = keel::exp(-0.5 * a * a) / keel::sqrt(2.0 * kPi);
    const keel::f64 surv = 0.5 * keel::erfc(a / keel::sqrt(2.0));  // Φ(-a)
    return (surv > 1e-300) ? pdf / surv : 0.0;
}

// z = -norminv(α/2) via Newton iteration on std::erf.
keel::f64 zNorm(keel::f64 alpha) {
    const keel::f64 y_target = 1.0 - alpha;
    keel::f64 e = y_target;
    for (keel::i32 it = 0; it < 50; ++it) {
        const keel::f64 f  = keel::erf(e) - y_target;
        const keel::f64 fp = (2.0 / keel::sqrt(kPi)) * keel::exp(-e * e);
        const keel::f64 step = f / fp;
        e -= step;
        if (keel::abs(step) < 1e-15) break;
    }
    return keel::sqrt(2.0) * e;
}

// Result of a normal-MLE fit on observations `y` with optional right-
// censoring + freq weights. Used by both normfit (raw x) and lognfit
// (y = log(x)).
struct NormalFitOut {
    keel::f64 mu, sd;
    keel::f64 mu_lo, mu_hi;
    keel::f64 sd_lo, sd_hi;
    bool   ok;
};

[[maybe_unused]] NormalFitOut
normal_fit_mle(const keel::Array<keel::f64> &y, const keel::Array<keel::f64> &fr, const keel::Array<keel::u8> &cn, keel::f64 alpha, keel::Allocator *mr)
{
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    NormalFitOut R{nan, nan, nan, nan, nan, nan, false};
    const keel::usize N = y.size();
    if (N < 2) return R;

    keel::f64 sumf = 0.0;
    keel::i32 nuncens = 0;
    for (keel::usize i = 0; i < N; ++i) {
        sumf += fr[i];
        if (!cn[i]) ++nuncens;
    }
    if (sumf < 2.0 || nuncens < 1) return R;

    const bool any_cens = (nuncens != (keel::i32)N);

    if (!any_cens) {
        // Closed-form weighted moments.
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < N; ++i) s += fr[i] * y[i];
        const keel::f64 mu = s / sumf;
        keel::f64 sq = 0.0;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 d = y[i] - mu;
            sq += fr[i] * d * d;
        }
        const keel::f64 sd  = keel::sqrt(sq / (sumf - 1.0));
        const keel::f64 dof = sumf - 1.0;
        const keel::f64 t   = tinv_scalar(1.0 - alpha / 2.0, dof, mr);
        const keel::f64 sem = sd / keel::sqrt(sumf);
        const keel::f64 chiU = chi2inv_scalar(1.0 - alpha / 2.0, dof, mr);
        const keel::f64 chiL = chi2inv_scalar(alpha / 2.0, dof, mr);
        R.mu = mu; R.sd = sd;
        R.mu_lo = mu - t * sem; R.mu_hi = mu + t * sem;
        R.sd_lo = keel::sqrt(dof * sd * sd / chiU);
        R.sd_hi = keel::sqrt(dof * sd * sd / chiL);
        R.ok = true;
        return R;
    }

    // Censored MLE via EM on truncated-normal moments. Init from
    // uncensored data only.
    keel::f64 mu = 0.0, sd = 0.0;
    {
        keel::f64 s_un = 0.0, w_un = 0.0;
        for (keel::usize i = 0; i < N; ++i) if (!cn[i]) {
            s_un += fr[i] * y[i]; w_un += fr[i];
        }
        mu = s_un / w_un;
        keel::f64 sq = 0.0;
        for (keel::usize i = 0; i < N; ++i) if (!cn[i]) {
            const keel::f64 d = y[i] - mu; sq += fr[i] * d * d;
        }
        sd = keel::sqrt(std::max(sq / std::max(w_un - 1.0, 1.0), 1e-12));
    }
    const keel::i32    maxIter = 200;
    const keel::f64 tolFun  = 1e-10;
    for (keel::i32 it = 0; it < maxIter; ++it) {
        keel::f64 sumfy = 0.0, sumfy2 = 0.0;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 yi = y[i], fi = fr[i];
            if (!cn[i]) {
                sumfy  += fi * yi;
                sumfy2 += fi * yi * yi;
            } else {
                const keel::f64 a   = (yi - mu) / sd;
                const keel::f64 lam = inv_mills(a);
                const keel::f64 Ey  = mu + sd * lam;
                const keel::f64 Eym2 = sd * sd * (1.0 + a * lam);
                const keel::f64 Ey2  = Eym2 + 2.0 * mu * sd * lam + mu * mu;
                sumfy  += fi * Ey;
                sumfy2 += fi * Ey2;
            }
        }
        const keel::f64 mu_new  = sumfy / sumf;
        const keel::f64 var_new = std::max(sumfy2 / sumf - mu_new * mu_new, 1e-300);
        const keel::f64 sd_new  = keel::sqrt(var_new);
        const keel::f64 delta   = keel::abs(mu_new - mu) + keel::abs(sd_new - sd);
        mu = mu_new; sd = sd_new;
        if (delta < tolFun) break;
    }

    // Analytic observed Fisher info at the MLE.
    keel::f64 Imm = 0.0, Ims = 0.0, Iss = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 fi = fr[i];
        const keel::f64 a  = (y[i] - mu) / sd;
        if (!cn[i]) {
            Imm +=  fi / (sd * sd);
            Ims +=  2.0 * fi * (y[i] - mu) / (sd * sd * sd);
            Iss += -fi / (sd * sd) + 3.0 * fi * (y[i] - mu) * (y[i] - mu) / (sd * sd * sd * sd);
        } else {
            const keel::f64 m  = inv_mills(a);
            const keel::f64 mp = m * (m - a);
            Imm += fi * mp / (sd * sd);
            Ims += fi * (a * mp + m) / (sd * sd);
            Iss += fi * a * (a * mp + 2.0 * m) / (sd * sd);
        }
    }
    const keel::f64 det = Imm * Iss - Ims * Ims;
    keel::f64 SEmu = nan, SEsigma = nan;
    if (det > 0.0) {
        SEmu    = keel::sqrt(Iss / det);
        SEsigma = keel::sqrt(Imm / det);
    }
    const keel::f64 z = zNorm(alpha);

    R.mu = mu; R.sd = sd;
    R.mu_lo = mu - z * SEmu;
    R.mu_hi = mu + z * SEmu;
    const keel::f64 SElogS = (sd > 0.0) ? SEsigma / sd : nan;
    R.sd_lo = sd * keel::exp(-z * SElogS);
    R.sd_hi = sd * keel::exp( z * SElogS);
    R.ok = true;
    return R;
}

} // anonymous
namespace {
[[maybe_unused]] keel::f64 betainv_scalar2(keel::f64 p, keel::f64 a, keel::f64 b, keel::Allocator *mr)
{
    if (a <= 0.0 || b <= 0.0) return keel::Limits<keel::f64>::quiet_nan();
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 1.0;
    return betainv(Value::scalar(p, mr), a, b, mr).toScalar();
}
}
namespace {
constexpr keel::f64 kLog2Pi = 1.8378770664093454835606594728112352;
}

// MLE negative-log-likelihood workers (def in fit.cpp, external). lognlike_full/
// explike_full fill an optional 2x2 asymptotic-covariance via double *avarOut.
keel::f64 lognlike_full(keel::f64 mu, keel::f64 sigma, const Value &x,
                     const Value &cens, const Value &freq, keel::f64 *avarOut);
keel::f64 explike_full(keel::f64 mu, const Value &x,
                    const Value &cens, const Value &freq, keel::f64 *avarOut);
keel::f64 wbllike_full(keel::f64 scale, keel::f64 shape, const Value &x,
                    const Value &cens, const Value &freq,
                    keel::Allocator *mr);
keel::f64 evlike_full(keel::f64 mu, keel::f64 sigma, const Value &x,
                   const Value &cens, const Value &freq,
                   keel::Allocator *mr);

} // namespace numkit::stats
