// toolboxes/stats/src/fit/fit.cpp
// Distribution MLE fitters with confidence intervals.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/stats/fit/fit.hpp>

#include <numkit/stats/distributions/chi2.hpp>
#include <numkit/stats/distributions/students_t.hpp>
#include <numkit/stats/distributions/beta.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "fit_detail.hpp"

namespace numkit::stats {


std::tuple<Value, Value, Value, Value>
normfit(const Value &x, keel::f64 alpha, const Value &cens, const Value &freq, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();

    // CI Values are MATLAB-style 2x1 column vectors (rowCI is misnamed
    // — preserved for compat with existing callers).
    auto fail = [&]() {
        return std::make_tuple(Value::scalar(nan, mr),
                               Value::scalar(nan, mr),
                               rowCI(nan, nan, mr),
                               rowCI(nan, nan, mr));
    };
    if (N == 0) return fail();
    if (N == 1) {
        const keel::f64 v = x.elemAsDouble(0);
        return std::make_tuple(Value::scalar(v, mr),
                               Value::scalar(0.0, mr),
                               rowCI(nan, nan, mr),
                               rowCI(nan, nan, mr));
    }
    if (!cens.isEmpty() && cens.numel() != N) return fail();
    if (!freq.isEmpty() && freq.numel() != N) return fail();
    const bool has_cens = !cens.isEmpty();
    const bool has_freq = !freq.isEmpty();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64>  y(N, scratch.allocator());
    keel::Array<keel::f64>  fr(N, 1.0, scratch.allocator());
    keel::Array<keel::u8> cn(N, 0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        y[i] = x.elemAsDouble(i);
        if (has_freq) fr[i] = freq.elemAsDouble(i);
        if (has_cens) cn[i] = cens.elemAsDouble(i) > 0.5 ? 1 : 0;
    }
    auto R = normal_fit_mle(y, fr, cn, alpha, mr);
    if (!R.ok) return fail();
    return std::make_tuple(Value::scalar(R.mu, mr),
                           Value::scalar(R.sd, mr),
                           rowCI(R.mu_lo, R.mu_hi, mr),
                           rowCI(R.sd_lo, R.sd_hi, mr));
}

// Backward-compat 2-arg form.
std::tuple<Value, Value, Value, Value>
normfit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return normfit(x, alpha, Value::Empty, Value::Empty, mr);
}

std::tuple<Value, Value>
poissfit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    if (N == 0) return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    keel::f64 S = 0.0;
    for (keel::usize i = 0; i < N; ++i) S += x.elemAsDouble(i);
    const keel::f64 lambda = S / keel::f64(N);
    // Exact CI via chi² inversion (Garwood).
    const keel::f64 lo = (S == 0.0) ? 0.0
                                 : chi2inv_scalar(alpha / 2.0, 2.0 * S, mr)       / (2.0 * N);
    const keel::f64 hi = chi2inv_scalar(1.0 - alpha / 2.0, 2.0 * (S + 1.0), mr) / (2.0 * N);
    return {Value::scalar(lambda, mr), rowCI(lo, hi, mr)};
}

std::tuple<Value, Value>
expfit(const Value &x, keel::f64 alpha, const Value &cens, const Value &freq, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    if (N == 0) return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    // Validate optional vector lengths.
    if (!cens.isEmpty() && cens.numel() != N)
        return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    if (!freq.isEmpty() && freq.numel() != N)
        return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    const bool has_cens = !cens.isEmpty();
    const bool has_freq = !freq.isEmpty();
    // Total observation time T = Σ(freq · x); event count D = Σ(freq · (1-cens)).
    keel::f64 T = 0.0, D = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        const keel::f64 fi = has_freq ? freq.elemAsDouble(i) : 1.0;
        const keel::f64 ci = has_cens ? cens.elemAsDouble(i) : 0.0;
        T += fi * xi;
        D += fi * (1.0 - ci);
    }
    if (!(D > 0.0))
        return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    const keel::f64 mu = T / D;
    // Exact CI via χ²(2D): 2T/μ̂ ~ χ²(2D); CI[μ] = [2T/χ²₁₋α/₂, 2T/χ²_α/₂].
    const keel::f64 dof  = 2.0 * D;
    const keel::f64 chiU = chi2inv_scalar(1.0 - alpha / 2.0, dof, mr);
    const keel::f64 chiL = chi2inv_scalar(alpha / 2.0, dof, mr);
    const keel::f64 lo = 2.0 * T / chiU;
    const keel::f64 hi = 2.0 * T / chiL;
    return {Value::scalar(mu, mr), rowCI(lo, hi, mr)};
}

std::tuple<Value, Value, Value, Value>
unifit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    if (N == 0) {
        return {Value::scalar(nan, mr), Value::scalar(nan, mr),
                rowCI(nan, nan, mr), rowCI(nan, nan, mr)};
    }
    keel::f64 mn = x.elemAsDouble(0), mx = mn;
    for (keel::usize i = 1; i < N; ++i) {
        const keel::f64 v = x.elemAsDouble(i);
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    const keel::f64 range = mx - mn;
    const keel::f64 delta = range * (std::pow(alpha, -1.0 / keel::f64(N)) - 1.0);
    return {Value::scalar(mn, mr),
            Value::scalar(mx, mr),
            rowCI(mn - delta, mn, mr),
            rowCI(mx, mx + delta, mr)};
}

// ── lognfit ───────────────────────────────────────────────────────────

std::tuple<Value, Value>
lognfit(const Value &x, keel::f64 alpha, const Value &cens, const Value &freq, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    Value parm = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    Value pci  = Value::matrix(2, 2, ValueType::DOUBLE, mr);
    keel::f64 *pd = parm.doubleDataMut();
    keel::f64 *cd = pci.doubleDataMut();
    auto fail = [&]() {
        for (keel::i32 i = 0; i < 2; ++i) pd[i] = nan;
        for (keel::i32 i = 0; i < 4; ++i) cd[i] = nan;
        return std::make_tuple(std::move(parm), std::move(pci));
    };
    if (N < 2) return fail();
    if (!cens.isEmpty() && cens.numel() != N) return fail();
    if (!freq.isEmpty() && freq.numel() != N) return fail();
    const bool has_cens = !cens.isEmpty();
    const bool has_freq = !freq.isEmpty();

    // Build y = log(x); reject non-positive x.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64>  y(N, scratch.allocator());
    keel::Array<keel::f64>  fr(N, 1.0, scratch.allocator());
    keel::Array<keel::u8> cn(N, 0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (!(xi > 0.0)) return fail();
        y[i] = std::log(xi);
        if (has_freq) fr[i] = freq.elemAsDouble(i);
        if (has_cens) cn[i] = cens.elemAsDouble(i) > 0.5 ? 1 : 0;
    }
    auto R = normal_fit_mle(y, fr, cn, alpha, mr);
    if (!R.ok) return fail();
    pd[0] = R.mu;     pd[1] = R.sd;
    cd[0] = R.mu_lo;  cd[1] = R.mu_hi;
    cd[2] = R.sd_lo;  cd[3] = R.sd_hi;
    return {std::move(parm), std::move(pci)};
}

// Backward-compat 2-arg form.
std::tuple<Value, Value>
lognfit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    return lognfit(x, alpha, Value::Empty, Value::Empty, mr);
}

// ── binofit ───────────────────────────────────────────────────────────


namespace {
void binofit_argcheck(keel::usize Nx, keel::usize Nn, bool scalarN)
{
    if (!scalarN && Nn != Nx)
        throw Error("binofit: x and n must be the same length",
                    0, 0, "binofit", "", "numkit:binofit:size");
}
} // namespace

std::tuple<Value, Value>
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
        if (!(N > 0.0)) {
            const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
            pd[i] = nan;
            cd[i]      = nan;
            cd[i + Nx] = nan;
            continue;
        }
        const keel::f64 p = k / N;
        pd[i] = p;
        // Clopper-Pearson exact CI.
        const keel::f64 lo = (k == 0.0) ? 0.0
                                     : betainv_scalar2(alpha / 2.0, k, N - k + 1.0, mr);
        const keel::f64 hi = (k == N)   ? 1.0
                                     : betainv_scalar2(1.0 - alpha / 2.0, k + 1.0, N - k, mr);
        // Column-major: pci(:,1) = lower, pci(:,2) = upper.
        cd[i]       = lo;
        cd[i + Nx]  = hi;
    }
    if (Nx == 1) {
        // Collapse phat to a true scalar to match MATLAB's [phat,pci] = binofit(x,n).
        phat = Value::scalar(pd[0], mr);
    }
    return {std::move(phat), std::move(pci)};
}

// ── raylfit ───────────────────────────────────────────────────────────

std::tuple<Value, Value>
raylfit(const Value &x, keel::f64 alpha, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    if (N == 0)
        return {Value::scalar(nan, mr), rowCI(nan, nan, mr)};
    keel::f64 s2 = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        s2 += xi * xi;
    }
    const keel::f64 sigma = std::sqrt(s2 / (2.0 * keel::f64(N)));
    const keel::f64 chiU = chi2inv_scalar(1.0 - alpha / 2.0, 2.0 * keel::f64(N), mr);
    const keel::f64 chiL = chi2inv_scalar(alpha / 2.0, 2.0 * keel::f64(N), mr);
    const keel::f64 lo = sigma * std::sqrt(2.0 * keel::f64(N) / chiU);
    const keel::f64 hi = sigma * std::sqrt(2.0 * keel::f64(N) / chiL);
    return {Value::scalar(sigma, mr), rowCI(lo, hi, mr)};
}

// ── Negative log-likelihoods ──────────────────────────────────────────


keel::f64 normlike(keel::f64 mu, keel::f64 sigma, const Value &x, const Value &cens, const Value &freq, keel::Allocator * /*mr*/)
{
    const keel::usize N = x.numel();
    if (N == 0) return 0.0;
    if (!(sigma > 0.0)) return std::numeric_limits<keel::f64>::quiet_NaN();
    const bool useC = cens.numel() > 0;
    const bool useF = freq.numel() > 0;
    if (useC && cens.numel() != N)
        throw Error("normlike: censoring must match the data length",
                    0, 0, "normlike", "", "numkit:normlike:cens");
    if (useF && freq.numel() != N)
        throw Error("normlike: freq must match the data length",
                    0, 0, "normlike", "", "numkit:normlike:freq");

    const keel::f64 inv_s   = 1.0 / sigma;
    const keel::f64 inv_2s2 = 0.5 * inv_s * inv_s;
    const keel::f64 logS    = std::log(sigma);
    const keel::f64 halfLog2pi = 0.5 * kLog2Pi;
    const keel::f64 sqrt2_inv  = 1.0 / std::sqrt(2.0);
    keel::f64 nL = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 w = useF ? freq.elemAsDouble(i) : 1.0;
        if (w == 0.0) continue;                                  // MATLAB drops zero-freq rows
        const keel::f64 xi = x.elemAsDouble(i);
        if (std::isnan(xi)) return std::numeric_limits<keel::f64>::quiet_NaN();
        const keel::f64 d = xi - mu;
        const keel::f64 z = d * inv_s;
        const bool censored = useC && (cens.elemAsDouble(i) != 0.0);
        if (censored) {
            // -log(S(z)) = -log(0.5·erfc(z/sqrt(2)))
            const keel::f64 s = 0.5 * std::erfc(z * sqrt2_inv);
            nL += w * (-std::log(s));
        } else {
            // -log(f(z)) = log(σ) + 0.5·log(2π) + 0.5·z²
            nL += w * (logS + halfLog2pi + d * d * inv_2s2);
        }
    }
    return nL;
}

keel::f64 explike(keel::f64 mu, const Value &x, keel::Allocator * /*mr*/)
{
    // Two-arg form: empty data ⇒ 0 (matches MATLAB R2025b),
    //               mu <= 0   ⇒ NaN.
    const keel::usize N = x.numel();
    if (N == 0) return 0.0;
    if (mu <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
    keel::f64 sx = 0.0;
    for (keel::usize i = 0; i < N; ++i) sx += x.elemAsDouble(i);
    return keel::f64(N) * std::log(mu) + sx / mu;
}

// Extended form for the adapter: cens + freq + scalar avar.
// Returns nL via the function value; if `avarOut` is non-null, fills
// it with the inverse observed Fisher info (1/I).
// Uncensored row, weight w:  contributes w·(log μ + x/μ)
//   ∂²nL/∂μ² += w · (-1/μ² + 2 x / μ³)
// Right-censored row, weight w: contributes w·(x/μ)
//   ∂²nL/∂μ² += w · (2 x / μ³)
// Empty (after freq=0 drops) ⇒ 0.
keel::f64 explike_full(keel::f64 mu, const Value &x,
                           const Value &cens, const Value &freq,
                           keel::f64 *avarOut)
{
    const keel::usize N = x.numel();
    if (N == 0) {
        if (avarOut) *avarOut = std::numeric_limits<keel::f64>::quiet_NaN();
        return 0.0;
    }
    if (mu <= 0.0) {
        if (avarOut) *avarOut = std::numeric_limits<keel::f64>::quiet_NaN();
        return std::numeric_limits<keel::f64>::quiet_NaN();
    }
    const bool useC = cens.numel() > 0;
    const bool useF = freq.numel() > 0;
    if (useC && cens.numel() != N)
        throw Error("explike: censoring must match the data length",
                    0, 0, "explike", "", "numkit:explike:cens");
    if (useF && freq.numel() != N)
        throw Error("explike: freq must match the data length",
                    0, 0, "explike", "", "numkit:explike:freq");

    const keel::f64 inv_mu  = 1.0 / mu;
    const keel::f64 inv_mu2 = inv_mu * inv_mu;
    const keel::f64 inv_mu3 = inv_mu2 * inv_mu;
    const keel::f64 logMu   = std::log(mu);

    keel::f64 nL = 0.0, I = 0.0;
    bool any = false;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 w = useF ? freq.elemAsDouble(i) : 1.0;
        if (w == 0.0) continue;
        any = true;
        const keel::f64 xi = x.elemAsDouble(i);
        const bool censored = useC && (cens.elemAsDouble(i) != 0.0);
        if (censored) {
            nL += w * (xi * inv_mu);
            I  += w * (2.0 * xi * inv_mu3);
        } else {
            nL += w * (logMu + xi * inv_mu);
            I  += w * (-inv_mu2 + 2.0 * xi * inv_mu3);
        }
    }
    if (avarOut) {
        if (!any || I == 0.0 || !std::isfinite(I))
            *avarOut = std::numeric_limits<keel::f64>::quiet_NaN();
        else
            *avarOut = 1.0 / I;
    }
    return any ? nL : 0.0;
}

keel::f64 lognlike(keel::f64 mu, keel::f64 sigma, const Value &x, keel::Allocator * /*mr*/)
{
    // Two-arg form. Edges (matching MATLAB R2025b):
    //   empty data ⇒ 0;  sigma <= 0 ⇒ NaN;  any x[i] <= 0 ⇒ NaN.
    const keel::usize N = x.numel();
    if (N == 0) return 0.0;
    if (sigma <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
    keel::f64 sumLogX = 0.0, ss = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (xi <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
        const keel::f64 lx = std::log(xi);
        sumLogX += lx;
        const keel::f64 d = lx - mu;
        ss += d * d;
    }
    return sumLogX + keel::f64(N) * std::log(sigma)
         + 0.5 * keel::f64(N) * kLog2Pi + ss / (2.0 * sigma * sigma);
}

// Extended lognlike: cens + freq + 2×2 aVar (inverse observed-Fisher).
// Hessian wrt (mu, sigma) is structurally identical to the normal
// Hessian on y = log x (the per-row `log x_i` baseline is a constant
// in (mu, sigma)). Same uncensored / right-censored split as normlike.
// `avarOut` (4 doubles, column-major 2×2) filled iff non-null.
keel::f64 lognlike_full(keel::f64 mu, keel::f64 sigma, const Value &x,
                            const Value &cens, const Value &freq,
                            keel::f64 *avarOut)
{
    const keel::f64 NaNd = std::numeric_limits<keel::f64>::quiet_NaN();
    auto setAvarNaN = [&]() {
        if (avarOut) { for (keel::i32 i = 0; i < 4; ++i) avarOut[i] = NaNd; }
    };
    const keel::usize N = x.numel();
    if (N == 0) { setAvarNaN(); return 0.0; }
    if (sigma <= 0.0) { setAvarNaN(); return NaNd; }
    const bool useC = cens.numel() > 0;
    const bool useF = freq.numel() > 0;
    if (useC && cens.numel() != N)
        throw Error("lognlike: censoring must match the data length",
                    0, 0, "lognlike", "", "numkit:lognlike:cens");
    if (useF && freq.numel() != N)
        throw Error("lognlike: freq must match the data length",
                    0, 0, "lognlike", "", "numkit:lognlike:freq");

    const keel::f64 inv_s   = 1.0 / sigma;
    const keel::f64 inv_s2  = inv_s * inv_s;
    const keel::f64 inv_s3  = inv_s2 * inv_s;
    const keel::f64 inv_s4  = inv_s2 * inv_s2;
    const keel::f64 inv_2s2 = 0.5 * inv_s2;
    const keel::f64 logS    = std::log(sigma);
    const keel::f64 halfL2pi = 0.5 * kLog2Pi;
    const keel::f64 sqrt2_inv  = 1.0 / std::sqrt(2.0);
    const keel::f64 sqrt2pi_inv = 1.0 / std::sqrt(2.0 * 3.14159265358979323846);

    keel::f64 nL = 0.0;
    keel::f64 I00 = 0.0, I01 = 0.0, I11 = 0.0;
    bool any = false;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 w = useF ? freq.elemAsDouble(i) : 1.0;
        if (w == 0.0) continue;
        const keel::f64 xi = x.elemAsDouble(i);
        if (xi <= 0.0) { setAvarNaN(); return NaNd; }
        any = true;
        const keel::f64 lx = std::log(xi);
        const keel::f64 d  = lx - mu;
        const keel::f64 z  = d * inv_s;
        const bool censored = useC && (cens.elemAsDouble(i) != 0.0);
        if (!censored) {
            // -log f = log x + log σ + 0.5·log 2π + d²/(2σ²)
            nL  += w * (lx + logS + halfL2pi + d * d * inv_2s2);
            I00 += w * inv_s2;
            I11 += w * (-inv_s2 + 3.0 * d * d * inv_s4);
            I01 += w * 2.0 * d * inv_s3;
        } else {
            // -log S(z) = -log(0.5·erfc(z/√2)). For Hessian use the
            // same formulas as right-censored normal:
            //   I_μμ += w · h'/σ²
            //   I_σσ += w · (2 z h + z² h')/σ²
            //   I_μσ += w · (z h' + h)/σ²
            //   h = φ(z)/S(z), h' = h(h - z)
            const keel::f64 S = 0.5 * std::erfc(z * sqrt2_inv);
            nL += w * (-std::log(S));
            const keel::f64 phi = sqrt2pi_inv * std::exp(-0.5 * z * z);
            const keel::f64 h   = phi / S;
            const keel::f64 hp  = h * (h - z);
            I00 += w * hp * inv_s2;
            I11 += w * (2.0 * z * h + z * z * hp) * inv_s2;
            I01 += w * (z * hp + h) * inv_s2;
        }
    }
    if (avarOut) {
        if (!any) {
            setAvarNaN();
        } else {
            const keel::f64 det = I00 * I11 - I01 * I01;
            if (det == 0.0 || !std::isfinite(det)) {
                setAvarNaN();
            } else {
                const keel::f64 inv = 1.0 / det;
                avarOut[0] =  I11 * inv;
                avarOut[1] = -I01 * inv;
                avarOut[2] = -I01 * inv;
                avarOut[3] =  I00 * inv;
            }
        }
    }
    return any ? nL : 0.0;
}

keel::f64 gamlike(keel::f64 a, keel::f64 b, const Value &x, keel::Allocator * /*mr*/)
{
    const keel::usize N = x.numel();
    if (N == 0) return std::numeric_limits<keel::f64>::infinity();
    if (a <= 0.0 || b <= 0.0)
        return std::numeric_limits<keel::f64>::quiet_NaN();
    keel::f64 sumLogX = 0.0, sx = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (xi <= 0.0) return std::numeric_limits<keel::f64>::infinity();
        sumLogX += std::log(xi);
        sx += xi;
    }
    return -(a - 1.0) * sumLogX + sx / b
         + keel::f64(N) * a * std::log(b) + keel::f64(N) * std::lgamma(a);
}

keel::f64 betalike(keel::f64 a, keel::f64 b, const Value &x, keel::Allocator * /*mr*/)
{
    const keel::usize N = x.numel();
    if (N == 0) return std::numeric_limits<keel::f64>::infinity();
    if (a <= 0.0 || b <= 0.0)
        return std::numeric_limits<keel::f64>::quiet_NaN();
    keel::f64 sumLog = 0.0, sumLog1m = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (xi <= 0.0 || xi >= 1.0)
            return std::numeric_limits<keel::f64>::quiet_NaN();
        sumLog   += std::log(xi);
        sumLog1m += std::log1p(-xi);
    }
    const keel::f64 logBeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    return -(a - 1.0) * sumLog - (b - 1.0) * sumLog1m + keel::f64(N) * logBeta;
}

keel::f64 wbllike(keel::f64 scale, keel::f64 shape, const Value &x, keel::Allocator * /*mr*/)
{
    const keel::usize N = x.numel();
    if (N == 0) return 0.0;
    if (!(scale > 0.0) || !(shape > 0.0))
        return std::numeric_limits<keel::f64>::quiet_NaN();
    keel::f64 sumLogX = 0.0, sumPow = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (xi <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
        sumLogX += std::log(xi);
        sumPow  += std::pow(xi / scale, shape);
    }
    return -keel::f64(N) * std::log(shape) + keel::f64(N) * shape * std::log(scale)
         - (shape - 1.0) * sumLogX + sumPow;
}

// Extended form: cens + freq. Weibull(scale, shape).
//   pdf:      f(x) = (shape/scale)·(x/scale)^(shape-1)·exp(-(x/scale)^shape)
//   survival: S(x) = exp(-(x/scale)^shape)
//   -log f:   -log(shape) + shape·log(scale) - (shape-1)·log(x) + (x/scale)^shape
//   -log S:   (x/scale)^shape
keel::f64 wbllike_full(keel::f64 scale, keel::f64 shape, const Value &x, const Value &cens, const Value &freq, keel::Allocator * /*mr*/)
{
    const keel::usize N = x.numel();
    if (N == 0) return 0.0;
    if (!(scale > 0.0) || !(shape > 0.0))
        return std::numeric_limits<keel::f64>::quiet_NaN();
    const bool useC = cens.numel() > 0;
    const bool useF = freq.numel() > 0;
    if (useC && cens.numel() != N)
        throw Error("wbllike: censoring must match the data length",
                    0, 0, "wbllike", "", "numkit:wbllike:cens");
    if (useF && freq.numel() != N)
        throw Error("wbllike: freq must match the data length",
                    0, 0, "wbllike", "", "numkit:wbllike:freq");
    const keel::f64 logA = std::log(scale);
    const keel::f64 logK = std::log(shape);
    keel::f64 nL = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 w = useF ? freq.elemAsDouble(i) : 1.0;
        if (w == 0.0) continue;
        const keel::f64 xi = x.elemAsDouble(i);
        if (!(xi > 0.0)) return std::numeric_limits<keel::f64>::quiet_NaN();
        const keel::f64 pow_term = std::pow(xi / scale, shape);
        const bool censored = useC && (cens.elemAsDouble(i) != 0.0);
        if (censored) nL += w * pow_term;
        else          nL += w * (-logK + shape * logA - (shape - 1.0) * std::log(xi) + pow_term);
    }
    return nL;
}

keel::f64 evlike(keel::f64 mu, keel::f64 sigma, const Value &x, keel::Allocator * /*mr*/)
{
    const keel::usize N = x.numel();
    if (N == 0) return 0.0;                                  // matches MATLAB convention
    if (!(sigma > 0.0)) return std::numeric_limits<keel::f64>::quiet_NaN();
    keel::f64 sLin = 0.0, sExp = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 t = (x.elemAsDouble(i) - mu) / sigma;
        sLin += t;
        sExp += std::exp(t);
    }
    return keel::f64(N) * std::log(sigma) - sLin + sExp;
}

// Extended form: cens + freq.
//   Gumbel-min (Type-I extreme value): density f(z) = (1/σ)·exp(z)·exp(-exp(z)).
//   Uncensored contribution: log(σ) - z + exp(z), weight w.
//   Right-censored contribution: -log S(z) = exp(z), weight w.
keel::f64 evlike_full(keel::f64 mu, keel::f64 sigma, const Value &x, const Value &cens, const Value &freq, keel::Allocator * /*mr*/)
{
    const keel::usize N = x.numel();
    if (N == 0) return 0.0;
    if (!(sigma > 0.0)) return std::numeric_limits<keel::f64>::quiet_NaN();
    const bool useC = cens.numel() > 0;
    const bool useF = freq.numel() > 0;
    if (useC && cens.numel() != N)
        throw Error("evlike: censoring must match the data length",
                    0, 0, "evlike", "", "numkit:evlike:cens");
    if (useF && freq.numel() != N)
        throw Error("evlike: freq must match the data length",
                    0, 0, "evlike", "", "numkit:evlike:freq");
    const keel::f64 inv_s = 1.0 / sigma;
    const keel::f64 logS  = std::log(sigma);
    keel::f64 nL = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 w = useF ? freq.elemAsDouble(i) : 1.0;
        if (w == 0.0) continue;
        const keel::f64 xi = x.elemAsDouble(i);
        if (std::isnan(xi)) return std::numeric_limits<keel::f64>::quiet_NaN();
        const keel::f64 z  = (xi - mu) * inv_s;
        const keel::f64 ez = std::exp(z);
        const bool censored = useC && (cens.elemAsDouble(i) != 0.0);
        if (censored)  nL += w * ez;                           // -log S(z)
        else           nL += w * (logS - z + ez);              // -log f(z)
    }
    return nL;
}

keel::f64 gevlike(keel::f64 k, keel::f64 sigma, keel::f64 mu, const Value &x, keel::Allocator * /*mr*/)
{
    const keel::usize N = x.numel();
    if (N == 0) return std::numeric_limits<keel::f64>::infinity();
    if (sigma <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
    if (k == 0.0) {
        keel::f64 sZ = 0.0, sExp = 0.0;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 z = (x.elemAsDouble(i) - mu) / sigma;
            sZ += z;
            sExp += std::exp(-z);
        }
        return keel::f64(N) * std::log(sigma) + sZ + sExp;
    }
    keel::f64 sLogT = 0.0, sTinvk = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 z = (x.elemAsDouble(i) - mu) / sigma;
        const keel::f64 t = 1.0 + k * z;
        if (t <= 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
        sLogT  += std::log(t);
        sTinvk += std::pow(t, -1.0 / k);
    }
    return keel::f64(N) * std::log(sigma) + (1.0 / k + 1.0) * sLogT + sTinvk;
}

keel::f64 gplike(keel::f64 k, keel::f64 sigma, const Value &x, keel::Allocator * /*mr*/)
{
    // MATLAB's gplike does NOT enforce x >= 0 — it only requires the
    // per-point support condition `1 + k*x/sigma > 0`. With k>0 that
    // permits negative x close enough to 0; with k=0 (exponential
    // limit) the formula admits any finite x. Edges (matching MATLAB
    // R2025b probe — note the asymmetry vs gevlike):
    //   sigma == 0 → NaN
    //   sigma  < 0 → -Inf
    //   per-point support violation → +Inf (not NaN)
    const keel::usize N = x.numel();
    if (N == 0) return std::numeric_limits<keel::f64>::infinity();
    if (sigma == 0.0) return std::numeric_limits<keel::f64>::quiet_NaN();
    if (sigma  < 0.0) return -std::numeric_limits<keel::f64>::infinity();
    if (k == 0.0) {
        keel::f64 sX = 0.0;
        for (keel::usize i = 0; i < N; ++i) sX += x.elemAsDouble(i);
        return keel::f64(N) * std::log(sigma) + sX / sigma;
    }
    keel::f64 sLogT = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        const keel::f64 t = 1.0 + k * xi / sigma;
        if (t <= 0.0) return std::numeric_limits<keel::f64>::infinity();
        sLogT += std::log(t);
    }
    return keel::f64(N) * std::log(sigma) + (1.0 / k + 1.0) * sLogT;
}

} // namespace numkit::stats
