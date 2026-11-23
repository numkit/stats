// toolboxes/stats/src/distributions/dist_fits_detail.hpp
//
// Private (src-only) compute substrate for the dist_fits distribution:
// the scalar *K kernels + elementwise template + local helpers, shared
// between the engine-free compute (public *pdf/*cdf/*inv) in dist_fits.cpp and
// its CallContext register half in dist_fits_reg.cpp. Kept in an anonymous
// namespace (internal linkage per TU) — pure stateless math, no ODR risk.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/array.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <tuple>
#include <utility>


namespace numkit::stats {

namespace {

// Local digamma ψ(x) via Bernoulli asymptotic + recurrence to x ≥ 6.
[[maybe_unused]] keel::f64 digamma(keel::f64 x)
{
    keel::f64 s = 0.0;
    while (x < 6.0) {
        s -= 1.0 / x;
        x += 1.0;
    }
    const keel::f64 x2 = 1.0 / (x * x);
    keel::f64 r = keel::log(x) - 0.5 / x;
    r -= x2 * (1.0 / 12.0
              - x2 * (1.0 / 120.0
                     - x2 * (1.0 / 252.0)));
    return r + s;
}

// Trigamma ψ'(x) via same recurrence path.
[[maybe_unused]] keel::f64 trigamma(keel::f64 x)
{
    keel::f64 s = 0.0;
    while (x < 6.0) {
        s += 1.0 / (x * x);
        x += 1.0;
    }
    const keel::f64 x2 = 1.0 / (x * x);
    keel::f64 r = 1.0 / x + 0.5 / (x * x);
    r += (x2 / x) * (1.0 / 6.0
                    - x2 * (1.0 / 30.0
                           - x2 * (1.0 / 42.0)));
    return r + s;
}

[[maybe_unused]] keel::Array<keel::f64> toFlat(const Value &x)
{
    keel::Array<keel::f64> v(x.numel());
    for (keel::usize i = 0; i < v.size(); ++i) v[i] = x.elemAsDouble(i);
    return v;
}

// ── Generic 2-D Wald CI helper for MLE fitters ──────────────────────
//
// Given an NLL functor f(t1, t2) and the MLE point (t1_hat, t2_hat),
// computes the 2×2 confidence matrix by:
//   1. Central-FD Hessian of NLL → observed Fisher information I.
//   2. V = I^{-1}; SE_lin_j = sqrt(V[j,j]).
//   3. Per-parameter Wald CI on the appropriate transformed scale
//      (LINEAR / LOG / LOGIT — MATLAB conventions), then inverse-
//      transform back. z_{α/2} from `norminv(1 - α/2)`.
//
// Returns a 2×2 matrix in MATLAB convention: [lo_row; hi_row] with
// parameters as columns.

enum class CITransform { LINEAR, LOG, LOGIT };

template <typename NllFn>
Value wald_ci_2d(NllFn &&nll, keel::f64 t1, keel::f64 t2,
                 CITransform tr1, CITransform tr2,
                 keel::f64 alpha, keel::Allocator *mr)
{
    // FD step: eps^{1/3} ≈ 6e-6 is theoretically optimal for 2nd-deriv
    // central FD; relax to 1e-4 for distributions with curved score
    // functions (digamma/trigamma) to suppress roundoff noise.
    const keel::f64 h1 = std::max(keel::abs(t1), 1.0) * 1e-4;
    const keel::f64 h2 = std::max(keel::abs(t2), 1.0) * 1e-4;
    const keel::f64 f00   = nll(t1, t2);
    const keel::f64 f1p   = nll(t1 + h1, t2);
    const keel::f64 f1m   = nll(t1 - h1, t2);
    const keel::f64 f2p   = nll(t1, t2 + h2);
    const keel::f64 f2m   = nll(t1, t2 - h2);
    const keel::f64 f12pp = nll(t1 + h1, t2 + h2);
    const keel::f64 f12mm = nll(t1 - h1, t2 - h2);
    const keel::f64 f12pm = nll(t1 + h1, t2 - h2);
    const keel::f64 f12mp = nll(t1 - h1, t2 + h2);
    const keel::f64 H11 = (f1p - 2.0 * f00 + f1m) / (h1 * h1);
    const keel::f64 H22 = (f2p - 2.0 * f00 + f2m) / (h2 * h2);
    const keel::f64 H12 = (f12pp - f12pm - f12mp + f12mm) / (4.0 * h1 * h2);
    const keel::f64 det = H11 * H22 - H12 * H12;
    Value out = Value::matrix(2, 2, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    if (keel::abs(det) < 1e-300 || H11 < 0.0 || H22 < 0.0) {
        for (keel::i32 i = 0; i < 4; ++i) od[i] = keel::Limits<keel::f64>::quiet_nan();
        return out;
    }
    const keel::f64 V11 = H22 / det;
    const keel::f64 V22 = H11 / det;
    const keel::f64 se1 = keel::sqrt(std::max(V11, 0.0));
    const keel::f64 se2 = keel::sqrt(std::max(V22, 0.0));

    // z_{α/2} from norminv(1 - α/2). Series via std::erf inversion is
    // unnecessary — use the closed-form via erfc^{-1} approximation:
    // For α = 0.05, z = 1.95996398454005423552.
    auto z_alpha = [](keel::f64 a) -> keel::f64 {
        // Wichura AS 241: high-accuracy norm-inv for p = 1 - a/2.
        const keel::f64 p = 1.0 - 0.5 * a;
        const keel::f64 q = p - 0.5;
        if (keel::abs(q) <= 0.425) {
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
        r = keel::sqrt(-keel::log(r));
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

    auto apply_ci = [z](keel::f64 t, keel::f64 se, CITransform tr) -> std::pair<keel::f64, keel::f64> {
        switch (tr) {
            case CITransform::LINEAR:
                return {t - z * se, t + z * se};
            case CITransform::LOG: {
                if (!(t > 0.0))
                    return {keel::Limits<keel::f64>::quiet_nan(),
                            keel::Limits<keel::f64>::quiet_nan()};
                const keel::f64 se_log = se / t;
                return {t * keel::exp(-z * se_log), t * keel::exp(z * se_log)};
            }
            case CITransform::LOGIT: {
                if (!(t > 0.0 && t < 1.0))
                    return {keel::Limits<keel::f64>::quiet_nan(),
                            keel::Limits<keel::f64>::quiet_nan()};
                const keel::f64 se_logit = se / (t * (1.0 - t));
                const keel::f64 logit = keel::log(t / (1.0 - t));
                const keel::f64 lo = logit - z * se_logit;
                const keel::f64 hi = logit + z * se_logit;
                return {1.0 / (1.0 + keel::exp(-lo)),
                        1.0 / (1.0 + keel::exp(-hi))};
            }
        }
        return {t, t};
    };
    auto [lo1, hi1] = apply_ci(t1, se1, tr1);
    auto [lo2, hi2] = apply_ci(t2, se2, tr2);
    // 2×2 column-major: [lo1; hi1] [lo2; hi2]
    od[0] = lo1; od[1] = hi1; od[2] = lo2; od[3] = hi2;
    return out;
}

} // namespace
namespace {

// NLL definitions for each fit. Used by FD-Hessian Wald CI.

[[maybe_unused]] keel::f64 nll_gam(keel::f64 a, keel::f64 b, const keel::Array<keel::f64> &xv)
{
    if (!(a > 0.0) || !(b > 0.0)) return keel::Limits<keel::f64>::infinity();
    const keel::f64 N = keel::as<keel::f64>(xv.size());
    keel::f64 sx = 0.0, slx = 0.0;
    for (keel::f64 v : xv) { sx += v; slx += keel::log(v); }
    return N * keel::lgamma(a) + N * a * keel::log(b)
         - (a - 1.0) * slx + sx / b;
}

[[maybe_unused]] keel::f64 nll_wbl(keel::f64 a, keel::f64 b, const keel::Array<keel::f64> &xv)
{
    if (!(a > 0.0) || !(b > 0.0)) return keel::Limits<keel::f64>::infinity();
    const keel::f64 N = keel::as<keel::f64>(xv.size());
    keel::f64 slx = 0.0, sw = 0.0;
    for (keel::f64 v : xv) {
        slx += keel::log(v);
        sw  += keel::pow(v / a, b);
    }
    return -N * keel::log(b) + N * b * keel::log(a)
         - (b - 1.0) * slx + sw;
}

[[maybe_unused]] keel::f64 nll_beta(keel::f64 a, keel::f64 b, const keel::Array<keel::f64> &xv)
{
    if (!(a > 0.0) || !(b > 0.0)) return keel::Limits<keel::f64>::infinity();
    const keel::f64 N = keel::as<keel::f64>(xv.size());
    keel::f64 slx = 0.0, sl1mx = 0.0;
    for (keel::f64 v : xv) {
        slx   += keel::log(v);
        sl1mx += keel::log(1.0 - v);
    }
    return N * (keel::lgamma(a) + keel::lgamma(b) - keel::lgamma(a + b))
         - (a - 1.0) * slx - (b - 1.0) * sl1mx;
}

[[maybe_unused]] keel::f64 nll_nbin(keel::f64 r, keel::f64 p, const keel::Array<keel::f64> &xv)
{
    if (!(r > 0.0) || !(p > 0.0) || !(p < 1.0))
        return keel::Limits<keel::f64>::infinity();
    const keel::f64 N = keel::as<keel::f64>(xv.size());
    keel::f64 term = 0.0;
    keel::f64 sx = 0.0;
    for (keel::f64 v : xv) {
        term -= keel::lgamma(v + r) - keel::lgamma(r) - keel::lgamma(v + 1.0);
        sx   += v;
    }
    return term - N * r * keel::log(p) - sx * keel::log(1.0 - p);
}

[[maybe_unused]] keel::f64 nll_ev(keel::f64 mu, keel::f64 sigma, const keel::Array<keel::f64> &xv)
{
    if (!(sigma > 0.0)) return keel::Limits<keel::f64>::infinity();
    const keel::f64 N = keel::as<keel::f64>(xv.size());
    // log f(x) = -log σ + t - exp(t), t = (x-μ)/σ. NLL is sum of -log f.
    keel::f64 st = 0.0, set = 0.0;
    for (keel::f64 v : xv) {
        const keel::f64 t = (v - mu) / sigma;
        st  += t;
        set += keel::exp(t);
    }
    return N * keel::log(sigma) - st + set;
}

// gpfit NLL is already defined as gp_nll above.

} // anonymous
namespace {

// Build 2×2 Wald CI from analytical Fisher information (H = Hessian
// of NLL at MLE).
[[maybe_unused]] Value wald_ci_from_hessian(keel::f64 t1, keel::f64 t2,
                           keel::f64 H11, keel::f64 H22, keel::f64 H12,
                           CITransform tr1, CITransform tr2,
                           keel::f64 alpha, keel::Allocator *mr)
{
    Value out = Value::matrix(2, 2, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 det = H11 * H22 - H12 * H12;
    if (keel::abs(det) < 1e-300 || H11 < 0.0 || H22 < 0.0) {
        for (keel::i32 i = 0; i < 4; ++i) od[i] = keel::Limits<keel::f64>::quiet_nan();
        return out;
    }
    const keel::f64 V11 = H22 / det;
    const keel::f64 V22 = H11 / det;
    const keel::f64 se1 = keel::sqrt(std::max(V11, 0.0));
    const keel::f64 se2 = keel::sqrt(std::max(V22, 0.0));
    // z_{α/2} — at default α=0.05 returns 1.959963984540054.
    const keel::f64 z = [](keel::f64 a) -> keel::f64 {
        const keel::f64 p = 1.0 - 0.5 * a;
        const keel::f64 q = p - 0.5;
        if (keel::abs(q) <= 0.425) {
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
        r = keel::sqrt(-keel::log(r));
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
    }(alpha);
    auto apply_ci = [z](keel::f64 t, keel::f64 se, CITransform tr) -> std::pair<keel::f64, keel::f64> {
        switch (tr) {
            case CITransform::LINEAR:
                return {t - z * se, t + z * se};
            case CITransform::LOG: {
                if (!(t > 0.0))
                    return {keel::Limits<keel::f64>::quiet_nan(),
                            keel::Limits<keel::f64>::quiet_nan()};
                const keel::f64 se_log = se / t;
                return {t * keel::exp(-z * se_log), t * keel::exp(z * se_log)};
            }
            case CITransform::LOGIT: {
                if (!(t > 0.0 && t < 1.0))
                    return {keel::Limits<keel::f64>::quiet_nan(),
                            keel::Limits<keel::f64>::quiet_nan()};
                const keel::f64 se_logit = se / (t * (1.0 - t));
                const keel::f64 logit = keel::log(t / (1.0 - t));
                const keel::f64 lo = logit - z * se_logit;
                const keel::f64 hi = logit + z * se_logit;
                return {1.0 / (1.0 + keel::exp(-lo)),
                        1.0 / (1.0 + keel::exp(-hi))};
            }
        }
        return {t, t};
    };
    auto [lo1, hi1] = apply_ci(t1, se1, tr1);
    auto [lo2, hi2] = apply_ci(t2, se2, tr2);
    od[0] = lo1; od[1] = hi1; od[2] = lo2; od[3] = hi2;
    return out;
}

} // anonymous

} // namespace numkit::stats
