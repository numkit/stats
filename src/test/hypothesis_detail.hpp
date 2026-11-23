// toolboxes/.../hypothesis_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by hypothesis.cpp + hypothesis_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/random.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <numeric>
#include <keel/tuple.h>


namespace numkit::stats {

namespace {

[[maybe_unused]] void mean_var(const Value &x, keel::f64 &mean_out, keel::f64 &var_out, keel::usize &n_out) {
    const keel::usize N = x.numel();
    n_out = N;
    if (N == 0) { mean_out = 0.0; var_out = 0.0; return; }
    keel::f64 s = 0.0;
    for (keel::usize i = 0; i < N; ++i) s += x.elemAsDouble(i);
    mean_out = s / keel::f64(N);
    if (N < 2) { var_out = 0.0; return; }
    keel::f64 sq = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 d = x.elemAsDouble(i) - mean_out;
        sq += d * d;
    }
    var_out = sq / keel::f64(N - 1);
}

inline TestTail parse_tail(keel::StringRef s, TestTail def) {
    if (s == "both")     return TestTail::Both;
    if (s == "right")    return TestTail::Right;
    if (s == "left")     return TestTail::Left;
    // kstest / kstest2 aliases:
    if (s == "unequal")  return TestTail::Both;
    if (s == "larger")   return TestTail::Right;
    if (s == "smaller")  return TestTail::Left;
    return def;
}

// Compute two-sided / one-sided p-value from a t-statistic and df.
[[maybe_unused]] keel::f64 tpvalue(keel::f64 tstat, keel::f64 df, TestTail tail, keel::Allocator *mr) {
    Value tv = Value::scalar(tstat, mr);
    Value cdf_v = tcdf(tv, df, mr);
    const keel::f64 cdf = cdf_v.toScalar();
    switch (tail) {
        case TestTail::Both:  return 2.0 * std::min(cdf, 1.0 - cdf);
        case TestTail::Right: return 1.0 - cdf;
        case TestTail::Left:  return cdf;
    }
    return 1.0;
}

[[maybe_unused]] keel::f64 zpvalue(keel::f64 z, TestTail tail, keel::Allocator *mr) {
    Value zv = Value::scalar(z, mr);
    Value cdf_v = normcdf(zv, 0.0, 1.0, mr);
    const keel::f64 cdf = cdf_v.toScalar();
    switch (tail) {
        case TestTail::Both:  return 2.0 * std::min(cdf, 1.0 - cdf);
        case TestTail::Right: return 1.0 - cdf;
        case TestTail::Left:  return cdf;
    }
    return 1.0;
}

} // anonymous
namespace {

// ── Exact Kolmogorov-Smirnov distributions ──────────────────────────
// One-sample p-values use the EXACT finite-n distribution (MATLAB's default):
// two-sided via the Marsaglia-Tsang-Wang (2003) matrix method, one-sided via
// the Birnbaum-Tingey (1951) closed form. Two-sample uses Stephens' corrected
// asymptotic. Critical values come from inverting the matching p-function.

void mMultiply(keel::ArrayRef<keel::f64> A, keel::ArrayRef<keel::f64> B,
               keel::Array<keel::f64> &C, keel::i32 m) {
    for (keel::i32 i = 0; i < m; ++i)
        for (keel::i32 j = 0; j < m; ++j) {
            keel::f64 s = 0.0;
            for (keel::i32 k = 0; k < m; ++k) s += A[i*m+k] * B[k*m+j];
            C[i*m+j] = s;
        }
}

// V = A^n with an explicit base-10 exponent eV (rescaled to dodge overflow).
void mPower(keel::ArrayRef<keel::f64> A, keel::i32 eA, keel::Array<keel::f64> &V,
            keel::i32 &eV, keel::i32 m, keel::i32 n) {
    if (n == 1) {
        V.resize(A.size());
        std::copy(A.begin(), A.end(), V.begin());
        eV = eA;
        return;
    }
    mPower(A, eA, V, eV, m, n / 2);
    keel::Array<keel::f64> B(keel::as<keel::usize>(m) * m);
    mMultiply(V.as_ref(), V.as_ref(), B, m);
    keel::i32 eB = 2 * eV;
    if (n % 2 == 0) { V = keel::move(B); eV = eB; }
    else { mMultiply(A, B.as_ref(), V, m); eV = eA + eB; }
    if (V[(m/2)*m + (m/2)] > 1e140) {
        for (keel::f64 &x : V) x *= 1e-140;
        eV += 140;
    }
}

// P(D_n < d) — exact two-sided Kolmogorov CDF (Marsaglia-Tsang-Wang 2003).
keel::f64 marsagliaK(keel::i32 n, keel::f64 d) {
    if (d <= 0.0) return 0.0;
    if (d >= 1.0) return 1.0;
    const keel::f64 nd = n * d;
    const keel::i32 k = keel::as<keel::i32>(nd) + 1;
    const keel::i32 m = 2 * k - 1;
    const keel::f64 h = k - nd;
    keel::Array<keel::f64> H(keel::as<keel::usize>(m) * m), Q(keel::as<keel::usize>(m) * m, 0.0);
    for (keel::i32 i = 0; i < m; ++i)
        for (keel::i32 j = 0; j < m; ++j)
            H[i*m+j] = (i - j + 1 < 0) ? 0.0 : 1.0;
    for (keel::i32 i = 0; i < m; ++i) {
        H[i*m + 0]     -= keel::pow(h, i + 1);
        H[(m-1)*m + i] -= keel::pow(h, m - i);
    }
    H[(m-1)*m + 0] += (2.0*h - 1.0 > 0.0) ? keel::pow(2.0*h - 1.0, m) : 0.0;
    for (keel::i32 i = 0; i < m; ++i)
        for (keel::i32 j = 0; j < m; ++j)
            if (i - j + 1 > 0)
                for (keel::i32 g = 1; g <= i - j + 1; ++g) H[i*m+j] /= g;
    keel::i32 eQ = 0;
    mPower(H.as_ref(), 0, Q, eQ, m, n);
    keel::f64 s = Q[(k-1)*m + (k-1)];
    for (keel::i32 i = 1; i <= n; ++i) {
        s = s * i / n;
        if (s < 1e-140) { s *= 1e140; eQ -= 140; }
    }
    return s * keel::pow(10.0, keel::as<keel::f64>(eQ));
}

// P(D_n^+ >= d) — exact one-sided KS survival (Birnbaum-Tingey 1951), log space.
keel::f64 birnbaumTingey(keel::i32 n, keel::f64 d) {
    if (d <= 0.0) return 1.0;
    if (d >= 1.0) return 0.0;
    const keel::f64 dn = keel::as<keel::f64>(n);
    const keel::i32 jmax = keel::as<keel::i32>(keel::floor(dn * (1.0 - d)));
    keel::f64 sum = 0.0;
    for (keel::i32 j = 0; j <= jmax; ++j) {
        const keel::f64 a = 1.0 - d - j / dn;
        const keel::f64 b = d + j / dn;
        if (a <= 0.0 && n - j > 0) continue;     // a^(n-j) == 0
        const keel::f64 logC = keel::lgamma(dn + 1.0) - keel::lgamma(j + 1.0)
                          - keel::lgamma(dn - j + 1.0);
        keel::f64 logterm = logC + (j - 1) * keel::log(b);
        if (n - j > 0) logterm += (dn - j) * keel::log(a);
        sum += keel::exp(logterm);
    }
    return std::min(1.0, std::max(0.0, d * sum));
}

// Two-sided one-sample p: exact for small statistics; MATLAB's corrected
// asymptotic 2·exp(-(2.000071 + .331/√n + 1.409/n)·n·D²) for large ones.
[[maybe_unused]] keel::f64 ksOneSampleTwoSidedP(keel::i32 n, keel::f64 D) {
    const keel::f64 s = keel::as<keel::f64>(n) * D * D;
    if (s > 7.24 || (s > 3.76 && n > 99))
        return std::min(1.0, 2.0 * keel::exp(-(2.000071 + 0.331/keel::sqrt((keel::f64)n)
                                              + 1.409/n) * s));
    return 1.0 - marsagliaK(n, D);
}

// One-sided one-sample p (exact Birnbaum-Tingey on the directional statistic).
[[maybe_unused]] keel::f64 ksOneSampleOneSidedP(keel::i32 n, keel::f64 D) { return birnbaumTingey(n, D); }

// Two-sample KS p (Stephens' corrected asymptotic): two-sided is the
// alternating series, one-sided the single leading term.
[[maybe_unused]] keel::f64 ksTwoSampleP(keel::f64 neff, keel::f64 D, bool twoSided) {
    const keel::f64 lambda = (keel::sqrt(neff) + 0.12 + 0.11/keel::sqrt(neff)) * D;
    if (!twoSided)
        return std::min(1.0, std::max(0.0, keel::exp(-2.0 * lambda * lambda)));
    keel::f64 total = 0.0;
    for (keel::i32 k = 1; k <= 101; ++k) {
        const keel::f64 term = keel::exp(-2.0 * lambda * lambda * k * k);
        total += (k & 1) ? term : -term;
        if (term < 1e-20 && k > 4) break;
    }
    return std::min(1.0, std::max(0.0, 2.0 * total));
}

// Critical value: invert a monotone-decreasing p(D) so that p(cv) = alpha.
template <typename PFn>
keel::f64 ksCriticalValue(PFn pfn, keel::f64 alpha) {
    keel::f64 lo = 0.0, hi = 1.0;
    for (keel::i32 it = 0; it < 100; ++it) {
        const keel::f64 mid = 0.5 * (lo + hi);
        if (pfn(mid) > alpha) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Evaluate a piecewise-linear empirical CDF at point x given a sorted
// reference grid (x_grid, F_grid). Outside the grid, clamp to 0 / 1.
[[maybe_unused]] keel::f64 interp_cdf(keel::ArrayRef<keel::f64> xg,
                  keel::ArrayRef<keel::f64> Fg, keel::f64 x) {
    if (xg.empty()) return 0.0;
    if (x <= xg.front()) return Fg.front();
    if (x >= xg.back())  return Fg.back();
    auto it = std::upper_bound(xg.begin(), xg.end(), x);
    const keel::usize k = (keel::usize)(it - xg.begin()) - 1;
    const keel::f64 t = (x - xg[k]) / (xg[k + 1] - xg[k]);
    return Fg[k] + t * (Fg[k + 1] - Fg[k]);
}

} // anonymous
namespace {

// Compute the JB statistic for a sample (used by both the main entry
// point and the Monte-Carlo H₀ simulator).
inline keel::f64 jb_stat(const keel::f64 *x, keel::usize N) {
    keel::f64 mean = 0.0;
    for (keel::usize i = 0; i < N; ++i) mean += x[i];
    mean /= keel::f64(N);
    keel::f64 m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 d = x[i] - mean;
        const keel::f64 d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= keel::f64(N); m3 /= keel::f64(N); m4 /= keel::f64(N);
    const keel::f64 S = (m2 > 0.0) ? m3 / keel::pow(m2, 1.5) : 0.0;
    const keel::f64 K = (m2 > 0.0) ? m4 / (m2 * m2) : 0.0;
    return keel::f64(N) / 6.0 * (S * S + 0.25 * (K - 3.0) * (K - 3.0));
}

// Monte-Carlo p / critval for JB at sample size N. Iterates batches
// until SE(p̂) < mctol or until iter cap. Uses a deterministic seed
// (so spec output is reproducible). Returns (p, critval) where p is
// MATLAB-capped at 0.5.
struct JBMCResult { keel::f64 p; keel::f64 cv; };
[[maybe_unused]] JBMCResult jb_montecarlo(keel::usize N, keel::f64 JB_obs, keel::f64 alpha, keel::f64 mctol)
{
    // Hard caps: at most 1e6 reps, at least 1000 batch.
    const keel::usize batch     = 1000;
    const keel::usize max_reps  = 1'000'000;
    const keel::f64 cap_p     = 0.5;  // MATLAB caps p at 0.5

    keel::random::Xoshiro256PlusPlus gen(12345ULL);  // deterministic
    keel::Array<keel::f64> buf(N);
    keel::Array<keel::f64> JB_h0;
    JB_h0.reserve(batch);

    keel::usize total = 0, exceeds = 0;
    while (total < max_reps) {
        const keel::usize this_batch = std::min(batch, max_reps - total);
        for (keel::usize i = 0; i < this_batch; ++i) {
            for (keel::usize k = 0; k < N; ++k) buf[k] = keel::random::normal_f64(gen);
            const keel::f64 j = jb_stat(buf.data(), N);
            JB_h0.push_back(j);
            if (j >= JB_obs) ++exceeds;
        }
        total += this_batch;
        const keel::f64 p_hat = keel::f64(exceeds) / keel::f64(total);
        const keel::f64 se = keel::sqrt(std::max(p_hat * (1.0 - p_hat), 1.0 / keel::f64(total))
                                    / keel::f64(total));
        if (se < mctol && total >= 3 * batch) break;
    }
    // Critical value: (1-alpha) quantile of the empirical H₀ distribution.
    std::sort(JB_h0.begin(), JB_h0.end());
    const keel::f64 rank = (1.0 - alpha) * (keel::f64(JB_h0.size()) - 1.0);
    const keel::usize lo = (keel::usize)keel::floor(rank);
    const keel::usize hi = std::min(lo + 1, JB_h0.size() - 1);
    const keel::f64 frac = rank - keel::f64(lo);
    const keel::f64 cv = JB_h0[lo] * (1.0 - frac) + JB_h0[hi] * frac;

    keel::f64 p = keel::f64(exceeds) / keel::f64(total);
    if (p > cap_p) p = cap_p;
    return {p, cv};
}

} // anonymous
namespace {

// Group buckets: parallel `ns` (per-group sample size) and `data`
// (concatenated values, with `offsets[i]..offsets[i+1]`).
struct Groups {
    keel::Array<keel::usize>  ns;        // size k
    keel::Array<keel::usize>  offsets;   // size k+1
    keel::Array<keel::f64>  data;      // total N
    explicit Groups(keel::Allocator *mr)
        : ns(mr), offsets(mr), data(mr) {}
};

[[maybe_unused]] Groups bucket_by_group(const Value &x, const Value &group, keel::Allocator *scratch_mr)
{
    const keel::usize Nx = x.numel();
    Groups G(scratch_mr);
    // First pass: collect distinct labels (preserving first-encounter
    // order) + per-label counts; ignore NaN values/labels.
    keel::Array<keel::f64> labels(scratch_mr);  // distinct group labels
    labels.reserve(8);
    keel::Array<keel::usize> counts(scratch_mr);
    counts.reserve(8);
    keel::Array<keel::i32> bucket_for(Nx, -1, scratch_mr);
    for (keel::usize i = 0; i < Nx; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        const keel::f64 gi = group.elemAsDouble(i);
        if (keel::is_nan(xi) || keel::is_nan(gi)) continue;
        keel::i32 b = -1;
        for (keel::usize j = 0; j < labels.size(); ++j)
            if (labels[j] == gi) { b = (keel::i32)j; break; }
        if (b < 0) {
            b = (keel::i32)labels.size();
            labels.push_back(gi);
            counts.push_back(0);
        }
        bucket_for[i] = b;
        ++counts[(keel::usize)b];
    }
    const keel::usize k = labels.size();
    G.ns.assign(counts);
    G.offsets.resize(k + 1);
    G.offsets[0] = 0;
    for (keel::usize i = 0; i < k; ++i) G.offsets[i + 1] = G.offsets[i] + counts[i];
    G.data.resize(G.offsets[k]);
    keel::Array<keel::usize> cursor(k, 0, scratch_mr);
    for (keel::usize i = 0; i < Nx; ++i) {
        const keel::i32 b = bucket_for[i];
        if (b < 0) continue;
        G.data[G.offsets[b] + cursor[b]++] = x.elemAsDouble(i);
    }
    return G;
}

// One-way ANOVA on Z values stored in groups buckets. Returns
// (F, df1, df2, p).
struct AnovaOut { keel::f64 F, df1, df2, p; };
[[maybe_unused]] AnovaOut anova1_on_groups(const keel::Array<keel::f64> &Z, const keel::Array<keel::usize> &offsets, const keel::Array<keel::usize> &ns, keel::Allocator *mr)
{
    const keel::usize k = ns.size();
    keel::usize N = 0;
    for (keel::usize i = 0; i < k; ++i) N += ns[i];

    // Group means.
    keel::Array<keel::f64> Zbar(k, 0.0, Z.allocator());
    keel::f64 grand = 0.0;
    for (keel::usize g = 0; g < k; ++g) {
        keel::f64 s = 0.0;
        for (keel::usize j = offsets[g]; j < offsets[g + 1]; ++j) s += Z[j];
        Zbar[g] = (ns[g] > 0) ? s / keel::f64(ns[g]) : 0.0;
        grand += s;
    }
    grand /= keel::f64(N);

    keel::f64 SSB = 0.0, SSW = 0.0;
    for (keel::usize g = 0; g < k; ++g) {
        const keel::f64 db = Zbar[g] - grand;
        SSB += keel::f64(ns[g]) * db * db;
        for (keel::usize j = offsets[g]; j < offsets[g + 1]; ++j) {
            const keel::f64 dw = Z[j] - Zbar[g];
            SSW += dw * dw;
        }
    }
    const keel::f64 df1 = keel::f64(k - 1);
    const keel::f64 df2 = keel::f64(N - k);
    const keel::f64 MSB = SSB / df1;
    const keel::f64 MSW = SSW / df2;
    const keel::f64 F = (MSW > 0.0) ? MSB / MSW : 0.0;
    Value Fv = Value::scalar(F, mr);
    const keel::f64 cdf = fcdf(Fv, df1, df2, mr).toScalar();
    return {F, df1, df2, std::max(0.0, 1.0 - cdf)};
}

inline keel::f64 median_sorted(keel::f64 *a, keel::usize n) {
    std::sort(a, a + n);
    if (n & 1u) return a[n / 2];
    return 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

} // anonymous
namespace {

inline keel::f64 log_binom(keel::f64 n, keel::f64 k) {
    if (k < 0.0 || k > n) return -keel::Limits<keel::f64>::infinity();
    return keel::lgamma(n + 1.0) - keel::lgamma(k + 1.0) - keel::lgamma(n - k + 1.0);
}

// Exact P(R = r) under H0 for a sequence with n1 ones and n0 zeros.
// r in [1, n1+n0]; returns 0 outside the support.
[[maybe_unused]] keel::f64 exact_runs_pmf(keel::i32 r, keel::i32 n1, keel::i32 n0)
{
    if (r < 2) return 0.0;
    const keel::f64 logC = log_binom(keel::f64(n1 + n0), keel::f64(n1));
    if (r % 2 == 0) {
        const keel::i32 k = r / 2;
        if (k < 1 || k > n1 || k > n0) return 0.0;
        const keel::f64 lp = keel::log(2.0) + log_binom(keel::f64(n1 - 1), keel::f64(k - 1))
                        + log_binom(keel::f64(n0 - 1), keel::f64(k - 1)) - logC;
        return keel::exp(lp);
    }
    const keel::i32 k = (r - 1) / 2;            // r = 2k+1
    keel::f64 t1 = -keel::Limits<keel::f64>::infinity();
    keel::f64 t2 = -keel::Limits<keel::f64>::infinity();
    // Start with 1: (k+1) ones-blocks, k zero-blocks → C(n1-1, k)·C(n0-1, k-1)
    if (k <= n1 - 1 && k - 1 >= 0 && k - 1 <= n0 - 1)
        t1 = log_binom(keel::f64(n1 - 1), keel::f64(k))
           + log_binom(keel::f64(n0 - 1), keel::f64(k - 1));
    // Start with 0: k ones-blocks, (k+1) zero-blocks → C(n1-1, k-1)·C(n0-1, k)
    if (k - 1 >= 0 && k - 1 <= n1 - 1 && k <= n0 - 1)
        t2 = log_binom(keel::f64(n1 - 1), keel::f64(k - 1))
           + log_binom(keel::f64(n0 - 1), keel::f64(k));
    if (keel::is_inf(t1) && keel::is_inf(t2)) return 0.0;
    const keel::f64 lmax = std::max(t1, t2);
    const keel::f64 lsum = lmax + keel::log(
          (keel::is_inf(t1) ? 0.0 : keel::exp(t1 - lmax))
        + (keel::is_inf(t2) ? 0.0 : keel::exp(t2 - lmax)));
    return keel::exp(lsum - logC);
}

} // anonymous

// vartestn_full: vartestn worker -> [p, stat, df1, df2]. Def in hypothesis.cpp.
keel::Tuple<Value, Value, Value, Value>
vartestn_full(const Value &x, const Value &group, keel::i32 test,
              keel::Allocator *mr);

keel::Tuple<Value, Value, Value, Value>
ttest_dim(const Value &x, keel::f64 m, keel::f64 alpha, TestTail tail, keel::i32 dim,
          keel::Allocator *mr);

keel::Tuple<Value, Value, Value, Value>
ttest2_dim(const Value &x, const Value &y, keel::f64 alpha, TestTail tail,
           keel::StringRef vartype, keel::i32 dim,
           keel::Allocator *mr);

keel::Tuple<Value, Value, Value, Value>
ztest_dim(const Value &x, keel::f64 m, keel::f64 sigma, keel::f64 alpha, TestTail tail,
          keel::i32 dim, keel::Allocator *mr);

keel::Tuple<Value, Value, Value, Value>
vartest_dim(const Value &x, keel::f64 v, keel::f64 alpha, TestTail tail, keel::i32 dim,
            keel::Allocator *mr);

keel::Tuple<Value, Value, Value, Value>
vartest2_dim(const Value &x, const Value &y, keel::f64 alpha, TestTail tail,
             keel::i32 dim, keel::Allocator *mr);

} // namespace numkit::stats
