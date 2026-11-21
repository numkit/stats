// toolboxes/stats/src/regress/nlinfit.cpp
//
// Nonlinear least-squares fit via Levenberg-Marquardt + parameter and
// prediction confidence intervals.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/math.h>
#include <numkit/stats/regress/regress.hpp>

#include <numkit/stats/distributions/students_t.hpp>     // tinv
#include <numkit/linalg/solvers.hpp>                     // linsolve
#include <numkit/linalg/properties.hpp>                  // inv

#include <numkit/value/error.hpp>
#include <numkit/value/fn_handle.hpp>

#include <algorithm>
#include <keel/static_array.h>
#include <cmath>
#include <cstring>
#include <limits>

namespace numkit::stats {

namespace {

// Pull a Value as a flat double vector (column-major contents).
keel::Array<keel::f64> toFlatDouble(const Value &v)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

// Build a Value from a column-major buffer.
Value flatToMatrix(const keel::f64 *data, keel::usize rows, keel::usize cols,
                    keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    std::memcpy(out.doubleDataMut(), data, rows * cols * sizeof(keel::f64));
    return out;
}

// Evaluate model(beta, X) → returns flat predictions length n.
keel::Array<keel::f64> evalModel(FnHandle model, keel::ArrayRef<keel::f64> beta,
                               keel::usize p, const Value &X,
                               keel::Allocator *mr)
{
    auto betaVal = flatToMatrix(beta.data(), p, 1, mr);
    keel::StaticArray<Value, 2> argv{ betaVal, X };
    Value yhat;
    model(keel::ArrayRef<Value>(argv.data(), 2), keel::MutArrayRef<Value>(&yhat, 1), mr);
    return toFlatDouble(yhat);
}

// Numerical Jacobian via central differences.
// Returns row-major matrix of size n × p stored column-major
// (column-major: J[col * n + row]).
keel::Array<keel::f64> numericalJacobian(FnHandle model,
                                       keel::ArrayRef<keel::f64> beta,
                                       keel::usize n, keel::usize p,
                                       const Value &X,
                                       keel::Allocator *mr)
{
    keel::Array<keel::f64> J(n * p);
    keel::Array<keel::f64> bplus(beta.size()), bminus(beta.size());
    std::copy(beta.begin(), beta.end(), bplus.begin());
    std::copy(beta.begin(), beta.end(), bminus.begin());
    for (keel::usize j = 0; j < p; ++j) {
        const keel::f64 scale = std::max(keel::abs(beta[j]), 1.0);
        const keel::f64 h = 1e-7 * scale;
        bplus[j]  = beta[j] + h;
        bminus[j] = beta[j] - h;
        auto fp = evalModel(model, bplus.as_ref(),  p, X, mr);
        auto fm = evalModel(model, bminus.as_ref(), p, X, mr);
        bplus[j]  = beta[j];
        bminus[j] = beta[j];
        const keel::f64 inv2h = 1.0 / (2.0 * h);
        for (keel::usize i = 0; i < n; ++i)
            J[j * n + i] = (fp[i] - fm[i]) * inv2h;
    }
    return J;
}

// Compute (J' · J + λ · diag(J' · J)) · dβ = J' · r  via Gauss elim.
// J is n × p column-major, r is length n.
keel::Array<keel::f64> lmStep(keel::ArrayRef<keel::f64> J,
                            keel::usize n, keel::usize p,
                            keel::ArrayRef<keel::f64> r,
                            keel::f64 lambda)
{
    // M = J' · J + λ · diag(J' · J)
    keel::Array<keel::f64> M(p * p, 0.0);
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < n; ++k)
                s += J[i * n + k] * J[j * n + k];
            M[i * p + j] = s;
        }
    for (keel::usize i = 0; i < p; ++i)
        M[i * p + i] *= (1.0 + lambda);
    // g = J' · r
    keel::Array<keel::f64> g(p, 0.0);
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize k = 0; k < n; ++k)
            g[i] += J[i * n + k] * r[k];
    // Gauss elimination (M is row-major p×p).
    for (keel::usize k = 0; k < p; ++k) {
        keel::usize piv = k;
        keel::f64 pmax = std::fabs(M[k * p + k]);
        for (keel::usize rr = k + 1; rr < p; ++rr) {
            const keel::f64 v = std::fabs(M[rr * p + k]);
            if (v > pmax) { pmax = v; piv = rr; }
        }
        if (pmax == 0.0) {
            // Singular — return zero step.
            return keel::Array<keel::f64>(p, 0.0);
        }
        if (piv != k) {
            for (keel::usize j = 0; j < p; ++j)
                std::swap(M[k * p + j], M[piv * p + j]);
            std::swap(g[k], g[piv]);
        }
        const keel::f64 pv = M[k * p + k];
        for (keel::usize rr = k + 1; rr < p; ++rr) {
            const keel::f64 f = M[rr * p + k] / pv;
            for (keel::usize j = k; j < p; ++j)
                M[rr * p + j] -= f * M[k * p + j];
            g[rr] -= f * g[k];
        }
    }
    // Back-substitute.
    for (keel::usize k = p; k-- > 0;) {
        keel::f64 s = g[k];
        for (keel::usize j = k + 1; j < p; ++j)
            s -= M[k * p + j] * g[j];
        g[k] = s / M[k * p + k];
    }
    return g;
}

// Compute (J' · J)^{-1} for the covariance. Returns p × p column-major.
keel::Array<keel::f64> invJtJ(keel::ArrayRef<keel::f64> J,
                            keel::usize n, keel::usize p)
{
    keel::Array<keel::f64> M(p * p, 0.0);
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < n; ++k)
                s += J[i * n + k] * J[j * n + k];
            M[i * p + j] = s;
        }
    // Invert by Gauss-Jordan with augmented identity.
    keel::Array<keel::f64> A(p * 2 * p, 0.0);
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j)
            A[i * 2 * p + j] = M[i * p + j];
    for (keel::usize i = 0; i < p; ++i)
        A[i * 2 * p + (p + i)] = 1.0;
    for (keel::usize k = 0; k < p; ++k) {
        keel::usize piv = k;
        keel::f64 pmax = std::fabs(A[k * 2 * p + k]);
        for (keel::usize rr = k + 1; rr < p; ++rr) {
            const keel::f64 v = std::fabs(A[rr * 2 * p + k]);
            if (v > pmax) { pmax = v; piv = rr; }
        }
        if (pmax == 0.0) {
            return keel::Array<keel::f64>(p * p, std::numeric_limits<keel::f64>::quiet_NaN());
        }
        if (piv != k) {
            for (keel::usize j = 0; j < 2 * p; ++j)
                std::swap(A[k * 2 * p + j], A[piv * 2 * p + j]);
        }
        const keel::f64 pv = A[k * 2 * p + k];
        for (keel::usize j = 0; j < 2 * p; ++j)
            A[k * 2 * p + j] /= pv;
        for (keel::usize rr = 0; rr < p; ++rr) {
            if (rr == k) continue;
            const keel::f64 f = A[rr * 2 * p + k];
            for (keel::usize j = 0; j < 2 * p; ++j)
                A[rr * 2 * p + j] -= f * A[k * 2 * p + j];
        }
    }
    // Extract inverse (column-major). A right half is row-major inverse.
    keel::Array<keel::f64> Inv(p * p);
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize j = 0; j < p; ++j)
            Inv[j * p + i] = A[i * 2 * p + (p + j)];
    return Inv;
}

} // namespace

namespace {
void nlinfit_argcheck(keel::usize p)
{
    if (p == 0)
        throw Error("nlinfit: beta0 cannot be empty",
                    0, 0, "nlinfit", "", "numkit:nlinfit:emptyBeta0");
}
} // namespace

NlinfitResult nlinfit(const Value &X, const Value &y,
                      FnHandle model, const Value &beta0,
                      keel::Allocator *mr)
{
    const keel::usize n = y.numel();
    const keel::usize p = beta0.numel();
    nlinfit_argcheck(p);

    keel::Array<keel::f64> beta = toFlatDouble(beta0);
    keel::Array<keel::f64> yv = toFlatDouble(y);
    keel::Array<keel::f64> rv(n);

    // Initial residual + SSE.
    auto compute_r = [&](keel::ArrayRef<keel::f64> b) {
        auto yhat = evalModel(model, b, p, X, mr);
        keel::Array<keel::f64> r(n);
        for (keel::usize i = 0; i < n; ++i) r[i] = yv[i] - yhat[i];
        return r;
    };
    rv = compute_r(beta.as_ref());
    keel::f64 sse = 0.0;
    for (keel::f64 v : rv) sse += v * v;

    keel::f64 lambda = 1e-3;
    const int maxIter = 200;
    const keel::f64 tolFun = 1e-10;
    const keel::f64 tolX   = 1e-10;

    keel::Array<keel::f64> J;
    for (int iter = 0; iter < maxIter; ++iter) {
        J = numericalJacobian(model, beta.as_ref(), n, p, X, mr);
        auto dbeta = lmStep(J.as_ref(), n, p, rv.as_ref(), lambda);

        // Trial step.
        keel::Array<keel::f64> betaNew(p);
        keel::f64 stepNorm2 = 0.0;
        keel::f64 betaNorm2 = 0.0;
        for (keel::usize j = 0; j < p; ++j) {
            betaNew[j] = beta[j] + dbeta[j];
            stepNorm2 += dbeta[j] * dbeta[j];
            betaNorm2 += betaNew[j] * betaNew[j];
        }
        auto rNew = compute_r(betaNew.as_ref());
        keel::f64 sseNew = 0.0;
        for (keel::f64 v : rNew) sseNew += v * v;

        if (sseNew < sse) {
            // Accept.
            const keel::f64 rel = std::sqrt(stepNorm2) / (std::sqrt(betaNorm2) + 1e-12);
            const keel::f64 relF = (sse - sseNew) / (sse + 1e-12);
            beta = keel::move(betaNew);
            rv   = keel::move(rNew);
            sse  = sseNew;
            lambda *= 0.1;
            if (rel < tolX || relF < tolFun) break;
        } else {
            lambda *= 10.0;
            if (lambda > 1e16) break;     // give up
        }
    }

    // Final Jacobian (refresh at the accepted beta).
    J = numericalJacobian(model, beta.as_ref(), n, p, X, mr);

    // MSE = SSE / (n - p).
    const keel::f64 mse = (n > p) ? sse / keel::as<keel::f64>(n - p) : sse;
    auto Inv = invJtJ(J.as_ref(), n, p);
    keel::Array<keel::f64> CovB(p * p);
    for (keel::usize i = 0; i < p * p; ++i) CovB[i] = mse * Inv[i];

    NlinfitResult out;
    out.beta = flatToMatrix(beta.data(), p, 1, mr);
    out.R    = flatToMatrix(rv.data(),   n, 1, mr);
    out.J    = flatToMatrix(J.data(),    n, p, mr);
    out.CovB = flatToMatrix(CovB.data(), p, p, mr);
    out.MSE  = Value::scalar(mse, mr);
    return out;
}

Value nlparci(const Value &beta, const Value &R, const Value &J,
              keel::f64 alpha, keel::Allocator *mr)
{
    const keel::usize p = beta.numel();
    const keel::usize n = R.numel();
    if (J.dims().rows() != n || J.dims().cols() != p)
        throw Error("nlparci: J must be (n × p) consistent with R, beta",
                    0, 0, "nlparci", "", "numkit:nlparci:shapeMismatch");
    if (n <= p)
        throw Error("nlparci: need n > p degrees of freedom",
                    0, 0, "nlparci", "", "numkit:nlparci:noDOF");

    // MSE = ||R||² / (n - p).
    keel::f64 sse = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 r = R.elemAsDouble(i);
        sse += r * r;
    }
    const keel::f64 mse = sse / keel::as<keel::f64>(n - p);

    // CovB = MSE · (J' · J)^{-1}.
    auto Jvec = toFlatDouble(J);
    auto Inv = invJtJ(Jvec.as_ref(), n, p);

    // t_{α/2, n-p}.
    Value pHi = Value::scalar(1.0 - alpha / 2.0, mr);
    const keel::f64 tcrit = tinv(pHi, keel::as<keel::f64>(n - p), mr).toScalar();

    auto out = Value::matrix(p, 2, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < p; ++i) {
        const keel::f64 se = std::sqrt(mse * Inv[i * p + i]);
        const keel::f64 b  = beta.elemAsDouble(i);
        od[0 * p + i] = b - tcrit * se;  // lower (col 0)
        od[1 * p + i] = b + tcrit * se;  // upper (col 1)
    }
    return out;
}

std::tuple<Value, Value>
nlpredci(FnHandle model, const Value &X, const Value &beta,
         const Value &R, const Value &J,
         keel::f64 alpha,
         keel::Allocator *mr)
{
    const keel::usize p = beta.numel();
    const keel::usize n = R.numel();
    const keel::usize m = X.dims().rows();
    if (J.dims().rows() != n || J.dims().cols() != p)
        throw Error("nlpredci: J must be (n × p) consistent with R, beta",
                    0, 0, "nlpredci", "", "numkit:nlpredci:shapeMismatch");
    if (n <= p)
        throw Error("nlpredci: need n > p degrees of freedom",
                    0, 0, "nlpredci", "", "numkit:nlpredci:noDOF");

    keel::f64 sse = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 r = R.elemAsDouble(i);
        sse += r * r;
    }
    const keel::f64 mse = sse / keel::as<keel::f64>(n - p);
    auto Jvec = toFlatDouble(J);
    auto Inv = invJtJ(Jvec.as_ref(), n, p);

    auto betaFlat = toFlatDouble(beta);

    // ypred at query points: just fun(beta, X).
    auto ypredFlat = evalModel(model, betaFlat.as_ref(), p, X, mr);

    // Per-row prediction sensitivity g_i = ∂fun/∂beta at X(i,:).
    // We compute by central diff on the full X (cheaper: one Jacobian
    // for the whole query set per parameter).
    keel::Array<keel::f64> Jq(m * p);
    auto bplus = betaFlat.clone();
    auto bminus = betaFlat.clone();
    for (keel::usize j = 0; j < p; ++j) {
        const keel::f64 scale = std::max(keel::abs(betaFlat[j]), 1.0);
        const keel::f64 h = 1e-7 * scale;
        bplus[j]  = betaFlat[j] + h;
        bminus[j] = betaFlat[j] - h;
        auto yp = evalModel(model, bplus.as_ref(),  p, X, mr);
        auto ym = evalModel(model, bminus.as_ref(), p, X, mr);
        bplus[j]  = betaFlat[j];
        bminus[j] = betaFlat[j];
        const keel::f64 inv2h = 1.0 / (2.0 * h);
        for (keel::usize i = 0; i < m; ++i)
            Jq[j * m + i] = (yp[i] - ym[i]) * inv2h;
    }

    // t critical.
    Value pHi = Value::scalar(1.0 - alpha / 2.0, mr);
    const keel::f64 tcrit = tinv(pHi, keel::as<keel::f64>(n - p), mr).toScalar();

    auto ypred = flatToMatrix(ypredFlat.data(), m, 1, mr);
    auto delta = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    keel::f64 *dd = delta.doubleDataMut();
    for (keel::usize i = 0; i < m; ++i) {
        // var_i = g_i' · Inv · g_i  (Inv is p × p, g_i is p × 1)
        keel::f64 var = 0.0;
        for (keel::usize a = 0; a < p; ++a)
            for (keel::usize b = 0; b < p; ++b)
                var += Jq[a * m + i] * Inv[a * p + b] * Jq[b * m + i];
        dd[i] = tcrit * std::sqrt(mse * var);
    }
    return { std::move(ypred), std::move(delta) };
}

} // namespace numkit::stats
