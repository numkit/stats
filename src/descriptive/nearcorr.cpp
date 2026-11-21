// toolboxes/stats/src/descriptive/nearcorr.cpp
//
// MATLAB nearcorr: nearest correlation matrix (Higham 2002 alternating
// projections with Dykstra's correction).
//
//   Y = nearcorr(A)
//   Y = nearcorr(A, 'Tolerance', tol, 'MaxIterations', n, 'Method', m)
//
// The full-entry contract (nearcorr_full) owns the input gate and the
// name-value parse, probed against the reference engine (build/probe71.out
// N01-N11):
//   * gate — one rejection class: numeric, nonempty, real, 2-D matrix,
//     symmetric, entries in [-1, 1], unit diagonal, double or single
//     (integer classes are rejected — the reference reaches the same
//     rejection inside its symmetry helper which only accepts float and
//     logical classes, and logical fails the isnumeric test);
//   * 'Tolerance'      double scalar > 0, default 1e-6;
//   * 'MaxIterations'  double scalar > 0 integer, default 200;
//   * 'Method'         'newton' (default) | 'projection', exact
//     case-insensitive match;
//   * 'Weights'        voiced stub — changes the minimizer, filed in
//     bugs/opened/stats/correlation-deferred-forms.md;
//   * any bad name/value → one optional-input rejection class.
//
// DELIBERATE DIVERGENCE (documented, same minimizer): both Method values
// run the alternating-projections solver. The reference's 'newton' path
// is a semismooth Newton iteration on the same KKT system — it converges
// to the same nearest correlation matrix by a different route; the
// reference's own convergence study (nearcorr.m refs) reports the same
// fixed point. Parity specs pin the returned matrix, not the iteration
// count.
//
// Algorithm (Higham, "Computing the Nearest Correlation Matrix --
// A Problem from Finance", IMA J. Numer. Anal. 22 (3): 329-343, 2002):
//   Y = A; dS = 0
//   loop:
//     R = Y - dS                 -- subtract Dykstra correction
//     X = proj_PSD(R)            -- eig, clamp negative eigvals to 0
//     dS = X - R                 -- update Dykstra correction
//     Y = proj_unit_diag(X)      -- set diagonal entries to 1
//     until max(||Y-Yold||/||Y||, ||X-Xold||/||X||, ||Y-X||/||Y||) <= tol
//   restore unit diagonal via X ./ sqrt(diag(X) * diag(X)') and symmetrise.
//
// Single inputs compute in double and cast down (the reference runs the
// iteration in single arithmetic; the fixed point agrees to single
// precision but the iteration path may differ in the last digits).

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/stats/descriptive/descriptive.hpp>
#include "descriptive/descriptive_extras_detail.hpp"

#include <numkit/linalg/eig.hpp>             // eig_symmetric (migrated)
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>

namespace numkit::stats {

namespace {

using argcheck::ClsDbl;
using argcheck::ClsSgl;
using argcheck::classMaskOf;

inline constexpr argcheck::ClassMask kFloatReal =
    classMaskOf(ClsDbl) | classMaskOf(ClsSgl);

// Project onto PSD cone via symmetric eigendecomposition.
// Symmetrise input first to guard against tiny asymmetry from the
// previous iteration.
keel::Array<double>
projPSD(const keel::Array<double> &M, keel::usize n, keel::Allocator *mr)
{
    // Symmetrise: M_sym = (M + M') / 2
    keel::Array<double> Msym(n * n);
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < n; ++i)
            Msym[i + j * n] = 0.5 * (M[i + j * n] + M[j + i * n]);
    Value M_v = Value::matrix(n, n, ValueType::DOUBLE, mr);
    std::copy(Msym.begin(), Msym.end(), M_v.doubleDataMut());

    auto [V, D] = ::numkit::linalg::eig_symmetric(M_v, mr);
    const double *vd = V.doubleData();
    const double *dd = D.doubleData();

    // Build sqrt(d_pos) and reconstruct V * diag(d_pos) * V'.
    keel::Array<double> sqrt_d(n);
    for (keel::usize i = 0; i < n; ++i)
        sqrt_d[i] = std::sqrt(std::max(dd[i + i * n], 0.0));

    // X = V * diag(d_pos) * V' = (V * diag(sqrt_d)) * (V * diag(sqrt_d))'
    keel::Array<double> VS(n * n);
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < n; ++i)
            VS[i + j * n] = vd[i + j * n] * sqrt_d[j];

    keel::Array<double> X(n * n, 0.0);
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < n; ++j) {
            double s = 0.0;
            for (keel::usize k = 0; k < n; ++k)
                s += VS[i + k * n] * VS[j + k * n];   // VS * VS'
            X[i + j * n] = s;
        }
    return X;
}

double frobNorm(const keel::Array<double> &M)
{
    double s = 0.0;
    for (double v : M) s += v * v;
    return std::sqrt(s);
}

void nearcorr_argcheck(const Value &A)
{
    const auto rejectA = [&](keel::StringRef why) {
        throw Error(keel::format("nearcorr: invalid value of 'A'. Expected A to be a "
                    "symmetric matrix with unit diagonal and entries in "
                    "[-1,1] of double or single class ({}).", why),
                    0, 0, "nearcorr", "", "numkit:nearcorr:invalidInputA");
    };
    if (!A.isNumeric() || A.numel() == 0 || A.dims().ndim() > 2)
        rejectA(keel::format("got a {}{}", mtypeName(A.type()),
                    A.numel() == 0 ? " array, which is empty or not a matrix"
                                   : " array, which is not a matrix"));
    if (!argcheck::maskAccepts(kFloatReal, argcheck::classBitOf(A.type())))
        rejectA("integer classes are not accepted; pass double or single");
    if (A.isComplex())
        rejectA("got a complex array");
    const keel::usize N = A.dims().rows();
    if (A.dims().cols() != N)
        rejectA("the matrix is not square");
    for (keel::usize j = 0; j < N; ++j)
        for (keel::usize i = 0; i < N; ++i) {
            const double v = A.elemAsDouble(i + j * N);
            if (v < -1.0 || v > 1.0)
                rejectA(keel::format("entry ({},{}) is outside [-1,1]", i + 1, j + 1));
            if (i < j && v != A.elemAsDouble(j + i * N))
                rejectA("the matrix is not symmetric");
        }
    for (keel::usize i = 0; i < N; ++i)
        if (A.elemAsDouble(i + i * N) != 1.0)
            rejectA(keel::format("diagonal entry ({},{}) is not 1", i + 1, i + 1));
}

} // namespace

Value nearcorr_full(const Value &A, keel::ArrayRef<Value> nv,
                    keel::Allocator *mr)
{
    nearcorr_argcheck(A);
    const keel::usize N = A.dims().rows();
    const bool wantSingle = (A.type() == ValueType::SINGLE);

    // ── name-value parse (probe71 N05-N07: one optional-input class) ────
    double tol = 1e-6;
    double maxIterD = 200.0;
    bool hasWeights = false;
    keel::Array<double> wScale;
    static constexpr keel::StringRef kNv[] = {"tolerance", "maxiterations",
                                              "method", "weights"};
    const auto badOpt = [&](keel::StringRef why) {
        throw Error(keel::format("nearcorr: error in optional input arguments ({}).", why),
                    0, 0, "nearcorr", "", "numkit:nearcorr:optionalInputError");
    };
    argcheck::forEachNvPair(
        nv, 0, keel::ArrayRef<keel::StringRef>(kNv), "nearcorr",
        "numkit:nearcorr:optionalInputError",
        [&](keel::usize hit, const Value &v) {
            switch (hit) {
            case 0: {  // Tolerance: double scalar > 0
                if (v.type() != ValueType::DOUBLE || !v.isScalar())
                    badOpt("'Tolerance' must be a double scalar");
                const double t = v.toScalar();
                if (!(t > 0.0))
                    badOpt(keel::format("'Tolerance' must be positive, got {}", t));
                tol = t;
                break;
            }
            case 1: {  // MaxIterations: double scalar > 0 integer
                if (v.type() != ValueType::DOUBLE || !v.isScalar())
                    badOpt("'MaxIterations' must be a double scalar");
                const double m = v.toScalar();
                if (!(m > 0.0) || m != std::floor(m))
                    badOpt(keel::format("'MaxIterations' must be a positive integer, got {}", m));
                maxIterD = m;
                break;
            }
            case 2: {  // Method: EXACT 'newton' | 'projection' (the
                       // reference validates with ismember, not a prefix)
                keel::String rawMv = v.toString();
                keel::String mv = rawMv.clone();
                for (char &c : mv)
                    c = keel::to_ascii_lower(c);
                if (mv != "newton" && mv != "projection")
                    badOpt(keel::format("'Method' must be 'newton' or 'projection', got '{}'", rawMv));
                break;  // both accepted — see the divergence note above
            }
            case 3: {  // Weights: positive length-N vector or symmetric positive N×N matrix
                if (!v.isNumeric() || v.isComplex() || v.numel() == 0
                    || v.dims().ndim() > 2
                    || !argcheck::maskAccepts(kFloatReal, argcheck::classBitOf(v.type())))
                    badOpt("'Weights' must be a real float vector or symmetric matrix");
                const bool isVec = v.dims().isVector() && v.numel() == N;
                const bool isMat = (v.dims().rows() == N && v.dims().cols() == N);
                if (!isVec && !isMat)
                    badOpt("'Weights' must be a length-N vector or N-by-N matrix");
                for (keel::usize k = 0; k < v.numel(); ++k) {
                    const double wk = v.elemAsDouble(k);
                    if (!(wk > 0.0) || !std::isfinite(wk))
                        badOpt("'Weights' entries must be finite and positive");
                }
                wScale.resize(N * N);
                if (isVec) {
                    for (keel::usize j = 0; j < N; ++j)
                        for (keel::usize i = 0; i < N; ++i)
                            wScale[i + j * N] = std::sqrt(v.elemAsDouble(i) * v.elemAsDouble(j));
                } else {
                    for (keel::usize j = 0; j < N; ++j)
                        for (keel::usize i = 0; i < N; ++i) {
                            if (i < j && v.elemAsDouble(i + j * N) != v.elemAsDouble(j + i * N))
                                badOpt("'Weights' matrix must be symmetric");
                            wScale[i + j * N] = std::sqrt(v.elemAsDouble(i + j * N));
                        }
                }
                hasWeights = true;
                break;
            }
            default: break;
            }
        });
    const long maxIter = keel::as<long>(maxIterD);

    // Trivial case: a 1×1 unit-diagonal input is already valid.
    if (N == 1)
        return wantSingle ? Value::scalar(keel::as<float>(1.0))
                          : Value::scalar(1.0);

    // ── alternating projections with Dykstra's correction ───────────────
    keel::Array<double> Yold(N * N), Xold;
    for (keel::usize k = 0; k < N * N; ++k) Yold[k] = A.elemAsDouble(k);
    Xold = Yold.clone();
    keel::Array<double> dS(N * N, 0.0), Y = Yold.clone();

    for (long it = 0; it <= maxIter; ++it) {
        // R = Yold - dS; X = proj_PSD_W(R); dS = X - R; Y = X, unit diag.
        keel::Array<double> R(N * N);
        for (keel::usize k = 0; k < N * N; ++k) R[k] = Yold[k] - dS[k];
        keel::Array<double> X;
        if (hasWeights) {
            keel::Array<double> Rw(N * N);
            for (keel::usize k = 0; k < N * N; ++k) Rw[k] = R[k] * wScale[k];
            keel::Array<double> Xw = projPSD(Rw, N, mr);
            X.resize(N * N);
            for (keel::usize k = 0; k < N * N; ++k) X[k] = Xw[k] / wScale[k];
        } else {
            X = projPSD(R, N, mr);
        }
        for (keel::usize k = 0; k < N * N; ++k) dS[k] = X[k] - R[k];
        Y.assign(X.as_ref());
        for (keel::usize i = 0; i < N; ++i) Y[i + i * N] = 1.0;

        // Exit conditions (the reference checks all three).
        const double normY = frobNorm(Y);
        keel::Array<double> d1(N * N), d2(N * N), d3(N * N);
        for (keel::usize k = 0; k < N * N; ++k) {
            d1[k] = Y[k] - Yold[k];
            d2[k] = X[k] - Xold[k];
            d3[k] = Y[k] - X[k];
        }
        const double cond1 = frobNorm(d1) / normY;
        const double cond2 = frobNorm(d2) / frobNorm(X);
        const double cond3 = frobNorm(d3) / normY;
        if (std::max(cond1, std::max(cond2, cond3)) <= tol) {
            Xold = std::move(X);
            break;
        }
        Yold.assign(Y.as_ref());
        Xold = std::move(X);
    }

    // Restore the unit diagonal: Y = X ./ sqrt(diag(X) * diag(X)'), then
    // symmetrise (small numerical drift is possible).
    keel::Array<double> dd(N);
    for (keel::usize i = 0; i < N; ++i) dd[i] = Xold[i + i * N];
    Value out = wantSingle ? Value::matrix(N, N, ValueType::SINGLE, mr)
                           : Value::matrix(N, N, ValueType::DOUBLE, mr);
    auto putElem = [&](keel::usize lin, double v) {
        if (wantSingle)
            out.singleDataMut()[lin] = keel::as<float>(v);
        else
            out.doubleDataMut()[lin] = v;
    };
    for (keel::usize j = 0; j < N; ++j)
        for (keel::usize i = 0; i < N; ++i) {
            const double v = Xold[i + j * N] / std::sqrt(dd[i] * dd[j]);
            putElem(i + j * N, 0.5 * (v + Xold[j + i * N] / std::sqrt(dd[i] * dd[j])));
        }
    return out;
}

Value nearcorr(const Value &A, keel::Allocator *mr)
{
    return nearcorr_full(A, keel::ArrayRef<Value>(), mr);
}

} // namespace numkit::stats
