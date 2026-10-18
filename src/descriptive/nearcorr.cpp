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

#include <numkit/stats/descriptive/descriptive.hpp>
#include "descriptive/descriptive_extras_detail.hpp"

#include <numkit/linalg/eig.hpp>             // eig_symmetric (migrated)
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

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
std::vector<double>
projPSD(const std::vector<double> &M, size_t n, std::pmr::memory_resource *mr)
{
    // Symmetrise: M_sym = (M + M') / 2
    std::vector<double> Msym(n * n);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            Msym[i + j * n] = 0.5 * (M[i + j * n] + M[j + i * n]);
    Value M_v = Value::matrix(n, n, ValueType::DOUBLE, mr);
    std::copy(Msym.begin(), Msym.end(), M_v.doubleDataMut());

    auto [V, D] = ::numkit::linalg::eig_symmetric(M_v, mr);
    const double *vd = V.doubleData();
    const double *dd = D.doubleData();

    // Build sqrt(d_pos) and reconstruct V * diag(d_pos) * V'.
    std::vector<double> sqrt_d(n);
    for (size_t i = 0; i < n; ++i)
        sqrt_d[i] = std::sqrt(std::max(dd[i + i * n], 0.0));

    // X = V * diag(d_pos) * V' = (V * diag(sqrt_d)) * (V * diag(sqrt_d))'
    std::vector<double> VS(n * n);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            VS[i + j * n] = vd[i + j * n] * sqrt_d[j];

    std::vector<double> X(n * n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += VS[i + k * n] * VS[j + k * n];   // VS * VS'
            X[i + j * n] = s;
        }
    return X;
}

double frobNorm(const std::vector<double> &M)
{
    double s = 0.0;
    for (double v : M) s += v * v;
    return std::sqrt(s);
}

} // namespace

Value nearcorr_full(const Value &A, Span<const Value> nv,
                    std::pmr::memory_resource *mr)
{
    // ── gate (one rejection class; every clause names what failed) ──────
    const auto rejectA = [&](const std::string &why) {
        throw Error("nearcorr: invalid value of 'A'. Expected A to be a "
                    "symmetric matrix with unit diagonal and entries in "
                    "[-1,1] of double or single class (" + why + ").",
                    0, 0, "nearcorr", "", "numkit:nearcorr:invalidInputA");
    };
    if (!A.isNumeric() || A.numel() == 0 || A.dims().ndim() > 2)
        rejectA(std::string("got a ") + mtypeName(A.type())
                    + (A.numel() == 0 ? " array, which is empty or not a matrix"
                                      : " array, which is not a matrix"));
    if (!argcheck::maskAccepts(kFloatReal, argcheck::classBitOf(A.type())))
        rejectA("integer classes are not accepted; pass double or single");
    if (A.isComplex())
        rejectA("got a complex array");
    const size_t N = A.dims().rows();
    if (A.dims().cols() != N)
        rejectA("the matrix is not square");
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < N; ++i) {
            const double v = A.elemAsDouble(i + j * N);
            if (v < -1.0 || v > 1.0)
                rejectA("entry (" + std::to_string(i + 1) + ","
                        + std::to_string(j + 1) + ") is outside [-1,1]");
            if (i < j && v != A.elemAsDouble(j + i * N))
                rejectA("the matrix is not symmetric");
        }
    for (size_t i = 0; i < N; ++i)
        if (A.elemAsDouble(i + i * N) != 1.0)
            rejectA("diagonal entry (" + std::to_string(i + 1) + ","
                    + std::to_string(i + 1) + ") is not 1");
    const bool wantSingle = (A.type() == ValueType::SINGLE);

    // ── name-value parse (probe71 N05-N07: one optional-input class) ────
    double tol = 1e-6;
    double maxIterD = 200.0;
    bool hasWeights = false;
    std::vector<double> wScale;
    static constexpr std::string_view kNv[] = {"tolerance", "maxiterations",
                                               "method", "weights"};
    const auto badOpt = [&](const std::string &why) {
        throw Error("nearcorr: error in optional input arguments (" + why
                    + ").",
                    0, 0, "nearcorr", "", "numkit:nearcorr:optionalInputError");
    };
    argcheck::forEachNvPair(
        nv, 0, Span<const std::string_view>(kNv), "nearcorr",
        "numkit:nearcorr:optionalInputError",
        [&](std::size_t hit, const Value &v) {
            switch (hit) {
            case 0: {  // Tolerance: double scalar > 0
                if (v.type() != ValueType::DOUBLE || !v.isScalar())
                    badOpt("'Tolerance' must be a double scalar");
                const double t = v.toScalar();
                if (!(t > 0.0))
                    badOpt("'Tolerance' must be positive, got "
                           + std::to_string(t));
                tol = t;
                break;
            }
            case 1: {  // MaxIterations: double scalar > 0 integer
                if (v.type() != ValueType::DOUBLE || !v.isScalar())
                    badOpt("'MaxIterations' must be a double scalar");
                const double m = v.toScalar();
                if (!(m > 0.0) || m != std::floor(m))
                    badOpt("'MaxIterations' must be a positive integer, got "
                           + std::to_string(m));
                maxIterD = m;
                break;
            }
            case 2: {  // Method: EXACT 'newton' | 'projection' (the
                       // reference validates with ismember, not a prefix)
                std::string mv = v.toString();
                for (char &c : mv)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                if (mv != "newton" && mv != "projection")
                    badOpt("'Method' must be 'newton' or 'projection', got '"
                           + v.toString() + "'");
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
                for (size_t k = 0; k < v.numel(); ++k) {
                    const double wk = v.elemAsDouble(k);
                    if (!(wk > 0.0) || !std::isfinite(wk))
                        badOpt("'Weights' entries must be finite and positive");
                }
                wScale.resize(N * N);
                if (isVec) {
                    for (size_t j = 0; j < N; ++j)
                        for (size_t i = 0; i < N; ++i)
                            wScale[i + j * N] = std::sqrt(v.elemAsDouble(i) * v.elemAsDouble(j));
                } else {
                    for (size_t j = 0; j < N; ++j)
                        for (size_t i = 0; i < N; ++i) {
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
    const long maxIter = static_cast<long>(maxIterD);

    // Trivial case: a 1×1 unit-diagonal input is already valid.
    if (N == 1)
        return wantSingle ? Value::scalar(static_cast<float>(1.0))
                          : Value::scalar(1.0);

    // ── alternating projections with Dykstra's correction ───────────────
    std::vector<double> Yold(N * N), Xold;
    for (size_t k = 0; k < N * N; ++k) Yold[k] = A.elemAsDouble(k);
    Xold = Yold;
    std::vector<double> dS(N * N, 0.0), Y = Yold;

    for (long it = 0; it <= maxIter; ++it) {
        // R = Yold - dS; X = proj_PSD_W(R); dS = X - R; Y = X, unit diag.
        std::vector<double> R(N * N);
        for (size_t k = 0; k < N * N; ++k) R[k] = Yold[k] - dS[k];
        std::vector<double> X;
        if (hasWeights) {
            std::vector<double> Rw(N * N);
            for (size_t k = 0; k < N * N; ++k) Rw[k] = R[k] * wScale[k];
            std::vector<double> Xw = projPSD(Rw, N, mr);
            X.resize(N * N);
            for (size_t k = 0; k < N * N; ++k) X[k] = Xw[k] / wScale[k];
        } else {
            X = projPSD(R, N, mr);
        }
        for (size_t k = 0; k < N * N; ++k) dS[k] = X[k] - R[k];
        Y = X;
        for (size_t i = 0; i < N; ++i) Y[i + i * N] = 1.0;

        // Exit conditions (the reference checks all three).
        const double normY = frobNorm(Y);
        std::vector<double> d1(N * N), d2(N * N), d3(N * N);
        for (size_t k = 0; k < N * N; ++k) {
            d1[k] = Y[k] - Yold[k];
            d2[k] = X[k] - Xold[k];
            d3[k] = Y[k] - X[k];
        }
        const double cond1 = frobNorm(d1) / normY;
        const double cond2 = frobNorm(d2) / frobNorm(X);
        const double cond3 = frobNorm(d3) / normY;
        if (std::max(cond1, std::max(cond2, cond3)) <= tol) {
            Xold = X;
            break;
        }
        Yold = Y;
        Xold = X;
    }

    // Restore the unit diagonal: Y = X ./ sqrt(diag(X) * diag(X)'), then
    // symmetrise (small numerical drift is possible).
    std::vector<double> dd(N);
    for (size_t i = 0; i < N; ++i) dd[i] = Xold[i + i * N];
    Value out = wantSingle ? Value::matrix(N, N, ValueType::SINGLE, mr)
                           : Value::matrix(N, N, ValueType::DOUBLE, mr);
    auto putElem = [&](std::size_t lin, double v) {
        if (wantSingle)
            out.singleDataMut()[lin] = static_cast<float>(v);
        else
            out.doubleDataMut()[lin] = v;
    };
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < N; ++i) {
            const double v = Xold[i + j * N] / std::sqrt(dd[i] * dd[j]);
            putElem(i + j * N, 0.5 * (v + Xold[j + i * N] / std::sqrt(dd[i] * dd[j])));
        }
    return out;
}

Value nearcorr(const Value &A, std::pmr::memory_resource *mr)
{
    return nearcorr_full(A, Span<const Value>(), mr);
}

} // namespace numkit::stats
