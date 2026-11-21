// toolboxes/stats/src/descriptive/partial_canon.cpp
//
// Multivariate-correlation primitives that residualise / decompose via
// linear-algebra ops:
//   partialcorri — partial correlation with X-column-specific controls
//   canoncorr    — canonical correlation analysis (QR + SVD)

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/linalg/decompositions.hpp>     // qr_decompose, svd_decompose

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace numkit::stats {

namespace {

// In-place Gaussian elimination + back-sub of (M · x = b), where M is
// p×p row-major. Returns false if singular (caller sets coef to 0).
bool solveGauss(double *M, double *b, keel::usize p)
{
    for (keel::usize k = 0; k < p; ++k) {
        keel::usize piv = k;
        double pmax = std::fabs(M[k * p + k]);
        for (keel::usize r = k + 1; r < p; ++r) {
            const double v = std::fabs(M[r * p + k]);
            if (v > pmax) { pmax = v; piv = r; }
        }
        if (pmax == 0.0) return false;
        if (piv != k) {
            for (keel::usize j = 0; j < p; ++j)
                std::swap(M[k * p + j], M[piv * p + j]);
            std::swap(b[k], b[piv]);
        }
        const double pivVal = M[k * p + k];
        for (keel::usize r = k + 1; r < p; ++r) {
            const double f = M[r * p + k] / pivVal;
            for (keel::usize j = k; j < p; ++j)
                M[r * p + j] -= f * M[k * p + j];
            b[r] -= f * b[k];
        }
    }
    // Back-substitute.
    for (keel::usize k = p; k-- > 0;) {
        double s = b[k];
        for (keel::usize j = k + 1; j < p; ++j)
            s -= M[k * p + j] * b[j];
        b[k] = s / M[k * p + k];
    }
    return true;
}

// Residualise `Wcol` (length n) on a column-major design matrix `C`
// (n × p with intercept already included). Returns residual vector
// (n elements) into `out`. Uses normal equations + Gauss elim.
//
// `MM` is `C' · C` (p × p) — caller computes once and reuses.
void residualiseColumn(const double *C, keel::usize n, keel::usize p,
                       const double *MM,
                       const double *Wcol,
                       double *out,
                       keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    // RHS b = C' · Wcol  (length p).
    keel::Array<double> b(p, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < p; ++i)
        for (keel::usize k = 0; k < n; ++k)
            b[i] += C[i * n + k] * Wcol[k];
    // Solve (M · coef = b).
    keel::Array<double> Mcopy(keel::ArrayRef<double>(MM, p * p), scratch.allocator());
    if (!solveGauss(Mcopy.data(), b.data(), p)) {
        std::fill(b.begin(), b.end(), 0.0);
    }
    // Residual = Wcol - C · coef.
    for (keel::usize k = 0; k < n; ++k) {
        double pred = 0.0;
        for (keel::usize i = 0; i < p; ++i)
            pred += C[i * n + k] * b[i];
        out[k] = Wcol[k] - pred;
    }
}

double pearsonOfCentered(const double *a, const double *b, keel::usize n)
{
    // Both inputs are already residuals (mean ≈ 0). Compute
    // sum(a·b) / sqrt(sum(a²) · sum(b²)).
    double sab = 0.0, saa = 0.0, sbb = 0.0;
    for (keel::usize k = 0; k < n; ++k) {
        sab += a[k] * b[k];
        saa += a[k] * a[k];
        sbb += b[k] * b[k];
    }
    if (saa <= 0.0 || sbb <= 0.0) return std::nan("");
    return sab / std::sqrt(saa * sbb);
}

void partialcorri_argcheck(const Value &Y, const Value &X, const Value &Z)
{
    if (X.dims().ndim() > 2 || Y.dims().ndim() > 2)
        throw Error("partialcorri: X, Y must be 2D matrices",
                    0, 0, "partialcorri", "", "numkit:partialcorri:notMatrix");
    const keel::usize n = Y.dims().dim(0);
    if (X.dims().dim(0) != n)
        throw Error("partialcorri: rows(X) must equal rows(Y)",
                    0, 0, "partialcorri", "", "numkit:partialcorri:dimMismatch");
    if (!Z.isEmpty()) {
        if (Z.dims().ndim() > 2)
            throw Error("partialcorri: Z must be a 2D matrix",
                        0, 0, "partialcorri", "", "numkit:partialcorri:notMatrix");
        if (Z.dims().dim(0) != n)
            throw Error("partialcorri: rows(Z) must equal rows(Y)",
                        0, 0, "partialcorri", "", "numkit:partialcorri:dimMismatch");
    }
}

[[maybe_unused]] void canoncorr_argcheck(const Value &X, const Value &Y)
{
    if (X.dims().ndim() > 2 || Y.dims().ndim() > 2)
        throw Error("canoncorr: X, Y must be 2D matrices",
                    0, 0, "canoncorr", "", "numkit:canoncorr:notMatrix");
    const keel::usize n = X.dims().dim(0);
    const keel::usize p = X.dims().dim(1);
    const keel::usize q = Y.dims().dim(1);
    if (Y.dims().dim(0) != n)
        throw Error("canoncorr: rows(Y) must equal rows(X)",
                    0, 0, "canoncorr", "", "numkit:canoncorr:dimMismatch");
    if (n < p + q)
        throw Error("canoncorr: insufficient observations (need n >= p + q)",
                    0, 0, "canoncorr", "", "numkit:canoncorr:tooFewRows");
}

} // namespace

Value partialcorri(const Value &Y, const Value &X, const Value &Z,
                   keel::Allocator *mr)
{
    partialcorri_argcheck(Y, X, Z);
    const keel::usize n  = keel::as<keel::usize>(Y.dims().dim(0));
    const keel::usize pY = keel::as<keel::usize>(Y.dims().dim(1));
    const keel::usize pX = keel::as<keel::usize>(X.dims().dim(1));
    const bool hasZ = !Z.isEmpty();
    const keel::usize pZ = hasZ ? keel::as<keel::usize>(Z.dims().dim(1)) : 0;

    const double *Xd = X.doubleData();
    const double *Yd = Y.doubleData();
    const double *Zd = hasZ ? Z.doubleData() : nullptr;

    auto Rout = Value::matrix(pY, pX, ValueType::DOUBLE, mr);
    double *R = Rout.doubleDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Control matrix per j: [1, X(:, ~j), Z] of size n × pCtrl where
    // pCtrl = 1 + (pX - 1) + pZ. We rebuild `C` from scratch for each
    // j — pX is typically small (≤ 20) and this keeps the code simple.
    const keel::usize pCtrl = 1u + (pX > 0 ? pX - 1u : 0u) + pZ;
    keel::Array<double> C(n * pCtrl, 0.0, scratch.allocator());
    keel::Array<double> MM(pCtrl * pCtrl, 0.0, scratch.allocator());
    keel::Array<double> y_res(n, 0.0, scratch.allocator());
    keel::Array<double> x_res(n, 0.0, scratch.allocator());

    for (keel::usize j = 0; j < pX; ++j) {
        // Build C column-major.
        std::fill(C.begin(), C.end(), 0.0);
        // Col 0: intercept (all ones).
        for (keel::usize k = 0; k < n; ++k) C[0 * n + k] = 1.0;
        // Cols 1..(pX-1): X without col j.
        keel::usize cidx = 1;
        for (keel::usize jj = 0; jj < pX; ++jj) {
            if (jj == j) continue;
            for (keel::usize k = 0; k < n; ++k)
                C[cidx * n + k] = Xd[jj * n + k];
            ++cidx;
        }
        // Cols (pX)..(pCtrl-1): Z columns.
        for (keel::usize jz = 0; jz < pZ; ++jz) {
            for (keel::usize k = 0; k < n; ++k)
                C[cidx * n + k] = Zd[jz * n + k];
            ++cidx;
        }

        // MM = C' · C (pCtrl × pCtrl, row-major for solveGauss).
        std::fill(MM.begin(), MM.end(), 0.0);
        for (keel::usize a = 0; a < pCtrl; ++a)
            for (keel::usize b = 0; b < pCtrl; ++b) {
                double s = 0.0;
                for (keel::usize k = 0; k < n; ++k)
                    s += C[a * n + k] * C[b * n + k];
                MM[a * pCtrl + b] = s;
            }

        // Residualise X(:, j) on C.
        residualiseColumn(C.data(), n, pCtrl, MM.data(),
                           Xd + j * n, x_res.data(), mr);

        // Residualise each Y(:, i) on C and correlate.
        for (keel::usize i = 0; i < pY; ++i) {
            residualiseColumn(C.data(), n, pCtrl, MM.data(),
                               Yd + i * n, y_res.data(), mr);
            R[j * pY + i] = pearsonOfCentered(y_res.data(),
                                              x_res.data(), n);
        }
    }
    return Rout;
}

CanoncorrResult canoncorr(const Value &X, const Value &Y,
                           keel::Allocator *mr)
{
    if (X.dims().ndim() > 2 || Y.dims().ndim() > 2)
        throw Error("canoncorr: X, Y must be 2D matrices",
                    0, 0, "canoncorr", "", "numkit:canoncorr:notMatrix");
    const keel::usize n = X.dims().dim(0);
    const keel::usize p = X.dims().dim(1);
    const keel::usize q = Y.dims().dim(1);
    if (Y.dims().dim(0) != n)
        throw Error("canoncorr: rows(Y) must equal rows(X)",
                    0, 0, "canoncorr", "", "numkit:canoncorr:dimMismatch");
    if (n < p + q)
        throw Error("canoncorr: insufficient observations (need n >= p + q)",
                    0, 0, "canoncorr", "", "numkit:canoncorr:tooFewRows");

    const keel::usize k = std::min(p, q);
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Centre X and Y.
    auto Xc = Value::matrix(n, p, ValueType::DOUBLE, mr);
    auto Yc = Value::matrix(n, q, ValueType::DOUBLE, mr);
    {
        const double *Xd = X.doubleData();
        double *Xcd = Xc.doubleDataMut();
        for (keel::usize j = 0; j < p; ++j) {
            double m = 0.0;
            for (keel::usize i = 0; i < n; ++i) m += Xd[j * n + i];
            m /= keel::as<double>(n);
            for (keel::usize i = 0; i < n; ++i)
                Xcd[j * n + i] = Xd[j * n + i] - m;
        }
        const double *Yd = Y.doubleData();
        double *Ycd = Yc.doubleDataMut();
        for (keel::usize j = 0; j < q; ++j) {
            double m = 0.0;
            for (keel::usize i = 0; i < n; ++i) m += Yd[j * n + i];
            m /= keel::as<double>(n);
            for (keel::usize i = 0; i < n; ++i)
                Ycd[j * n + i] = Yd[j * n + i] - m;
        }
    }

    // Full QR on centred Xc, Yc. QX is n × n orthogonal; RX is n × p
    // upper-triangular. We only need the first p columns of QX (call
    // it QX1) and the top p × p block of RX (call it RX1).
    auto [QX, RX] = ::numkit::linalg::qr_decompose(Xc, mr);
    auto [QY, RY] = ::numkit::linalg::qr_decompose(Yc, mr);

    // Take the leading `cols` columns of a matrix. The row count MUST come
    // from the source matrix, not the observation count `n`: this helper is
    // reused below on U (p × p) and V (q × q), where p, q < n. Hardcoding `n`
    // rows there read past the U/V buffers — an out-of-bounds access (a
    // non-deterministic SEH 0xc0000005 crash, or silently wrong values).
    auto sliceCols = [&](const Value &Q, keel::usize cols) {
        const keel::usize rows = Q.dims().rows();
        auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        const double *qd = Q.doubleData();
        for (keel::usize j = 0; j < cols; ++j)
            for (keel::usize i = 0; i < rows; ++i)
                od[j * rows + i] = qd[j * rows + i];
        return out;
    };
    auto QX1 = sliceCols(QX, p);
    auto QY1 = sliceCols(QY, q);

    // M = QX1' · QY1, size p × q.
    auto M = Value::matrix(p, q, ValueType::DOUBLE, mr);
    {
        double *Md = M.doubleDataMut();
        const double *qxd = QX1.doubleData();
        const double *qyd = QY1.doubleData();
        for (keel::usize a = 0; a < p; ++a)
            for (keel::usize b = 0; b < q; ++b) {
                double s = 0.0;
                for (keel::usize i = 0; i < n; ++i)
                    s += qxd[a * n + i] * qyd[b * n + i];
                Md[b * p + a] = s;
            }
    }

    // SVD: M = U · diag(s) · V'. U is p × p, V is q × q, s length min(p,q).
    auto [U, S, V] = ::numkit::linalg::svd_decompose(M, mr);

    // A_top = RX(1:p, 1:p) \ U(:, 1:k)
    //   ↔ solve RX1 · A_top = U(:, 1:k) for A_top (p × k).
    // RX1 is upper-triangular → back-sub.
    auto solveR = [&](const Value &R, const Value &RHS, keel::usize cols, keel::usize dim) {
        auto Out = Value::matrix(dim, cols, ValueType::DOUBLE, mr);
        const double *Rd = R.doubleData();
        const double *Bd = RHS.doubleData();
        double *Od = Out.doubleDataMut();
        for (keel::usize col = 0; col < cols; ++col) {
            for (keel::usize i = dim; i-- > 0;) {
                double s = Bd[col * dim + i];   // B is dim × cols when sliced
                for (keel::usize j = i + 1; j < dim; ++j)
                    s -= Rd[j * R.dims().rows() + i] * Od[col * dim + j];
                const double diag = Rd[i * R.dims().rows() + i];
                Od[col * dim + i] = (std::fabs(diag) > 0.0) ? s / diag : 0.0;
            }
        }
        return Out;
    };

    auto Uk = sliceCols(U, k);
    auto Vk = sliceCols(V, k);
    auto A  = solveR(RX, Uk, k, p);
    auto B  = solveR(RY, Vk, k, q);

    // r is the first k singular values of S. S is p × q diagonal matrix
    // from svd_decompose; the singular values live on its main diagonal.
    auto r = Value::matrix(k, 1, ValueType::DOUBLE, mr);
    {
        double *rd = r.doubleDataMut();
        const double *Sd = S.doubleData();
        const keel::usize Sr = S.dims().rows();
        for (keel::usize i = 0; i < k; ++i) {
            // Clamp to [0, 1] — FP drift can push it slightly outside.
            const double si = Sd[i * Sr + i];
            rd[i] = std::min(1.0, std::max(0.0, si));
        }
    }

    return { std::move(A), std::move(B), std::move(r) };
}

} // namespace numkit::stats
