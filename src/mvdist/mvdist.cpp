// toolboxes/stats/src/mvdist/mvdist.cpp
//
// mvnpdf / mvtpdf / mnpdf — closed-form multivariate PDFs/PMFs.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/stats.h>
#include <numkit/stats/mvdist/mvdist.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>

namespace numkit::stats {

namespace {

bool isMatrixValue(const Value &V)
{
    return V.dims().ndim() <= 2 && V.numel() > 0;
}

Value castDown(const Value &X, bool single, keel::Allocator *mr)
{
    if (!single || X.type() == ValueType::SINGLE) return X;
    const keel::usize r = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize c = keel::as<keel::usize>(X.dims().dim(1));
    Value out = Value::matrix(r, c, ValueType::SINGLE, mr);
    const keel::usize cnt = r * c;
    for (keel::usize k = 0; k < cnt; ++k)
        out.singleDataMut()[k] = keel::as<keel::f32>(X.elemAsDouble(k));
    return out;
}

void mvnpdf_argcheck(const Value &X)
{
    if (!isMatrixValue(X))
        throw Error(keel::format("mvnpdf: X must be a non-empty matrix (got a {} array).",
                                 mtypeName(X.type())),
                    0, 0, "mvnpdf", "", "numkit:mvnpdf:invalidData");
    const keel::usize d = keel::as<keel::usize>(X.dims().dim(1));
    if (d < 1)
        throw Error("mvnpdf: X must have at least one column.",
                    0, 0, "mvnpdf", "", "numkit:mvnpdf:tooFewDimensions");
}

} // namespace

Value mvnpdf(const Value &X, const Value &mu, const Value &Sigma, keel::Allocator *mr)
{
    mvnpdf_argcheck(X);
    keel::usize n = keel::as<keel::usize>(X.dims().dim(0));
    keel::usize d = keel::as<keel::usize>(X.dims().dim(1));
    bool single = (X.type() == ValueType::SINGLE);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> x0(scratch.allocator());

    if (mu.isEmpty()) {
        x0.resize(n * d);
        for (keel::usize k = 0; k < n * d; ++k) x0[k] = X.elemAsDouble(k);
    } else if (mu.isScalar()) {
        const keel::f64 m = mu.elemAsDouble(0);
        x0.resize(n * d);
        for (keel::usize k = 0; k < n * d; ++k) x0[k] = X.elemAsDouble(k) - m;
    } else if (isMatrixValue(mu)) {
        const keel::usize n2 = keel::as<keel::usize>(mu.dims().dim(0));
        const keel::usize d2 = keel::as<keel::usize>(mu.dims().dim(1));
        if (d2 != d)
            throw Error(keel::format("mvnpdf: X and MU must have the same number of "
                                     "columns (X has {}, MU has {}).", d, d2),
                        0, 0, "mvnpdf", "", "numkit:mvnpdf:colSizeMismatch");
        if (n2 == 1 || n2 == n) {
            x0.resize(n * d);
            for (keel::usize j = 0; j < d; ++j)
                for (keel::usize r = 0; r < n; ++r)
                    x0[r + j * n] = X.elemAsDouble(r + j * n)
                                  - mu.elemAsDouble((n2 == 1 ? 0 : r) + j * n2);
        } else if (n == 1) {
            n = n2;
            x0.resize(n * d);
            for (keel::usize j = 0; j < d; ++j)
                for (keel::usize r = 0; r < n; ++r)
                    x0[r + j * n] = X.elemAsDouble(j) - mu.elemAsDouble(r + j * n);
        } else {
            throw Error(keel::format("mvnpdf: MU must be a row vector or have one row "
                                     "per X row (X has {} rows, MU has {}).", n, n2),
                        0, 0, "mvnpdf", "", "numkit:mvnpdf:rowSizeMismatch");
        }
    } else {
        throw Error("mvnpdf: MU must be a scalar or a matrix.",
                    0, 0, "mvnpdf", "", "numkit:mvnpdf:badMu");
    }
    single = single || (mu.type() == ValueType::SINGLE);

    keel::usize pages = 1;
    bool sigmaIsDiag = false;
    keel::Array<keel::f64> diagVar(scratch.allocator());
    keel::Array<keel::f64> sigBuf(scratch.allocator());
    keel::Array<keel::Array<keel::f64>> pageSig(scratch.allocator());

    if (Sigma.isEmpty()) {
        if (d == 1 && X.numel() > 1) {
            d = n;
            n = 1;
        }
        sigmaIsDiag = true;
        diagVar.assign(d, 1.0);
    } else {
        keel::usize sz1 = keel::as<keel::usize>(Sigma.dims().dim(0));
        const keel::usize sz2 = keel::as<keel::usize>(Sigma.dims().dim(1));
        if (Sigma.dims().ndim() <= 2 && sz1 == 1 && sz2 > 1) {
            sigmaIsDiag = true;
            sz1 = sz2;
        }
        if (!sigmaIsDiag && d == 1 && X.numel() > 1 && sz1 == n) {
            d = n;
            n = 1;
        }
        single = single || (Sigma.type() == ValueType::SINGLE);
        pages = Sigma.dims().ndim() >= 3 ? keel::as<keel::usize>(Sigma.dims().dim(2)) : 1;

        if (pages == 1) {
            if (sz1 != sz2)
                throw Error("mvnpdf: SIGMA must be a square covariance "
                            "matrix, the diagonal of one as a row vector, "
                            "or a stack of matrices.",
                            0, 0, "mvnpdf", "", "numkit:mvnpdf:badCovariance");
            if (sz1 != d)
                throw Error(keel::format("mvnpdf: SIGMA must be {}-by-{} to match X "
                                         "(got {}-by-{}).", d, d, sz1, sz2),
                            0, 0, "mvnpdf", "", "numkit:mvnpdf:covSizeMismatch");
            if (sigmaIsDiag) {
                diagVar.resize(d);
                for (keel::usize j = 0; j < d; ++j) {
                    const keel::f64 v = Sigma.elemAsDouble(j);
                    if (!(v > 0.0))
                        throw Error(keel::format("mvnpdf: all elements of the diagonal "
                                                 "SIGMA must be positive (got {} at position {}).",
                                                 v, j + 1),
                                    0, 0, "mvnpdf", "", "numkit:mvnpdf:badDiagSigma");
                    diagVar[j] = v;
                }
            } else {
                sigBuf.resize(d * d);
                for (keel::usize k = 0; k < d * d; ++k) sigBuf[k] = Sigma.elemAsDouble(k);
            }
        } else {
            if (n == 1) {
                keel::Array<keel::f64> xr(pages * d, scratch.allocator());
                for (keel::usize j = 0; j < d; ++j)
                    for (keel::usize r = 0; r < pages; ++r)
                        xr[r + j * pages] = x0[j];
                x0 = keel::move(xr);
                n = pages;
            }
            if (sz1 != sz2)
                throw Error("mvnpdf: SIGMA must be a square covariance "
                            "matrix, the diagonal of one as a row vector, "
                            "or a stack of matrices.",
                            0, 0, "mvnpdf", "", "numkit:mvnpdf:badCovarianceMultiple");
            if (sz1 != d)
                throw Error(keel::format("mvnpdf: each SIGMA page must be {}-by-{} to match X.",
                                         d, d),
                            0, 0, "mvnpdf", "", "numkit:mvnpdf:covSizeMismatchMultiple");
            if (pages != n)
                throw Error(keel::format("mvnpdf: SIGMA must have one page per X row "
                                         "(X has {} rows, SIGMA has {} pages).", n, pages),
                            0, 0, "mvnpdf", "", "numkit:mvnpdf:covSizeMismatchPages");
            if (sigmaIsDiag) {
                for (keel::usize j = 0; j < d; ++j) {
                    for (keel::usize p = 0; p < pages; ++p) {
                        if (!(Sigma.elemAsDouble(j + p * d) > 0.0))
                            throw Error("mvnpdf: all elements of the "
                                        "diagonal SIGMA must be positive.",
                                        0, 0, "mvnpdf", "", "numkit:mvnpdf:badDiagSigma");
                    }
                }
            } else {
                pageSig.reserve(pages);
                for (keel::usize p = 0; p < pages; ++p) {
                    keel::Array<keel::f64> mat(d * d, 0.0, scratch.allocator());
                    for (keel::usize k = 0; k < d * d; ++k)
                        mat[k] = Sigma.elemAsDouble(k + p * d * d);
                    pageSig.push_back(keel::move(mat));
                }
            }
        }
    }

    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    if (pages == 1) {
        if (sigmaIsDiag) {
            (void)keel::stats::mvnpdf_diag(x0.as_ref(), n, d, diagVar.as_ref(),
                                           keel::MutArrayRef<keel::f64>(od, n));
        } else {
            const auto st = keel::stats::mvnpdf_full(
                x0.as_ref(), n, d, sigBuf.as_ref(),
                keel::MutArrayRef<keel::f64>(od, n), scratch.allocator());
            if (st == keel::stats::StatsStatus::kSingularMatrix)
                throw Error("mvnpdf: SIGMA must be a square, symmetric, "
                            "positive semi-definite matrix.",
                            0, 0, "mvnpdf", "", "numkit:mvnpdf:badMatrixSigma");
        }
    } else if (sigmaIsDiag) {
        keel::Array<keel::f64> xrow(d, scratch.allocator());
        keel::Array<keel::f64> dv(d, scratch.allocator());
        for (keel::usize r = 0; r < n; ++r) {
            for (keel::usize j = 0; j < d; ++j) {
                xrow[j] = x0[r + j * n];
                dv[j] = Sigma.elemAsDouble(j + r * d);
            }
            (void)keel::stats::mvnpdf_diag(xrow.as_ref(), 1, d, dv.as_ref(),
                                           keel::MutArrayRef<keel::f64>(od + r, 1));
        }
    } else {
        keel::Array<keel::f64> xrow(d, scratch.allocator());
        for (keel::usize r = 0; r < n; ++r) {
            for (keel::usize j = 0; j < d; ++j) xrow[j] = x0[r + j * n];
            const auto st = keel::stats::mvnpdf_full(
                xrow.as_ref(), 1, d, pageSig[r].as_ref(),
                keel::MutArrayRef<keel::f64>(od + r, 1), scratch.allocator());
            if (st == keel::stats::StatsStatus::kSingularMatrix)
                throw Error("mvnpdf: SIGMA must be a square, symmetric, "
                            "positive semi-definite matrix.",
                            0, 0, "mvnpdf", "", "numkit:mvnpdf:badMatrixSigmaMultiple");
        }
    }
    return castDown(out, single, mr);
}

Value mvtpdfCompute(const Value &X, const Value &C, const Value &df,
                    bool single, keel::Allocator *mr)
{
    const keel::usize n = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize d = keel::as<keel::usize>(X.dims().dim(1));
    const keel::usize cr = keel::as<keel::usize>(C.dims().dim(0));
    const keel::usize cc = keel::as<keel::usize>(C.dims().dim(1));
    if (cr != cc)
        throw Error(keel::format("mvtpdf: C must be a square matrix (got {}-by-{}).", cr, cc),
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:badCorrelationNotSquare");
    if (cr != d)
        throw Error(keel::format("mvtpdf: C must be a square matrix with size equal to "
                                 "the number of columns of X ({}-by-{}, got {}-by-{}).",
                                 d, d, cr, cc),
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:inputSizeMismatchC");
    single = single || (C.type() == ValueType::SINGLE);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> dfv(n, scratch.allocator());
    if (df.isScalar()) {
        std::fill(dfv.begin(), dfv.end(), df.elemAsDouble(0));
    } else if (df.dims().isVector() && df.numel() == n) {
        for (keel::usize r = 0; r < n; ++r) dfv[r] = df.elemAsDouble(r);
    } else {
        throw Error("mvtpdf: DF must be a scalar or a vector with one "
                    "element for each row of X.",
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:inputSizeMismatchDF");
    }
    for (keel::f64 v : dfv)
        if (!(v > 0.0))
            throw Error(keel::format("mvtpdf: DF must be positive (got {}).", v),
                        0, 0, "mvtpdf", "", "numkit:mvtpdf:invalidDF");
    single = single || (df.type() == ValueType::SINGLE);

    keel::Array<keel::f64> xb(n * d, scratch.allocator());
    for (keel::usize i = 0; i < n * d; ++i) xb[i] = X.elemAsDouble(i);
    keel::Array<keel::f64> cb(d * d, scratch.allocator());
    for (keel::usize i = 0; i < d * d; ++i) cb[i] = C.elemAsDouble(i);

    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    const auto st = keel::stats::mvtpdf_full(
        xb.as_ref(), n, d, cb.as_ref(), dfv.as_ref(),
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n), scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("mvtpdf: C must be a symmetric positive "
                    "(semi-)definite matrix.",
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:badCorrelationSymPos");
    return castDown(out, single, mr);
}

Value mvtpdf(const Value &X, const Value &C, const Value &df, keel::Allocator *mr)
{
    if (X.dims().ndim() != 2)
        throw Error("mvtpdf: X must be a matrix.",
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:invalidData");
    const keel::usize n = keel::as<keel::usize>(X.dims().dim(0));
    const keel::usize d = keel::as<keel::usize>(X.dims().dim(1));
    if (d < 1)
        throw Error("mvtpdf: X must have at least one column.",
                    0, 0, "mvtpdf", "", "numkit:mvtpdf:tooFewDimensions");
    const bool single = (X.type() == ValueType::SINGLE);

    const keel::usize cr = C.dims().dim(0);
    if (X.dims().isVector() && cr == n) {
        Value Xt = Value::matrix(1, n, ValueType::DOUBLE, mr);
        for (keel::usize k = 0; k < n; ++k)
            Xt.doubleDataMut()[k] = X.elemAsDouble(k);
        return mvtpdfCompute(Xt, C, df, single, mr);
    }
    return mvtpdfCompute(X, C, df, single, mr);
}

Value mnpdf(const Value &X, const Value &P, keel::Allocator *mr)
{
    Value x = X, p = P;
    const bool single = (X.type() == ValueType::SINGLE) || (P.type() == ValueType::SINGLE);
    const keel::f64 epsD = keel::Limits<keel::f64>::kEpsilon;

    if (p.dims().dim(1) == 1 && p.dims().dim(0) > 1) {
        keel::f64 s = 0.0;
        const keel::usize len = keel::as<keel::usize>(p.numel());
        for (keel::usize k = 0; k < len; ++k) s += p.elemAsDouble(k);
        if (keel::abs(s - 1.0) <= keel::as<keel::f64>(len) * epsD) {
            Value pt = Value::matrix(1, len, ValueType::DOUBLE, mr);
            for (keel::usize k = 0; k < len; ++k) pt.doubleDataMut()[k] = p.elemAsDouble(k);
            p = pt;
            if (x.dims().dim(1) == 1 && x.dims().dim(0) > 1) {
                Value xt = Value::matrix(1, x.numel(), ValueType::DOUBLE, mr);
                for (keel::usize k = 0; k < keel::as<keel::usize>(x.numel()); ++k)
                    xt.doubleDataMut()[k] = x.elemAsDouble(k);
                x = xt;
            }
        }
    }

    keel::usize m = keel::as<keel::usize>(x.dims().dim(0));
    const keel::usize k = keel::as<keel::usize>(x.dims().dim(1));
    if (k < 1)
        throw Error("mnpdf: X must have at least one column.",
                    0, 0, "mnpdf", "", "numkit:mnpdf:noCategories");

    const keel::usize mm = keel::as<keel::usize>(p.dims().dim(0));
    const keel::usize kk = keel::as<keel::usize>(p.dims().dim(1));
    if (kk != k)
        throw Error(keel::format("mnpdf: P and X must have the same number of columns "
                                 "(X has {}, P has {}).", k, kk),
                    0, 0, "mnpdf", "", "numkit:mnpdf:colSizeMismatch");

    if (mm != 1 && m != 1 && mm != m)
        throw Error(keel::format("mnpdf: P and X must have the same number of rows "
                                 "(X has {}, P has {}).", m, mm),
                    0, 0, "mnpdf", "", "numkit:mnpdf:rowSizeMismatch");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xb(scratch.allocator());
    if (m == 1 && mm > 1) {
        xb.resize(mm * k);
        for (keel::usize j = 0; j < k; ++j)
            for (keel::usize r = 0; r < mm; ++r)
                xb[r + j * mm] = x.elemAsDouble(j);
        m = mm;
    } else {
        xb.resize(m * k);
        for (keel::usize i = 0; i < m * k; ++i) xb[i] = x.elemAsDouble(i);
    }

    keel::Array<keel::f64> pb(mm * k, scratch.allocator());
    for (keel::usize i = 0; i < mm * k; ++i) pb[i] = p.elemAsDouble(i);

    Value out = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    (void)keel::stats::mnpdf_rows(xb.as_ref(), m, k, pb.as_ref(), mm,
                                  keel::MutArrayRef<keel::f64>(out.doubleDataMut(), m));
    return castDown(out, single, mr);
}

} // namespace numkit::stats
