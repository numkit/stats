// toolboxes/stats/src/regress/regress.cpp
//
// OLS (regress), weighted/generalized least squares (lscov), and Ridge
// regression, delegating numerical kernels to keel::stats.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>

#include <numkit/stats/regress/regress.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "regress_detail.hpp"

namespace numkit::stats {

namespace {

void copyToF64(const Value &v, keel::Array<keel::f64> &out)
{
    const keel::usize n = v.numel();
    out.resize(n);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
}

void regress_full_argcheck(const Value &y, const Value &X)
{
    const keel::usize N = y.numel();
    const keel::usize p = X.dims().cols();
    if (X.dims().rows() != N || N == 0 || p == 0)
        throw Error("regress: X must be N×p with same N as y",
                    0, 0, "regress", "", "numkit:regress:size");
}

void lscov_argcheck(const Value &A, const Value &b, const Value &w)
{
    const keel::usize N = b.numel();
    const keel::usize p = A.dims().cols();
    if (A.dims().rows() != N || N == 0 || p == 0)
        throw Error("lscov: A must be N×p with same N as b",
                    0, 0, "lscov", "", "numkit:lscov:size");
    if (!w.isEmpty()) {
        const bool isVecN = (w.numel() == N);
        const bool isMatNN = (w.dims().rows() == N && w.dims().cols() == N);
        if (!isVecN && !isMatNN)
            throw Error("lscov: w must be a length-N vector or N×N covariance matrix",
                        0, 0, "lscov", "", "numkit:lscov:w");
    }
}

void ridge_argcheck(const Value &y, const Value &X, const Value &kVec)
{
    const keel::usize N = y.numel();
    const keel::usize p = X.dims().cols();
    if (X.dims().rows() != N || N == 0 || p == 0)
        throw Error("ridge: X must be N×p with same N as y",
                    0, 0, "ridge", "", "numkit:ridge:size");
    if (kVec.numel() == 0)
        throw Error("ridge: k must be non-empty",
                    0, 0, "ridge", "", "numkit:ridge:k");
}

} // namespace

std::tuple<Value, Value, Value, Value, Value>
regress_full(const Value &y, const Value &X, keel::f64 alpha, keel::Allocator *mr)
{
    regress_full_argcheck(y, X);
    const keel::usize N = y.numel();
    const keel::usize p = X.dims().cols();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> yv(scratch.allocator()), xv(scratch.allocator());
    copyToF64(y, yv);
    copyToF64(X, xv);

    Value bV     = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value bintV  = Value::matrix(p, 2, ValueType::DOUBLE, mr);
    Value rV     = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    Value rintV  = Value::matrix(N, 2, ValueType::DOUBLE, mr);
    Value statsV = Value::matrix(1, 4, ValueType::DOUBLE, mr);

    keel::stats::RegressStats rs{};
    const auto st = keel::stats::regress(
        yv.as_ref(), xv.as_ref(), N, p, alpha,
        keel::MutArrayRef<keel::f64>(bV.doubleDataMut(), p),
        keel::MutArrayRef<keel::f64>(bintV.doubleDataMut(), 2 * p),
        keel::MutArrayRef<keel::f64>(rV.doubleDataMut(), N),
        keel::MutArrayRef<keel::f64>(rintV.doubleDataMut(), 2 * N),
        rs, scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("regress: design matrix is rank-deficient",
                    0, 0, "regress", "", "numkit:regress:rank");

    keel::f64 *sd = statsV.doubleDataMut();
    sd[0] = rs.r2;
    sd[1] = rs.f;
    sd[2] = rs.p_f;
    sd[3] = rs.sigma2;
    return {std::move(bV), std::move(bintV), std::move(rV), std::move(rintV), std::move(statsV)};
}

std::tuple<Value, Value, Value, Value>
regress(const Value &y, const Value &X, keel::f64 alpha, keel::Allocator *mr)
{
    auto [b, bint, r, rint, stats] = regress_full(y, X, alpha, mr);
    (void)rint;
    return {std::move(b), std::move(bint), std::move(r), std::move(stats)};
}

std::tuple<Value, Value, Value, Value>
lscov(const Value &A, const Value &b, const Value &w, keel::Allocator *mr)
{
    lscov_argcheck(A, b, w);
    const keel::usize N = b.numel();
    const keel::usize p = A.dims().cols();
    const bool weighted = !w.isEmpty();
    const bool fullCov = weighted && (w.dims().rows() == N && w.dims().cols() == N && N > 1);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> av(scratch.allocator()), bv(scratch.allocator()), wv(scratch.allocator());
    copyToF64(A, av);
    copyToF64(b, bv);
    if (weighted) copyToF64(w, wv);

    Value xV    = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value stdxV = Value::matrix(p, 1, ValueType::DOUBLE, mr);
    Value SV    = Value::matrix(p, p, ValueType::DOUBLE, mr);
    keel::f64 mse = 0.0;

    const auto st = keel::stats::lscov(
        av.as_ref(), bv.as_ref(), N, p, wv.as_ref(), fullCov,
        keel::MutArrayRef<keel::f64>(xV.doubleDataMut(), p),
        keel::MutArrayRef<keel::f64>(stdxV.doubleDataMut(), p),
        mse,
        keel::MutArrayRef<keel::f64>(SV.doubleDataMut(), p * p),
        scratch.allocator());
    if (st == keel::stats::StatsStatus::kInvalidArgument)
        throw Error("lscov: covariance matrix V must be symmetric positive definite",
                    0, 0, "lscov", "", "numkit:lscov:covNotSPD");
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("lscov: design matrix is rank-deficient",
                    0, 0, "lscov", "", "numkit:lscov:rank");

    return {std::move(xV), std::move(stdxV), Value::scalar(mse, mr), std::move(SV)};
}

Value ridge(const Value &y, const Value &X, const Value &kVec, bool scaled, keel::Allocator *mr)
{
    ridge_argcheck(y, X, kVec);
    const keel::usize N = y.numel();
    const keel::usize p = X.dims().cols();
    const keel::usize Nk = kVec.numel();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> yv(scratch.allocator()), xv(scratch.allocator()), kv(scratch.allocator());
    copyToF64(y, yv);
    copyToF64(X, xv);
    copyToF64(kVec, kv);

    const keel::usize outRows = scaled ? p : (p + 1);
    Value B = Value::matrix(outRows, Nk, ValueType::DOUBLE, mr);
    const auto st = keel::stats::ridge(
        yv.as_ref(), xv.as_ref(), N, p, kv.as_ref(), scaled,
        keel::MutArrayRef<keel::f64>(B.doubleDataMut(), outRows * Nk),
        scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("ridge: regularised normal equations not PD",
                    0, 0, "ridge", "", "numkit:ridge:psd");
    return B;
}

} // namespace numkit::stats
