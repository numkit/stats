// toolboxes/stats/src/dim/pca.cpp
//
// Principal Component Analysis (pca / pcacov / pcares),
// delegating numerical kernels to keel::stats.

#include <keel/arena.h>
#include <keel/tuple.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>

#include <numkit/stats/dim/pca.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "pca_detail.hpp"

namespace numkit::stats {

namespace {

void pca_argcheck(const Value &X)
{
    const keel::usize N = X.dims().rows();
    const keel::usize D = X.dims().cols();
    if (N < 2 || D < 1)
        throw Error("pca: need at least 2 rows and 1 column",
                    0, 0, "pca", "", "numkit:pca:size");
}

} // namespace

keel::Tuple<Value, Value, Value, Value, Value, Value>
pca(const Value &X, keel::Allocator *mr)
{
    pca_argcheck(X);
    const keel::usize N = X.dims().rows();
    const keel::usize D = X.dims().cols();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(N * D, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N * D; ++i) xv[i] = X.elemAsDouble(i);

    Value coeff_v     = Value::matrix(D, D, ValueType::DOUBLE, mr);
    Value score_v     = Value::matrix(N, D, ValueType::DOUBLE, mr);
    Value latent_v    = Value::matrix(D, 1, ValueType::DOUBLE, mr);
    Value tsq_v       = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    Value explained_v = Value::matrix(D, 1, ValueType::DOUBLE, mr);
    Value mu_v        = Value::matrix(1, D, ValueType::DOUBLE, mr);

    (void)keel::stats::pca(
        xv.as_ref(), N, D,
        keel::MutArrayRef<keel::f64>(coeff_v.doubleDataMut(), D * D),
        keel::MutArrayRef<keel::f64>(score_v.doubleDataMut(), N * D),
        keel::MutArrayRef<keel::f64>(latent_v.doubleDataMut(), D),
        keel::MutArrayRef<keel::f64>(tsq_v.doubleDataMut(), N),
        keel::MutArrayRef<keel::f64>(explained_v.doubleDataMut(), D),
        keel::MutArrayRef<keel::f64>(mu_v.doubleDataMut(), D),
        scratch.allocator());

    return keel::make_tuple(keel::move(coeff_v), keel::move(score_v),
                           keel::move(latent_v), keel::move(tsq_v),
                           keel::move(explained_v), keel::move(mu_v));
}

keel::Tuple<Value, Value, Value>
pcacov(const Value &C, keel::Allocator *mr)
{
    const keel::usize D = C.dims().rows();
    if (C.dims().cols() != D)
        throw Error("pcacov: C must be square", 0, 0, "pcacov", "",
                    "numkit:pcacov:size");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> cv(D * D, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < D * D; ++i) cv[i] = C.elemAsDouble(i);

    Value coeff_v     = Value::matrix(D, D, ValueType::DOUBLE, mr);
    Value latent_v    = Value::matrix(D, 1, ValueType::DOUBLE, mr);
    Value explained_v = Value::matrix(D, 1, ValueType::DOUBLE, mr);

    (void)keel::stats::pcacov(
        cv.as_ref(), D,
        keel::MutArrayRef<keel::f64>(coeff_v.doubleDataMut(), D * D),
        keel::MutArrayRef<keel::f64>(latent_v.doubleDataMut(), D),
        keel::MutArrayRef<keel::f64>(explained_v.doubleDataMut(), D),
        scratch.allocator());

    return keel::make_tuple(keel::move(coeff_v), keel::move(latent_v),
                           keel::move(explained_v));
}

keel::Tuple<Value, Value>
pcares_full(const Value &X, keel::i32 ndim, keel::Allocator *mr)
{
    pca_argcheck(X);
    const keel::usize N = X.dims().rows();
    const keel::usize D = X.dims().cols();
    if (ndim < 0 || keel::as<keel::usize>(ndim) > D)
        throw Error("pcares: ndim must be in 0..D", 0, 0, "pcares", "",
                    "numkit:pcares:badndim");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(N * D, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N * D; ++i) xv[i] = X.elemAsDouble(i);

    Value res   = Value::matrix(N, D, ValueType::DOUBLE, mr);
    Value recon = Value::matrix(N, D, ValueType::DOUBLE, mr);
    (void)keel::stats::pcares(
        xv.as_ref(), N, D, keel::as<keel::usize>(ndim),
        keel::MutArrayRef<keel::f64>(res.doubleDataMut(), N * D),
        keel::MutArrayRef<keel::f64>(recon.doubleDataMut(), N * D),
        scratch.allocator());
    return {keel::move(res), keel::move(recon)};
}

Value pcares(const Value &X, keel::i32 ndim, keel::Allocator *mr)
{
    auto [res, recon] = pcares_full(X, ndim, mr);
    (void)recon;
    return res;
}

} // namespace numkit::stats
