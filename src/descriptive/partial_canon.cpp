// toolboxes/stats/src/descriptive/partial_canon.cpp
// Thin Value wrappers over keel::stats::partialcorri and keel::stats::canoncorr.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::stats {

namespace {

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

void canoncorr_argcheck(const Value &X, const Value &Y)
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

    auto Rout = Value::matrix(pY, pX, ValueType::DOUBLE, mr);
    (void)keel::stats::partialcorri(
        keel::ArrayRef<keel::f64>(Y.doubleData(), n * pY), n, pY,
        keel::ArrayRef<keel::f64>(X.doubleData(), n * pX), pX,
        hasZ ? keel::ArrayRef<keel::f64>(Z.doubleData(), n * pZ) : keel::ArrayRef<keel::f64>{}, pZ,
        keel::MutArrayRef<keel::f64>(Rout.doubleDataMut(), pY * pX), mr);
    return Rout;
}

CanoncorrResult canoncorr(const Value &X, const Value &Y,
                           keel::Allocator *mr)
{
    canoncorr_argcheck(X, Y);
    const keel::usize n = X.dims().dim(0);
    const keel::usize p = X.dims().dim(1);
    const keel::usize q = Y.dims().dim(1);
    const keel::usize k = keel::min(p, q);

    auto A = Value::matrix(p, k, ValueType::DOUBLE, mr);
    auto B = Value::matrix(q, k, ValueType::DOUBLE, mr);
    auto r = Value::matrix(k, 1, ValueType::DOUBLE, mr);
    (void)keel::stats::canoncorr(
        keel::ArrayRef<keel::f64>(X.doubleData(), n * p), n, p,
        keel::ArrayRef<keel::f64>(Y.doubleData(), n * q), q,
        keel::MutArrayRef<keel::f64>(A.doubleDataMut(), p * k),
        keel::MutArrayRef<keel::f64>(B.doubleDataMut(), q * k),
        keel::MutArrayRef<keel::f64>(r.doubleDataMut(), k), mr);
    return { keel::move(A), keel::move(B), keel::move(r) };
}

} // namespace numkit::stats
