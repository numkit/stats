// toolboxes/stats/src/spline/spline.cpp

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/interp.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/tuple.h>
#include <keel/utf8.h>
#include <numkit/stats/spline/spline.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::stats {

namespace {
void aveknt_argcheck(keel::usize N, keel::i32 k)
{
    if (k < 2 || keel::as<keel::usize>(k) > N)
        throw Error("aveknt: order k must satisfy 2 ≤ k ≤ length(t)",
                    0, 0, "aveknt", "", "numkit:aveknt:k");
}
} // namespace

Value aveknt(const Value &t, keel::i32 k, keel::Allocator *mr)
{
    const keel::usize N = t.numel();
    aveknt_argcheck(N, k);
    const keel::usize M = N - keel::as<keel::usize>(k);
    Value out = Value::matrix(1, M, ValueType::DOUBLE, mr);
    if (M == 0) return out;
    keel::Array<keel::f64> tv(N, mr);
    for (keel::usize i = 0; i < N; ++i) tv[i] = t.elemAsDouble(i);
    (void)keel::interp::aveknt(tv.as_ref(), k, keel::MutArrayRef<keel::f64>(out.doubleDataMut(), M));
    return out;
}

Value augknt(const Value &knots, keel::i32 k, keel::Allocator *mr)
{
    const keel::usize N = knots.numel();
    if (k < 1 || N == 0)
        throw Error("augknt: k must be ≥ 1 and knots non-empty",
                    0, 0, "augknt", "", "numkit:augknt:k");
    const keel::usize outN = N + 2 * (keel::as<keel::usize>(k) - 1);
    Value out = Value::matrix(1, outN, ValueType::DOUBLE, mr);
    keel::Array<keel::f64> kv(N, mr);
    for (keel::usize i = 0; i < N; ++i) kv[i] = knots.elemAsDouble(i);
    (void)keel::interp::augknt(kv.as_ref(), k, keel::MutArrayRef<keel::f64>(out.doubleDataMut(), outN));
    return out;
}

Value brk2knt(const Value &breaks, const Value &mults, keel::Allocator *mr)
{
    const keel::usize N = breaks.numel();
    if (mults.numel() != N)
        throw Error("brk2knt: breaks and mults must have same length",
                    0, 0, "brk2knt", "", "numkit:brk2knt:size");
    keel::Array<keel::f64> bv(N, mr);
    keel::Array<keel::usize> mv(N, mr);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 mi = mults.elemAsDouble(i);
        if (!(mi >= 0.0))
            throw Error("brk2knt: multiplicities must be non-negative",
                        0, 0, "brk2knt", "", "numkit:brk2knt:m");
        bv[i] = breaks.elemAsDouble(i);
        mv[i] = keel::as<keel::usize>(mi);
    }
    keel::Array<keel::f64> knots(mr);
    (void)keel::interp::brk2knt(bv.as_ref(), mv.as_ref(), knots, mr);
    Value out = Value::matrix(1, knots.size(), ValueType::DOUBLE, mr);
    if (!knots.empty()) {
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < knots.size(); ++i) od[i] = knots[i];
    }
    return out;
}

keel::Tuple<Value, Value>
knt2brk(const Value &knots, keel::Allocator *mr)
{
    const keel::usize N = knots.numel();
    if (N == 0) {
        Value bv = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        Value mv = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return {keel::move(bv), keel::move(mv)};
    }
    keel::Array<keel::f64> kv(N, mr);
    for (keel::usize i = 0; i < N; ++i) kv[i] = knots.elemAsDouble(i);
    keel::Array<keel::f64> b(mr);
    keel::Array<keel::usize> m(mr);
    keel::interp::knt2brk(kv.as_ref(), b, m, mr);

    const keel::usize K = b.size();
    Value bv = Value::matrix(1, K, ValueType::DOUBLE, mr);
    Value mv = Value::matrix(1, K, ValueType::DOUBLE, mr);
    keel::f64 *bd = bv.doubleDataMut();
    keel::f64 *md = mv.doubleDataMut();
    for (keel::usize i = 0; i < K; ++i) { bd[i] = b[i]; md[i] = keel::f64(m[i]); }
    return {keel::move(bv), keel::move(mv)};
}

Value ppmak(const Value &breaks, const Value &coefs, keel::i32 d, keel::Allocator *mr)
{
    const keel::usize L1 = breaks.numel();
    if (L1 < 2)
        throw Error("ppmak: breaks must have at least 2 entries",
                    0, 0, "ppmak", "", "numkit:ppmak:breaks");
    // Breaks must be strictly increasing (the reference rejects
    // decreasing sequences and repeated extremes, probe78 S04/S05).
    for (keel::usize i = 0; i + 1 < L1; ++i)
        if (!(breaks.elemAsDouble(i) < breaks.elemAsDouble(i + 1)))
            throw Error(keel::format("ppmak: the breaks sequence must be strictly "
                                     "increasing (breaks({}) = {} is not less than breaks({}) = {}).",
                                     i + 1, breaks.elemAsDouble(i),
                                     i + 2, breaks.elemAsDouble(i + 1)),
                        0, 0, "ppmak", "", "numkit:ppmak:breaks");
    const keel::usize L  = L1 - 1;
    const keel::usize cR = coefs.dims().rows();
    const keel::usize cC = coefs.dims().cols();
    if (d < 1) d = 1;
    keel::usize effD = keel::as<keel::usize>(d);
    keel::usize effK = cC;
    bool reshapeDbyLK = false;
    if (cR != effD * L) {
        if (cR >= 1 && cC >= L && (cC % L) == 0 && (d == 1 || cR == effD)) {
            effD = cR;
            effK = cC / L;
            reshapeDbyLK = true;
        } else {
            throw Error("ppmak: coefs must be d·L × K (rows = dim·pieces)",
                        0, 0, "ppmak", "", "numkit:ppmak:coefs");
        }
    }

    Value bv = Value::matrix(1, L1, ValueType::DOUBLE, mr);
    {
        keel::f64 *bd = bv.doubleDataMut();
        for (keel::usize i = 0; i < L1; ++i) bd[i] = breaks.elemAsDouble(i);
    }
    const keel::usize outR = effD * L;
    Value cv = Value::matrix(outR, effK, ValueType::DOUBLE, mr);
    {
        keel::f64 *cd = cv.doubleDataMut();
        if (reshapeDbyLK) {
            for (keel::usize m = 0; m < effK; ++m) {
                for (keel::usize j = 0; j < L; ++j) {
                    const keel::usize srcCol = m * L + j;
                    for (keel::usize r = 0; r < effD; ++r) {
                        cd[(r + j * effD) + m * outR] = coefs.elemAsDouble(r + srcCol * cR);
                    }
                }
            }
        } else {
            for (keel::usize i = 0; i < cR * cC; ++i) cd[i] = coefs.elemAsDouble(i);
        }
    }

    Value s = Value::structure(mr);
    s.field("form")   = Value::fromString("pp", mr);
    s.field("breaks") = keel::move(bv);
    s.field("coefs")  = keel::move(cv);
    s.field("pieces") = Value::scalar(keel::f64(L),    mr);
    s.field("order")  = Value::scalar(keel::f64(effK), mr);
    s.field("dim")    = Value::scalar(keel::f64(effD), mr);
    return s;
}

Value fnval(const Value &pp, const Value &xv, keel::Allocator *mr)
{
    if (!pp.hasField("form") || pp.field("form").toString() != "pp")
        throw Error("fnval: only pp form supported in this release",
                    0, 0, "fnval", "", "numkit:fnval:form");
    const Value &breaks = pp.field("breaks");
    const Value &coefs  = pp.field("coefs");
    const keel::usize L  = keel::as<keel::usize>(pp.field("pieces").toScalar());
    const keel::usize K  = keel::as<keel::usize>(pp.field("order").toScalar());
    const keel::usize d  = keel::as<keel::usize>(pp.field("dim").toScalar());
    const keel::usize cR = coefs.dims().rows();
    if (cR != d * L || coefs.dims().cols() != K || breaks.numel() != L + 1)
        throw Error("fnval: pp struct fields are inconsistent",
                    0, 0, "fnval", "", "numkit:fnval:struct");

    const keel::usize Nx = xv.numel();
    Value out;
    if (d == 1) {
        const auto &dx = xv.dims();
        out = Value::matrix(dx.rows(), dx.cols(), ValueType::DOUBLE, mr);
    } else {
        out = Value::matrix(d, Nx, ValueType::DOUBLE, mr);
    }
    if (Nx == 0) return out;

    keel::Array<keel::f64> bv(L + 1, mr);
    for (keel::usize i = 0; i <= L; ++i) bv[i] = breaks.elemAsDouble(i);
    keel::Array<keel::f64> cv(cR * K, mr);
    for (keel::usize i = 0; i < cR * K; ++i) cv[i] = coefs.elemAsDouble(i);
    keel::Array<keel::f64> qv(Nx, mr);
    for (keel::usize i = 0; i < Nx; ++i) qv[i] = xv.elemAsDouble(i);
    (void)keel::interp::ppval_nd(
        bv.as_ref(), cv.as_ref(), L, K, d, qv.as_ref(),
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), d * Nx));
    return out;
}

namespace {

struct PPView {
    Value breaks;
    Value coefs;
    keel::usize L, K, d;
};

PPView readPP(const Value &pp)
{
    if (!pp.hasField("form") || pp.field("form").toString() != "pp")
        throw Error("fnder/fnint: only pp form supported",
                    0, 0, "fn", "", "numkit:fn:form");
    PPView v;
    v.breaks = pp.field("breaks");
    v.coefs  = pp.field("coefs");
    v.L = keel::as<keel::usize>(pp.field("pieces").toScalar());
    v.K = keel::as<keel::usize>(pp.field("order").toScalar());
    v.d = keel::as<keel::usize>(pp.field("dim").toScalar());
    return v;
}

Value buildPP(const Value &breaks, keel::ArrayRef<keel::f64> coefsCM, keel::usize cR, keel::usize cC, keel::usize L, keel::usize d, keel::Allocator *mr)
{
    Value bv = Value::matrix(1, breaks.numel(), ValueType::DOUBLE, mr);
    {
        keel::f64 *bd = bv.doubleDataMut();
        for (keel::usize i = 0; i < breaks.numel(); ++i) bd[i] = breaks.elemAsDouble(i);
    }
    Value cv = Value::matrix(cR, cC, ValueType::DOUBLE, mr);
    {
        keel::f64 *cd = cv.doubleDataMut();
        for (keel::usize i = 0; i < cR * cC; ++i) cd[i] = coefsCM[i];
    }
    Value s = Value::structure(mr);
    s.field("form")   = Value::fromString("pp", mr);
    s.field("breaks") = keel::move(bv);
    s.field("coefs")  = keel::move(cv);
    s.field("pieces") = Value::scalar(keel::f64(L), mr);
    s.field("order")  = Value::scalar(keel::f64(cC), mr);
    s.field("dim")    = Value::scalar(keel::f64(d), mr);
    return s;
}

} // anonymous

Value fnder(const Value &pp, keel::i32 order, keel::Allocator *mr)
{
    if (order < 0)
        throw Error("fnder: order must be >= 0",
                    0, 0, "fnder", "", "numkit:fnder:order");
    PPView v = readPP(pp);
    const keel::usize cR = v.d * v.L;
    keel::Array<keel::f64> cv(cR * v.K, mr);
    for (keel::usize i = 0; i < cv.size(); ++i) cv[i] = v.coefs.elemAsDouble(i);
    keel::Array<keel::f64> nc(mr);
    keel::usize newK = 0;
    (void)keel::interp::pp_deriv(cv.as_ref(), v.L, v.K, v.d, order, nc, newK, mr);
    return buildPP(v.breaks, nc.as_ref(), cR, newK, v.L, v.d, mr);
}

Value fnint(const Value &pp, keel::Allocator *mr)
{
    PPView v = readPP(pp);
    const keel::usize cR = v.d * v.L;
    const keel::usize newK = v.K + 1;
    keel::Array<keel::f64> bv(v.L + 1, mr);
    for (keel::usize i = 0; i <= v.L; ++i) bv[i] = v.breaks.elemAsDouble(i);
    keel::Array<keel::f64> cv(cR * v.K, mr);
    for (keel::usize i = 0; i < cv.size(); ++i) cv[i] = v.coefs.elemAsDouble(i);
    keel::Array<keel::f64> nc(cR * newK, 0.0, mr);
    (void)keel::interp::pp_integ(bv.as_ref(), cv.as_ref(), v.L, v.K, v.d, nc.as_mut_ref());
    return buildPP(v.breaks, nc.as_ref(), cR, newK, v.L, v.d, mr);
}

Value csapi(const Value &x, const Value &y, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    if (y.numel() != N || N < 2)
        throw Error("csapi: x and y must be vectors of the same length ≥ 2",
                    0, 0, "csapi", "", "numkit:csapi:size");
    keel::Array<keel::f64> xv(N, mr), yv(N, mr);
    for (keel::usize i = 0; i < N; ++i) {
        xv[i] = x.elemAsDouble(i);
        yv[i] = y.elemAsDouble(i);
    }
    for (keel::usize i = 0; i + 1 < N; ++i) {
        if (xv[i + 1] == xv[i])
            throw Error("csapi: singular system (duplicate breaks?)",
                        0, 0, "csapi", "", "numkit:csapi:singular");
    }

    const keel::usize L = N - 1;
    keel::Array<keel::f64> coefs(L * 4, 0.0, mr);
    if (!keel::interp::spline_pp_coefs(xv.as_ref(), yv.as_ref(), coefs.as_mut_ref(), mr))
        throw Error("csapi: singular system (duplicate breaks?)",
                    0, 0, "csapi", "", "numkit:csapi:singular");

    Value bv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    keel::f64 *bd = bv.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) bd[i] = xv[i];
    return buildPP(bv, coefs.as_ref(), L, 4, L, 1, mr);
}

Value fncmb(const Value &pp1, keel::f64 c1, const Value &pp2, keel::f64 c2,
            keel::Allocator *mr)
{
    PPView v1 = readPP(pp1);
    const keel::usize cR = v1.coefs.dims().rows();
    const keel::usize cC = v1.coefs.dims().cols();
    keel::Array<keel::f64> nc(cR * cC, 0.0);
    if (pp2.isEmpty()) {
        for (keel::usize i = 0; i < cR * cC; ++i)
            nc[i] = c1 * v1.coefs.elemAsDouble(i);
        return buildPP(v1.breaks, nc.as_ref(), cR, cC, v1.L, v1.d, mr);
    }
    PPView v2 = readPP(pp2);
    if (v2.L != v1.L || v2.K != v1.K || v2.d != v1.d)
        throw Error("fncmb: pp1 and pp2 must share pieces / order / dim",
                    0, 0, "fncmb", "", "numkit:fncmb:shape");
    if (v1.breaks.numel() != v2.breaks.numel())
        throw Error("fncmb: pp1 and pp2 must have same breaks",
                    0, 0, "fncmb", "", "numkit:fncmb:breaks");
    for (keel::usize i = 0; i < v1.breaks.numel(); ++i)
        if (v1.breaks.elemAsDouble(i) != v2.breaks.elemAsDouble(i))
            throw Error("fncmb: pp1 and pp2 must have same breaks",
                        0, 0, "fncmb", "", "numkit:fncmb:breaks");
    for (keel::usize i = 0; i < cR * cC; ++i)
        nc[i] = c1 * v1.coefs.elemAsDouble(i) + c2 * v2.coefs.elemAsDouble(i);
    return buildPP(v1.breaks, nc.as_ref(), cR, cC, v1.L, v1.d, mr);
}

Value fnbrk(const Value &pp, keel::StringRef part_in, keel::Allocator *mr)
{
    if (!pp.hasField("form"))
        throw Error("fnbrk: input must be a spline struct",
                    0, 0, "fnbrk", "", "numkit:fnbrk:struct");
    keel::String part = keel::String::from(part_in);
    part.to_ascii_lower_in_place();
    auto copyV = [&](const Value &v) {
        if (v.type() == ValueType::CHAR || v.isString()) {
            keel::String sv = v.toString();
            return Value::fromString(sv.as_ref(), mr);
        }
        const auto &d = v.dims();
        Value out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
        const keel::usize n = v.numel();
        if (n == 0) return out;
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) od[i] = v.elemAsDouble(i);
        return out;
    };
    if (part == "form")   return copyV(pp.field("form"));
    if (part == "breaks") return copyV(pp.field("breaks"));
    if (part == "coefs")  return copyV(pp.field("coefs"));
    if (part == "pieces" || part == "l")
        return Value::scalar(pp.field("pieces").toScalar(), mr);
    if (part == "order"  || part == "k")
        return Value::scalar(pp.field("order").toScalar(), mr);
    if (part == "dim"    || part == "d")
        return Value::scalar(pp.field("dim").toScalar(), mr);
    // Unknown part names fall back to the breaks in the reference
    // (probe78 S13); keep that acceptance instead of erroring.
    return copyV(pp.field("breaks"));
}

} // namespace numkit::stats
