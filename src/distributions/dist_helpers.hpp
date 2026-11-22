// toolboxes/stats/src/distributions/dist_helpers.hpp
//
// Shared helpers for the distribution-CDF adapters. Currently exposes
// the `'upper'` flag detection that all MATLAB *cdf functions accept:
//
//   p = <dist>cdf(x, ..., 'upper')   →   1 - F(x; ...)
//
// Used by every cdf_reg adapter under toolboxes/stats/src/distributions/.

#pragma once

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/builtin/elmat.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace numkit::stats::detail {

// If the last positional arg is a string equal to 'upper' (any case),
// strip it and set `upper = true`. Returns the effective trailing-arg
// count. Adapters should consume args[0..count-1] for positional
// parsing and ignore args[count..] (which is at most the 'upper' flag).
inline keel::usize stripUpperFlag(keel::ArrayRef<Value> args, bool &upper)
{
    upper = false;
    if (args.empty()) return 0;
    const Value &last = args[args.size() - 1];
    if (!last.isChar() && !last.isString()) return args.size();
    keel::String s = last.toString();
    s.to_ascii_lower_in_place();
    if (s == "upper") {
        upper = true;
        return args.size() - 1;
    }
    return args.size();
}

// Apply the upper-tail transform in place: y = 1 - y for every finite
// element. NaN passes through unchanged.
inline void applyUpperInPlace(Value &y)
{
    if (y.type() != ValueType::DOUBLE) return;
    keel::f64 *d = y.doubleDataMut();
    const keel::usize n = y.numel();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = d[i];
        if (!std::isnan(v)) d[i] = 1.0 - v;
    }
}

// ── *stat vectorisation helpers ───────────────────────────────────────
//
// Every `<dist>stat(args...)` returns `tuple<double, double>` (mean,
// variance). MATLAB vectorises these element-wise with broadcasting
// (equal sizes OR one scalar). The helpers below take a scalar impl
// and apply the broadcasting boilerplate once.
//
// Edge: 0-dim scalar inputs use a fast path (no allocation).

template <class Stat1>
inline void emit_vec_stat_1arg(keel::ArrayRef<Value> args, keel::usize nargout,
                                keel::MutArrayRef<Value> outs, keel::Allocator *mr,
                                const char *fnName, Stat1 stat_impl)
{
    if (args.empty())
        throw Error(keel::format("{}: requires 1 arg", fnName),
                    0, 0, fnName, "", "numkit:nargin");
    const Value &p = args[0];
    if (p.isScalar()) {
        auto [m, v] = stat_impl(p.toScalar());
        outs[0] = Value::scalar(m, mr);
        if (nargout > 1) outs[1] = Value::scalar(v, mr);
        return;
    }
    const auto &d = p.dims();
    Value out_m = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    Value out_v = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *pm = out_m.doubleDataMut();
    keel::f64 *pv = out_v.doubleDataMut();
    const keel::usize n = p.numel();
    for (keel::usize i = 0; i < n; ++i) {
        auto [m, v] = stat_impl(p.elemAsDouble(i));
        pm[i] = m;
        pv[i] = v;
    }
    outs[0] = std::move(out_m);
    if (nargout > 1) outs[1] = std::move(out_v);
}

template <class Stat2>
inline void emit_vec_stat_2arg(keel::ArrayRef<Value> args, keel::usize nargout,
                                keel::MutArrayRef<Value> outs, keel::Allocator *mr,
                                const char *fnName, Stat2 stat_impl)
{
    if (args.size() < 2)
        throw Error(keel::format("{}: requires 2 args", fnName),
                    0, 0, fnName, "", "numkit:nargin");
    const Value &av = args[0];
    const Value &bv = args[1];
    const keel::usize na = av.numel();
    const keel::usize nb = bv.numel();
    if (na == 1 && nb == 1) {
        auto [m, v] = stat_impl(av.toScalar(), bv.toScalar());
        outs[0] = Value::scalar(m, mr);
        if (nargout > 1) outs[1] = Value::scalar(v, mr);
        return;
    }
    if (na > 1 && nb > 1 && na != nb)
        throw Error(keel::format("{}: args must be same size or scalar", fnName),
                    0, 0, fnName, "", "numkit:dim");
    const Value &ref = (na >= nb) ? av : bv;
    const auto &d = ref.dims();
    Value out_m = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    Value out_v = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *pm = out_m.doubleDataMut();
    keel::f64 *pv = out_v.doubleDataMut();
    const keel::usize n = ref.numel();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 a = av.elemAsDouble(na == 1 ? 0 : i);
        const keel::f64 b = bv.elemAsDouble(nb == 1 ? 0 : i);
        auto [m, v] = stat_impl(a, b);
        pm[i] = m;
        pv[i] = v;
    }
    outs[0] = std::move(out_m);
    if (nargout > 1) outs[1] = std::move(out_v);
}

template <class Stat3>
inline void emit_vec_stat_3arg(keel::ArrayRef<Value> args, keel::usize nargout,
                                keel::MutArrayRef<Value> outs, keel::Allocator *mr,
                                const char *fnName, Stat3 stat_impl)
{
    if (args.size() < 3)
        throw Error(keel::format("{}: requires 3 args", fnName),
                    0, 0, fnName, "", "numkit:nargin");
    const Value &av = args[0];
    const Value &bv = args[1];
    const Value &cv = args[2];
    const keel::usize na = av.numel(), nb = bv.numel(), nc = cv.numel();
    const keel::usize nmax = std::max({na, nb, nc});
    if (nmax == 1) {
        auto [m, v] = stat_impl(av.toScalar(), bv.toScalar(), cv.toScalar());
        outs[0] = Value::scalar(m, mr);
        if (nargout > 1) outs[1] = Value::scalar(v, mr);
        return;
    }
    auto sizeOK = [&](keel::usize n){ return n == 1 || n == nmax; };
    if (!sizeOK(na) || !sizeOK(nb) || !sizeOK(nc))
        throw Error(keel::format("{}: args must be same size or scalar", fnName),
                    0, 0, fnName, "", "numkit:dim");
    const Value &ref = (na == nmax) ? av : (nb == nmax ? bv : cv);
    const auto &d = ref.dims();
    Value out_m = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    Value out_v = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *pm = out_m.doubleDataMut();
    keel::f64 *pv = out_v.doubleDataMut();
    for (keel::usize i = 0; i < nmax; ++i) {
        const keel::f64 a = av.elemAsDouble(na == 1 ? 0 : i);
        const keel::f64 b = bv.elemAsDouble(nb == 1 ? 0 : i);
        const keel::f64 c = cv.elemAsDouble(nc == 1 ? 0 : i);
        auto [m, v] = stat_impl(a, b, c);
        pm[i] = m;
        pv[i] = v;
    }
    outs[0] = std::move(out_m);
    if (nargout > 1) outs[1] = std::move(out_v);
}

// ── *pdf / *cdf / *inv parameter-broadcast helpers ───────────────────
//
// The distribution density/cdf/quantile functions broadcast ALL arguments
// (the data x AND the distribution parameters mu/sigma/a/b/df/…) to a common
// size: a scalar expands; equal non-scalar sizes element-align; mismatched
// non-scalar sizes error like MATLAB ("Non-scalar arguments must match in
// size."). Each function supplies a scalar kernel (its formula, evaluated for
// one element, owning its own per-element domain handling, e.g. sigma<=0 →
// NaN). The helpers below apply that kernel under MATLAB broadcasting.
//
// NOTE (consistent with the *stat helpers above): broadcasting is by element
// COUNT, not strict shape — two same-numel/different-shape non-scalars align
// by linear index rather than erroring. Rare; matches existing behaviour.

// Resolve an optional parameter arg with a scalar default WITHOUT copying a
// present argument: returns a const ref to args[idx] when supplied and
// non-empty, else fills `holder` with the scalar default and refs that.
inline const Value &dist_param(keel::ArrayRef<Value> args, keel::usize idx, keel::f64 defv,
                               keel::Allocator *mr, Value &holder)
{
    if (idx < args.size() && !args[idx].isEmpty()) return args[idx];
    holder = Value::scalar(defv, mr);
    return holder;
}

// Allocate (or, for a 0-element operand, produce an empty) DOUBLE Value with
// the same shape as `x`. Used both for empty→empty and for sizing a
// broadcast result buffer.
inline Value dist_empty_like(const Value &x, keel::Allocator *mr)
{
    const auto &d = x.dims();
    if (d.is3D())
        return Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    return Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
}

// Validate MATLAB broadcasting across a set of element counts: every count
// must be 1 or the common maximum N (0/empty is handled by the caller before
// this). Throws MATLAB-style on a non-scalar size clash. Returns N.
inline keel::usize dist_match_numel(std::initializer_list<keel::usize> sizes,
                                    const char *fnName)
{
    keel::usize N = 0;
    for (keel::usize s : sizes) N = std::max(N, s);
    for (keel::usize s : sizes)
        if (s != 1 && s != N)
            throw Error(keel::format("{}: Non-scalar arguments must match in size.", fnName),
                        0, 0, fnName, "", "numkit:dist:size");
    return N;
}

// One data arg + one parameter: kernel(double x, double p) -> double.
template <class K>
inline Value broadcast_dist2(const Value &av, const Value &bv,
                             keel::Allocator *mr,
                             const char *fnName, K kernel)
{
    const keel::usize na = av.numel(), nb = bv.numel();
    if (na == 0) return dist_empty_like(av, mr);
    if (nb == 0) return dist_empty_like(bv, mr);
    if (na == 1 && nb == 1)
        return Value::scalar(kernel(av.toScalar(), bv.toScalar()), mr);
    if (na > 1 && nb > 1 && na != nb)
        throw Error(keel::format("{}: Non-scalar arguments must match in size.", fnName),
                    0, 0, fnName, "", "numkit:dist:size");
    const Value &ref = (na >= nb) ? av : bv;
    const auto &d = ref.dims();
    Value out = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = ref.numel();
    for (keel::usize i = 0; i < n; ++i)
        od[i] = kernel(av.elemAsDouble(na == 1 ? 0 : i),
                       bv.elemAsDouble(nb == 1 ? 0 : i));
    return out;
}

// One data arg + two parameters: kernel(double x, double p1, double p2).
template <class K>
inline Value broadcast_dist3(const Value &av, const Value &bv, const Value &cv,
                             keel::Allocator *mr,
                             const char *fnName, K kernel)
{
    const keel::usize na = av.numel(), nb = bv.numel(), nc = cv.numel();
    if (na == 0) return dist_empty_like(av, mr);
    if (nb == 0) return dist_empty_like(bv, mr);
    if (nc == 0) return dist_empty_like(cv, mr);
    const keel::usize nmax = std::max({na, nb, nc});
    if (nmax == 1)
        return Value::scalar(kernel(av.toScalar(), bv.toScalar(), cv.toScalar()), mr);
    auto ok = [&](keel::usize k) { return k == 1 || k == nmax; };
    if (!ok(na) || !ok(nb) || !ok(nc))
        throw Error(keel::format("{}: Non-scalar arguments must match in size.", fnName),
                    0, 0, fnName, "", "numkit:dist:size");
    const Value &ref = (na == nmax) ? av : (nb == nmax ? bv : cv);
    const auto &d = ref.dims();
    Value out = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < nmax; ++i)
        od[i] = kernel(av.elemAsDouble(na == 1 ? 0 : i),
                       bv.elemAsDouble(nb == 1 ? 0 : i),
                       cv.elemAsDouble(nc == 1 ? 0 : i));
    return out;
}

// One data arg + three parameters: kernel(double x, double p1, p2, p3).
template <class K>
inline Value broadcast_dist4(const Value &av, const Value &bv, const Value &cv,
                             const Value &dv, keel::Allocator *mr,
                             const char *fnName, K kernel)
{
    const keel::usize na = av.numel(), nb = bv.numel(), nc = cv.numel(), nd = dv.numel();
    if (na == 0) return dist_empty_like(av, mr);
    if (nb == 0) return dist_empty_like(bv, mr);
    if (nc == 0) return dist_empty_like(cv, mr);
    if (nd == 0) return dist_empty_like(dv, mr);
    const keel::usize nmax = std::max({na, nb, nc, nd});
    if (nmax == 1)
        return Value::scalar(
            kernel(av.toScalar(), bv.toScalar(), cv.toScalar(), dv.toScalar()), mr);
    auto ok = [&](keel::usize k) { return k == 1 || k == nmax; };
    if (!ok(na) || !ok(nb) || !ok(nc) || !ok(nd))
        throw Error(keel::format("{}: Non-scalar arguments must match in size.", fnName),
                    0, 0, fnName, "", "numkit:dist:size");
    const Value &ref = (na == nmax) ? av : (nb == nmax ? bv : (nc == nmax ? cv : dv));
    const auto &d = ref.dims();
    Value out = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < nmax; ++i)
        od[i] = kernel(av.elemAsDouble(na == 1 ? 0 : i),
                       bv.elemAsDouble(nb == 1 ? 0 : i),
                       cv.elemAsDouble(nc == 1 ? 0 : i),
                       dv.elemAsDouble(nd == 1 ? 0 : i));
    return out;
}

// ── RNG size-arg parser ──────────────────────────────────────────────
// Parses MATLAB-style trailing size arguments shared by every *rnd
// distribution adapter:
//
//   <fn>(...params)              → scalar (1×1)
//   <fn>(...params, n)           → n × n
//   <fn>(...params, [m n ...])   → m × n × ... (vector size)
//   <fn>(...params, m, n, ...)   → m × n × ... (multi-arg size)
//
// `start` is the index of the first size-argument candidate (0-based).
// On exit, sets rows/cols (with cols = product of dims 2..k so the 2-D
// compute kernel generates the right number of draws), and
// reshape_rng_nd() reshapes the 2-D result to the full N-D shape when
// 3+ dimensions were requested.
inline void parse_rng_size(keel::ArrayRef<Value> args, keel::usize start,
                            keel::usize &rows, keel::usize &cols, const char *fn)
{
    rows = 1;
    cols = 1;
    if (args.size() <= start)
        return;
    const Value &a0 = args[start];
    // Vector size form: numel > 1. Only valid when it is the sole size arg.
    if (a0.numel() > 1) {
        if (args.size() > start + 1)
            throw Error(keel::format("{}: too many input arguments — a size "
                        "vector cannot be followed by additional size inputs.", fn),
                        0, 0, fn, "", keel::format("numkit:{}:maxnargin", fn));
        argcheck::arrayDimReal(a0, fn);
        const keel::usize n = a0.numel();
        rows = argcheck::arrayDim(a0.elemAsDouble(0), fn);
        cols = 1;
        for (keel::usize i = 1; i < n; ++i)
            cols *= argcheck::arrayDim(a0.elemAsDouble(i), fn);
        return;
    }
    if (args.size() == start + 1) {
        if (a0.isEmpty()) {
            rows = 0;
            cols = 0;
        } else {
            rows = argcheck::arrayDim(a0, fn);
            cols = rows;  // n×n
        }
        return;
    }
    // 2 or more scalar (or empty) size args: validate every one and fold
    // dims[1..] into cols for the 2-D kernel.
    for (keel::usize i = start; i < args.size(); ++i) {
        if (args[i].numel() > 1)
            throw Error(keel::format("{}: size inputs must be scalars when "
                        "more than one size argument is supplied.", fn),
                        0, 0, fn, "", keel::format("numkit:{}:sizeNotScalar", fn));
    }
    rows = a0.isEmpty() ? 0 : argcheck::arrayDim(a0, fn);
    cols = 1;
    for (keel::usize i = start + 1; i < args.size(); ++i) {
        const keel::usize d = args[i].isEmpty() ? 0 : argcheck::arrayDim(args[i], fn);
        cols *= d;
    }
}

inline Value reshape_rng_nd(Value v, keel::ArrayRef<Value> args, keel::usize start,
                            keel::Allocator *mr, const char *fn)
{
    if (args.size() <= start)
        return v;
    keel::Array<keel::usize> shape;
    if (args.size() == start + 1 && args[start].numel() > 1) {
        const keel::usize n = args[start].numel();
        shape.reserve(n);
        for (keel::usize i = 0; i < n; ++i)
            shape.push_back(argcheck::arrayDim(args[start].elemAsDouble(i), fn));
    } else if (args.size() >= start + 3) {
        shape.reserve(args.size() - start);
        for (keel::usize i = start; i < args.size(); ++i)
            shape.push_back(args[i].isEmpty() ? 0 : argcheck::arrayDim(args[i], fn));
    } else {
        return v;
    }
    while (shape.size() > 2 && shape.back() == 1)
        shape.pop_back();
    if (shape.size() <= 2)
        return v;
    return builtin::reshape(std::move(v), keel::ArrayRef<keel::usize>(shape.data(), shape.size()), mr);
}

} // namespace numkit::stats::detail
