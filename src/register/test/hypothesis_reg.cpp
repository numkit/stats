// toolboxes/signal/src/test/hypothesis_reg.cpp
//
// CallContext register half of test/hypothesis.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/distributions/binomial.hpp>
#include <numkit/stats/distributions/chi2.hpp>
#include <numkit/stats/distributions/fisher_f.hpp>
#include <numkit/stats/distributions/normal.hpp>
#include <numkit/stats/distributions/students_t.hpp>
#include <numkit/stats/test/hypothesis.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "test/hypothesis_detail.hpp"
#include <numkit/value/argcheck.hpp>


namespace numkit::stats {

namespace detail {

namespace {
[[maybe_unused]] TestTail parse_tail_arg(keel::ArrayRef<Value> args, keel::usize i, TestTail def) {
    if (i >= args.size()) return def;
    const Value &v = args[i];
    if (v.isChar() || v.isString()) return parse_tail(v.toString(), def);
    return def;
}

keel::f64 parse_alpha(keel::ArrayRef<Value> args, keel::usize i, keel::f64 def) {
    if (i >= args.size() || args[i].isEmpty()) return def;
    if (args[i].isChar() || args[i].isString()) return def;
    return args[i].toScalar();
}
}

void ttest_reg(keel::ArrayRef<Value> args, keel::usize nargout,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "ttest");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();

    // Detect paired form: ttest(x, y) where y is a non-scalar numeric
    // array matching x's length. Pre-difference x - y and run vs m=0.
    Value xData = args[0];
    bool pairedConsumed = false;
    if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()
        && !args[1].isEmpty() && args[1].numel() > 1) {
        const Value &y = args[1];
        if (y.numel() != args[0].numel())
            throw Error("ttest: paired vectors must have equal length",
                        0, 0, "ttest", "", "numkit:ttest:pairedLen");
        const keel::usize r = args[0].dims().rows();
        const keel::usize c = args[0].dims().cols();
        Value diff = Value::matrix(r, c, ValueType::DOUBLE, mr);
        keel::f64 *dst = diff.doubleDataMut();
        for (keel::usize i = 0; i < y.numel(); ++i)
            dst[i] = args[0].elemAsDouble(i) - y.elemAsDouble(i);
        xData = keel::move(diff);
        pairedConsumed = true;
    }

    keel::f64 m = 0.0;
    if (!pairedConsumed && args.size() >= 2 && !args[1].isChar()
        && !args[1].isString() && !args[1].isEmpty()) {
        m = args[1].toScalar();
    }
    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    keel::i32 dim = 0;
    keel::usize i = (args.size() >= 2 && (args[1].isChar() || args[1].isString())) ? 1 : 2;
    while (i < args.size()) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) {
            keel::String sl = a.toString();
            sl.to_ascii_lower_in_place();
            if (sl == "alpha" && i + 1 < args.size()) {
                alpha = args[i + 1].toScalar(); i += 2;
            } else if (sl == "tail" && i + 1 < args.size()) {
                keel::String tv = args[i + 1].toString();
                tail = parse_tail(tv.as_ref(), tail); i += 2;
            } else if (sl == "dim") {
                if (i + 1 >= args.size())
                    throw Error("ttest: missing value for 'Dim'",
                                0, 0, "ttest", "", "numkit:ttest:dim");
                dim = argcheck::expectedPositiveInteger(
                    argcheck::expectedReal(args[i + 1], keel::as<keel::i32>(i + 2), "dim", "ttest"),
                    keel::as<keel::i32>(i + 2), "dim", "ttest");
                i += 2;
            } else {
                tail = parse_tail(sl.as_ref(), tail); ++i;
            }
        } else {
            alpha = a.toScalar(); ++i;
        }
    }
    auto [h, p, ci, t] = ttest_dim(xData, m, alpha, tail, dim, mr);
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = keel::move(ci);
    if (nargout > 3) outs[3] = keel::move(t);   // stats struct {tstat, df, sd}
}

void ttest2_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "ttest2");  // no max: name-value tail in the reference
    keel::f64 alpha = parse_alpha(args, 2, 0.05);
    TestTail tail = TestTail::Both;
    // Reference engine default is 'equal' (pooled variance), NOT Welch.
    keel::String vartype = keel::String::from("equal");
    keel::i32 dim = 0;
    for (keel::usize i = 2; i + 1 < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            keel::String k = args[i].toString();
            k.to_ascii_lower_in_place();
            if      (k == "tail")    { keel::String tv = args[i + 1].toString(); tail = parse_tail(tv.as_ref(), TestTail::Both); }
            else if (k == "vartype") vartype = args[i + 1].toString();
            else if (k == "alpha")   alpha = args[i + 1].toScalar();
            else if (k == "dim") {
                dim = argcheck::expectedPositiveInteger(
                    argcheck::expectedReal(args[i + 1], keel::as<keel::i32>(i + 2), "dim", "ttest2"),
                    keel::as<keel::i32>(i + 2), "dim", "ttest2");
            }
        }
    }
    auto [h, p, ci, t] = ttest2_dim(args[0], args[1], alpha, tail, vartype.as_ref(), dim, ctx.engine->allocator());
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = keel::move(ci);
    if (nargout > 3) outs[3] = keel::move(t);
}

void ztest_reg(keel::ArrayRef<Value> args, keel::usize nargout,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "ztest");  // no max: name-value tail in the reference
    const keel::f64 m     = args[1].toScalar();
    const keel::f64 sigma = args[2].toScalar();
    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    keel::i32 dim = 0;
    keel::usize i = 3;
    while (i < args.size()) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) {
            keel::String sl = a.toString();
            sl.to_ascii_lower_in_place();
            if (sl == "alpha" && i + 1 < args.size()) { alpha = args[i + 1].toScalar(); i += 2; }
            else if (sl == "tail" && i + 1 < args.size()) { keel::String tv = args[i + 1].toString(); tail = parse_tail(tv.as_ref(), tail); i += 2; }
            else if (sl == "dim") {
                if (i + 1 >= args.size())
                    throw Error("ztest: missing value for 'Dim'",
                                0, 0, "ztest", "", "numkit:ztest:dim");
                dim = argcheck::expectedPositiveInteger(
                    argcheck::expectedReal(args[i + 1], keel::as<keel::i32>(i + 2), "dim", "ztest"),
                    keel::as<keel::i32>(i + 2), "dim", "ztest");
                i += 2;
            }
            else { tail = parse_tail(sl.as_ref(), tail); ++i; }
        } else { alpha = a.toScalar(); ++i; }
    }
    auto [h, p, ci, z] = ztest_dim(args[0], m, sigma, alpha, tail, dim, ctx.engine->allocator());
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = keel::move(ci);
    if (nargout > 3) outs[3] = keel::move(z);
}

void vartest_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "vartest");  // no max: name-value tail in the reference
    const keel::f64 v = args[1].toScalar();
    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    keel::i32 dim = 0;
    keel::usize i = 2;
    while (i < args.size()) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) {
            keel::String sl = a.toString();
            sl.to_ascii_lower_in_place();
            if (sl == "alpha" && i + 1 < args.size()) { alpha = args[i + 1].toScalar(); i += 2; }
            else if (sl == "tail" && i + 1 < args.size()) { keel::String tv = args[i + 1].toString(); tail = parse_tail(tv.as_ref(), tail); i += 2; }
            else if (sl == "dim") {
                if (i + 1 >= args.size())
                    throw Error("vartest: missing value for 'Dim'",
                                0, 0, "vartest", "", "numkit:vartest:dim");
                dim = argcheck::expectedPositiveInteger(
                    argcheck::expectedReal(args[i + 1], keel::as<keel::i32>(i + 2), "dim", "vartest"),
                    keel::as<keel::i32>(i + 2), "dim", "vartest");
                i += 2;
            }
            else { tail = parse_tail(sl.as_ref(), tail); ++i; }
        } else { alpha = a.toScalar(); ++i; }
    }
    auto [h, p, ci, T] = vartest_dim(args[0], v, alpha, tail, dim, ctx.engine->allocator());
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = keel::move(ci);
    if (nargout > 3) outs[3] = keel::move(T);
}

void vartest2_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "vartest2");  // no max: name-value tail in the reference
    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    keel::i32 dim = 0;
    keel::usize i = 2;
    while (i < args.size()) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) {
            keel::String sl = a.toString();
            sl.to_ascii_lower_in_place();
            if (sl == "alpha" && i + 1 < args.size()) { alpha = args[i + 1].toScalar(); i += 2; }
            else if (sl == "tail" && i + 1 < args.size()) { keel::String tv = args[i + 1].toString(); tail = parse_tail(tv.as_ref(), tail); i += 2; }
            else if (sl == "dim") {
                if (i + 1 >= args.size())
                    throw Error("vartest2: missing value for 'Dim'",
                                0, 0, "vartest2", "", "numkit:vartest2:dim");
                dim = argcheck::expectedPositiveInteger(
                    argcheck::expectedReal(args[i + 1], keel::as<keel::i32>(i + 2), "dim", "vartest2"),
                    keel::as<keel::i32>(i + 2), "dim", "vartest2");
                i += 2;
            }
            else { tail = parse_tail(sl.as_ref(), tail); ++i; }
        } else { alpha = a.toScalar(); ++i; }
    }
    auto [h, p, ci, F] = vartest2_dim(args[0], args[1], alpha, tail, dim, ctx.engine->allocator());
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = keel::move(ci);
    if (nargout > 3) outs[3] = keel::move(F);
}

void kstest_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "kstest");  // no max: name-value tail in the reference
    Value cdf = (args.size() >= 2 && !(args[1].isChar() || args[1].isString()))
                  ? args[1] : Value();  // empty default
    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    // Walk trailing args: positional alpha (numeric scalar), positional
    // tail string, and Name-Value pairs ('Alpha', value | 'Tail', value).
    keel::usize i = (cdf.numel() > 0 || (args.size() >= 2 && (args[1].isChar() || args[1].isString()))) ? 2 : 1;
    while (i < args.size()) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) {
            keel::String sl = a.toString();
            sl.to_ascii_lower_in_place();
            if (sl == "alpha" && i + 1 < args.size()) {
                alpha = args[i + 1].toScalar();
                i += 2;
            } else if (sl == "tail" && i + 1 < args.size()) {
                keel::String tv = args[i + 1].toString();
                tail = parse_tail(tv.as_ref(), tail);
                i += 2;
            } else {
                // positional tail string
                tail = parse_tail(sl.as_ref(), tail);
                ++i;
            }
        } else {
            // positional alpha
            alpha = a.toScalar();
            ++i;
        }
    }
    auto [h, p, D, cv] = kstest(args[0], cdf, alpha, tail, ctx.engine->allocator());
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = keel::move(D);
    if (nargout > 3) outs[3] = keel::move(cv);
}

void kstest2_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "kstest2");  // no max: name-value tail in the reference
    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    keel::usize i = 2;
    while (i < args.size()) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) {
            keel::String sl = a.toString();
            sl.to_ascii_lower_in_place();
            if (sl == "alpha" && i + 1 < args.size()) {
                alpha = args[i + 1].toScalar(); i += 2;
            } else if (sl == "tail" && i + 1 < args.size()) {
                keel::String tv = args[i + 1].toString();
                tail = parse_tail(tv.as_ref(), tail); i += 2;
            } else {
                tail = parse_tail(sl.as_ref(), tail); ++i;
            }
        } else {
            alpha = a.toScalar(); ++i;
        }
    }
    auto [h, p, D, cv] = kstest2(args[0], args[1], alpha, tail, ctx.engine->allocator());
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = keel::move(D);
    if (nargout > 3) outs[3] = keel::move(cv);
}

void jbtest_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "jbtest");
    keel::f64 alpha = parse_alpha(args, 1, 0.05);
    // 3rd arg = mctol (Monte-Carlo standard-error tolerance).
    const keel::f64 mctol = (args.size() > 2 && !args[2].isEmpty())
                         ? args[2].toScalar()
                         : keel::Limits<keel::f64>::quiet_nan();
    auto [h, p, JB, cv] = jbtest(args[0], alpha, mctol, ctx.engine->allocator());
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = keel::move(JB);
    if (nargout > 3) outs[3] = keel::move(cv);
}

void fishertest_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "fishertest");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    for (keel::usize i = 1; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) break;
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        const Value &v = args[i + 1];
        if      (name == "alpha") alpha = v.toScalar();
        else if (name == "tail")  { keel::String tv = v.toString(); tail = parse_tail(tv.as_ref(), TestTail::Both); }
    }
    auto [h, p, OR, lo, hi] = fishertest(args[0], alpha, tail, mr);
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) {
        Value s = Value::structure(mr);
        s.field("OddsRatio") = OR;
        Value ci = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        keel::f64 *cd = ci.doubleDataMut();
        cd[0] = lo.toScalar();
        cd[1] = hi.toScalar();
        s.field("ConfidenceInterval") = keel::move(ci);
        outs[2] = keel::move(s);
    }
}

void chi2gof_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "chi2gof");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        s.to_ascii_lower_in_place();
        return s;
    };

    Value freq, expected, edges_arg, ctrs_arg;
    keel::i32 nbins = 10;
    keel::i32 nparams = -1;       // -1 = use default (2 if auto-fit, 0 if explicit O/E)
    keel::f64 alpha = 0.05;
    keel::f64 emin = 5.0;

    bool freq_set = false, expected_set = false, edges_set = false;
    bool nbins_set = false, ctrs_set = false, nparams_set = false;
    bool cdf_supplied = false;

    for (keel::usize i = 1; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) break;
        const keel::String rawName = args[i].toString();
        const keel::String name = lower(rawName.as_ref());
        const Value &v = args[i + 1];
        if      (name == "frequency") { freq = v;     freq_set = true; }
        else if (name == "expected")  { expected = v; expected_set = true; }
        else if (name == "edges")     { edges_arg = v; edges_set = true; }
        else if (name == "nbins")     { nbins = (keel::i32)v.toScalar(); nbins_set = true; }
        else if (name == "ctrs")      { ctrs_arg = v; ctrs_set = true; }
        else if (name == "nparams")   { nparams = (keel::i32)v.toScalar(); nparams_set = true; }
        else if (name == "emin")      { emin = v.toScalar(); }
        else if (name == "alpha")     { alpha = v.toScalar(); }
        else if (name == "cdf")       { cdf_supplied = true; }
    }

    if (cdf_supplied)
        throw Error("chi2gof: 'CDF' function-handle argument is not yet "
                    "supported in nk; supply 'Expected' or rely on "
                    "the default normal auto-fit instead",
                    0, 0, "chi2gof", "", "numkit:chi2gof:cdf_nyi");

    // Path A: explicit Frequency + Expected (existing behavior).
    if (freq_set && expected_set) {
        const keel::i32 np = nparams_set ? nparams : 0;
        auto [p, h, chi2, df] = chi2gof(freq, expected, np, alpha, mr);
        outs[0] = keel::move(h);
        if (nargout > 1) outs[1] = keel::move(p);
        if (nargout > 2) {
            const keel::usize K = freq.numel();
            Value s = Value::structure(mr);
            s.field("chi2stat") = chi2;
            s.field("df")       = df;
            // Synthesize edges from the first arg if it's monotone numeric;
            // else default to 1:K with width 1.
            Value edges_out = Value::matrix(1, K + 1, ValueType::DOUBLE, mr);
            keel::f64 *ep = edges_out.doubleDataMut();
            const Value &xv = args[0];
            const keel::f64 x0 = xv.elemAsDouble(0);
            const keel::f64 xK = xv.elemAsDouble(K - 1);
            const keel::f64 dx = (K > 1) ? (xK - x0) / keel::f64(K - 1) : 1.0;
            for (keel::usize i = 0; i <= K; ++i) ep[i] = x0 + (keel::f64(i) - 0.5) * dx;
            s.field("edges") = edges_out;
            s.field("O")     = freq;
            s.field("E")     = expected;
            outs[2] = keel::move(s);
        }
        return;
    }

    // Path B: auto-binning. Need data vector.
    const Value &x = args[0];
    const keel::usize N = x.numel();
    if (N < 2)
        throw Error("chi2gof: data vector must have at least 2 elements",
                    0, 0, "chi2gof", "", "numkit:chi2gof:size");

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Compute mean / std of x.
    keel::f64 mean = 0.0;
    for (keel::usize i = 0; i < N; ++i) mean += x.elemAsDouble(i);
    mean /= keel::f64(N);
    keel::f64 sq = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 d = x.elemAsDouble(i) - mean;
        sq += d * d;
    }
    const keel::f64 sd = keel::sqrt(sq / keel::f64(N - 1));

    // Build edges.
    keel::Array<keel::f64> edges(scratch.allocator());
    if (edges_set) {
        const keel::usize M = edges_arg.numel();
        edges.resize(M);
        for (keel::usize i = 0; i < M; ++i) edges[i] = edges_arg.elemAsDouble(i);
    } else if (ctrs_set) {
        // Centres → derive edges as midpoints + extrapolation.
        const keel::usize M = ctrs_arg.numel();
        edges.resize(M + 1);
        const keel::f64 c0 = ctrs_arg.elemAsDouble(0);
        const keel::f64 c1 = ctrs_arg.elemAsDouble(1);
        edges[0] = c0 - 0.5 * (c1 - c0);
        for (keel::usize i = 0; i + 1 < M; ++i) {
            edges[i + 1] = 0.5 * (ctrs_arg.elemAsDouble(i)
                                  + ctrs_arg.elemAsDouble(i + 1));
        }
        const keel::f64 cN1 = ctrs_arg.elemAsDouble(M - 1);
        const keel::f64 cN2 = ctrs_arg.elemAsDouble(M - 2);
        edges[M] = cN1 + 0.5 * (cN1 - cN2);
    } else {
        // NBins equally-spaced from min(x) to max(x).
        keel::f64 xmin = x.elemAsDouble(0), xmax = xmin;
        for (keel::usize i = 1; i < N; ++i) {
            const keel::f64 v = x.elemAsDouble(i);
            if (v < xmin) xmin = v;
            if (v > xmax) xmax = v;
        }
        const keel::i32 K = keel::max(2, nbins);
        edges.resize(K + 1);
        for (keel::i32 i = 0; i <= K; ++i)
            edges[i] = xmin + (xmax - xmin) * keel::f64(i) / keel::f64(K);
    }

    // Build O = histogram of x against edges. MATLAB chi2gof's binning
    // rule depends on whether edges came from the user (left-closed,
    // last bin right-inclusive — standard histcounts) or auto-binning
    // (right-closed first bin extended, right-inclusive everywhere).
    // Verified vs R2025b on (-3:0.05:3) data with both Edges= and NBins=.
    const keel::usize K0 = edges.size() - 1;
    keel::Array<keel::f64> O(K0, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = x.elemAsDouble(i);
        if (v < edges[0] || v > edges[K0]) continue;
        for (keel::usize b = 0; b < K0; ++b) {
            bool in;
            if (edges_set) {
                // Left-closed standard histcounts.
                in = (b == K0 - 1)
                    ? (v >= edges[b] && v <= edges[b + 1])
                    : (v >= edges[b] && v <  edges[b + 1]);
            } else {
                // Right-closed (auto-binning).
                in = (b == 0)
                    ? (v <= edges[1])
                    : (v > edges[b] && v <= edges[b + 1]);
            }
            if (in) { O[b] += 1.0; break; }
        }
    }

    // Build E under N(mean, sd) by default (via norm CDF).
    keel::Array<keel::f64> E(K0, 0.0, scratch.allocator());
    auto Phi = [](keel::f64 z) {
        return 0.5 * (1.0 + keel::erf(z / keel::sqrt(2.0)));
    };
    // Total mass under tail-extended bins so that Σ E = N (matches MATLAB).
    // The first bin extends to -∞, the last to +∞ (chi2gof convention).
    for (keel::usize b = 0; b < K0; ++b) {
        const keel::f64 z_lo = (b == 0)        ? -keel::Limits<keel::f64>::infinity()
                                            : (edges[b]     - mean) / sd;
        const keel::f64 z_hi = (b == K0 - 1)   ?  keel::Limits<keel::f64>::infinity()
                                            : (edges[b + 1] - mean) / sd;
        const keel::f64 F_lo = keel::is_finite(z_lo) ? Phi(z_lo) : 0.0;
        const keel::f64 F_hi = keel::is_finite(z_hi) ? Phi(z_hi) : 1.0;
        E[b] = keel::f64(N) * (F_hi - F_lo);
    }

    // Apply EMin: merge tail bins with E < emin (working from each end
    // inward, merging the small bin into its inward neighbour). MATLAB
    // also merges contiguous low-E interior runs, but tail-only is the
    // common case; this matches the MATLAB reference output.
    auto merge_left = [&]() {
        while (O.size() > 1 && E[0] < emin) {
            O[1] += O[0]; E[1] += E[0];
            O.erase(0); E.erase(0);
            edges.erase(1);  // remove inner edge
        }
    };
    auto merge_right = [&]() {
        while (O.size() > 1 && E.back() < emin) {
            O[O.size() - 2] += O.back(); E[E.size() - 2] += E.back();
            O.pop_back(); E.pop_back();
            edges.erase(edges.size() - 2);  // remove inner edge
        }
    };
    merge_left();
    merge_right();

    // Compute chi2.
    keel::f64 chi2 = 0.0;
    for (keel::usize b = 0; b < O.size(); ++b) {
        if (E[b] > 0.0) {
            const keel::f64 d = O[b] - E[b];
            chi2 += d * d / E[b];
        }
    }
    const keel::i32 K_final = (keel::i32)O.size();
    // Default NParams = 2 (mean + std estimated from data) unless user
    // overrode or supplied explicit Edges (MATLAB still defaults to 2).
    const keel::i32 np = nparams_set ? nparams : 2;
    const keel::f64 df = keel::f64(K_final) - 1.0 - keel::f64(np);
    Value chi2v = Value::scalar(chi2, mr);
    const keel::f64 cdf = (df > 0.0) ? chi2cdf(chi2v, df, mr).toScalar() : 1.0;
    const keel::f64 p = keel::max(0.0, 1.0 - cdf);
    const keel::i32 h = (p < alpha) ? 1 : 0;

    outs[0] = Value::scalar(keel::f64(h), mr);
    if (nargout > 1) outs[1] = Value::scalar(p, mr);
    if (nargout > 2) {
        Value s = Value::structure(mr);
        s.field("chi2stat") = Value::scalar(chi2, mr);
        s.field("df")       = Value::scalar(df, mr);
        // Pack edges / O / E into Value rows.
        Value edges_out = Value::matrix(1, edges.size(), ValueType::DOUBLE, mr);
        keel::copy(edges.begin(), edges.end(), edges_out.doubleDataMut());
        s.field("edges") = edges_out;
        Value O_out = Value::matrix(1, O.size(), ValueType::DOUBLE, mr);
        keel::copy(O.begin(), O.end(), O_out.doubleDataMut());
        s.field("O") = O_out;
        Value E_out = Value::matrix(1, E.size(), ValueType::DOUBLE, mr);
        keel::copy(E.begin(), E.end(), E_out.doubleDataMut());
        s.field("E") = E_out;
        outs[2] = keel::move(s);
    }
    (void)nbins_set;  // documented arg, no separate code path needed
    (void)ctrs_set;
    (void)expected_set;
}

void vartestn_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "vartestn");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    auto lower = [](keel::StringRef sv) {
        keel::String s = keel::String::from(sv);
        s.to_ascii_lower_in_place();
        return s;
    };
    // Parse: vartestn(X[, GROUP], N-V...). The 2nd arg is GROUP iff it's
    // a non-string vector. If it's a string, no group given (matrix
    // input form: each column = group).
    keel::i32 test = 0;  // Bartlett default
    keel::usize nv_start = 1;
    Value X = args[0];
    Value G;
    bool have_group = false;
    if (args.size() >= 2 && !(args[1].isChar() || args[1].isString())) {
        G = args[1];
        have_group = true;
        nv_start = 2;
    }
    for (keel::usize i = nv_start; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) break;
        const keel::String rawName = args[i].toString();
        const keel::String name = lower(rawName.as_ref());
        if (name == "testtype") {
            const keel::String rawV = args[i + 1].toString();
            const keel::String v = lower(rawV.as_ref());
            if      (v == "bartlett")        test = 0;
            else if (v == "levenequadratic") test = 1;
            else if (v == "leveneabsolute")  test = 2;
            else if (v == "brownforsythe")   test = 3;
            else if (v == "obrien")          test = 4;
            else throw Error(keel::format("vartestn: unknown TestType '{}'", v),
                             0, 0, "vartestn", "", "numkit:vartestn:badtype");
        }
        // 'display' / 'alpha' silently ignored (Display has no console
        // effect; alpha doesn't change p/stat output).
    }

    // Matrix-input form: build (values, group) where group encodes the
    // column index of each observation.
    if (!have_group) {
        const keel::usize R = X.dims().rows();
        const keel::usize C = X.dims().cols();
        if (R == 0 || C < 2)
            throw Error("vartestn: matrix input must have >=2 columns",
                        0, 0, "vartestn", "", "numkit:vartestn:size");
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::f64> vv(R * C, scratch.allocator());
        keel::Array<keel::f64> gg(R * C, scratch.allocator());
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize r = 0; r < R; ++r) {
                vv[c * R + r] = X.elemAsDouble(c * R + r);
                gg[c * R + r] = keel::f64(c + 1);
            }
        Value Vx = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        Value Vg = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        keel::copy(vv.begin(), vv.end(), Vx.doubleDataMut());
        keel::copy(gg.begin(), gg.end(), Vg.doubleDataMut());
        X = keel::move(Vx);
        G = keel::move(Vg);
    }

    auto [p, stat, df1, df2] = vartestn_full(X, G, test, mr);
    outs[0] = keel::move(p);
    if (nargout > 1) {
        Value s = Value::structure(mr);
        if (test == 0) {
            s.field("chisqstat") = stat;
            s.field("df")        = df1;
        } else {
            s.field("fstat") = stat;
            // df is a 1×2 row vector [df1 df2].
            Value dfv = Value::matrix(1, 2, ValueType::DOUBLE, mr);
            keel::f64 *dp = dfv.doubleDataMut();
            dp[0] = df1.toScalar();
            dp[1] = df2.toScalar();
            s.field("df") = dfv;
        }
        outs[1] = keel::move(s);
    }
}

void runstest_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "runstest");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();

    // arg[1] is positional v (scalar), 'ud' string for up-down test, or a
    // name-value start.
    keel::f64 v = keel::Limits<keel::f64>::quiet_nan();   // sentinel: use median(x)
    bool up_down = false;
    keel::usize i = 1;
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        keel::String sl = args[i].toString();
        sl.to_ascii_lower_in_place();
        if (sl == "ud") { up_down = true; ++i; }
        // otherwise leave for the Name-Value loop below.
    } else if (i < args.size() && !args[i].isEmpty()) {
        v = args[i].toScalar();
        ++i;
    }

    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    keel::String method;

    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString()) break;
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        const Value &val = args[i + 1];
        if      (name == "alpha")  alpha = val.toScalar();
        else if (name == "tail")   { keel::String tv = val.toString(); tail = parse_tail(tv.as_ref(), TestTail::Both); }
        else if (name == "method") method = val.toString();
        i += 2;
    }

    // For 'ud' (up-down test): replace x with sign(diff(x)) — the
    // resulting binary sequence above/below 0 counts ascent/descent
    // runs, which is exactly what MATLAB's runstest('ud') does.
    Value xUsed = args[0];
    if (up_down) {
        const Value &x = args[0];
        const keel::usize Nx = x.numel();
        if (Nx < 2) {
            Value diffSign = Value::matrix(0, 0, ValueType::DOUBLE, mr);
            xUsed = keel::move(diffSign);
        } else {
            keel::Array<keel::f64> diffs;
            diffs.reserve(Nx - 1);
            for (keel::usize k = 1; k < Nx; ++k) {
                const keel::f64 xa = x.elemAsDouble(k - 1);
                const keel::f64 xb = x.elemAsDouble(k);
                if (keel::is_nan(xa) || keel::is_nan(xb)) continue;
                diffs.push_back((xb > xa) ? 1.0 : ((xb < xa) ? -1.0 : 0.0));
            }
            Value sgn = Value::matrix(1, diffs.size(), ValueType::DOUBLE, mr);
            if (!diffs.empty())
                keel::copy(diffs.begin(), diffs.end(), sgn.doubleDataMut());
            xUsed = keel::move(sgn);
        }
        v = 0.0;       // reference value for the sign sequence
    }

    auto [p, h, R, n1, n0, z] = runstest(xUsed, v, alpha, tail, method.as_ref(), mr);
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) {
        Value s = Value::structure(mr);
        s.field("nruns") = R;
        s.field("n1")    = n1;
        s.field("n0")    = n0;
        if (!keel::is_nan(z.toScalar())) s.field("z") = z;
        outs[2] = keel::move(s);
    }
}

void ranksum_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "ranksum");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();

    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    keel::String method;

    keel::usize i = 2;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()) {
        alpha = args[i].toScalar();
        ++i;
    }
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString()) break;
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        const Value &v = args[i + 1];
        if      (name == "alpha")  alpha = v.toScalar();
        else if (name == "tail")   { keel::String tv = v.toString(); tail = parse_tail(tv.as_ref(), TestTail::Both); }
        else if (name == "method") method = v.toString();
        i += 2;
    }

    auto [p, h, rs, z] = ranksum(args[0], args[1], alpha, tail, method.as_ref(), mr);
    outs[0] = keel::move(p);
    if (nargout > 1) outs[1] = keel::move(h);
    if (nargout > 2) {
        Value s = Value::structure(mr);
        if (!keel::is_nan(z.toScalar())) s.field("zval") = z;
        s.field("ranksum") = rs;
        outs[2] = keel::move(s);
    }
}

void signrank_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "signrank");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();

    Value y_or_m = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    keel::usize i = 1;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()) {
        y_or_m = args[i];
        ++i;
    }

    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;
    keel::String method;

    if (i < args.size() && !args[i].isChar() && !args[i].isString()) {
        alpha = args[i].toScalar();
        ++i;
    }
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString()) break;
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        const Value &v = args[i + 1];
        if      (name == "alpha")  alpha = v.toScalar();
        else if (name == "tail")   { keel::String tv = v.toString(); tail = parse_tail(tv.as_ref(), TestTail::Both); }
        else if (name == "method") method = v.toString();
        i += 2;
    }

    auto [p, h, sr, z] = signrank(args[0], y_or_m, alpha, tail, method.as_ref(), mr);
    outs[0] = keel::move(p);
    if (nargout > 1) outs[1] = keel::move(h);
    if (nargout > 2) {
        Value s = Value::structure(mr);
        s.field("signedrank") = sr;
        // zval only present for approximate method (NaN otherwise).
        if (!keel::is_nan(z.toScalar())) s.field("zval") = z;
        outs[2] = keel::move(s);
    }
}

void signtest_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "signtest");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();

    // arg[1] may be: missing, scalar median, or paired y vector. Skip it
    // if it's a string (start of name-value list).
    Value y_or_m = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    keel::usize i = 1;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()) {
        y_or_m = args[i];
        ++i;
    }

    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;

    // Optional positional alpha next (legacy 3-arg form), then name-value.
    if (i < args.size() && !args[i].isChar() && !args[i].isString()) {
        alpha = args[i].toScalar();
        ++i;
    }
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString()) break;
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        const Value &v = args[i + 1];
        if      (name == "alpha")  alpha = v.toScalar();
        else if (name == "tail")   { keel::String tv = v.toString(); tail = parse_tail(tv.as_ref(), TestTail::Both); }
        else if (name == "method") { /* exact / approximate — both rely on binocdf */ }
        i += 2;
    }

    auto [p, h, sig] = signtest(args[0], y_or_m, alpha, tail, mr);
    outs[0] = keel::move(p);
    if (nargout > 1) outs[1] = keel::move(h);
    if (nargout > 2) {
        Value s = Value::structure(mr);
        // MATLAB R2025b stats struct shape: {zval, sign}. zval is NaN
        // for the exact (binomial) path — currently always taken;
        // 'approximate' would populate zval with the normal-approx z.
        s.field("zval") = Value::scalar(keel::Limits<keel::f64>::quiet_nan(), mr);
        s.field("sign") = sig;
        outs[2] = keel::move(s);
    }
}

// ── lillietest (Lilliefors normality test) ─────────────────────────
// Tests H0: x ~ N(mu, sigma^2) for unspecified mu, sigma. Uses KS
// statistic against a fitted normal CDF (mean and std estimated
// from sample). p-value via Stephens (1974) approximation.

void lillietest_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "lillietest");  // no max: name-value tail in the reference
    keel::f64 alpha = 0.05;
    if (args.size() >= 2 && !args[1].isEmpty()) alpha = args[1].toScalar();
    auto [h, p, kstat, critval] =
        lillietest(args[0], alpha, ctx.engine->allocator());
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = keel::move(kstat);
    if (nargout > 3) outs[3] = keel::move(critval);
}

void ansaribradley_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "ansaribradley");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();

    keel::f64 alpha = 0.05;
    TestTail tail = TestTail::Both;

    keel::usize i = 2;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()) {
        alpha = args[i].toScalar();
        ++i;
    }
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString()) break;
        keel::String name = args[i].toString();
        name.to_ascii_lower_in_place();
        const Value &v = args[i + 1];
        if      (name == "alpha") alpha = v.toScalar();
        else if (name == "tail")  { keel::String tv = v.toString(); tail = parse_tail(tv.as_ref(), TestTail::Both); }
        i += 2;
    }

    auto [h, p, W, Wstar] = ansaribradley(args[0], args[1], alpha, tail, mr);
    outs[0] = keel::move(h);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) {
        Value s = Value::structure(mr);
        s.field("W")     = W;
        s.field("Wstar") = Wstar;
        outs[2] = keel::move(s);
    }
}

} // namespace detail

} // namespace numkit::stats
