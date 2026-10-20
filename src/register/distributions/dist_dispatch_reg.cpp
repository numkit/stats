// toolboxes/stats/src/distributions/dist_dispatch_reg.cpp
//
// Generic distribution dispatchers: cdf / pdf / icdf / random. Each takes a
// distribution NAME as the first argument and forwards the remaining arguments
// to the matching per-family builtin (normcdf / poisspdf / norminv / normrnd …)
// — these already exist and are parity-validated, so this is pure register-level
// glue (a name → family-prefix table + a findExternal forward). MATLAB:
//   cdf('Normal', x, mu, sigma)  ==  normcdf(x, mu, sigma)
//   icdf(...)  -> <fam>inv ,  random(...) -> <fam>rnd .
// bugs/stats/distribution-dispatchers.
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include "distributions/dist_helpers.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace numkit::stats {

namespace detail {

namespace {

// MATLAB distribution name (lower-cased) → nk family prefix. Covers every
// family that ships <prefix>{cdf,pdf,inv,rnd}; canonical names + common aliases.
const std::unordered_map<std::string, std::string> &distPrefixMap()
{
    static const std::unordered_map<std::string, std::string> m = {
        {"normal", "norm"}, {"norm", "norm"}, {"gaussian", "norm"},
        {"poisson", "poiss"}, {"poiss", "poiss"},
        {"binomial", "bino"}, {"bino", "bino"},
        {"exponential", "exp"}, {"exp", "exp"},
        {"gamma", "gam"}, {"gam", "gam"},
        {"beta", "beta"},
        {"uniform", "unif"}, {"unif", "unif"},
        {"discrete uniform", "unid"}, {"unid", "unid"},
        {"geometric", "geo"}, {"geo", "geo"},
        {"negative binomial", "nbin"}, {"nbin", "nbin"},
        {"hypergeometric", "hyge"}, {"hyge", "hyge"},
        {"weibull", "wbl"}, {"wbl", "wbl"},
        {"lognormal", "logn"}, {"logn", "logn"},
        {"extreme value", "ev"}, {"ev", "ev"},
        {"generalized extreme value", "gev"}, {"gev", "gev"},
        {"generalized pareto", "gp"}, {"gp", "gp"},
        {"rayleigh", "rayl"}, {"rayl", "rayl"},
        {"nakagami", "naka"}, {"naka", "naka"},
        {"rician", "rice"}, {"rice", "rice"},
        {"chisquare", "chi2"}, {"chi-square", "chi2"}, {"chi2", "chi2"},
        {"t", "t"}, {"students t", "t"},
        {"f", "f"},
        {"noncentral chi-square", "ncx2"}, {"ncx2", "ncx2"},
    };
    return m;
}

std::string toLowerStr(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool isDistStruct(const Value &v)
{
    return v.isStruct() && v.hasField("DistributionName") && v.hasField("__nparams");
}

Value evalDistStructBase(const Value &pd, const char *suffix, const Value &xVal,
                         CallContext &ctx)
{
    auto *mr = ctx.engine->resource();
    const std::string dist = toLowerStr(pd.field("DistributionName").toString());
    const int nparams = static_cast<int>(pd.field("__nparams").toScalar());
    const double p1 = (nparams >= 1) ? pd.field("p1").toScalar() : 0.0;
    const double p2 = (nparams >= 2) ? pd.field("p2").toScalar() : 1.0;
    const double p3 = (nparams >= 3) ? pd.field("p3").toScalar() : 1.0;

    if (dist == "tlocationscale") {
        // mu = p1, sigma = p2, nu = p3
        const std::string sfx(suffix);
        if (sfx == "pdf") {
            const ExternalFunc *fn = ctx.engine->findExternal("tpdf", ctx.env);
            if (!fn) throw Error("pdf: 'tpdf' is not available", 0, 0, "pdf", "", "numkit:pdf:missing");
            Value z = xVal.isEmpty() ? xVal : Value::matrix(xVal.dims().rows(), xVal.dims().cols(), ValueType::DOUBLE, mr);
            const size_t n = xVal.numel();
            for (size_t i = 0; i < n; ++i)
                z.doubleDataMut()[i] = (xVal.elemAsDouble(i) - p1) / p2;
            Value callArgs[2] = {z, Value::scalar(p3, mr)};
            Value out;
            (*fn)(Span<const Value>(callArgs, 2), 1, Span<Value>(&out, 1), ctx);
            for (size_t i = 0; i < out.numel(); ++i)
                out.doubleDataMut()[i] /= p2;
            return out;
        }
        if (sfx == "cdf") {
            const ExternalFunc *fn = ctx.engine->findExternal("tcdf", ctx.env);
            if (!fn) throw Error("cdf: 'tcdf' is not available", 0, 0, "cdf", "", "numkit:cdf:missing");
            Value z = xVal.isEmpty() ? xVal : Value::matrix(xVal.dims().rows(), xVal.dims().cols(), ValueType::DOUBLE, mr);
            const size_t n = xVal.numel();
            for (size_t i = 0; i < n; ++i)
                z.doubleDataMut()[i] = (xVal.elemAsDouble(i) - p1) / p2;
            Value callArgs[2] = {z, Value::scalar(p3, mr)};
            Value out;
            (*fn)(Span<const Value>(callArgs, 2), 1, Span<Value>(&out, 1), ctx);
            return out;
        }
        if (sfx == "inv") {
            const ExternalFunc *fn = ctx.engine->findExternal("tinv", ctx.env);
            if (!fn) throw Error("icdf: 'tinv' is not available", 0, 0, "icdf", "", "numkit:icdf:missing");
            Value callArgs[2] = {xVal, Value::scalar(p3, mr)};
            Value out;
            (*fn)(Span<const Value>(callArgs, 2), 1, Span<Value>(&out, 1), ctx);
            for (size_t i = 0; i < out.numel(); ++i)
                out.doubleDataMut()[i] = p1 + p2 * out.doubleDataMut()[i];
            return out;
        }
    }

    const auto &m = distPrefixMap();
    auto it = m.find(dist);
    if (it == m.end())
        throw Error(std::string(suffix) + ": unsupported distribution '" + dist + "'",
                    0, 0, suffix, "", "numkit:dist:unsupported");
    const std::string target = it->second + suffix;
    const ExternalFunc *fn = ctx.engine->findExternal(target, ctx.env);
    if (!fn)
        throw Error(std::string(suffix) + ": '" + target + "' is not available",
                    0, 0, suffix, "", "numkit:dist:missing");
    std::vector<Value> callArgs;
    callArgs.push_back(xVal);
    if (nparams >= 1) callArgs.push_back(Value::scalar(p1, mr));
    if (nparams >= 2) callArgs.push_back(Value::scalar(p2, mr));
    if (nparams >= 3) callArgs.push_back(Value::scalar(p3, mr));
    Value out;
    (*fn)(Span<const Value>(callArgs.data(), callArgs.size()), 1, Span<Value>(&out, 1), ctx);
    return out;
}

void dispatchDistStruct(const char *fnName, const char *suffix,
                        Span<const Value> args, size_t nargout, Span<Value> outs,
                        CallContext &ctx)
{
    auto *mr = ctx.engine->resource();
    const Value &pd = args[0];
    const bool isTrunc = pd.hasField("IsTruncated") && pd.field("IsTruncated").toBool();
    const double tLo = pd.hasField("TruncLower") ? pd.field("TruncLower").toScalar()
                                                 : -std::numeric_limits<double>::infinity();
    const double tHi = pd.hasField("TruncUpper") ? pd.field("TruncUpper").toScalar()
                                                 : std::numeric_limits<double>::infinity();
    const std::string sfx(suffix);

    if (sfx == "rnd") {
        if (!isTrunc) {
            const std::string dist = toLowerStr(pd.field("DistributionName").toString());
            const int nparams = static_cast<int>(pd.field("__nparams").toScalar());
            const double p1 = (nparams >= 1) ? pd.field("p1").toScalar() : 0.0;
            const double p2 = (nparams >= 2) ? pd.field("p2").toScalar() : 1.0;
            const double p3 = (nparams >= 3) ? pd.field("p3").toScalar() : 1.0;
            if (dist == "tlocationscale") {
                const ExternalFunc *fn = ctx.engine->findExternal("trnd", ctx.env);
                if (!fn) throw Error("random: 'trnd' is not available", 0, 0, "random", "", "numkit:random:missing");
                std::vector<Value> callArgs;
                callArgs.push_back(Value::scalar(p3, mr));
                for (size_t i = 1; i < args.size(); ++i) callArgs.push_back(args[i]);
                Value out;
                (*fn)(Span<const Value>(callArgs.data(), callArgs.size()), 1, Span<Value>(&out, 1), ctx);
                for (size_t i = 0; i < out.numel(); ++i)
                    out.doubleDataMut()[i] = p1 + p2 * out.doubleDataMut()[i];
                outs[0] = std::move(out);
                return;
            }
            const auto &m = distPrefixMap();
            auto it = m.find(dist);
            if (it == m.end())
                throw Error("random: unsupported distribution '" + dist + "'",
                            0, 0, "random", "", "numkit:random:dist");
            const ExternalFunc *fn = ctx.engine->findExternal(it->second + "rnd", ctx.env);
            if (!fn)
                throw Error("random: '" + it->second + "rnd' is not available",
                            0, 0, "random", "", "numkit:random:missing");
            std::vector<Value> callArgs;
            if (nparams >= 1) callArgs.push_back(Value::scalar(p1, mr));
            if (nparams >= 2) callArgs.push_back(Value::scalar(p2, mr));
            if (nparams >= 3) callArgs.push_back(Value::scalar(p3, mr));
            for (size_t i = 1; i < args.size(); ++i) callArgs.push_back(args[i]);
            (*fn)(Span<const Value>(callArgs.data(), callArgs.size()), nargout, outs, ctx);
            return;
        }
        // Truncated random: inverse-CDF transform icdf_base(pl + (pu - pl) * rand(sz))
        const double pl = std::isinf(tLo) && tLo < 0.0
            ? 0.0 : evalDistStructBase(pd, "cdf", Value::scalar(tLo, mr), ctx).toScalar();
        const double pu = std::isinf(tHi) && tHi > 0.0
            ? 1.0 : evalDistStructBase(pd, "cdf", Value::scalar(tHi, mr), ctx).toScalar();
        const double Z = std::max(pu - pl, 0.0);
        size_t rows = 1, cols = 1;
        parse_rng_size(args, 1, rows, cols, "random");
        Value u = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
        double *ud = u.doubleDataMut();
        const size_t n = rows * cols;
        auto &rng = ctx.engine->rng();
        for (size_t i = 0; i < n; ++i)
            ud[i] = pl + Z * rng.nextUniform();
        Value out = evalDistStructBase(pd, "inv", u, ctx);
        outs[0] = reshape_rng_nd(std::move(out), args, 1, mr, "random");
        return;
    }

    const Value &xVal = args[1];
    if (!isTrunc) {
        outs[0] = evalDistStructBase(pd, suffix, xVal, ctx);
        return;
    }

    const double pl = std::isinf(tLo) && tLo < 0.0
        ? 0.0 : evalDistStructBase(pd, "cdf", Value::scalar(tLo, mr), ctx).toScalar();
    const double pu = std::isinf(tHi) && tHi > 0.0
        ? 1.0 : evalDistStructBase(pd, "cdf", Value::scalar(tHi, mr), ctx).toScalar();
    const double Z = std::max(pu - pl, 1e-300);

    if (sfx == "pdf") {
        Value out = evalDistStructBase(pd, "pdf", xVal, ctx);
        double *od = out.doubleDataMut();
        const size_t n = out.numel();
        for (size_t i = 0; i < n; ++i) {
            const double xi = xVal.elemAsDouble(i);
            if (xi < tLo || xi > tHi) od[i] = 0.0;
            else od[i] /= Z;
        }
        outs[0] = std::move(out);
        return;
    }
    if (sfx == "cdf") {
        Value out = evalDistStructBase(pd, "cdf", xVal, ctx);
        double *od = out.doubleDataMut();
        const size_t n = out.numel();
        for (size_t i = 0; i < n; ++i) {
            const double xi = xVal.elemAsDouble(i);
            if (xi <= tLo) od[i] = 0.0;
            else if (xi >= tHi) od[i] = 1.0;
            else od[i] = std::clamp((od[i] - pl) / Z, 0.0, 1.0);
        }
        outs[0] = std::move(out);
        return;
    }
    if (sfx == "inv") {
        Value pScaled = xVal.isEmpty() ? xVal
            : Value::matrix(xVal.dims().rows(), xVal.dims().cols(), ValueType::DOUBLE, mr);
        const size_t n = xVal.numel();
        for (size_t i = 0; i < n; ++i) {
            const double pi = xVal.elemAsDouble(i);
            pScaled.doubleDataMut()[i] = (pi < 0.0 || pi > 1.0)
                ? std::numeric_limits<double>::quiet_NaN()
                : (pl + pi * (pu - pl));
        }
        outs[0] = evalDistStructBase(pd, "inv", pScaled, ctx);
        return;
    }
    (void)fnName;
}

// Resolve the distribution name in args[0], forward args[1..] to the family
// builtin <prefix><suffix> (cdf/pdf/inv/rnd) found via the engine.
void dispatchDist(const char *fnName, const char *suffix,
                  Span<const Value> args, size_t nargout, Span<Value> outs,
                  CallContext &ctx)
{
    if (!args.empty() && isDistStruct(args[0])) {
        dispatchDistStruct(fnName, suffix, args, nargout, outs, ctx);
        return;
    }
    const std::string fn = fnName;
    if (args.empty() || (!args[0].isChar() && !args[0].isString()))
        throw Error(fn + ": first argument must be a distribution name or object",
                    0, 0, fnName, "", "numkit:" + fn + ":name");

    std::string name = toLowerStr(args[0].toString());

    const auto &m = distPrefixMap();
    auto it = m.find(name);
    if (it == m.end())
        throw Error(fn + ": unsupported distribution '" + args[0].toString() + "'",
                    0, 0, fnName, "", "numkit:" + fn + ":dist");

    // The bare-name resolver finds the function (e.g. "normcdf" →
    // stats.distributions.normcdf) without the removed compat.* prefix.
    const std::string target = it->second + suffix;
    const ExternalFunc *target_fn = ctx.engine->findExternal(target, ctx.env);
    if (!target_fn)
        throw Error(fn + ": '" + it->second + suffix + "' is not available",
                    0, 0, fnName, "", "numkit:" + fn + ":missing");

    (*target_fn)(args.subspan(1), nargout, outs, ctx);
}

} // namespace

void makedist_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "makedist");
    auto *mr = ctx.engine->resource();
    if (!args[0].isChar() && !args[0].isString())
        throw Error("makedist: first argument must be a distribution name",
                    0, 0, "makedist", "", "numkit:makedist:name");

    const std::string rawName = args[0].toString();
    const std::string dist = toLowerStr(rawName);

    int nparams = 2;
    double p1 = 0.0, p2 = 1.0, p3 = 1.0;
    std::string n1 = "p1", n2 = "p2", n3 = "p3";

    if (dist == "normal" || dist == "norm" || dist == "gaussian" || dist == "lognormal" || dist == "logn") {
        nparams = 2; p1 = 0.0; p2 = 1.0; n1 = "mu"; n2 = "sigma";
    } else if (dist == "tlocationscale") {
        nparams = 3; p1 = 0.0; p2 = 1.0; p3 = 5.0; n1 = "mu"; n2 = "sigma"; n3 = "nu";
    } else if (dist == "gamma" || dist == "gam" || dist == "beta" || dist == "weibull" || dist == "wbl") {
        nparams = 2; p1 = 1.0; p2 = 1.0; n1 = "a"; n2 = "b";
    } else if (dist == "uniform" || dist == "unif") {
        nparams = 2; p1 = 0.0; p2 = 1.0; n1 = "lower"; n2 = "upper";
    } else if (dist == "exponential" || dist == "exp") {
        nparams = 1; p1 = 1.0; n1 = "mu";
    } else if (dist == "poisson" || dist == "poiss") {
        nparams = 1; p1 = 1.0; n1 = "lambda";
    } else if (dist == "t" || dist == "students t" || dist == "chisquare" || dist == "chi-square" || dist == "chi2") {
        nparams = 1; p1 = 1.0; n1 = "nu";
    } else if (dist == "rayleigh" || dist == "rayl") {
        nparams = 1; p1 = 1.0; n1 = "b";
    } else if (dist == "binomial" || dist == "bino") {
        nparams = 2; p1 = 1.0; p2 = 0.5; n1 = "N"; n2 = "p";
    } else if (dist == "extreme value" || dist == "ev") {
        nparams = 2; p1 = 0.0; p2 = 1.0; n1 = "mu"; n2 = "sigma";
    } else if (dist == "generalized extreme value" || dist == "gev") {
        nparams = 3; p1 = 0.0; p2 = 1.0; p3 = 0.0; n1 = "k"; n2 = "sigma"; n3 = "mu";
    } else if (dist == "generalized pareto" || dist == "gp") {
        nparams = 3; p1 = 1.0; p2 = 1.0; p3 = 0.0; n1 = "k"; n2 = "sigma"; n3 = "theta";
    } else {
        const auto &m = distPrefixMap();
        if (m.find(dist) == m.end())
            throw Error("makedist: unsupported distribution '" + rawName + "'",
                        0, 0, "makedist", "", "numkit:makedist:dist");
    }

    if (args.size() > 1) {
        if (args[1].isChar() || args[1].isString()) {
            for (size_t i = 1; i + 1 < args.size(); i += 2) {
                const std::string k = toLowerStr(args[i].toString());
                const double val = args[i + 1].toScalar();
                if (k == toLowerStr(n1) || k == "p1") p1 = val;
                else if (k == toLowerStr(n2) || k == "p2") p2 = val;
                else if (k == toLowerStr(n3) || k == "p3") p3 = val;
            }
        } else {
            if (args.size() >= 2) p1 = args[1].toScalar();
            if (args.size() >= 3) p2 = args[2].toScalar();
            if (args.size() >= 4) p3 = args[3].toScalar();
        }
    }

    Value pd = Value::structure(mr);
    pd.setFieldAll("DistributionName", Value::fromString(rawName, mr));
    pd.setFieldAll("__nparams", Value::scalar(static_cast<double>(nparams), mr));
    pd.setFieldAll("p1", Value::scalar(p1, mr));
    pd.setFieldAll("p2", Value::scalar(p2, mr));
    pd.setFieldAll("p3", Value::scalar(p3, mr));
    pd.setFieldAll(n1, Value::scalar(p1, mr));
    if (nparams >= 2) pd.setFieldAll(n2, Value::scalar(p2, mr));
    if (nparams >= 3) pd.setFieldAll(n3, Value::scalar(p3, mr));
    pd.setFieldAll("IsTruncated", Value::logicalScalar(false, mr));
    pd.setFieldAll("TruncLower", Value::scalar(-std::numeric_limits<double>::infinity(), mr));
    pd.setFieldAll("TruncUpper", Value::scalar(std::numeric_limits<double>::infinity(), mr));
    outs[0] = std::move(pd);
}

void truncate_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "truncate");
    if (!isDistStruct(args[0]))
        throw Error("truncate: first argument must be a probability distribution object",
                    0, 0, "truncate", "", "numkit:truncate:pd");
    auto *mr = ctx.engine->resource();
    const double lo = args[1].toScalar();
    const double hi = args[2].toScalar();
    if (!(lo < hi))
        throw Error("truncate: lower bound must be strictly less than upper bound",
                    0, 0, "truncate", "", "numkit:truncate:bounds");
    Value pd = args[0];
    pd.setFieldAll("IsTruncated", Value::logicalScalar(true, mr));
    pd.setFieldAll("TruncLower", Value::scalar(lo, mr));
    pd.setFieldAll("TruncUpper", Value::scalar(hi, mr));
    outs[0] = std::move(pd);
}

void cdf_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "cdf");  // no max: name-value tail in the reference
    dispatchDist("cdf", "cdf", args, nargout, outs, ctx);
}

void pdf_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "pdf");  // no max: name-value tail in the reference
    dispatchDist("pdf", "pdf", args, nargout, outs, ctx);
}

void icdf_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "icdf");  // no max: name-value tail in the reference
    dispatchDist("icdf", "inv", args, nargout, outs, ctx);
}

void random_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "random");  // no max: name-value tail in the reference
    dispatchDist("random", "rnd", args, nargout, outs, ctx);
}

} // namespace detail

} // namespace numkit::stats
