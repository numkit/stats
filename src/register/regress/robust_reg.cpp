// toolboxes/signal/src/regress/robust_reg.cpp
//
// CallContext register half of regress/robust.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/distributions/chi2.hpp>      // chi2inv
#include <numkit/stats/regress/regress.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "regress/robust_detail.hpp"
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void robustfit_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 7, "robustfit");
    RobustWeight w = RobustWeight::Bisquare;
    if (args.size() >= 3 && args[2].isChar()) {
        keel::String s = args[2].toString();
        for (char &c : s) c = keel::to_ascii_lower(c);
        if      (s == "andrews")  w = RobustWeight::Andrews;
        else if (s == "bisquare") w = RobustWeight::Bisquare;
        else if (s == "cauchy")   w = RobustWeight::Cauchy;
        else if (s == "fair")     w = RobustWeight::Fair;
        else if (s == "huber")    w = RobustWeight::Huber;
        else if (s == "logistic") w = RobustWeight::Logistic;
        else if (s == "ols")      w = RobustWeight::Ols;
        else if (s == "talwar")   w = RobustWeight::Talwar;
        else if (s == "welsch")   w = RobustWeight::Welsch;
        else
            throw Error("robustfit: weight must be one of 'andrews', "
                        "'bisquare', 'cauchy', 'fair', 'huber', 'logistic', "
                        "'ols', 'talwar', 'welsch'",
                        0, 0, "robustfit", "", "numkit:robustfit:badWeight");
    }
    keel::f64 tune = std::numeric_limits<keel::f64>::quiet_NaN();
    if (args.size() >= 4 && !args[3].isEmpty())
        tune = args[3].toScalar();

    // MATLAB robustfit adds a constant (intercept) term by default; the
    // 5th argument 'const' = 'on' (default) | 'off' toggles it. With the
    // intercept, b = [b0; slopes...] (b0 first). nk previously fit the
    // raw X with no intercept, returning the wrong number of coefficients.
    bool addConst = true;
    if (args.size() >= 5 && args[4].isChar()) {
        const keel::String c = args[4].toString();
        if (c == "off" || c == "Off" || c == "OFF")       addConst = false;
        else if (c == "on" || c == "On" || c == "ON")     addConst = true;
        else
            throw Error("robustfit: const must be 'on' or 'off'",
                        0, 0, "robustfit", "", "numkit:robustfit:badConst");
    }

    Value Xin = args[0];
    if (addConst) {
        const keel::usize n = Xin.dims().rows();
        const keel::usize p = Xin.dims().cols();
        Value Xaug = Value::matrix(n, p + 1, ValueType::DOUBLE,
                                   ctx.engine->allocator());
        keel::f64 *xa = Xaug.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) xa[i] = 1.0;        // intercept col
        for (keel::usize j = 0; j < p; ++j)
            for (keel::usize i = 0; i < n; ++i)
                xa[(j + 1) * n + i] = Xin.elemAsDouble(j * n + i);
        Xin = std::move(Xaug);
    }

    auto r = robustfit(Xin, args[1], w, tune, ctx.engine->allocator());
    outs[0] = std::move(r.b);
    if (nargout > 1) outs[1] = std::move(r.s);
}

void robustcov_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "robustcov");  // no max: name-value tail in the reference
    auto r = robustcov(args[0], ctx.engine->allocator());
    outs[0] = std::move(r.sigma);
    if (nargout > 1) outs[1] = std::move(r.mu);
}

} // namespace detail

} // namespace numkit::stats
