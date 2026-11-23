// toolboxes/signal/src/distributions/multivariate_reg.cpp
//
// CallContext register half of distributions/multivariate.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/math.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>  // RngContext + rand/randn/randi/randperm (session-scoped, no global/mutex)
#include <numkit/builtin/specfun.hpp>  // betainc (used by mvtcdf d=1)
#include <numkit/stats/distributions/multivariate.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/multivariate_detail.hpp"
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::stats {

namespace detail {

void mvncdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "mvncdf");  // no max: options-struct tail in the reference
    // Marshalling only — the positional forms (X[, mu[, Sigma]] and the
    // box (xl, xu, mu, Sigma)), the options-struct strip and every
    // shape/PSD gate live in mvncdf_full.
    outs[0] = mvncdf_full(args, ctx.engine->allocator());
}

void mvnrnd_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "mvnrnd");
    // cases: extract a validated integer count (fractional / NaN /
    // complex reject); empty or absent → derive (empty Value). Negative
    // counts flow through as-is — the contract turns them into 0×d.
    Value cases;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        const keel::f64 n = argcheck::expectedReal(args[2], 3, "n", "mvnrnd");
        if (keel::is_nan(n) || n != keel::floor(n))
            throw Error(keel::format("mvnrnd: n must be a real integer scalar (got {}).", n),
                        0, 0, "mvnrnd", "", "numkit:mvnrnd:badCases");
        cases = Value::scalar(n, ctx.engine->allocator());
    }
    auto r = mvnrnd_full(ctx.engine->rng(), args[0], args[1], cases,
                         nargout >= 2, ctx.engine->allocator());
    outs[0] = keel::move(r.r);
    if (nargout >= 2) outs[1] = keel::move(r.T);
}

void mvtrnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "mvtrnd");
    Value cases;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        const keel::f64 n = argcheck::expectedReal(args[2], 3, "cases", "mvtrnd");
        if (keel::is_nan(n) || n != keel::floor(n))
            throw Error(keel::format("mvtrnd: cases must be a real integer scalar (got {}).", n),
                        0, 0, "mvtrnd", "", "numkit:mvtrnd:badCases");
        cases = Value::scalar(n, ctx.engine->allocator());
    }
    outs[0] = mvtrnd_full(ctx.engine->rng(), args[0], args[1], cases,
                          ctx.engine->allocator());
}

void mnrnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "mnrnd");
    const keel::usize N = argcheck::roundLength(args[0], 1, "N", "mnrnd");
    keel::usize m = 1;
    if (args.size() >= 3 && !args[2].isEmpty())
        m = argcheck::arrayDim(args[2], "mnrnd");
    outs[0] = mnrnd(ctx.engine->rng(), N, args[1], m, ctx.engine->allocator());
}

void wishrnd_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "wishrnd");
    const keel::f64 df = args[1].toScalar();
    const Value D_in = (args.size() > 2) ? args[2] : Value::Empty;
    auto [W, D] = wishrnd_factor(ctx.engine->rng(), args[0], df, D_in, ctx.engine->allocator());
    outs[0] = keel::move(W);
    if (nargout >= 2) outs[1] = keel::move(D);
}

void iwishrnd_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "iwishrnd");
    const keel::f64 df = args[1].toScalar();
    const Value DI_in = (args.size() > 2) ? args[2] : Value::Empty;
    auto [W, DI] = iwishrnd_factor(ctx.engine->rng(), args[0], df, DI_in, ctx.engine->allocator());
    outs[0] = keel::move(W);
    if (nargout >= 2) outs[1] = keel::move(DI);
}

void mvtcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "mvtcdf");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    // 3 args: mvtcdf(X, C, df) — upper-tail form.
    // 4 args: mvtcdf(L, U, C, df) — box form.
    // 5 args: mvtcdf(L, U, C, df, tol) — box form with tolerance.
    if (args.size() == 3) {
        outs[0] = mvtcdf(args[0], args[1], args[2].toScalar(), 0.01, mr);
    } else if (args.size() == 4) {
        outs[0] = mvtcdf_box(args[0], args[1], args[2],
                             args[3].toScalar(), 0.01, mr);
    } else if (args.size() == 5) {
        outs[0] = mvtcdf_box(args[0], args[1], args[2],
                             args[3].toScalar(), args[4].toScalar(), mr);
    } else {
        throw Error("mvtcdf: requires (X, C, df) or (L, U, C, df[, tol])",
                    0, 0, "mvtcdf", "", "numkit:mvtcdf:nargin");
    }
}

} // namespace detail

} // namespace numkit::stats
