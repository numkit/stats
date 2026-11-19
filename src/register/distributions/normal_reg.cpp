// toolboxes/signal/src/distributions/normal_reg.cpp
//
// CallContext register half of distributions/normal.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/array.h>
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>  // RngContext + rand/randn/randi/randperm (session-scoped, no global/mutex)
#include <numkit/stats/distributions/normal.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"            // stripUpperFlag / applyUpperInPlace
#include "distributions/normal_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void normpdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "normpdf");
    auto *mr = ctx.engine->allocator();
    Value hmu, hsig;
    const Value &mu  = dist_param(args, 1, 0.0, mr, hmu);
    const Value &sig = dist_param(args, 2, 1.0, mr, hsig);
    outs[0] = broadcast_dist3(args[0], mu, sig, mr, "normpdf", normpdfK);
}

void normcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "normcdf");
    auto *mr = ctx.engine->allocator();
    bool upper = false;
    const keel::ArrayRef<Value> a = args.slice(0, stripUpperFlag(args, upper));
    Value hmu, hsig;
    const Value &mu  = dist_param(a, 1, 0.0, mr, hmu);
    const Value &sig = dist_param(a, 2, 1.0, mr, hsig);
    Value v = broadcast_dist3(a[0], mu, sig, mr, "normcdf", normcdfK);
    if (upper) applyUpperInPlace(v);
    outs[0] = std::move(v);
}

void norminv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "norminv");
    auto *mr = ctx.engine->allocator();
    Value hmu, hsig;
    const Value &mu  = dist_param(args, 1, 0.0, mr, hmu);
    const Value &sig = dist_param(args, 2, 1.0, mr, hsig);
    outs[0] = broadcast_dist3(args[0], mu, sig, mr, "norminv", norminvK);
}

void normrnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "normrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    auto *mr = ctx.engine->allocator();
    const Value &muVal = args[0];
    const Value &sigVal = args[1];
    if (muVal.isScalar() && sigVal.isScalar()) {
        const double mu = muVal.toScalar();
        const double sigma = sigVal.toScalar();
        keel::usize rows, cols;
        parse_rng_size(args, 2, rows, cols, "normrnd");
        outs[0] = reshape_rng_nd(normrnd(ctx.engine->rng(), mu, sigma, rows, cols, mr), args, 2, mr, "normrnd");
        return;
    }
    const keel::usize nmu = muVal.numel();
    const keel::usize nsig = sigVal.numel();
    keel::usize rows = 0, cols = 0;
    if (args.size() > 2) {
        parse_rng_size(args, 2, rows, cols, "normrnd");
    } else {
        const Value &ref = (nmu >= nsig) ? muVal : sigVal;
        rows = ref.dims().rows();
        cols = ref.dims().cols();
        for (int d = 2; d < ref.dims().ndim(); ++d)
            cols *= ref.dims().dim(d);
    }
    const keel::usize n = rows * cols;
    if ((nmu != 1 && nmu != n) || (nsig != 1 && nsig != n))
        throw Error("normrnd: Non-scalar arguments must match in size.",
                    0, 0, "normrnd", "", "numkit:normrnd:size");
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    auto &rng = ctx.engine->rng();
    for (keel::usize i = 0; i < n; ++i) {
        const double m = muVal.elemAsDouble(nmu == 1 ? 0 : i);
        const double s = sigVal.elemAsDouble(nsig == 1 ? 0 : i);
        od[i] = (s < 0.0) ? std::numeric_limits<double>::quiet_NaN()
                          : (m + s * rng.nextNormal());
    }
    if (args.size() > 2) {
        outs[0] = reshape_rng_nd(std::move(out), args, 2, mr, "normrnd");
    } else {
        const Value &ref = (nmu >= nsig) ? muVal : sigVal;
        if (ref.dims().ndim() > 2) {
            keel::Array<keel::usize> shape(static_cast<keel::usize>(ref.dims().ndim()));
            for (int d = 0; d < ref.dims().ndim(); ++d)
                shape[d] = ref.dims().dim(d);
            outs[0] = builtin::reshape(std::move(out), keel::ArrayRef<keel::usize>(shape.data(), shape.size()), mr);
        } else {
            outs[0] = std::move(out);
        }
    }
}

void normstat_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "normstat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->allocator(), "normstat",
                       [](double mu, double sigma) { return normstat(mu, sigma); });
}

} // namespace detail

} // namespace numkit::stats
