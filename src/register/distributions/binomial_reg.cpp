// toolboxes/signal/src/distributions/binomial_reg.cpp
//
// CallContext register half of distributions/binomial.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>
#include <numkit/stats/distributions/binomial.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/binomial_detail.hpp"
#include "distributions/dist_helpers.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>

namespace numkit::stats {

namespace detail {

void binopdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "binopdf");
    auto *mr = ctx.engine->allocator();
    const Value &n = args[1];
    const Value &p = args[2];
    if (n.isScalar() && p.isScalar())
        outs[0] = binopdf(args[0], n.toScalar(), p.toScalar(), mr);
    else
        outs[0] = broadcast_dist3(args[0], n, p, mr, "binopdf", binopdfK);
}

void binocdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "binocdf");
    bool upper = false;
    const keel::ArrayRef<Value> a = args.slice(0, stripUpperFlag(args, upper));
    if (a.size() < 3)
        throw Error("binocdf: requires (k, n, p[, 'upper'])", 0, 0, "binocdf", "", "numkit:binocdf:nargin");
    auto *mr = ctx.engine->allocator();
    const Value &n = a[1];
    const Value &p = a[2];
    Value v;
    if (n.isScalar() && p.isScalar()) {
        v = binocdf(a[0], n.toScalar(), p.toScalar(), mr);
    } else {
        // Per-element F(k_i; n_i, p_i) (bino_cdf_scalar handles the k edges +
        // betainc); invalid (n, p) → NaN. Same per-element-betainc cost as the
        // existing scalar binocdf over a vector k.
        const Value &k = a[0];
        const keel::usize nk = k.numel(), nn = n.numel(), np = p.numel();
        if (nk == 0 || nn == 0 || np == 0) {
            v = dist_empty_like(nk == 0 ? k : (nn == 0 ? n : p), mr);
        } else {
            const keel::usize N = dist_match_numel({nk, nn, np}, "binocdf");
            const Value &ref = (nn == N) ? n : (np == N ? p : k);
            v = dist_empty_like(ref, mr);
            keel::f64 *od = v.doubleDataMut();
            const keel::f64 NaN = keel::Limits<keel::f64>::quiet_nan();
            for (keel::usize i = 0; i < N; ++i) {
                const keel::f64 ni = n.elemAsDouble(nn == 1 ? 0 : i);
                const keel::f64 pi = p.elemAsDouble(np == 1 ? 0 : i);
                const keel::f64 ki = k.elemAsDouble(nk == 1 ? 0 : i);
                od[i] = bino_params_ok(ni, pi) ? bino_cdf_scalar(ki, ni, pi, mr) : NaN;
            }
        }
    }
    if (upper) applyUpperInPlace(v);
    outs[0] = keel::move(v);
}

void binoinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "binoinv");
    auto *mr = ctx.engine->allocator();
    const Value &n = args[1];
    const Value &p = args[2];
    if (n.isScalar() && p.isScalar()) {
        outs[0] = binoinv(args[0], n.toScalar(), p.toScalar(), mr);
        return;
    }
    const Value &pin = args[0];
    const keel::usize nq = pin.numel(), nn = n.numel(), np = p.numel();
    if (nq == 0 || nn == 0 || np == 0) {
        outs[0] = dist_empty_like(nq == 0 ? pin : (nn == 0 ? n : p), mr);
        return;
    }
    const keel::usize N = dist_match_numel({nq, nn, np}, "binoinv");
    const Value &ref = (nn == N) ? n : (np == N ? p : pin);
    Value out = dist_empty_like(ref, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 NaN = keel::Limits<keel::f64>::quiet_nan();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 ni = n.elemAsDouble(nn == 1 ? 0 : i);
        const keel::f64 pi = p.elemAsDouble(np == 1 ? 0 : i);
        const keel::f64 qi = pin.elemAsDouble(nq == 1 ? 0 : i);
        od[i] = bino_params_ok(ni, pi) ? bino_inv_scalar(qi, ni, pi) : NaN;
    }
    outs[0] = keel::move(out);
}

void binornd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "binornd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    const keel::f64 n = args[0].toScalar();
    const keel::f64 p = args[1].toScalar();
    keel::usize rows, cols;
    parse_rng_size(args, 2, rows, cols, "binornd");
    outs[0] = reshape_rng_nd(binornd(ctx.engine->rng(), n, p, rows, cols, ctx.engine->allocator()), args, 2, ctx.engine->allocator(), "binornd");
}

void binostat_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "binostat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->allocator(), "binostat",
                       [](keel::f64 n, keel::f64 p) { return binostat(n, p); });
}

} // namespace detail

} // namespace numkit::stats
