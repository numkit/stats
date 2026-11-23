// toolboxes/signal/src/distributions/gamma_dist_reg.cpp
//
// CallContext register half of distributions/gamma_dist.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>
#include <numkit/stats/distributions/gamma_dist.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "distributions/dist_helpers.hpp"
#include "distributions/gamma_dist_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>

namespace numkit::stats {

namespace detail {

void gampdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "gampdf");
    auto *mr = ctx.engine->allocator();
    const Value &a = args[1];
    const Value &b = args[2];
    if (a.isScalar() && b.isScalar())
        outs[0] = gampdf(args[0], a.toScalar(), b.toScalar(), mr);
    else
        outs[0] = broadcast_dist3(args[0], a, b, mr, "gampdf", gampdfK);
}

void gamcdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "gamcdf");
    bool upper = false;
    const keel::ArrayRef<Value> a0 = args.slice(0, stripUpperFlag(args, upper));
    if (a0.size() < 3)
        throw Error("gamcdf: requires (x, a, b[, 'upper'])", 0, 0, "gamcdf", "", "numkit:gamcdf:nargin");
    auto *mr = ctx.engine->allocator();
    const Value &a = a0[1];
    const Value &b = a0[2];
    Value v;
    if (a.isScalar() && b.isScalar()) {
        v = gamcdf(a0[0], a.toScalar(), b.toScalar(), mr);
    } else {
        // F(x; a, b) = gammainc(x/b clamped at 0, a). xs broadcasts (x, b)
        // (b<=0 → NaN), then gammainc broadcasts (xs, a) and gives NaN where
        // a<=0 — matching the scalar path's a<=0||b<=0 → NaN.
        Value xs = broadcast_dist2(a0[0], b, mr, "gamcdf", [](keel::f64 xi, keel::f64 bi) {
            if (!(bi > 0.0)) return keel::Limits<keel::f64>::quiet_nan();
            return (xi <= 0.0) ? 0.0 : xi / bi;
        });
        v = ::numkit::builtin::gammainc(xs, a, mr);
    }
    if (upper) applyUpperInPlace(v);
    outs[0] = keel::move(v);
}

void gaminv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "gaminv");
    auto *mr = ctx.engine->allocator();
    const Value &p = args[0];
    const Value &a = args[1];
    const Value &b = args[2];
    if (a.isScalar() && b.isScalar()) {
        outs[0] = gaminv(p, a.toScalar(), b.toScalar(), mr);   // unchanged fast path
        return;
    }
    // Broadcast: q = gammaincinv(p, a) (broadcasts (p, a)); then scale by b and
    // apply the degenerate quantile per element, matching the scalar gaminv:
    // a<0 or b<=0 → NaN; a==0 → 0 (p∈[0,1]) / NaN; else b·gammaincinv(p, a).
    const keel::usize np = p.numel(), na = a.numel(), nb = b.numel();
    if (np == 0 || na == 0 || nb == 0) {
        outs[0] = dist_empty_like(np == 0 ? p : (na == 0 ? a : b), mr);
        return;
    }
    const keel::usize N = dist_match_numel({np, na, nb}, "gaminv");
    Value q = ::numkit::builtin::gammaincinv(p, a, mr);
    const keel::usize nq = q.numel();
    const Value &ref = (np == N) ? p : (na == N ? a : b);
    Value out = dist_empty_like(ref, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 NaN = keel::Limits<keel::f64>::quiet_nan();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 pi = p.elemAsDouble(np == 1 ? 0 : i);
        const keel::f64 ai = a.elemAsDouble(na == 1 ? 0 : i);
        const keel::f64 bi = b.elemAsDouble(nb == 1 ? 0 : i);
        const keel::f64 qi = q.elemAsDouble(nq == 1 ? 0 : i);
        od[i] = (ai < 0.0 || !(bi > 0.0))
                    ? NaN
                    : (ai == 0.0 ? ((pi >= 0.0 && pi <= 1.0) ? 0.0 : NaN) : bi * qi);
    }
    outs[0] = keel::move(out);
}

void gamrnd_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "gamrnd");  // no max: trailing size args (N-D tail; nk caps at two — bugs/opened/stats/rnd-nd-size-tails.md)
    auto *mr = ctx.engine->allocator();
    if (args.size() == 2 && (!args[0].isScalar() || !args[1].isScalar())) {
        const Value &a = args[0], &b = args[1];
        const keel::usize na = a.numel(), nb = b.numel();
        if (na == 0 || nb == 0) {
            outs[0] = dist_empty_like(na == 0 ? a : b, mr);
            return;
        }
        const keel::usize N = dist_match_numel({na, nb}, "gamrnd");
        const Value &ref = (na == N) ? a : b;
        Value out = dist_empty_like(ref, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 ai = a.elemAsDouble(na == 1 ? 0 : i);
            const keel::f64 bi = b.elemAsDouble(nb == 1 ? 0 : i);
            Value sample = gamrnd(ctx.engine->rng(), ai, bi, 1, 1, mr);
            od[i] = sample.elemAsDouble(0);
        }
        outs[0] = keel::move(out);
        return;
    }
    const keel::f64 a = args[0].toScalar();
    const keel::f64 b = args[1].toScalar();
    keel::usize rows, cols;
    parse_rng_size(args, 2, rows, cols, "gamrnd");
    outs[0] = reshape_rng_nd(gamrnd(ctx.engine->rng(), a, b, rows, cols, mr), args, 2, mr, "gamrnd");
}

void gamstat_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "gamstat");
    emit_vec_stat_2arg(args, nargout, outs, ctx.engine->allocator(), "gamstat",
                       [](keel::f64 a, keel::f64 b) { return gamstat(a, b); });
}

// ── randg — undocumented but very widely used "raw" gamma RNG ─────────
//
// `randg(ctx.engine->rng(), shape, ...)` is shorthand for gamma(shape, 1) — i.e. scale = 1.
// Forms:
//   r = randg(ctx.engine->rng(), shape)            — scalar (or per-element if shape is an
//                                  array)
//   r = randg(ctx.engine->rng(), shape, n)         — n×n matrix
//   r = randg(ctx.engine->rng(), shape, m, n)      — m×n matrix
//   r = randg(ctx.engine->rng(), shape, [m n])     — same
//
// Implementation: delegate to gamrnd with b = 1. When shape is an array
// and no explicit size args are given, draw one sample per shape entry
// (each with its own shape parameter).
void randg_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, "randg");  // no max: trailing size args (builtin in the reference build)
    auto *mr = ctx.engine->allocator();
    if (args.empty()) {
        outs[0] = randg(ctx.engine->rng(), 1.0, 1, 1, mr);
        return;
    }
    const Value &shape = args[0];
    // Per-element form: array shape AND no extra size args.
    if (!shape.isScalar() && args.size() == 1) {
        outs[0] = randg(ctx.engine->rng(), shape, mr);
        return;
    }
    // Scalar shape — pull size from the remaining args (or default 1×1).
    keel::usize rows = 1, cols = 1;
    parse_rng_size(args, 1, rows, cols, "randg");
    outs[0] = reshape_rng_nd(randg(ctx.engine->rng(), shape.toScalar(), rows, cols, mr), args, 1, ctx.engine->allocator(), "randg");
}

} // namespace detail

} // namespace numkit::stats
