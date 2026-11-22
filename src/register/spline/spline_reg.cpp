// toolboxes/signal/src/spline/spline_reg.cpp
//
// CallContext register half of spline/spline.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/spline/spline.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void aveknt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "aveknt");
    outs[0] = aveknt(args[0], argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "k", "aveknt"), 2, "k", "aveknt"),
        ctx.engine->allocator());
}

void augknt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "augknt");
    outs[0] = augknt(args[0], argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "k", "augknt"), 2, "k", "augknt"),
        ctx.engine->allocator());
}

void brk2knt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "brk2knt");
    outs[0] = brk2knt(args[0], args[1], ctx.engine->allocator());
}

void ppmak_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "ppmak");
    keel::i32 d = 1;
    if (args.size() >= 3 && !args[2].isEmpty())
        d = argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "d", "ppmak"), 3, "d", "ppmak");
    outs[0] = ppmak(args[0], args[1], d, ctx.engine->allocator());
}

void fnval_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "fnval");  // no max: name-value tail in the reference
    outs[0] = fnval(args[0], args[1], ctx.engine->allocator());
}

void fnder_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "fnder");
    keel::i32 order = 1;
    if (args.size() >= 2 && !args[1].isEmpty())
        order = keel::as<keel::i32>(argcheck::roundLength(args[1], 2, "order", "fnder"));
    outs[0] = fnder(args[0], order, ctx.engine->allocator());
}

void fnint_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "fnint");
    outs[0] = fnint(args[0], ctx.engine->allocator());
}

void csapi_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "csapi");
    outs[0] = csapi(args[0], args[1], ctx.engine->allocator());
}

void fncmb_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "fncmb");
    auto *mr = ctx.engine->allocator();
    auto isStruct = [](const Value &v) { return v.isStruct(); };
    if (args.size() == 2) {
        // fncmb(pp, c) or fncmb(c, pp)
        if (isStruct(args[0]) && !isStruct(args[1])) {
            outs[0] = fncmb(args[0], args[1].toScalar(), Value::Empty, 0.0, mr);
        } else if (!isStruct(args[0]) && isStruct(args[1])) {
            outs[0] = fncmb(args[1], args[0].toScalar(), Value::Empty, 0.0, mr);
        } else {
            throw Error("fncmb: 2-arg form requires (pp, scalar) or (scalar, pp)",
                        0, 0, "fncmb", "", "numkit:fncmb:nargin");
        }
        return;
    }
    if (args.size() == 4) {
        outs[0] = fncmb(args[0], args[1].toScalar(), args[2], args[3].toScalar(), mr);
        return;
    }
    throw Error("fncmb: requires (pp, c) or (pp1, c1, pp2, c2)",
                0, 0, "fncmb", "", "numkit:fncmb:nargin");
}

void fnbrk_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "fnbrk");  // no max: name-value tail in the reference
    if (!args[1].isChar() && !args[1].isString())
        throw Error("fnbrk: numkit only supports the named-part form",
                    0, 0, "fnbrk", "", "numkit:fnbrk:type");
    keel::String part = args[1].toString();
    outs[0] = fnbrk(args[0], part.as_ref(), ctx.engine->allocator());
}

void knt2brk_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "knt2brk");
    auto [b, m] = knt2brk(args[0], ctx.engine->allocator());
    outs[0] = std::move(b);
    if (nargout > 1) outs[1] = std::move(m);
}

} // namespace detail

} // namespace numkit::stats
