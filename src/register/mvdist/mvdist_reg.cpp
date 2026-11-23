// toolboxes/signal/src/mvdist/mvdist_reg.cpp
//
// CallContext register half of mvdist/mvdist.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/mvdist/mvdist.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void mvnpdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "mvnpdf");
    auto *mr = ctx.engine->allocator();
    Value empty = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    const Value &mu  = (args.size() >= 2) ? args[1] : empty;
    const Value &sig = (args.size() >= 3) ? args[2] : empty;
    outs[0] = mvnpdf(args[0], mu, sig, mr);
}

void mnpdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "mnpdf");
    outs[0] = mnpdf(args[0], args[1], ctx.engine->allocator());
}

void mvtpdf_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "mvtpdf");
    // df flows through as a Value — the scalar-or-per-row contract
    // (plus the C standardization and PSD gate) lives in compute.
    outs[0] = mvtpdf(args[0], args[1], args[2], ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::stats
