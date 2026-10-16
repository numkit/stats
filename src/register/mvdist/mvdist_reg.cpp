// toolboxes/signal/src/mvdist/mvdist_reg.cpp
//
// CallContext register half of mvdist/mvdist.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/mvdist/mvdist.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::stats {

namespace detail {

void mvnpdf_reg(Span<const Value> args, size_t /*nargout*/,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "mvnpdf");
    auto *mr = ctx.engine->resource();
    Value empty = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    const Value &mu  = (args.size() >= 2) ? args[1] : empty;
    const Value &sig = (args.size() >= 3) ? args[2] : empty;
    outs[0] = mvnpdf(args[0], mu, sig, mr);
}

void mnpdf_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "mnpdf");
    outs[0] = mnpdf(args[0], args[1], ctx.engine->resource());
}

void mvtpdf_reg(Span<const Value> args, size_t /*nargout*/,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "mvtpdf");
    // df flows through as a Value — the scalar-or-per-row contract
    // (plus the C standardization and PSD gate) lives in compute.
    outs[0] = mvtpdf(args[0], args[1], args[2], ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::stats
