// toolboxes/signal/src/descriptive/partial_canon_reg.cpp
//
// CallContext register half of descriptive/partial_canon.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/linalg/decompositions.hpp>     // qr_decompose, svd_decompose
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
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

void partialcorri_reg(Span<const Value> args, size_t /*nargout*/,
                       Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "partialcorri");  // no max: name-value tail in the reference
    const Value Z = (args.size() >= 3) ? args[2] : Value();
    outs[0] = partialcorri(args[0], args[1], Z, ctx.engine->resource());
}

void canoncorr_reg(Span<const Value> args, size_t nargout,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "canoncorr");
    auto res = canoncorr(args[0], args[1], ctx.engine->resource());
    outs[0] = std::move(res.A);
    if (nargout > 1) outs[1] = std::move(res.B);
    if (nargout > 2) outs[2] = std::move(res.r);
}

} // namespace detail

} // namespace numkit::stats
