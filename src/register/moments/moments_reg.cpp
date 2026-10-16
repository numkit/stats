// toolboxes/signal/src/moments/moments_reg.cpp
//
// CallContext register half of moments/moments.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/moments/moments.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

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

void skewness_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "skewness");
    int normFlag = 1;  // MATLAB default
    int dim = 0;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const double fv = argcheck::expectedReal(args[1], 2, "flag", "skewness");
        argcheck::expectedFlag01(fv, 2, "flag", "skewness");
        normFlag = static_cast<int>(fv);
    }
    if (args.size() >= 3 && !args[2].isEmpty())
        dim = argcheck::dimArg(args[2], "skewness");
    outs[0] = skewness(args[0], normFlag, dim, ctx.engine->resource());
}

void kurtosis_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "kurtosis");
    int normFlag = 1;
    int dim = 0;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const double fv = argcheck::expectedReal(args[1], 2, "flag", "kurtosis");
        argcheck::expectedFlag01(fv, 2, "flag", "kurtosis");
        normFlag = static_cast<int>(fv);
    }
    if (args.size() >= 3 && !args[2].isEmpty())
        dim = argcheck::dimArg(args[2], "kurtosis");
    outs[0] = kurtosis(args[0], normFlag, dim, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::stats
