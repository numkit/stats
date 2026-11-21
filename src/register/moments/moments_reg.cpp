// toolboxes/signal/src/moments/moments_reg.cpp
//
// CallContext register half of moments/moments.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/moments/moments.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

void skewness_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "skewness");
    keel::i32 normFlag = 1;  // MATLAB default
    keel::i32 dim = 0;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const keel::f64 fv = argcheck::expectedReal(args[1], 2, "flag", "skewness");
        argcheck::expectedFlag01(fv, 2, "flag", "skewness");
        normFlag = keel::as<keel::i32>(fv);
    }
    if (args.size() >= 3 && !args[2].isEmpty())
        dim = argcheck::dimArg(args[2], "skewness");
    outs[0] = skewness(args[0], normFlag, dim, ctx.engine->allocator());
}

void kurtosis_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "kurtosis");
    keel::i32 normFlag = 1;
    keel::i32 dim = 0;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const keel::f64 fv = argcheck::expectedReal(args[1], 2, "flag", "kurtosis");
        argcheck::expectedFlag01(fv, 2, "flag", "kurtosis");
        normFlag = keel::as<keel::i32>(fv);
    }
    if (args.size() >= 3 && !args[2].isEmpty())
        dim = argcheck::dimArg(args[2], "kurtosis");
    outs[0] = kurtosis(args[0], normFlag, dim, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::stats
