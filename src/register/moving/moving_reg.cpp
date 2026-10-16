// toolboxes/signal/src/moving/moving_reg.cpp
//
// CallContext register half of moving/moving.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/moving/moving.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include <numkit/ops/helpers.hpp>            // createLike, createForDims (toolboxes/builtin/src/)
#include "moving/moving_detail.hpp"
#include <numkit/ops/reductions.hpp>  // numkit::ops::firstNonSingletonDim, validateDim
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

void movmean_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "movmean");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto kBuf = decodeWindowValueToScratch(args[1], "movmean", scratch);
    auto opt = parseMovExtras(args, 2, "movmean");
    outs[0] = movmean_impl(args[0], Span<const size_t>(kBuf.data(), kBuf.size()), opt, mr);
}

void movsum_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "movsum");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto kBuf = decodeWindowValueToScratch(args[1], "movsum", scratch);
    auto opt = parseMovExtras(args, 2, "movsum");
    outs[0] = movsum_impl(args[0], Span<const size_t>(kBuf.data(), kBuf.size()), opt, mr);
}

void movmin_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "movmin");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto kBuf = decodeWindowValueToScratch(args[1], "movmin", scratch);
    auto opt = parseMovExtras(args, 2, "movmin");
    outs[0] = movmin_impl(args[0], Span<const size_t>(kBuf.data(), kBuf.size()), opt, mr);
}

void movmax_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "movmax");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto kBuf = decodeWindowValueToScratch(args[1], "movmax", scratch);
    auto opt = parseMovExtras(args, 2, "movmax");
    outs[0] = movmax_impl(args[0], Span<const size_t>(kBuf.data(), kBuf.size()), opt, mr);
}

void movprod_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "movprod");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto kBuf = decodeWindowValueToScratch(args[1], "movprod", scratch);
    auto opt = parseMovExtras(args, 2, "movprod");
    outs[0] = movprod_impl(args[0], Span<const size_t>(kBuf.data(), kBuf.size()), opt, mr);
}

void movmedian_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "movmedian");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto kBuf = decodeWindowValueToScratch(args[1], "movmedian", scratch);
    auto opt = parseMovExtras(args, 2, "movmedian");
    outs[0] = movmedian_impl(args[0], Span<const size_t>(kBuf.data(), kBuf.size()), opt, mr);
}

void movvar_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "movvar");  // no max: name-value tail in the reference
    int normFlag = 0;
    size_t extras_start = 2;
    if (args.size() >= 3 && !args[2].isChar() && !args[2].isString()
        && args[2].isScalar()) {
        // Could be normFlag or dim — disambiguate: normFlag is 0 or 1.
        const double v = args[2].toScalar();
        if (v == 0.0 || v == 1.0) {
            normFlag = static_cast<int>(v);
            extras_start = 3;
        }
    }
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto kBuf = decodeWindowValueToScratch(args[1], "movvar", scratch);
    auto opt = parseMovExtras(args, extras_start, "movvar");
    outs[0] = movvar_impl(args[0], Span<const size_t>(kBuf.data(), kBuf.size()), normFlag, opt, mr);
}

void movstd_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "movstd");  // no max: name-value tail in the reference
    int normFlag = 0;
    size_t extras_start = 2;
    if (args.size() >= 3 && !args[2].isChar() && !args[2].isString()
        && args[2].isScalar()) {
        const double v = args[2].toScalar();
        if (v == 0.0 || v == 1.0) {
            normFlag = static_cast<int>(v);
            extras_start = 3;
        }
    }
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto kBuf = decodeWindowValueToScratch(args[1], "movstd", scratch);
    auto opt = parseMovExtras(args, extras_start, "movstd");
    outs[0] = movstd_impl(args[0], Span<const size_t>(kBuf.data(), kBuf.size()), normFlag, opt, mr);
}

void movmad_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "movmad");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto kBuf = decodeWindowValueToScratch(args[1], "movmad", scratch);
    auto opt = parseMovExtras(args, 2, "movmad");
    outs[0] = movmad_impl(args[0], Span<const size_t>(kBuf.data(), kBuf.size()), opt, mr);
}

void smoothdata_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "smoothdata");  // no max: name-value tail in the reference
    std::string method = "movmean";
    int k = 0;
    if (args.size() >= 2) {
        if (args[1].isChar() || args[1].isString())
            method = args[1].toString();
        else {
            k = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(args[1], 2, "window", "smoothdata"),
                2, "window", "smoothdata");
        }
    }
    if (args.size() >= 3) {
        if (k == 0 && (args[2].isScalar() || args[2].numel() == 1)) {
            k = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(args[2], 3, "window", "smoothdata"),
                3, "window", "smoothdata");
        }
    }
    outs[0] = smoothdata(args[0], method, k, 0, ctx.engine->resource());
}

void hampel_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "hampel");
    int k = 3;
    if (args.size() >= 2) {
        k = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "k", "hampel"), 2, "k", "hampel");
    }
    const double nsigmas = (args.size() >= 3) ? args[2].toScalar() : 3.0;
    auto *mr = ctx.engine->resource();
    const Value &x = args[0];

    if (nargout < 2) {
        outs[0] = hampel(x, k, nsigmas, mr);
        return;
    }

    // [y, i, xmedian, xsigma] = hampel(...). i is a logical mask of the
    // replaced (outlier) points; xmedian/xsigma are the local median and
    // 1.4826·MAD estimate at every sample. vs MATLAB R2025b.
    hampelValidate(x, k, nsigmas);
    const std::size_t n = x.numel();
    auto y    = createLike(x, ValueType::DOUBLE,  mr);
    auto imask = createLike(x, ValueType::LOGICAL, mr);
    auto xmed = createLike(x, ValueType::DOUBLE,  mr);
    auto xsig = createLike(x, ValueType::DOUBLE,  mr);
    if (n > 0)
        hampelCore(x.doubleData(), n, k, nsigmas, y.doubleDataMut(),
                   imask.logicalDataMut(), xmed.doubleDataMut(),
                   xsig.doubleDataMut(), mr);

    outs[0] = y;
    outs[1] = imask;
    if (nargout >= 3) outs[2] = xmed;
    if (nargout >= 4) outs[3] = xsig;
}

} // namespace detail

} // namespace numkit::stats
