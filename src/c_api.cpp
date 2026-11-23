// src/c_api.cpp
//
// Self-contained C-ABI implementation for `numkit/stats`.
#include <keel/array.h>
#include <keel/memory.h>
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/stats/c_api.h>
#include <numkit/stats/library.hpp>
#include <numkit/ops/rng_context.hpp>
#include <numkit/sys/lite_context.hpp>
#include <numkit/value/c_api_bridge.hpp>

#include <exception>

struct numkit_stats_ctx_s {
    numkit::ops::RngContext rng;
    numkit::sys::LitePluginContext ctx;

    numkit_stats_ctx_s() : ctx(&rng)
    {
        numkit::StatsLibrary::install(ctx);
    }
};

namespace {

constexpr const char *kToolboxNs = "stats";

template <typename InvokeFn>
int runDispatch(const numkit_val *const *args,
                keel::usize nargs,
                keel::usize nargout,
                numkit_val **outs,
                numkit_error *err,
                InvokeFn &&doInvoke) noexcept
{
using numkit::Value;
    using numkit::c_abi::boxValue;
    using numkit::c_abi::clearError;
    using numkit::c_abi::setError;
    using numkit::c_abi::unboxValue;

    clearError(err);
    try {
        if (nargout > 0 && !outs) {
            setError(err, 1, "outs array must be non-null when nargout > 0");
            return 1;
        }

        Value inStack[8];
        keel::Array<Value> inHeap;
        Value *inPtr = inStack;
        if (nargs > 8) {
            inHeap.resize(nargs);
            inPtr = inHeap.data();
        }
        for (keel::usize i = 0; i < nargs; ++i) {
            inPtr[i] = args && args[i] ? unboxValue(args[i]) : Value::Empty;
        }

        keel::usize outSlots = nargout == 0 ? 1 : nargout;
        Value outStack[4];
        keel::Array<Value> outHeap;
        Value *outPtr = outStack;
        if (outSlots > 4) {
            outHeap.resize(outSlots);
            outPtr = outHeap.data();
        }

        doInvoke(keel::ArrayRef<Value>(inPtr, nargs), nargout, keel::MutArrayRef<Value>(outPtr, outSlots));

        for (keel::usize i = 0; i < nargout; ++i) {
            outs[i] = boxValue(keel::move(outPtr[i]));
        }
        return 0;
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return 1;
    } catch (...) {
        setError(err, 1, "Unknown C++ exception in numkit_stats");
        return 1;
    }
}

} // namespace

extern "C" {

numkit_stats_ctx *numkit_stats_ctx_create(numkit_error *err)
{
    numkit::c_abi::clearError(err);
    try {
        return new numkit_stats_ctx_s();
    } catch (const std::exception &ex) {
        numkit::c_abi::setError(err, 1, ex.what());
        return nullptr;
    }
}

void numkit_stats_ctx_destroy(numkit_stats_ctx *ctx)
{
    delete ctx;
}

void numkit_stats_seed_rng(numkit_stats_ctx *ctx, keel::u64 seed)
{
    if (ctx) ctx->rng.seed(seed);
}

int numkit_stats_has_function(const numkit_stats_ctx *ctx, const char *name)
{
    if (!ctx || !name || !*name) return 0;
    return ctx->ctx.hasFunction(kToolboxNs, name) ? 1 : 0;
}

int numkit_stats_register_fn(numkit_stats_ctx *ctx,
                            const char *name,
                            numkit_host_fn fn,
                            void *user_data,
                            numkit_error *err)
{
    numkit::c_abi::clearError(err);
    if (!ctx || !name || !*name || !fn) {
        numkit::c_abi::setError(err, 1, "Invalid arguments to numkit_stats_register_fn");
        return 1;
    }
    try {
        auto cb = [fn, user_data](keel::ArrayRef<numkit::Value> args,
                                  keel::usize nargout,
                                  keel::MutArrayRef<numkit::Value> outs,
                                  numkit::CallContext & /*c*/) {
            numkit::c_abi::invokeHostCallback(fn, user_data, args, nargout, outs);
        };
        ctx->ctx.registerFunction(kToolboxNs, name, keel::move(cb));
        return 0;
    } catch (const std::exception &ex) {
        numkit::c_abi::setError(err, 1, ex.what());
        return 1;
    }
}

const numkit_fn *numkit_stats_resolve(const numkit_stats_ctx *ctx, const char *name)
{
    if (!ctx || !name || !*name) return nullptr;
    const numkit::ExternalFunc *fn = ctx->ctx.resolve(kToolboxNs, name);
    return reinterpret_cast<const numkit_fn *>(fn);
}

int numkit_stats_invoke(numkit_stats_ctx *ctx,
                       const numkit_fn *fn,
                       const numkit_val *const *args,
                       keel::usize nargs,
                       keel::usize nargout,
                       numkit_val **outs,
                       numkit_error *err)
{
    if (!ctx || !fn) {
        numkit::c_abi::setError(err, 1, "Null context or pre-resolved function pointer");
        return 1;
    }
    const auto *extFn = reinterpret_cast<const numkit::ExternalFunc *>(fn);
    return runDispatch(args, nargs, nargout, outs, err,
                       [&](keel::ArrayRef<numkit::Value> inSpan,
                           keel::usize nOut,
                           keel::MutArrayRef<numkit::Value> outSpan) {
                           ctx->ctx.invokeResolved(extFn, inSpan, nOut, outSpan);
                       });
}

int numkit_stats_call(numkit_stats_ctx *ctx,
                     const char *name,
                     const numkit_val *const *args,
                     keel::usize nargs,
                     keel::usize nargout,
                     numkit_val **outs,
                     numkit_error *err)
{
    if (!ctx || !name || !*name) {
        numkit::c_abi::setError(err, 1, "Null context or empty function name");
        return 1;
    }
    return runDispatch(args, nargs, nargout, outs, err,
                       [&](keel::ArrayRef<numkit::Value> inSpan,
                           keel::usize nOut,
                           keel::MutArrayRef<numkit::Value> outSpan) {
                           ctx->ctx.invoke(kToolboxNs, name, inSpan, nOut, outSpan);
                       });
}

keel::usize numkit_stats_format(const numkit_stats_ctx *ctx,
                          const numkit_val *v,
                          char *buf,
                          keel::usize buf_cap)
{
    if (!ctx || !v) {
        if (buf && buf_cap > 0) buf[0] = '\0';
        return 0;
    }
    try {
        keel::String s = ctx->ctx.formatValue(numkit::c_abi::unboxValue(v));
        if (!buf || buf_cap == 0) return s.size();
        keel::usize n = keel::min(s.size(), buf_cap - 1);
        if (n > 0) keel::mem_copy(buf, s.data(), n);
        buf[n] = '\0';
        return s.size();
    } catch (...) {
        if (buf && buf_cap > 0) buf[0] = '\0';
        return 0;
    }
}

} // extern "C"
