/* include/numkit/stats/c_api.h
 *
 * Self-contained C-ABI for the `numkit/stats` toolbox.
 *
 * Supports both:
 *   1. Pre-resolved O(1) dispatch (`numkit_stats_resolve` + `numkit_stats_invoke`)
 *      with zero string hashing on the hot path.
 *   2. Dynamic name/method dispatch (`numkit_stats_call`) for MATLAB overloads,
 *      Name-Value pairs, and toolbox object methods.
 */
#ifndef NUMKIT_STATS_C_API_H
#define NUMKIT_STATS_C_API_H

#include <numkit/value/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(NUMKIT_PLUGIN_BUILD)
#    define NUMKIT_STATS_API __declspec(dllexport)
#  elif defined(NUMKIT_STATS_STATIC)
#    define NUMKIT_STATS_API
#  else
#    define NUMKIT_STATS_API __declspec(dllimport)
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define NUMKIT_STATS_API __attribute__((visibility("default")))
#  else
#    define NUMKIT_STATS_API
#  endif
#endif

/* Opaque isolated `stats` toolbox context */
typedef struct numkit_stats_ctx_s numkit_stats_ctx;

/* Create an isolated `stats` toolbox context (engine-free, zero interpreter overhead). */
NUMKIT_STATS_API numkit_stats_ctx *numkit_stats_ctx_create(numkit_error *err);

/* Destroy `stats` toolbox context and release its resources. */
NUMKIT_STATS_API void numkit_stats_ctx_destroy(numkit_stats_ctx *ctx);

/* Seed the `stats` context's random number stream (MATLAB `rng(seed)` semantics). */
NUMKIT_STATS_API void numkit_stats_seed_rng(numkit_stats_ctx *ctx, uint64_t seed);

/* Return 1 if function `name` is registered in `stats`, 0 otherwise. */
NUMKIT_STATS_API int numkit_stats_has_function(const numkit_stats_ctx *ctx, const char *name);

/* Register a host-language callback (`numkit_host_fn`) into this context. */
NUMKIT_STATS_API int numkit_stats_register_fn(numkit_stats_ctx *ctx,
                                            const char *name,
                                            numkit_host_fn fn,
                                            void *user_data,
                                            numkit_error *err);

/* Resolve function `name` once into an O(1) function pointer (`NULL` if not found). */
NUMKIT_STATS_API const numkit_fn *numkit_stats_resolve(const numkit_stats_ctx *ctx,
                                                     const char *name);

/* Invoke a pre-resolved function pointer directly with zero lookup overhead. */
NUMKIT_STATS_API int numkit_stats_invoke(numkit_stats_ctx *ctx,
                                       const numkit_fn *fn,
                                       const numkit_val *const *args,
                                       size_t nargs,
                                       size_t nargout,
                                       numkit_val **outs,
                                       numkit_error *err);

/* Dynamically invoke `stats` function or object method `name`. */
NUMKIT_STATS_API int numkit_stats_call(numkit_stats_ctx *ctx,
                                     const char *name,
                                     const numkit_val *const *args,
                                     size_t nargs,
                                     size_t nargout,
                                     numkit_val **outs,
                                     numkit_error *err);

/* Format a value or `stats` class object (e.g. `sym`, `tf`, `ss`) to UTF-8 text. */
NUMKIT_STATS_API size_t numkit_stats_format(const numkit_stats_ctx *ctx,
                                          const numkit_val *v,
                                          char *buf,
                                          size_t buf_cap);

#ifdef __cplusplus
}
#endif

#endif /* NUMKIT_STATS_C_API_H */
