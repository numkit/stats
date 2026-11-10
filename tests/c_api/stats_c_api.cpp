#include <numkit/stats/c_api.h>
#include <gtest/gtest.h>

TEST(NumkitStatsCApi, NormcdfAndMedian)
{
    const int64_t base = numkit_debug_live_values();
    numkit_error err{};
    numkit_stats_ctx *ctx = numkit_stats_ctx_create(&err);
    ASSERT_NE(ctx, nullptr) << err.message;

    const numkit_fn *normcdfFn = numkit_stats_resolve(ctx, "normcdf");
    ASSERT_NE(normcdfFn, nullptr);
    numkit_val *x = numkit_val_scalar(0.0);
    const numkit_val *args[1] = {x};
    numkit_val *p = nullptr;
    ASSERT_EQ(numkit_stats_invoke(ctx, normcdfFn, args, 1, 1, &p, &err), 0) << err.message;
    EXPECT_NEAR(numkit_val_as_double(p, &err), 0.5, 1e-12);

    numkit_val_release(p);
    numkit_val_release(x);
    numkit_stats_ctx_destroy(ctx);
    EXPECT_EQ(numkit_debug_live_values(), base);
}
