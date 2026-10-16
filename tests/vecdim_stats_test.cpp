// toolboxes/stats/tests/vecdim_stats_test.cpp
//
// Dual-engine guards for the vecdim form of the stats reductions
// (median/mode/var/std/iqr/bounds/moment/trimmean/mape/rmse) — the
// shared numkit::vecdim parser + collapse path. Reference values probed
// against R2025b on 2026-09-17 (bugs/opened/stats/vecdim-rank-model.md,
// vecdim-partial.md, trimmean-vecdim.md — now fixed).

#include "dual_engine_fixture.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

class VecdimStatsTest : public m_test::DualEngineTest
{
};

INSTANTIATE_DUAL(VecdimStatsTest);

// ── rank-model fixes: vectors / scalars / rank-4+ ───────────────────

TEST_P(VecdimStatsTest, MedianVecdimVectorAndScalar)
{
    EXPECT_DOUBLE_EQ(evalScalar("median(1:5, [1 2]);"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("median(3, [1 2]);"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("median(ones(2,3,4,5), [1 2 3 4]);"), 1.0);
}

TEST_P(VecdimStatsTest, MedianVecdimPartialValuesAndShape)
{
    // Per-j median of the (i,k) blocks: {1,2,7,8,13,14,19,20} etc.
    eval("m = median(reshape(1:24, [2 3 4]), [1 3]);");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(m);")), 3);
    EXPECT_DOUBLE_EQ(evalScalar("m(1);"), 10.5);
    EXPECT_DOUBLE_EQ(evalScalar("m(2);"), 12.5);
    EXPECT_DOUBLE_EQ(evalScalar("m(3);"), 14.5);
    // Output shape: listed dims → 1 at their positions, trailing 1s
    // trimmed (probed: median(ones(2,3,4),[2 3]) is 2x1).
    eval("s = size(median(ones(2,3,4), [2 3]));");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(s);")), 2);
    EXPECT_DOUBLE_EQ(evalScalar("s(1);"), 2.0);
    EXPECT_DOUBLE_EQ(evalScalar("s(2);"), 1.0);
}

TEST_P(VecdimStatsTest, IqrBoundsVecdim)
{
    EXPECT_DOUBLE_EQ(evalScalar("iqr(1:5, [1 2]);"), 2.5);
    eval("[lo, hi] = bounds(1:5, [1 2]);");
    EXPECT_DOUBLE_EQ(evalScalar("lo;"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("hi;"), 5.0);
    // Partial: bounds over dims [2 3] of 2x3x4 keeps dim 1 (2x1 outs).
    eval("[lo2, hi2] = bounds(ones(2,3,4), [2 3]);");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(lo2);")), 2);
    EXPECT_DOUBLE_EQ(evalScalar("all(lo2 == 1 & hi2 == 1);"), 1.0);
}

TEST_P(VecdimStatsTest, MedianVecdimOmitnan)
{
    // NaN handling carries over through the same kernel.
    EXPECT_DOUBLE_EQ(evalScalar(
        "median([1 NaN 3; 4 5 6], [1 2], 'omitnan');"), 4.0);
}

TEST_P(VecdimStatsTest, VarStdVecdim)
{
    // Sample variance of the j-blocks: 362/7 = 51.714285714285715.
    eval("v = var(reshape(1:24, [2 3 4]), 0, [1 3]);");
    EXPECT_NEAR(evalScalar("v(1);"), 362.0 / 7.0, 1e-12);
    EXPECT_NEAR(evalScalar("v(3);"), 362.0 / 7.0, 1e-12);
    // moment uses population normalization: 362/8 = 45.25 (probed).
    eval("mo = moment(reshape(1:24, [2 3 4]), 2, [1 3]);");
    EXPECT_DOUBLE_EQ(evalScalar("mo(1);"), 45.25);
    // std = sqrt(var) along the same set.
    eval("sd = std(reshape(1:24, [2 3 4]), 0, [1 3]);");
    EXPECT_NEAR(evalScalar("sd(1);"), sqrt(362.0 / 7.0), 1e-12);
}

TEST_P(VecdimStatsTest, VarWeightedVecdimDeferred)
{
    // Weight-vector reductions stay 2-D-only; a dim-set with weights
    // hits the explicit N-D deferral.
    EXPECT_THROW(eval("var(reshape(1:24, [2 3 4]), ones(1,8), [1 3]);"),
                 std::exception);
}

TEST_P(VecdimStatsTest, ModeVecdimPartial)
{
    eval("[v, c] = mode(ones(2,3,4), [1 3]);");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(v);")), 3);
    EXPECT_DOUBLE_EQ(evalScalar("all(v == 1 & c == 8);"), 1.0);
}

TEST_P(VecdimStatsTest, TrimmeanVecdim)
{
    EXPECT_DOUBLE_EQ(evalScalar("trimmean(1:5, 10, [1 2]);"), 3.0);
    // Partial on 3-D keeps the non-listed dim.
    eval("t = trimmean(reshape(1:24, [2 3 4]), 10, [1 3]);");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(t);")), 3);
}

TEST_P(VecdimStatsTest, MapeRmseVecdim)
{
    // mape of A vs 2A is 50 for every slice (probed).
    eval("A = reshape(1:24, [2 3 4]);");
    eval("mp = mape(A, 2*A, [1 3]);");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(mp);")), 3);
    EXPECT_DOUBLE_EQ(evalScalar("mp(1);"), 50.0);
    // Per-slice rmse of A vs 2A == sqrt(mean of the j=1 block squares):
    // (1+4+49+64+169+196+361+400)/8 = 155.5.
    eval("rr = rmse(A, 2*A, [1 3]);");
    EXPECT_NEAR(evalScalar("rr(1);"), std::sqrt(155.5), 1e-12);
}
