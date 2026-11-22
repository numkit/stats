// toolboxes/stats/tests/ksdensity_test.cpp
// ksdensity.

#include <numkit/core/standard_engine.hpp>
#include <keel/math.h>
#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace numkit;

class KsdensityTest : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override {
                engine.eval("x = [1 2 2.5 3 3.5 4 5 6 7 9]';");
        engine.eval("pts = (0:0.5:10)';");
    }
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

// 2026-05-08 — gap closure: bandwidth uses MAD/0.6745 with IQR
// fallback (was just IQR), matching MATLAB R2025b exactly.
TEST_F(KsdensityTest, DefaultBandwidthMatchesMATLAB)
{
    eval("[f, xi, bw] = ksdensity(x, pts);");
    EXPECT_NEAR(evalScalar("bw"), 1.4862684537, 1e-6);
}

TEST_F(KsdensityTest, NormalKernelPdf)
{
    eval("f = ksdensity(x, pts);");
    EXPECT_NEAR(evalScalar("f(1)"), 0.044779, 1e-5);
    EXPECT_NEAR(evalScalar("f(5)"), 0.126313, 1e-5);
}

// gap closure: 4 kernel types with σ²=1 normalization.
TEST_F(KsdensityTest, BoxKernel)
{
    eval("f = ksdensity(x, pts, 'Kernel', 'box');");
    EXPECT_NEAR(evalScalar("f(1)"), 0.058268, 1e-5);
    EXPECT_NEAR(evalScalar("f(5)"), 0.116537, 1e-5);
}

TEST_F(KsdensityTest, TriangleKernel)
{
    eval("f = ksdensity(x, pts, 'Kernel', 'triangle');");
    EXPECT_NEAR(evalScalar("f(1)"), 0.046801, 1e-5);
    EXPECT_NEAR(evalScalar("f(5)"), 0.124372, 1e-5);
}

TEST_F(KsdensityTest, EpanechnikovKernel)
{
    eval("f = ksdensity(x, pts, 'Kernel', 'epanechnikov');");
    EXPECT_NEAR(evalScalar("f(1)"), 0.048894, 1e-5);
    EXPECT_NEAR(evalScalar("f(5)"), 0.122215, 1e-5);
}

// gap closure: Function modes.
TEST_F(KsdensityTest, FunctionCdf)
{
    eval("f = ksdensity(x, pts, 'Function', 'cdf');");
    EXPECT_NEAR(evalScalar("f(1)"), 0.042102, 1e-5);
    EXPECT_NEAR(evalScalar("f(5)"), 0.213963, 1e-5);
    EXPECT_NEAR(evalScalar("f(numel(f))"), 0.972376, 1e-4);
}

TEST_F(KsdensityTest, FunctionSurvivor)
{
    eval("f = ksdensity(x, pts, 'Function', 'survivor');");
    // s = 1 - cdf
    EXPECT_NEAR(evalScalar("f(1)"), 0.957898, 1e-5);
}

// gap closure: Weights.
TEST_F(KsdensityTest, Weights)
{
    eval("w = ones(10, 1); w(1:5) = 2;");
    eval("f = ksdensity(x, pts, 'Weights', w);");
    EXPECT_NEAR(evalScalar("f(1)"), 0.059159, 1e-5);
    EXPECT_NEAR(evalScalar("f(5)"), 0.158307, 1e-5);
}

// DEEP-PROBE c170: 'Function','icdf' (inverse CDF). The 2nd arg are
// PROBABILITIES; the result is the inverse of the smoothed CDF, solved via
// Newton's method (matching MATLAB R2025b to ~1e-10). nk previously
// threw "icdf is not yet supported".
TEST_F(KsdensityTest, FunctionIcdf)
{
    eval("ic = ksdensity(x, [0.1 0.25 0.5 0.75 0.9], 'Function', 'icdf');");
    EXPECT_NEAR(evalScalar("ic(1)"), 0.9191050959, 1e-6);
    EXPECT_NEAR(evalScalar("ic(2)"), 2.2760353116, 1e-6);
    EXPECT_NEAR(evalScalar("ic(3)"), 4.0173996405, 1e-6);
    EXPECT_NEAR(evalScalar("ic(4)"), 6.1567629614, 1e-6);
    EXPECT_NEAR(evalScalar("ic(5)"), 8.2055341632, 1e-6);
}

// icdf is the inverse of cdf: icdf(cdf(x0)) == x0.
TEST_F(KsdensityTest, IcdfRoundTrip)
{
    eval("c = ksdensity(x, 4.0, 'Function', 'cdf');");
    eval("rt = ksdensity(x, c, 'Function', 'icdf');");
    EXPECT_NEAR(evalScalar("rt"), 4.0, 1e-6);
}

// icdf boundary / out-of-range probabilities.
TEST_F(KsdensityTest, IcdfBoundaries)
{
    eval("e = ksdensity(x, [0 1 -0.1 1.1], 'Function', 'icdf');");
    EXPECT_TRUE(keel::is_inf(evalScalar("e(1)")) && evalScalar("e(1)") < 0.0); // p=0  -> -Inf
    EXPECT_TRUE(keel::is_inf(evalScalar("e(2)")) && evalScalar("e(2)") > 0.0); // p=1  -> +Inf
    EXPECT_TRUE(keel::is_nan(evalScalar("e(3)")));                             // p<0  -> NaN
    EXPECT_TRUE(keel::is_nan(evalScalar("e(4)")));                             // p>1  -> NaN
}

// gap closure: Censoring rejected with clear error.
TEST_F(KsdensityTest, CensoringRejected)
{
    bool threw = false;
    try { eval("ksdensity(x, pts, 'Censoring', zeros(10, 1));"); }
    catch (const std::exception &) { threw = true; }
    EXPECT_TRUE(threw);
}

TEST_F(KsdensityTest, ZeroOutputsPlotsAndPreservesAns)
{
    eval("42; ksdensity(x);");
    ASSERT_NE(engine.getVariable("ans"), nullptr);
    EXPECT_DOUBLE_EQ(engine.getVariable("ans")->toScalar(), 42.0);
}

TEST_F(KsdensityTest, OutputShapeMatchesPtsAndSupportBoundaryCorrection)
{
    // Column pts produces column f and xi.
    eval("pcol = [1; 3; 5; 7]; [fc, xic] = ksdensity(x, pcol);");
    ASSERT_NE(engine.getVariable("fc"), nullptr);
    EXPECT_EQ(engine.getVariable("fc")->dims().rows(), 4u);
    EXPECT_EQ(engine.getVariable("fc")->dims().cols(), 1u);
    EXPECT_EQ(engine.getVariable("xic")->dims().rows(), 4u);
    EXPECT_EQ(engine.getVariable("xic")->dims().cols(), 1u);

    // Positive support with reflection and log boundary correction.
    eval("xp = [0.5 1.0 1.2 1.8 2.5 3.1]'; q = [0 0.5 1.0 2.0];");
    eval("[fr, ~, bwr] = ksdensity(xp, q, 'Support', 'positive', 'BoundaryCorrection', 'reflection');");
    EXPECT_NEAR(evalScalar("bwr"), 0.823070064834, 1e-9);
    EXPECT_DOUBLE_EQ(evalScalar("fr(1)"), 0.0);
    EXPECT_NEAR(evalScalar("fr(2)"), 0.29747574025, 1e-9);
    EXPECT_NEAR(evalScalar("fr(3)"), 0.317322070687, 1e-9);
    EXPECT_NEAR(evalScalar("fr(4)"), 0.283966917851, 1e-9);

    eval("[fl, ~, bwl] = ksdensity(xp, q, 'Support', 'positive', 'BoundaryCorrection', 'log');");
    EXPECT_NEAR(evalScalar("bwl"), 0.502780981394, 1e-9);
    EXPECT_DOUBLE_EQ(evalScalar("fl(1)"), 0.0);
    EXPECT_NEAR(evalScalar("fl(2)"), 0.437072173053, 1e-9);
    EXPECT_NEAR(evalScalar("fl(3)"), 0.409619971529, 1e-9);
    EXPECT_NEAR(evalScalar("fl(4)"), 0.236336389543, 1e-9);

    // Two-sided [0 4] support with reflection.
    eval("[fb, ~, bwb] = ksdensity(xp, [0 0.5 1.0 2.0 4.0], 'Support', [0 4], 'BoundaryCorrection', 'reflection');");
    EXPECT_NEAR(evalScalar("bwb"), 0.823070064834, 1e-9);
    EXPECT_DOUBLE_EQ(evalScalar("fb(1)"), 0.0);
    EXPECT_NEAR(evalScalar("fb(2)"), 0.297475791351, 1e-9);
    EXPECT_NEAR(evalScalar("fb(3)"), 0.317323173229, 1e-9);
    EXPECT_NEAR(evalScalar("fb(4)"), 0.284139442736, 1e-9);
    EXPECT_DOUBLE_EQ(evalScalar("fb(5)"), 0.0);
}
