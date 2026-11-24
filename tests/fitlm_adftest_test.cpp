// stats/tests/fitlm_adftest_test.cpp
//
// Regression guards for fitlm, predict, and adftest after moving them from L2
// runtime/containers.cpp into the L3 stats plugin (consuming L0 TablePayload).

#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class FitlmAdftestTest : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override { engine.eval("rng(0);"); }
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

TEST_F(FitlmAdftestTest, FitlmMatrixAndPredict)
{
    eval("x = [1; 2; 3; 4; 5]; y = 2 + 3*x; mdl = fitlm(x, y);");
    EXPECT_NEAR(evalScalar("mdl.Coefficients.Estimate(1)"), 2.0, 1e-10);
    EXPECT_NEAR(evalScalar("mdl.Coefficients.Estimate(2)"), 3.0, 1e-10);
    EXPECT_NEAR(evalScalar("mdl.Rsquared.Ordinary"), 1.0, 1e-10);
    eval("yp = predict(mdl, [6; 10]);");
    EXPECT_NEAR(evalScalar("yp(1)"), 20.0, 1e-10);
    EXPECT_NEAR(evalScalar("yp(2)"), 32.0, 1e-10);
}

TEST_F(FitlmAdftestTest, FitlmTableFormulaAndPredict)
{
    eval("x1 = [1; 2; 3; 4; 5; 6]; x2 = [2; 1; 4; 3; 6; 5]; "
         "y = 1.5 + 2.0*x1 - 0.5*x2; "
         "T = table(x1, x2, y); "
         "mdl = fitlm(T, 'y ~ x1 + x2');");
    EXPECT_NEAR(evalScalar("mdl.Coefficients.Estimate(1)"), 1.5, 1e-10);
    EXPECT_NEAR(evalScalar("mdl.Coefficients.Estimate(2)"), 2.0, 1e-10);
    EXPECT_NEAR(evalScalar("mdl.Coefficients.Estimate(3)"), -0.5, 1e-10);
    eval("Tnew = table([10; 20], [4; 8], 'VariableNames', {'x1', 'x2'}); "
         "yp = predict(mdl, Tnew);");
    EXPECT_NEAR(evalScalar("yp(1)"), 1.5 + 20.0 - 2.0, 1e-10);
    EXPECT_NEAR(evalScalar("yp(2)"), 1.5 + 40.0 - 4.0, 1e-10);
}

TEST_F(FitlmAdftestTest, AdftestStationaryVsRandomWalk)
{
    eval("e = randn(200, 1); [h1, p1, stat1, cv1] = adftest(e);");
    EXPECT_EQ(static_cast<int>(evalScalar("h1")), 1);
    EXPECT_LT(evalScalar("p1"), 0.05);
    EXPECT_LT(evalScalar("stat1"), evalScalar("cv1"));

    eval("rw = cumsum(randn(200, 1)); [h2, p2] = adftest(rw, 'Model', 'ARD');");
    EXPECT_EQ(static_cast<int>(evalScalar("h2")), 0);
    EXPECT_GT(evalScalar("p2"), 0.05);
}
