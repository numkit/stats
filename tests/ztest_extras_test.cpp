// toolboxes/stats/tests/ztest_extras_test.cpp
// — Name-Value Alpha/Tail parsing.

#include <numkit/core/standard_engine.hpp>
#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class ZtestExtrasTest : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override
    {
                engine.eval("x = [1.2 2.4 3.1 4.5 5.0]';");
    }
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

TEST_F(ZtestExtrasTest, AlphaNVChangesCI)
{
    eval("[~,~,c95] = ztest(x, 3, 1, 'Alpha', 0.05); [~,~,c99] = ztest(x, 3, 1, 'Alpha', 0.01);");
    EXPECT_GT(evalScalar("(c99(2) - c99(1))"),
              evalScalar("(c95(2) - c95(1))"));
}

TEST_F(ZtestExtrasTest, TailNVRecognised)
{
    EXPECT_NO_THROW(eval("ztest(x, 3, 1, 'Tail', 'right');"));
    EXPECT_NO_THROW(eval("ztest(x, 3, 1, 'Tail', 'left');"));
}

TEST_F(ZtestExtrasTest, Dim1AndDim2)
{
    eval("M = [x, x + 1];");
    eval("[h1, p1, ci1, z1] = ztest(M, 3, 1, 'Dim', 1);");
    eval("[hc1, pc1, cic1, zc1] = ztest(M(:,1), 3, 1);");
    eval("[hc2, pc2, cic2, zc2] = ztest(M(:,2), 3, 1);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 1)")), 1);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 2)")), 2);
    EXPECT_NEAR(evalScalar("p1(1)"), evalScalar("pc1"), 1e-12);
    EXPECT_NEAR(evalScalar("p1(2)"), evalScalar("pc2"), 1e-12);
    EXPECT_NEAR(evalScalar("z1(1)"), evalScalar("zc1"), 1e-12);
    EXPECT_NEAR(evalScalar("z1(2)"), evalScalar("zc2"), 1e-12);

    eval("[h2, p2, ci2, z2] = ztest(M', 3, 1, 'Dim', 2);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 1)")), 2);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 2)")), 1);
    EXPECT_NEAR(evalScalar("p2(1)"), evalScalar("pc1"), 1e-12);
    EXPECT_NEAR(evalScalar("p2(2)"), evalScalar("pc2"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(1, 1)"), evalScalar("cic1(1)"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(2, 2)"), evalScalar("cic2(2)"), 1e-12);
}

TEST_F(ZtestExtrasTest, BasicValuesUnchanged)
{
    eval("[h, p, ~, z] = ztest(x, 3, 1);");
    // X mean = 3.24; std error = 1/sqrt(5) = 0.4472; z = (3.24-3)/0.4472 ≈ 0.5367
    EXPECT_NEAR(evalScalar("z"), 0.5367, 1e-3);
}
