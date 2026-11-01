// toolboxes/stats/tests/vartest_extras_test.cpp
// — Name-Value parsing
// for Alpha/Tail.

#include <numkit/core/standard_engine.hpp>
#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class VartestExtrasTest : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override
    {
                engine.eval("x = [1.2 2.4 3.1 4.5 5.0]';");
        engine.eval("y = [0.8 1.9 2.7 4.0 4.5]';");
    }
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

TEST_F(VartestExtrasTest, AlphaNVChangesCI)
{
    eval("[~,~,c95] = vartest(x, 1, 'Alpha', 0.05); [~,~,c99] = vartest(x, 1, 'Alpha', 0.01);");
    EXPECT_GT(evalScalar("(c99(2) - c99(1))"),
              evalScalar("(c95(2) - c95(1))"));
}

TEST_F(VartestExtrasTest, TailNVRecognised)
{
    EXPECT_NO_THROW(eval("vartest(x, 1, 'Tail', 'right');"));
    EXPECT_NO_THROW(eval("vartest(x, 1, 'Tail', 'left');"));
}

TEST_F(VartestExtrasTest, Dim1AndDim2)
{
    eval("M = [x, y];");
    eval("[h1, p1, ci1, st1] = vartest(M, 1, 'Dim', 1);");
    eval("[hx, px, cix, stx] = vartest(x, 1);");
    eval("[hy, py, ciy, sty] = vartest(y, 1);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 1)")), 1);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 2)")), 2);
    EXPECT_NEAR(evalScalar("p1(1)"), evalScalar("px"), 1e-12);
    EXPECT_NEAR(evalScalar("p1(2)"), evalScalar("py"), 1e-12);
    EXPECT_NEAR(evalScalar("st1.chisqstat(1)"), evalScalar("stx.chisqstat"), 1e-12);
    EXPECT_NEAR(evalScalar("st1.chisqstat(2)"), evalScalar("sty.chisqstat"), 1e-12);

    eval("[h2, p2, ci2, st2] = vartest(M', 1, 'Dim', 2);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 1)")), 2);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 2)")), 1);
    EXPECT_NEAR(evalScalar("p2(1)"), evalScalar("px"), 1e-12);
    EXPECT_NEAR(evalScalar("p2(2)"), evalScalar("py"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(1, 1)"), evalScalar("cix(1)"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(2, 2)"), evalScalar("ciy(2)"), 1e-12);
}

TEST_F(VartestExtrasTest, Vartest2AlphaNVRecognised)
{
    EXPECT_NO_THROW(eval("vartest2(x, y, 'Alpha', 0.01);"));
}

TEST_F(VartestExtrasTest, Vartest2TailNV)
{
    EXPECT_NO_THROW(eval("vartest2(x, y, 'Tail', 'right');"));
}

TEST_F(VartestExtrasTest, Vartest2Dim1AndDim2)
{
    eval("X = [x, x * 1.5]; Y = [y, y * 1.2];");
    eval("[h1, p1, ci1, st1] = vartest2(X, Y, 'Dim', 1);");
    eval("[hc1, pc1, cic1, stc1] = vartest2(X(:,1), Y(:,1));");
    eval("[hc2, pc2, cic2, stc2] = vartest2(X(:,2), Y(:,2));");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 1)")), 1);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 2)")), 2);
    EXPECT_NEAR(evalScalar("p1(1)"), evalScalar("pc1"), 1e-12);
    EXPECT_NEAR(evalScalar("p1(2)"), evalScalar("pc2"), 1e-12);
    EXPECT_NEAR(evalScalar("st1.fstat(1)"), evalScalar("stc1.fstat"), 1e-12);
    EXPECT_NEAR(evalScalar("st1.fstat(2)"), evalScalar("stc2.fstat"), 1e-12);

    eval("[h2, p2, ci2, st2] = vartest2(X', Y', 'Dim', 2);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 1)")), 2);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 2)")), 1);
    EXPECT_NEAR(evalScalar("p2(1)"), evalScalar("pc1"), 1e-12);
    EXPECT_NEAR(evalScalar("p2(2)"), evalScalar("pc2"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(1, 1)"), evalScalar("cic1(1)"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(2, 2)"), evalScalar("cic2(2)"), 1e-12);
}

TEST_F(VartestExtrasTest, BasicValuesUnchanged)
{
    // Sanity: existing positional behaviour must not regress.
    eval("[h, p, ~, T] = vartest(x, 1);");
    EXPECT_NEAR(evalScalar("p"), 0.0966, 1e-3);
}
