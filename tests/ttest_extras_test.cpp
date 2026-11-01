// toolboxes/stats/tests/ttest_extras_test.cpp
// Reference values from MATLAB R2025b probe.

#include <numkit/core/standard_engine.hpp>
#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class TtestExtrasTest : public ::testing::Test
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

// ── ttest paired form ─────────────────────────────────────────────────

TEST_F(TtestExtrasTest, PairedFormDoesNotThrow)
{
    EXPECT_NO_THROW(eval("ttest(x, y);"));
}

TEST_F(TtestExtrasTest, PairedFormProducesValidTstat)
{
    // 4th output is the MATLAB stats struct {tstat, df, sd}.
    eval("[h, p, ci, st] = ttest(x, y);");
    EXPECT_GT(evalScalar("st.tstat"), 10.0);
    EXPECT_EQ(static_cast<int>(evalScalar("st.df")), 4);   // n-1 = 4
    EXPECT_GT(evalScalar("st.sd"), 0.0);
    EXPECT_EQ(static_cast<int>(evalScalar("h")), 1);
}

// ── ttest Name-Value Alpha ────────────────────────────────────────────

TEST_F(TtestExtrasTest, AlphaNVChangesCI)
{
    // 99% CI vs 95% CI must differ in width.
    eval("[~,~,ci95] = ttest(x, 4, 'Alpha', 0.05); [~,~,ci99] = ttest(x, 4, 'Alpha', 0.01);");
    EXPECT_GT(evalScalar("(ci99(2) - ci99(1))"),
              evalScalar("(ci95(2) - ci95(1))"));
}

TEST_F(TtestExtrasTest, TailNVRecognised)
{
    EXPECT_NO_THROW(eval("ttest(x, 4, 'Tail', 'right');"));
    EXPECT_NO_THROW(eval("ttest(x, 4, 'Tail', 'left');"));
}

// ── ttest2 default = equal (pooled) ────────────────────────────────────

TEST_F(TtestExtrasTest, Ttest2DefaultIsEqualPooled)
{
    eval("[~,~,~,te] = ttest2(x, y); [~,~,~,td] = ttest2(x, y, 'Vartype', 'equal');");
    EXPECT_DOUBLE_EQ(evalScalar("te.tstat"), evalScalar("td.tstat"));
}

TEST_F(TtestExtrasTest, Ttest2EqualValueMatchesMATLAB)
{
    eval("[~,~,~,st] = ttest2(x, y);");
    EXPECT_NEAR(evalScalar("st.tstat"), 0.475466, 1e-5);
}

// The 4th output of the parametric tests is MATLAB's stats struct. vs R2025b.
// ttest/ttest2 -> {tstat,df,sd}; vartest -> {chisqstat,df};
// vartest2 -> {fstat,df1,df2}; ztest -> bare zval scalar (unchanged).
TEST_F(TtestExtrasTest, ParametricStatsStructs)
{
    eval("a = 1:10; b = [2 3 4 5 6 8 9 11 13 15];");
    eval("[~,~,~,s1] = ttest(a, 5);");
    EXPECT_NEAR(evalScalar("s1.tstat"), 0.5222328, 1e-6);
    EXPECT_DOUBLE_EQ(evalScalar("s1.df"), 9.0);
    EXPECT_NEAR(evalScalar("s1.sd"), 3.0276504, 1e-6);
    eval("[~,~,~,s2] = ttest2(a, b);");        // default 'equal' -> pooled sd
    EXPECT_NEAR(evalScalar("s2.tstat"), -1.2478312, 1e-6);
    EXPECT_DOUBLE_EQ(evalScalar("s2.df"), 18.0);
    EXPECT_NEAR(evalScalar("s2.sd"), 3.7631250, 1e-6);
    eval("[~,~,~,s4] = vartest(a, 5);");
    EXPECT_NEAR(evalScalar("s4.chisqstat"), 16.5, 1e-9);
    EXPECT_DOUBLE_EQ(evalScalar("s4.df"), 9.0);
    eval("[~,~,~,s5] = vartest2(a, b);");
    EXPECT_NEAR(evalScalar("s5.fstat"), 0.4785384, 1e-6);
    EXPECT_DOUBLE_EQ(evalScalar("s5.df1"), 9.0);
    EXPECT_DOUBLE_EQ(evalScalar("s5.df2"), 9.0);
    // ztest's 4th output stays a bare scalar (zval), not a struct.
    eval("[~,~,~,zv] = ztest(a, 5, 2);");
    EXPECT_EQ(static_cast<int>(evalScalar("isstruct(zv)")), 0);
    EXPECT_NEAR(evalScalar("zv"), 0.7905694, 1e-6);
}

TEST_F(TtestExtrasTest, TtestDim1AndDim2)
{
    eval("M = [x, y];");
    eval("[h1, p1, ci1, st1] = ttest(M, 2, 'Dim', 1);");
    eval("[hx, px, cix, stx] = ttest(x, 2);");
    eval("[hy, py, ciy, sty] = ttest(y, 2);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 1)")), 1);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 2)")), 2);
    EXPECT_DOUBLE_EQ(evalScalar("h1(1)"), evalScalar("hx"));
    EXPECT_DOUBLE_EQ(evalScalar("h1(2)"), evalScalar("hy"));
    EXPECT_NEAR(evalScalar("p1(1)"), evalScalar("px"), 1e-12);
    EXPECT_NEAR(evalScalar("p1(2)"), evalScalar("py"), 1e-12);
    EXPECT_NEAR(evalScalar("ci1(1, 1)"), evalScalar("cix(1)"), 1e-12);
    EXPECT_NEAR(evalScalar("ci1(2, 2)"), evalScalar("ciy(2)"), 1e-12);
    EXPECT_NEAR(evalScalar("st1.tstat(1)"), evalScalar("stx.tstat"), 1e-12);
    EXPECT_NEAR(evalScalar("st1.tstat(2)"), evalScalar("sty.tstat"), 1e-12);

    eval("[h2, p2, ci2, st2] = ttest(M', 2, 'Dim', 2);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 1)")), 2);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 2)")), 1);
    EXPECT_NEAR(evalScalar("p2(1)"), evalScalar("px"), 1e-12);
    EXPECT_NEAR(evalScalar("p2(2)"), evalScalar("py"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(1, 1)"), evalScalar("cix(1)"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(2, 2)"), evalScalar("ciy(2)"), 1e-12);
}

TEST_F(TtestExtrasTest, Ttest2Dim1AndDim2)
{
    eval("X = [x, x + 1]; Y = [y, y + 0.5];");
    eval("[h1, p1, ci1, st1] = ttest2(X, Y, 'Dim', 1);");
    eval("[hc1, pc1, cic1, stc1] = ttest2(X(:,1), Y(:,1));");
    eval("[hc2, pc2, cic2, stc2] = ttest2(X(:,2), Y(:,2));");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 1)")), 1);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h1, 2)")), 2);
    EXPECT_NEAR(evalScalar("p1(1)"), evalScalar("pc1"), 1e-12);
    EXPECT_NEAR(evalScalar("p1(2)"), evalScalar("pc2"), 1e-12);
    EXPECT_NEAR(evalScalar("st1.tstat(1)"), evalScalar("stc1.tstat"), 1e-12);
    EXPECT_NEAR(evalScalar("st1.tstat(2)"), evalScalar("stc2.tstat"), 1e-12);

    eval("[h2, p2, ci2, st2] = ttest2(X', Y', 'Dim', 2);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 1)")), 2);
    EXPECT_EQ(static_cast<int>(evalScalar("size(h2, 2)")), 1);
    EXPECT_NEAR(evalScalar("p2(1)"), evalScalar("pc1"), 1e-12);
    EXPECT_NEAR(evalScalar("p2(2)"), evalScalar("pc2"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(1, 1)"), evalScalar("cic1(1)"), 1e-12);
    EXPECT_NEAR(evalScalar("ci2(2, 2)"), evalScalar("cic2(2)"), 1e-12);
}
