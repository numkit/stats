// toolboxes/stats/tests/friedman_test.cpp
//
// Regression guard for friedman() — Friedman nonparametric two-way ANOVA by
// ranks. Expected p-values from MATLAB R2025b. bugs/stats/friedman.
// nk returns [p, Q, df] (statistic + df), not MATLAB's (tbl, stats); the
// primary p matches MATLAB exactly.

#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class FriedmanTest : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override {}
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

TEST_F(FriedmanTest, NoTies)
{
    // MATLAB: friedman([1 2 3;2 3 4;3 4 5;1 3 5], 1) = 0.0183156388887, Q=8, df=2.
    EXPECT_NEAR(evalScalar("friedman([1 2 3;2 3 4;3 4 5;1 3 5], 1)"),
                0.0183156388887, 1e-10);
    eval("[p, Q, df] = friedman([1 2 3;2 3 4;3 4 5;1 3 5], 1);");
    EXPECT_NEAR(evalScalar("Q"), 8.0, 1e-10);
    EXPECT_DOUBLE_EQ(evalScalar("df"), 2.0);
}

TEST_F(FriedmanTest, WithTies)
{
    // MATLAB: friedman([7 9 8;6 5 7;9 7 6;8 8 9;5 6 5], 1) = 0.846481724891.
    // Exercises the tie correction C = 1 - sum(t^3-t)/(n*(k^3-k)).
    EXPECT_NEAR(evalScalar("friedman([7 9 8;6 5 7;9 7 6;8 8 9;5 6 5], 1)"),
                0.846481724891, 1e-10);
}

TEST_F(FriedmanTest, DefaultRepsOmitted)
{
    // reps defaults to 1 when omitted.
    EXPECT_NEAR(evalScalar("friedman([1 2 3;2 3 4;3 4 5;1 3 5])"),
                0.0183156388887, 1e-10);
}

TEST_F(FriedmanTest, ReplicatedBlocks)
{
    // 3 blocks, reps = 2 replicates per cell, 3 columns.
    // Within each 2x3 block, col 1 has ranks {1,2}, col 2 has {3,4}, col 3 has {5,6} (no ties).
    // R = [9, 21, 33], Q = 12/(3*3*4*7)*(81+441+1089) - 3*3*7 = 96/7, df = 2.
    eval("X = [1 3 5; 2 4 6; 10 30 50; 20 40 60; 100 300 500; 200 400 600];");
    eval("[p, Q, df] = friedman(X, 2);");
    EXPECT_NEAR(evalScalar("Q"), 96.0 / 7.0, 1e-10);
    EXPECT_DOUBLE_EQ(evalScalar("df"), 2.0);
    EXPECT_NEAR(evalScalar("p"), std::exp(-48.0 / 7.0), 1e-10);

    // With ties inside each replicate cell: Q = (96/7) / (32/35) = 15.0.
    eval("Xt = [1 2 3; 1 2 3; 2 3 4; 2 3 4; 3 4 5; 3 4 5];");
    eval("[pt, Qt, dft] = friedman(Xt, 2);");
    EXPECT_NEAR(evalScalar("Qt"), 15.0, 1e-10);
    EXPECT_DOUBLE_EQ(evalScalar("dft"), 2.0);
    EXPECT_NEAR(evalScalar("pt"), std::exp(-7.5), 1e-10);
}
