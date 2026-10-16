// Sampling-family type/behavior contracts, probed against the
// reference engine R2025b (build/probe75.m S/D/L batteries):
// randsample / datasample / lhsnorm. RNG-dependent draw VALUES are not
// bit-identical (randi/randperm cascades pending — see
// bugs/opened/stats/sampling-deferred-forms.md); the contracts pin
// forms, classes, shapes and deterministic invariants.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/stats/resample/resample.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace {

class SamplingContractTest : public ::testing::Test {
public:
    numkit::StandardEngine engine;
    numkit::Value eval(const std::string &c) { return engine.eval(c); }
    double num(const std::string &c)
    {
        return eval(c).elemAsDouble(0);
    }
    std::string cls(const std::string &c)
    {
        return eval("class(" + c + ")").toString();
    }
    std::string idOf(const std::string &c)
    {
        try {
            engine.eval(c);
        } catch (const numkit::Error &e) {
            return e.identifier();
        }
        return "<no-error>";
    }
};

// ── randsample ────────────────────────────────────────────────────────

TEST_F(SamplingContractTest, RandsampleFormsAndClasses)
{
    // k > n without replacement (S01).
    EXPECT_EQ(idOf("randsample(10, 11)"),
              "numkit:randsample:sampleTooLarge");
    // Bad REPLACE (S02): the legal set is true/false/1/0/'true'/'false'.
    EXPECT_EQ(idOf("randsample(10, 5, 2)"),
              "numkit:randsample:badReplaceValue");
    EXPECT_EQ(num("randsample(10, 2, 'true') > 0"), 1.0);   // parses
    // Weight vector classes (S03/S04).
    EXPECT_EQ(idOf("randsample(10, 5, true, [1 2 3])"),
              "numkit:randsample:inputSizeMismatch");
    EXPECT_EQ(idOf("randsample(10, 5, true, [1 -2 3 4 5 6 7 8 9 10])"),
              "numkit:randsample:invalidWeights");
    // Population must be a vector (S05).
    EXPECT_EQ(idOf("randsample(zeros(2,2), 2)"),
              "numkit:randsample:badPopulation");
    // Empty population (S06/S07).
    EXPECT_EQ(idOf("randsample(0, 1, true)"),
              "numkit:randsample:emptyPopulation");
    EXPECT_EQ(num("numel(randsample(0, 0, true))"), 0.0);
    // Weighted without replacement (S08).
    EXPECT_EQ(idOf("randsample(10, 5, false, ones(1, 10))"),
              "numkit:randsample:noWeighting");
    // k = 0 → 0×1 (S09).
    EXPECT_EQ(num("numel(randsample(10, 0))"), 0.0);
    // Full-population permutation: sum invariant (S10).
    engine.eval("rng(1); y = randsample([10 20 30 40 50], 5);");
    EXPECT_NEAR(num("sum(y)"), 150.0, 1e-9);
    // Count form → column (S11); row population → row (S12).
    engine.eval("rng(2); c = randsample(10, 3, true);");
    EXPECT_EQ(num("size(c,1)"), 3.0);
    EXPECT_EQ(num("size(c,2)"), 1.0);
    engine.eval("rng(3); r = randsample(1:5, 3, false);");
    EXPECT_EQ(num("size(r,1)"), 1.0);
    EXPECT_EQ(num("size(r,2)"), 3.0);
    // Single population stays single.
    EXPECT_EQ(cls("randsample(single([1 2 3 4 5]), 2, false)"), "single");
}

// ── datasample ────────────────────────────────────────────────────────

TEST_F(SamplingContractTest, DatasampleFormsAndClasses)
{
    // k can exceed n WITH replacement (the default — D01: no error).
    EXPECT_EQ(num("numel(datasample(1:10, 11))"), 11.0);
    // Bad k (D02) and the logical-only REPLACE (D03).
    EXPECT_EQ(idOf("datasample(1:10, 2.5)"), "numkit:datasample:invalidK");
    EXPECT_EQ(idOf("datasample(1:10, 2, 3, 'Replace', 1)"),
              "numkit:datasample:invalidReplace");
    EXPECT_EQ(num("datasample(1:10, 2, 'Replace', true) > 0"), 1.0);
    // Weights classes (D04).
    EXPECT_EQ(idOf("datasample(1:10, 2, 'Weights', [1 2])"),
              "numkit:datasample:inputSizeMismatch");
    EXPECT_EQ(idOf("datasample(1:10, 2, 'Weights', ones(1,9))"),
              "numkit:datasample:inputSizeMismatch");
    // Empty data (D05).
    EXPECT_EQ(idOf("datasample(zeros(1,0), 1)"),
              "numkit:datasample:emptyData");
    // Too few positive weights without replacement (D06).
    EXPECT_EQ(idOf("datasample(1:10, 2, 'Replace', false, "
                   "'Weights', [1 0 0 0 0 0 0 0 0 0])"),
              "numkit:datasample:tooFewPosWeights");
    // Full permutation sum invariant (D07) and the k>n error without
    // replacement.
    engine.eval("rng(1); y = datasample(1:10, 10, 'Replace', false);");
    EXPECT_NEAR(num("sum(y)"), 55.0, 1e-9);
    EXPECT_EQ(idOf("datasample(1:10, 11, 'Replace', false)"),
              "numkit:datasample:sampleTooLarge");
    // Dimension defaults by shape (D08/D09): column data → rows,
    // row data → columns.
    engine.eval("y8 = datasample(ones(3,1), 2);");
    EXPECT_EQ(num("size(y8,1)"), 2.0);
    EXPECT_EQ(num("size(y8,2)"), 1.0);
    engine.eval("y9 = datasample([1 2 3], 2);");
    EXPECT_EQ(num("size(y9,1)"), 1.0);
    EXPECT_EQ(num("size(y9,2)"), 2.0);
    // Explicit dim positional (non-string 3rd arg).
    engine.eval("y2 = datasample([1 2 3; 4 5 6], 1, 2);");
    EXPECT_EQ(num("size(y2,1)"), 2.0);
    EXPECT_EQ(num("size(y2,2)"), 1.0);
    // Weighted with replacement draws only from positive-weight slots.
    engine.eval("rng(4); yw = datasample(1:10, 200, 'Replace', true, "
                "'Weights', [0 1 0 0 0 0 0 0 0 0]);");
    EXPECT_NEAR(num("all(yw == 2)"), 1.0, 1e-12);
}

// ── lhsnorm ───────────────────────────────────────────────────────────

TEST_F(SamplingContractTest, LhsnormContract)
{
    engine.eval("rng(1); L = lhsnorm([0 0], [1 .5; .5 1], 500);");
    EXPECT_EQ(num("size(L,1)"), 500.0);
    EXPECT_EQ(num("size(L,2)"), 2.0);
    // Rank-correlation structure survives the LHS transform (loose —
    // the exact values are RNG-bound).
    engine.eval("CL = corr(L);");
    EXPECT_NEAR(num("CL(1,2)"), 0.5, 0.15);
    engine.eval("rng(2); L2 = lhsnorm([0 0], eye(2), 3, 'off'); "
                "rng(3); L3 = lhsnorm([0 0], eye(2), 3, 7);");
    EXPECT_EQ(num("size(L2,1)"), 3.0);
    EXPECT_EQ(num("size(L3,2)"), 2.0);   // non-'on' → off branch, no error
    // Shape gate.
    EXPECT_THROW(eval("lhsnorm([0 0], eye(3), 4);"), std::exception);
}

} // namespace
