// Correlation-family type/behavior contracts, probed against the
// reference engine R2025b (probe batteries build/probe71{,b}.m for the
// rejection classes, build/probe72.m for the full-precision expected
// values): nearcorr / corr / partialcorr. One TEST_F per probed
// contract branch.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace {

class CorrContractTest : public ::testing::Test {
public:
    numkit::StandardEngine engine;
    numkit::Value eval(const std::string &c) { return engine.eval(c); }
    double num(const std::string &c)
    {
        return eval(c).elemAsDouble(0);
    }
    double at(const std::string &c, std::size_t i)
    {
        return eval(c).elemAsDouble(i);
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
    void SetUp() override
    {
        // The probe battery's shared data (probe71b/probe72).
        engine.eval("X = [1 2 3; 4 5 6; 7 8 10; 2 4 5; 5 1 3];");
        engine.eval("Z = [1 2; 3 4; 5 6; 1 1; 2 2];");
        engine.eval("Y = [1 2; 3 5; 7 9; 2 3; 6 2];");
    }
};

// ── nearcorr ──────────────────────────────────────────────────────────

TEST_F(CorrContractTest, NearcorrInputGate)
{
    // One rejection class covers every gate clause (probe71 N02-N04,
    // N10); int8 fails the float requirement, logical fails isnumeric.
    EXPECT_EQ(idOf("nearcorr([1 0.5; 0.4 1])"), "numkit:nearcorr:invalidInputA");
    EXPECT_EQ(idOf("nearcorr([1 2; 2 1])"), "numkit:nearcorr:invalidInputA");
    EXPECT_EQ(idOf("nearcorr([0 0.5; 0.5 0])"), "numkit:nearcorr:invalidInputA");
    EXPECT_EQ(idOf("nearcorr(zeros(2))"), "numkit:nearcorr:invalidInputA");
    EXPECT_EQ(idOf("nearcorr(int8([1 0; 0 1]))"), "numkit:nearcorr:invalidInputA");
    EXPECT_NEAR(num("all(nearcorr(eye(2)) == eye(2))"), 1.0, 1e-12);  // valid passes
}

TEST_F(CorrContractTest, NearcorrOptionalInputs)
{
    // Bad NV values/names → one optional-input class (probe71 N05-N07).
    // NOTE: the scalar form single(1e-6) is promoted to double by the
    // engine's argument marshalling before the gate sees it — only the
    // array form preserves the class (noted in the family bug file).
    EXPECT_EQ(idOf("nearcorr(eye(2), 'Tolerance', single([1e-6]))"),
              "numkit:nearcorr:optionalInputError");
    EXPECT_EQ(idOf("nearcorr(eye(2), 'Tolerance', 0)"),
              "numkit:nearcorr:optionalInputError");
    EXPECT_EQ(idOf("nearcorr(eye(2), 'MaxIterations', 2.5)"),
              "numkit:nearcorr:optionalInputError");
    EXPECT_EQ(idOf("nearcorr(eye(2), 'Method', 'proj')"),
              "numkit:nearcorr:optionalInputError");  // exact match, no prefix
    EXPECT_EQ(idOf("nearcorr(eye(2), 'Bogus', 1)"),
              "numkit:nearcorr:optionalInputError");
    // Weights → voiced stub (filed: correlation-deferred-forms.md).
    EXPECT_EQ(idOf("nearcorr(eye(2), 'Weights', [1 1]')"),
              "numkit:nearcorr:weightsUnsupported");
    // Both Method values accepted and agree with the default result.
    engine.eval("A3 = [1 0.9 0.5; 0.9 1 0.3; 0.5 0.3 1];");
    engine.eval("Yn = nearcorr(A3); Yp = nearcorr(A3, 'Method', 'projection'); "
                "Yw = nearcorr(A3, 'Method', 'newton', 'Tolerance', 1e-10);");
    EXPECT_NEAR(at("Yn", 1), 0.9, 1e-9);
    EXPECT_NEAR(at("Yp", 1), 0.9, 1e-9);
    EXPECT_NEAR(at("Yw", 1), 0.9, 1e-9);
    // A non-PSD input is actually repaired: perturb then project back.
    engine.eval("B3 = [1 1 1; 1 1 1; 1 1 1] * 0.5 + eye(3) * 0.5; "
                "B3(1,2) = 0.95; B3(2,1) = 0.95; "
                "Rr = nearcorr(B3);");
    EXPECT_NEAR(at("Rr", 0), 1.0, 1e-9);
    EXPECT_NEAR(num("min(eig(Rr)) >= -1e-9"), 1.0, 1e-12);
}

TEST_F(CorrContractTest, NearcorrSingle)
{
    engine.eval("S1 = nearcorr(single([1 0.9; 0.9 1]));");
    EXPECT_EQ(cls("S1"), "single");
    EXPECT_NEAR(at("S1", 1), 0.9f, 1e-6);
}

// ── corr: Weights ─────────────────────────────────────────────────────

TEST_F(CorrContractTest, CorrWeightsContract)
{
    // Weighted Pearson (probe72 C03).
    engine.eval("RW = corr(X, Y, 'Weights', [1 1 1 1 2]');");
    EXPECT_NEAR(at("RW", 0), 0.9557790087219501, 1e-9);
    EXPECT_NEAR(at("RW", 3), 0.62236569297639743, 1e-9);
    EXPECT_NEAR(at("RW", 4), 0.95647780183741093, 1e-9);
    // Negative weight → nonnegative rejection (probe71b C04).
    EXPECT_EQ(idOf("corr(X, Y, 'Weights', [1 1 1 1 -2]')"),
              "numkit:corr:expectedNonnegative");
    // Wrong size → size rejection blaming Weights (probe71b C05).
    EXPECT_EQ(idOf("corr(X, Y, 'Weights', [1 1 1]')"),
              "numkit:corr:incorrectSize");
    EXPECT_EQ(idOf("corr(X, Y, 'Weights', ones(2, 3))"),
              "numkit:corr:incorrectSize");
    // Non-float weights → class rejection.
    EXPECT_EQ(idOf("corr(X, Y, 'Weights', int32([1 1 1 1 1]'))"),
              "numkit:corr:invalidWeight");
    // Weighted p-values are NaN by definition (probe72 CWpNaN).
    engine.eval("[rw, pw] = corr(X, Y, 'Weights', [1 1 1 1 2]');");
    EXPECT_TRUE(std::isnan(at("pw", 0)));
    EXPECT_TRUE(std::isnan(at("pw", 5)));
    // Zero-weight rows drop in EVERY Rows mode (probe72 CW0a/CW0b).
    engine.eval("W0 = [1 0 1 1 1]'; "
                "Ca = corr(X, Y, 'Rows', 'complete', 'Weights', W0); "
                "Cb = corr(X, Y, 'Rows', 'all',      'Weights', W0); "
                "Cp = corr(X, Y, 'Rows', 'pairwise', 'Weights', W0);");
    EXPECT_NEAR(at("Ca", 0), 0.98681073936895169, 1e-9);
    EXPECT_NEAR(at("Cb", 0), 0.98681073936895169, 1e-9);
    EXPECT_NEAR(at("Cp", 0), 0.98681073936895169, 1e-9);
    // Weights + rank type → voiced stub (filed).
    EXPECT_EQ(idOf("corr(X, Y, 'Type', 'Spearman', 'Weights', ones(5, 1))"),
              "numkit:corr:weightsRankUnsupported");
}

// ── corr: Tail ────────────────────────────────────────────────────────

TEST_F(CorrContractTest, CorrTailOneSidedPValues)
{
    // One-sided Pearson p-values (probe72 C08/C09); r is tail-invariant.
    engine.eval("[rl, pl] = corr(X, Y, 'Tail', 'left');");
    EXPECT_NEAR(at("rl", 0), 0.96281342346277299, 1e-9);
    EXPECT_NEAR(at("pl", 0), 0.99571998056184563, 1e-9);
    EXPECT_NEAR(at("pl", 3), 0.92895617375729223, 1e-9);
    engine.eval("[rr, pr] = corr(X, Y, 'Tail', 'right');");
    EXPECT_NEAR(at("pr", 0), 0.0042800194381543508, 1e-9);
    EXPECT_NEAR(at("pr", 3), 0.07104382624270783, 1e-9);
    // 'l' prefix and the 'ne' alias land on the same values (C15/CNE).
    engine.eval("[~, pl2] = corr(X, Y, 'Tail', 'l'); "
                "[~, pne] = corr(X, Y, 'Tail', 'ne');");
    EXPECT_NEAR(num("pl2(1) - pl(1)"), 0.0, 1e-12);
    EXPECT_NEAR(at("pne", 0), 0.0085600388763087017, 1e-9);
    // Bad tail value → its own class (probe71b pattern).
    EXPECT_EQ(idOf("corr(X, Y, 'Tail', 'bogus')"), "numkit:corr:unknownTail");
    // One-sided at |r| = 1: left→1 / right→0 for positive fit.
    engine.eval("[r1, p1l] = corr([1;2;3], [2;4;6], 'Tail', 'left'); "
                "[~,   p1r] = corr([1;2;3], [2;4;6], 'Tail', 'right');");
    EXPECT_NEAR(num("p1l"), 1.0, 1e-8);  // fp: r is 1 - O(1e-9)
    EXPECT_NEAR(num("p1r"), 0.0, 1e-8);
}

// ── corr: shapes / classes / n<2 ──────────────────────────────────────

TEST_F(CorrContractTest, CorrSmallNAndClasses)
{
    // n = 2: every column pair is an exact fit → r = 1, p NaN off the
    // diagonal and 1 on it (probe72 C06/C07).
    engine.eval("[r2, p2] = corr(X([1 2],:));");
    EXPECT_NEAR(at("r2", 0), 1.0, 1e-12);
    EXPECT_NEAR(at("r2", 4), 1.0, 1e-12);
    EXPECT_TRUE(std::isnan(at("p2", 1)));
    EXPECT_TRUE(std::isnan(at("p2", 3)));
    EXPECT_NEAR(at("p2", 0), 1.0, 1e-12);
    // Single input → single output (C13; double-compute cast, ~1e-7).
    engine.eval("S3 = corr(single(X), Y);");
    EXPECT_EQ(cls("S3"), "single");
    EXPECT_NEAR(at("S3", 0), 0.96281337738037109, 1e-6);
    // Complex Pearson returns complex rho (C10)...
    engine.eval("CX = corr(complex(X, X), Y);");
    EXPECT_EQ(cls("CX"), "double");
    EXPECT_NEAR(at("CX", 0), 0.68081190074796161, 1e-9);
    // ...but errors on p-values (C11) and on rank types.
    EXPECT_EQ(idOf("[cp1, cp2] = corr(complex(X, X), Y);"),
              "numkit:corr:pValComplexData");
    EXPECT_EQ(idOf("corr(complex(X, X), Y, 'Type', 'Spearman');"),
              "numkit:corr:rankCorComplexData");
    // Bad Rows value → its own class (C14).
    EXPECT_EQ(idOf("corr(X, Y, 'Rows', 'bogus')"), "numkit:corr:unknownRows");
}

TEST_F(CorrContractTest, CorrSpearmanPValues)
{
    // Exact Spearman p-values with ties in the cross form (probe72 C12).
    engine.eval("[rs, ps] = corr(X, Y, 'Type', 'Spearman');");
    EXPECT_NEAR(at("rs", 0), 1.0, 1e-12);
    EXPECT_NEAR(at("rs", 3), 0.56428809364683474, 1e-9);
    EXPECT_NEAR(at("ps", 0), 0.016666666666666666, 1e-9);
    EXPECT_NEAR(at("ps", 3), 0.4, 1e-9);
    EXPECT_NEAR(at("ps", 4), 0.033333333333333333, 1e-9);
    // 'sp' prefix works; unknown type value rejects.
    engine.eval("rs2 = corr(X, Y, 'Type', 'sp');");
    EXPECT_NEAR(num("rs2(2) - rs(2)"), 0.0, 1e-12);
    EXPECT_EQ(idOf("corr(X, Y, 'Type', 'bogus')"), "numkit:corr:unknownType");
}

// ── partialcorr ───────────────────────────────────────────────────────

TEST_F(CorrContractTest, PartialcorrGates)
{
    // Numeric-matrix gate: logical fails isnumeric (P03).
    EXPECT_EQ(idOf("partialcorr(logical(X))"),
              "numkit:partialcorr:inputsMustBeMatrices");
    // Integer-only inputs fail the float requirement (P02).
    EXPECT_EQ(idOf("partialcorr(int8(X))"),
              "numkit:partialcorr:inputsMustBeFloats");
    // Row mismatches (P04/P05).
    EXPECT_EQ(idOf("partialcorr(X, [1 2 3])"),
              "numkit:partialcorr:inputSizeMismatch");
    EXPECT_EQ(idOf("partialcorr(X, Z, [1 2])"),
              "numkit:partialcorr:inputSizeMismatch");
    // Kendall is a parity error (P06).
    EXPECT_EQ(idOf("partialcorr(X, Z, 'Type', 'kendall')"),
              "numkit:partialcorr:kendall");
}

TEST_F(CorrContractTest, PartialcorrValuesAndOptions)
{
    // combinedXZ default (P01).
    engine.eval("R1 = partialcorr(X);");
    EXPECT_NEAR(at("R1", 1), -0.89588471005352044, 1e-9);
    EXPECT_NEAR(at("R1", 2), 0.92309308324574424, 1e-9);
    EXPECT_NEAR(at("R1", 5), 0.99481826471109192, 1e-9);
    // Spearman ranks before residualising; 'sp' prefix matches (P07/P08).
    engine.eval("R7 = partialcorr(X, Z, 'Type', 'spearman'); "
                "R8 = partialcorr(X, Z, 'Type', 'sp');");
    EXPECT_NEAR(at("R7", 1), -0.53582588123382013, 1e-9);
    EXPECT_NEAR(at("R7", 5), 1.0, 1e-12);
    EXPECT_NEAR(num("R8(2) - R7(2)"), 0.0, 1e-12);
    // Rows 'comp' prefix (P09) and Tail parses (P11/P12 — inert while
    // the p-value output stays a filed missing output).
    engine.eval("R9 = partialcorr(X, Z, 'Rows', 'comp'); "
                "R11 = partialcorr(X, Z, 'Tail', 'left'); "
                "R12 = partialcorr(X, Z, 'Tail', 'lt');");
    EXPECT_NEAR(at("R9", 1), -1.0, 1e-12);
    EXPECT_NEAR(at("R9", 2), -0.98764038293883094, 1e-9);
    EXPECT_NEAR(num("R11(2) - R9(2)"), 0.0, 1e-12);
    EXPECT_NEAR(num("R12(3) - R9(3)"), 0.0, 1e-12);
    // Unknown option values / names (P10/P13).
    EXPECT_EQ(idOf("partialcorr(X, Z, 'Rows', 'bogus')"),
              "numkit:partialcorr:unknownRows");
    EXPECT_EQ(idOf("partialcorr(X, Z, 'Type', 'bogus')"),
              "numkit:partialcorr:unknownType");
    EXPECT_EQ(idOf("partialcorr(X, Z, 'Tail', 'bogus')"),
              "numkit:partialcorr:unknownTail");
    EXPECT_EQ(idOf("partialcorr(X, Z, 'Foo', 1)"),
              "numkit:partialcorr:badParamName");
    // Single input → single output (P14; double-compute cast, ~1e-6).
    engine.eval("R14 = partialcorr(single(X), Z);");
    EXPECT_EQ(cls("R14"), "single");
    EXPECT_NEAR(at("R14", 2), -0.98764038293883094, 1e-6);
    // combinedXZ 'pairwise' == 'complete' (P15 == P01).
    engine.eval("R15 = partialcorr(X, 'Rows', 'pairwise');");
    EXPECT_NEAR(num("R15(2) - R1(2)"), 0.0, 1e-12);
    // (X, Y, Z) cross form — pearson (PC1) and spearman (PC2).
    engine.eval("RC1 = partialcorr(X, Y, Z); "
                "RC2 = partialcorr(X, Y, Z, 'Type', 'Spearman');");
    EXPECT_NEAR(at("RC1", 0), 0.90289389814326915, 1e-9);
    EXPECT_NEAR(at("RC1", 3), -0.97737564830250234, 1e-9);
    EXPECT_NEAR(at("RC2", 0), 1.0, 1e-12);
    EXPECT_NEAR(at("RC2", 3), -0.53582588123382058, 1e-9);
}

} // namespace
