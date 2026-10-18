// Multivariate-distribution family type/behavior contracts, probed
// against the reference engine R2025b (build/probe73.m for the
// rejection classes, build/probe74.m for the full-precision expected
// values): mvnpdf / mvnrnd / mvncdf / mvtrnd / mvtpdf / mnpdf. One
// TEST_F per probed contract branch.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/stats/mvdist/mvdist.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace {

class MvContractTest : public ::testing::Test {
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
        engine.eval("X2 = [1 2; 3 4; -1 0.5; 2 -2];");
        engine.eval("mu2 = [0.5 -1];");
        engine.eval("S2 = [4 1; 1 9];");
    }
};

// ── mvnpdf ────────────────────────────────────────────────────────────

TEST_F(MvContractTest, MvnpdfValuesAndForms)
{
    engine.eval("P1 = mvnpdf(X2, mu2, S2);");
    EXPECT_NEAR(at("P1", 0), 0.016258774841126453, 1e-12);
    EXPECT_NEAR(at("P1", 3), 0.018227263496876936, 1e-12);
    // Column X without Sigma → one d = n observation row (P02).
    EXPECT_NEAR(num("mvnpdf([1;2;3;4;5])"), 1.1519989328102472e-14, 1e-24);
    // Scalar mu (P03).
    engine.eval("P3 = mvnpdf(X2, 0.5, S2);");
    EXPECT_NEAR(at("P3", 0), 0.023404269606654655, 1e-12);
    // Diagonal Sigma row (P04).
    engine.eval("P4 = mvnpdf(X2, mu2, [4 9]);");
    EXPECT_NEAR(at("P4", 0), 0.015593727395206357, 1e-12);
    EXPECT_NEAR(at("P4", 2), 0.017670008075423876, 1e-12);
    // 3-D Sigma stack (P11) and the single-row replication (P12).
    engine.eval("P11 = mvnpdf([1 2; 3 -1], mu2, cat(3,S2,[2 0;0 5]));");
    EXPECT_NEAR(at("P11", 0), 0.016258774841126453, 1e-12);
    EXPECT_NEAR(at("P11", 1), 0.010549575963443355, 1e-12);
    engine.eval("P12 = mvnpdf([1 2], mu2, cat(3,S2,[2 0;0 5]));");
    EXPECT_NEAR(at("P12", 0), 0.016258774841126453, 1e-12);
    EXPECT_NEAR(at("P12", 1), 0.019222580699137948, 1e-12);
    // Semidefinite Sigma: no error, degenerate density → 0 (P15).
    engine.eval("P15 = mvnpdf(X2, mu2, [2 2; 2 2]);");
    EXPECT_NEAR(at("P15", 0), 0.0, 1e-12);
    // Single in → single out (P10).
    EXPECT_EQ(cls("mvnpdf(single(X2), single(mu2), single(S2))"), "single");
}

TEST_F(MvContractTest, MvnpdfGates)
{
    EXPECT_EQ(idOf("mvnpdf(X2, mu2, [4;9])"),
              "numkit:mvnpdf:badCovariance");          // column diag (P05)
    EXPECT_EQ(idOf("mvnpdf(X2, mu2, [4 -9])"),
              "numkit:mvnpdf:badDiagSigma");            // negative diag (P06)
    EXPECT_EQ(idOf("mvnpdf(X2, mu2, [4 0])"),
              "numkit:mvnpdf:badDiagSigma");            // zero diag (P14)
    EXPECT_EQ(idOf("mvnpdf(X2, mu2, [4 5; 5 4])"),
              "numkit:mvnpdf:badMatrixSigma");          // non-PSD (P07)
    EXPECT_EQ(idOf("mvnpdf(X2, [1 2 3], S2)"),
              "numkit:mvnpdf:colSizeMismatch");         // mu cols (P08/P09)
    EXPECT_EQ(idOf("mvnpdf(rand(2,2,2), mu2, S2)"),
              "numkit:mvnpdf:invalidData");             // 3-D X (P13)
}

// ── mvnrnd ────────────────────────────────────────────────────────────

TEST_F(MvContractTest, MvnrndFormsAndContract)
{
    // Basic draw: shapes + distributional sanity (seeded values are not
    // bit-identical — the randn cascade is a filed gap).
    engine.eval("rng(1); R1 = mvnrnd(mu2, S2, 4000);");
    EXPECT_EQ(num("size(R1,1)"), 4000);
    EXPECT_NEAR(num("abs(mean(R1(:,1)) - 0.5) < 0.15"), 1.0, 1e-12);
    // Diagonal Sigma row (R02).
    engine.eval("rng(2); R2 = mvnrnd(mu2, [4 9], 100);");
    EXPECT_EQ(num("size(R2,2)"), 2);
    // Zero variance forces the exact mean (R03).
    engine.eval("rng(3); R3 = mvnrnd(mu2, [4 0], 4);");
    EXPECT_NEAR(at("R3", 5), -1.0, 0.0);
    EXPECT_NEAR(at("R3", 7), -1.0, 0.0);
    // Column mu + matrix Sigma → row reinterpretation (R04).
    engine.eval("rng(4); R4 = mvnrnd([0.5;-1], S2);");
    EXPECT_EQ(num("size(R4,1)*10 + size(R4,2)"), 12);
    // cases vs rows(mu) contract (R06).
    EXPECT_EQ(idOf("mvnrnd([1 2;3 4], S2, 3)"),
              "numkit:mvnrnd:inputSizeMismatchMu");
    // Empty mu → TooFewInputs (R10).
    EXPECT_EQ(idOf("mvnrnd([], S2)"), "numkit:mvnrnd:tooFewInputs");
    // 3-D stack (R08) + pages mismatch (R11).
    engine.eval("rng(7); R8 = mvnrnd(mu2, cat(3,S2,[2 0;0 5]));");
    EXPECT_EQ(num("size(R8,1)"), 2);
    EXPECT_EQ(idOf("mvnrnd(mu2, cat(3,S2,[2 0;0 5]), 3)"),
              "numkit:mvnrnd:inputSizeMismatchSigmaDimension");
    // cases 0 / negative → 0×d empty (R12/R13).
    EXPECT_EQ(num("size(mvnrnd(mu2, S2, 0), 1)"), 0);
    EXPECT_EQ(num("size(mvnrnd(mu2, S2, -1), 2)"), 2);
    // T output = upper Cholesky factor (R09, probe74 RT*).
    engine.eval("[R9, T9] = mvnrnd(mu2, S2, 3);");
    EXPECT_NEAR(at("T9", 0), 2.0, 1e-12);
    EXPECT_NEAR(at("T9", 2), 0.5, 1e-12);
    EXPECT_NEAR(at("T9", 3), 2.9580398915498081, 1e-9);
    // Diag T (probe74 RTd*).
    engine.eval("[~, Td] = mvnrnd(mu2, [4 9], 3);");
    EXPECT_NEAR(at("Td", 0), 2.0, 1e-12);
    EXPECT_NEAR(at("Td", 1), 0.0, 1e-12);
    EXPECT_NEAR(at("Td", 3), 3.0, 1e-12);
    // Single in → single out (R07).
    EXPECT_EQ(cls("mvnrnd(single(mu2), single(S2), 2)"), "single");
    // Rank-deficient PSD Sigma: T has shape r × d (1 × 2) with T'*T == Sigma.
    engine.eval("[Rpsd, Tpsd] = mvnrnd([0 0], [4 2; 2 1], 5);");
    EXPECT_EQ(num("size(Tpsd, 1)"), 1.0);
    EXPECT_EQ(num("size(Tpsd, 2)"), 2.0);
    EXPECT_NEAR(at("Tpsd", 0), 2.0, 1e-12);
    EXPECT_NEAR(at("Tpsd", 1), 1.0, 1e-12);
    EXPECT_NEAR(num("max(max(abs(Tpsd' * Tpsd - [4 2; 2 1])))"), 0.0, 1e-12);
}

// ── mvncdf ────────────────────────────────────────────────────────────

TEST_F(MvContractTest, MvncdfValuesAndForms)
{
    // Column X, no Sigma → row reinterpretation (C01).
    EXPECT_NEAR(num("mvncdf([0; 0.5; -1])"), 0.054852076188749425, 1e-9);
    // Bivariate closed form (C02).
    engine.eval("P2 = mvncdf(X2, mu2, S2);");
    EXPECT_NEAR(at("P2", 0), 0.51964095663612198, 1e-9);
    EXPECT_NEAR(at("P2", 1), 0.8551777425245003, 1e-9);
    // Box form (C03).
    engine.eval("P3 = mvncdf([-1 0; 0 1], [1 1], S2);");
    EXPECT_NEAR(at("P3", 0), 0.074261110388731316, 1e-9);
    EXPECT_NEAR(at("P3", 1), 0.17775972044209676, 1e-9);
    // Diagonal Sigma row (C05) + scalar mu (C07).
    engine.eval("P5 = mvncdf(X2, mu2, [4 9]);");
    EXPECT_NEAR(at("P5", 0), 0.50371842155132984, 1e-9);
    EXPECT_NEAR(at("P5", 3), 0.28571582739780865, 1e-9);
    engine.eval("P7 = mvncdf(X2, 0.5, S2);");
    EXPECT_NEAR(at("P7", 0), 0.43698700440802107, 1e-9);
    // d = 3 identity via Genz SOV quasi-Monte Carlo (< 1e-4 accuracy).
    EXPECT_NEAR(num("mvncdf(1:3, 0, eye(3))"), 0.82109415046357204, 1e-4);
    // d = 3 correlated one-sided + box with statset options struct.
    engine.eval("S3 = [1 0.5 0.25; 0.5 1 0.5; 0.25 0.5 1]; "
                "opt = struct('TolFun', 1e-6, 'MaxFunEvals', 25000); "
                "p3a = mvncdf([0 0 0], [0 0 0], S3, opt); "
                "p3b = mvncdf([-1 -1 -1], [1 1 1], [0 0 0], S3, opt);");
    // Exact orthant probability for equicorrelated/AR(1) 3-D normal at origin:
    // 1/8 + (arcsin(0.5) + arcsin(0.25) + arcsin(0.5)) / (4*pi) = 0.2283978223508305
    EXPECT_NEAR(num("p3a"), 0.2283978223508305, 2e-4);
    EXPECT_GT(num("p3b"), 0.35);
    EXPECT_LT(num("p3b"), 0.45);
    // Single in → single out (C09).
    EXPECT_EQ(cls("mvncdf(single(X2), single(mu2), single(S2))"), "single");
}

TEST_F(MvContractTest, MvncdfGates)
{
    EXPECT_EQ(idOf("mvncdf(X2, mu2, [4 5; 5 4])"),
              "numkit:mvncdf:badMatrixSigma");         // non-PSD (C06)
    EXPECT_EQ(idOf("mvncdf(X2, [1;2;3;4;5], S2)"),
              "numkit:mvncdf:inputSizeMismatch");       // mu mismatch (C08)
}

// ── mvtrnd / mvtpdf ───────────────────────────────────────────────────

TEST_F(MvContractTest, MvtrndContract)
{
    engine.eval("rng(11); T1 = mvtrnd([1 .5; .5 1], 3, 5000);");
    EXPECT_EQ(num("size(T1,1)"), 5000);
    // df vector per case (T02), cases default 1 (T07).
    engine.eval("rng(12); T2 = mvtrnd([1 .5; .5 1], [3; 5], 2);");
    EXPECT_EQ(num("size(T2,1)"), 2);
    engine.eval("rng(14); T7 = mvtrnd([1 .5; .5 1], 3);");
    EXPECT_EQ(num("size(T7,1)*10 + size(T7,2)"), 12);
    // Auto-normalization of a non-unit diagonal (T06): the scaled C
    // gives correlation-consistent samples.
    engine.eval("rng(13); T6 = mvtrnd([4 2; 2 9], 3, 4000); "
                "C6 = corr(T6); ok6 = abs(C6(1,2) - 1/3) < 0.05;");
    EXPECT_NEAR(num("ok6"), 1.0, 1e-12);
    // Gates (T03/T04/T05).
    EXPECT_EQ(idOf("mvtrnd([1 .5; .5 1], -3, 4)"), "numkit:mvtrnd:badDF");
    EXPECT_EQ(idOf("mvtrnd([1 2 3; 4 5 6], 3, 4)"),
              "numkit:mvtrnd:badCorrelationNotSquare");
    EXPECT_EQ(idOf("mvtrnd([1 2; 2 1], 3, 4)"),
              "numkit:mvtrnd:badCorrelationSymPos");
}

TEST_F(MvContractTest, MvtpdfValuesAndForms)
{
    engine.eval("F1 = mvtpdf([1 2; 0 -1], [1 .5; .5 1], 3);");
    EXPECT_NEAR(at("F1", 0), 0.022097723580933724, 1e-12);
    EXPECT_NEAR(at("F1", 1), 0.073288763208531504, 1e-12);
    // Column X reinterpretation (T09).
    EXPECT_NEAR(num("mvtpdf([1; 0.5], [1 .5; .5 1], 3)"),
                0.089524655489191127, 1e-12);
    // C auto-normalization (T10).
    EXPECT_NEAR(num("mvtpdf([1 2], [4 2; 2 9], 3)"),
                0.019419469332860109, 1e-12);
    // df vector (T11).
    engine.eval("F4 = mvtpdf([1 2; 0 -1], [1 .5; .5 1], [3; 5]);");
    EXPECT_NEAR(at("F4", 0), 0.022097723580933724, 1e-12);
    EXPECT_NEAR(at("F4", 1), 0.080347302968180323, 1e-12);
    // Gates (T12-T16).
    EXPECT_EQ(idOf("mvtpdf([1 2], [1 .5; .5 1], -3)"),
              "numkit:mvtpdf:invalidDF");
    EXPECT_EQ(idOf("mvtpdf([1 2; 0 -1], [1 .5; .5 1], [3 5 7])"),
              "numkit:mvtpdf:inputSizeMismatchDF");
    EXPECT_EQ(idOf("mvtpdf([1 2 3], eye(2), 3)"),
              "numkit:mvtpdf:inputSizeMismatchC");
    EXPECT_EQ(idOf("mvtpdf([1 2], [1 2;2 1], 3)"),
              "numkit:mvtpdf:badCorrelationSymPos");
    EXPECT_EQ(idOf("mvtpdf(rand(2,2,2), eye(2), 3)"),
              "numkit:mvtpdf:invalidData");
}

// ── mnpdf ─────────────────────────────────────────────────────────────

TEST_F(MvContractTest, MnpdfValuesAndSemantics)
{
    EXPECT_NEAR(num("mnpdf([2 1 0], [0.3 0.5 0.2])"),
                0.13499999999999993, 1e-12);
    // Column p summing to 1 transposes (M02).
    EXPECT_NEAR(num("mnpdf([2; 1; 0], [0.3; 0.5; 0.2])"),
                0.13499999999999993, 1e-12);
    // Negative x → 0; fractional x or bad p → NaN (M03-M05).
    EXPECT_NEAR(num("mnpdf([2 -1 0], [0.3 0.5 0.2])"), 0.0, 0.0);
    EXPECT_TRUE(std::isnan(num("mnpdf([2 1 0.5], [0.3 0.5 0.2])")));
    EXPECT_TRUE(std::isnan(num("mnpdf([2 1 0], [0.3 0.5 0.4])")));
    // Row replication both ways (M06/M07).
    engine.eval("M6 = mnpdf([2 1 0; 1 0 2], [0.3 0.5 0.2]);");
    EXPECT_NEAR(at("M6", 0), 0.13499999999999993, 1e-12);
    EXPECT_NEAR(at("M6", 1), 0.03599999999999999, 1e-12);
    engine.eval("M7 = mnpdf([2 1 0], [0.3 0.5 0.2; 0.1 0.8 0.1]);");
    EXPECT_NEAR(at("M7", 0), 0.13499999999999993, 1e-12);
    EXPECT_NEAR(at("M7", 1), 0.024000000000000007, 1e-12);
    // Single in → single out (M10).
    EXPECT_EQ(cls("mnpdf(single([2 1 0]), single([0.3 0.5 0.2]))"),
              "single");
    // Gates (M08/M11).
    EXPECT_EQ(idOf("mnpdf([2 1], [0.3 0.5 0.2])"),
              "numkit:mnpdf:colSizeMismatch");
    EXPECT_EQ(idOf("mnpdf(zeros(1,0), [0.3 0.5 0.2])"),
              "numkit:mnpdf:noCategories");
}

} // namespace
