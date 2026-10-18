// Regression-family type/behavior contracts, probed against the
// reference engine R2025b (build/probe76.m R/N/G batteries):
// ridge / nlparci / glmval. One TEST_F per probed contract branch.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/stats/regress/regress.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace {

class RegressionContractTest : public ::testing::Test {
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
        engine.eval("X = [1 2; 2 3; 3 5; 4 4; 5 6];");
        engine.eval("y = [2.1; 3.2; 4.9; 5.1; 6.8];");
    }
};

TEST_F(RegressionContractTest, RidgeFormsAndGates)
{
    engine.eval("b1 = ridge(y, X, 1);");
    EXPECT_NEAR(at("b1", 0), 0.86883841360772307, 1e-9);
    EXPECT_NEAR(at("b1", 1), 0.77848762331719956, 1e-9);
    // Multiple k values (R02).
    engine.eval("b2 = ridge(y, X, [0 1 10]);");
    EXPECT_NEAR(at("b2", 0), 1.0901536144264654, 1e-9);
    EXPECT_NEAR(at("b2", 4), 0.40855300539587963, 1e-9);
    EXPECT_EQ(num("size(b2,2)"), 3.0);
    // Unscaled flag 0 → p+1 rows (R03); flag 1 == default (R08).
    engine.eval("b3 = ridge(y, X, 1, 0);");
    EXPECT_NEAR(at("b3", 0), 0.80205980066445282, 1e-9);
    EXPECT_EQ(num("size(b3,1)"), 3.0);
    engine.eval("b8 = ridge(y, X, 1, 1); b8d = ridge(y, X, 1);");
    EXPECT_NEAR(num("b8(1) - b8d(1)"), 0.0, 1e-12);
    // Gates (R04/R05/R06).
    EXPECT_EQ(idOf("ridge(y, X, 1, 2)"), "numkit:ridge:badScalingFlag");
    EXPECT_EQ(idOf("ridge(y(1:3), X, 1)"), "numkit:ridge:inputSizeMismatch");
    EXPECT_EQ(idOf("ridge([X(:,1) X(:,1)], X, 1)"), "numkit:ridge:invalidData");
    // NaN row removal (R07).
    engine.eval("b7 = ridge([y; NaN], [X; 1 1], 1);");
    EXPECT_NEAR(at("b7", 0), 0.86883841360772307, 1e-9);
}

TEST_F(RegressionContractTest, NlparciFormsAndGates)
{
    engine.eval("J3 = [1 2; 1 3; 1 4];");
    engine.eval("n1 = nlparci([1;2], [0.1;-0.1;0.2], J3);");
    EXPECT_NEAR(at("n1", 0), -5.8425006577493708, 1e-9);
    EXPECT_NEAR(at("n1", 1), -0.20077921744268723, 1e-9);
    EXPECT_NEAR(at("n1", 2), 7.8425006577493708, 1e-9);
    // Alpha (N02) and the covar form ('covar' is a parseArgs PREFIX of
    // 'covariance', N03).
    engine.eval("n2 = nlparci([1;2], [0.1;-0.1;0.2], J3, 0.1);");
    EXPECT_NEAR(at("n2", 0), -2.400059245782014, 1e-9);
    EXPECT_NEAR(at("n2", 2), 4.400059245782014, 1e-9);
    engine.eval("n3 = nlparci([1;2], [0.1;-0.1;0.2], 'covar', eye(2));");
    EXPECT_NEAR(at("n3", 0), -11.706204736174696, 1e-9);
    EXPECT_NEAR(at("n3", 3), 14.706204736174696, 1e-9);
    // Gates (N04/N05/N06).
    EXPECT_EQ(idOf("nlparci(complex([1;2],[1;1]), [0.1;-0.1], "
                   "[1 2;1 3])"),
              "numkit:nlparci:complexParams");
    EXPECT_EQ(idOf("nlparci([1;2], [0.1;-0.1], [1 2;1 3], 1.5)"),
              "numkit:nlparci:badAlpha");
    EXPECT_EQ(idOf("nlparci([1;2], [0.1;-0.1;0.2], [1 2;1 3])"),
              "numkit:nlparci:inputSizeMismatch");
}

TEST_F(RegressionContractTest, GlmvalFormsAndGates)
{
    engine.eval("g1 = glmval([1; 2], [1; 2; 3], 'logit');");
    EXPECT_NEAR(at("g1", 0), 0.95257412682243336, 1e-9);
    EXPECT_NEAR(at("g1", 2), 0.9990889488055994, 1e-9);
    // Scalar beta with the constant on broadcasts over the
    // const-prepended row (V02: [2 2 4 6]).
    engine.eval("g2 = glmval(2, [1 2 3], 'identity');");
    EXPECT_EQ(num("numel(g2)"), 4.0);
    EXPECT_NEAR(at("g2", 0), 2.0, 1e-12);
    EXPECT_NEAR(at("g2", 3), 6.0, 1e-12);
    // Size NV (V03) and the legacy positional (V04: a non-string 4th
    // arg is consumed as STATS, so 10 becomes the confidence level and
    // N stays 1).
    engine.eval("g3 = glmval([1; 2], [1; 2; 3], 'logit', 'Size', 10);");
    EXPECT_NEAR(at("g3", 0), 9.5257412682243334, 1e-9);
    engine.eval("g4 = glmval([1; 2], [1; 2; 3], 'logit', 0.9, 10);");
    EXPECT_NEAR(at("g4", 0), 0.95257412682243336, 1e-9);
    // Gates (V05/V06/V07).
    EXPECT_EQ(idOf("glmval([1; 2], [1; 2; 3], 'bogus')"),
              "numkit:glmval:unrecognizedLink");
    EXPECT_EQ(idOf("glmval([1; 2], [1; 2; 3], 'logit', "
                   "'Confidence', .9, 'Alpha', .1)"),
              "numkit:glmval:argCombination");
    EXPECT_EQ(idOf("glmval([1; 2], [1; 2; 3], 'logit', "
                   "'Size', 2, 'BinomialSize', 3)"),
              "numkit:glmval:argCombination");
    EXPECT_EQ(idOf("[lo, hi] = glmval([1; 2], [1; 2; 3], 'logit');"),
              "numkit:glmval:needSTATS");
    // Wrong-argument-size classes.
    EXPECT_EQ(idOf("glmval([1; 2], [1; 2; 3], 'logit', "
                   "'Size', [1 2 3])"),
              "numkit:glmval:wrongArgSizeBinomialSize");
    EXPECT_EQ(idOf("glmval([1; 2], [1; 2; 3], 'logit', "
                   "'Offset', [1 2 3])"),
              "numkit:glmval:wrongArgSizeOffset");

    // Confidence bounds ([yhat, dylo, dyhi]) from glmfit stats and Simultaneous NV.
    engine.eval("x = (1:5)'; y = 2 + 3*x + [0.1; -0.1; 0.05; -0.05; 0];"
                "[b_fit, dev_fit, st_fit] = glmfit(x, y, 'normal');"
                "[yh, dlo, dhi] = glmval(b_fit, [2; 4], 'identity', st_fit);"
                "[yhs, dlos, dhis] = glmval(b_fit, [2; 4], 'identity', st_fit, 'Simultaneous', true);");
    EXPECT_NEAR(at("yh", 0), 8.0, 0.2);
    EXPECT_GT(at("dlo", 0), 0.0);
    EXPECT_NEAR(at("dlo", 0), at("dhi", 0), 1e-12);
    // Simultaneous bounds must be strictly wider than pointwise bounds (p = 2 > 1).
    EXPECT_GT(at("dlos", 0), at("dlo", 0));

    // Custom link struct with function handle Inverse.
    engine.eval("S = struct('Link', @(mu) log(mu), 'Derivative', @(mu) 1./mu, 'Inverse', @(eta) exp(eta));"
                "yc = glmval([0; 1], [1; 2], S);");
    EXPECT_NEAR(at("yc", 0), std::exp(1.0), 1e-12);
    EXPECT_NEAR(at("yc", 1), std::exp(2.0), 1e-12);
}

} // namespace

