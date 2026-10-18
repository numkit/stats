// toolboxes/stats/tests/mle_test.cpp
//
// Regression guard for mle (closed-form max-likelihood estimator).

#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class MleTest : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override {}
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

TEST_F(MleTest, NormalDefault)
{
    eval("x = [1.2; 0.8; 1.5; 0.9; 1.1; 1.3; 1.0; 0.7; 1.4; 1.0]; m = mle(x);");
    EXPECT_NEAR(evalScalar("m(1)"), 1.09, 1e-10);
    EXPECT_NEAR(evalScalar("m(2)"), 0.246779, 1e-5);
}

TEST_F(MleTest, NormalExplicit)
{
    eval("x = [1 2 3 4 5]; m = mle(x, 'distribution', 'normal');");
    EXPECT_NEAR(evalScalar("m(1)"), 3.0, 1e-12);
    // sigma = sqrt(mean((x - 3)^2)) = sqrt(2)
    EXPECT_NEAR(evalScalar("m(2)"), std::sqrt(2.0), 1e-12);
}

TEST_F(MleTest, Exponential)
{
    eval("xe = [0.5; 1.0; 1.5; 2.0; 2.5; 0.3; 0.8; 1.2; 1.7; 0.6]; "
         "m = mle(xe, 'distribution', 'exponential');");
    EXPECT_NEAR(evalScalar("m(1)"), 1.21, 1e-10);
}

TEST_F(MleTest, Poisson)
{
    eval("xp = [0; 1; 2; 3; 4; 5; 1; 2; 3; 2]; "
         "m = mle(xp, 'distribution', 'poisson');");
    EXPECT_NEAR(evalScalar("m(1)"), 2.3, 1e-12);
}

TEST_F(MleTest, Lognormal)
{
    eval("xl = exp([0.1; 0.2; 0.3; 0.4; 0.5]); "
         "m = mle(xl, 'distribution', 'lognormal');");
    EXPECT_NEAR(evalScalar("m(1)"), 0.3, 1e-12);
}

TEST_F(MleTest, ExponentialNegativeRejected)
{
    EXPECT_THROW(eval("mle([-1; 2; 3], 'distribution', 'exponential');"), std::exception);
}

TEST_F(MleTest, LognormalZeroRejected)
{
    EXPECT_THROW(eval("mle([0; 1; 2], 'distribution', 'lognormal');"), std::exception);
}

TEST_F(MleTest, PoissonNonIntegerRejected)
{
    EXPECT_THROW(eval("mle([0.5; 1.5; 2.5], 'distribution', 'poisson');"), std::exception);
}

TEST_F(MleTest, UnknownDistRejected)
{
    EXPECT_THROW(eval("mle([1 2 3], 'distribution', 'gamma');"), std::exception);
}

TEST_F(MleTest, CustomPdfLogpdfNloglf)
{
    eval("x = [1; 2; 3; 4; 5];");
    eval("pdf_fn = @(d, mu) (1/sqrt(2*pi)) * exp(-(d - mu).^2 / 2);");
    eval("[phat, pci] = mle(x, 'pdf', pdf_fn, 'start', 1.0);");
    EXPECT_NEAR(evalScalar("phat(1)"), 3.0, 1e-4);
    EXPECT_LT(evalScalar("pci(1, 1)"), 3.0);
    EXPECT_GT(evalScalar("pci(2, 1)"), 3.0);

    eval("logpdf_fn = @(d, mu) -0.5 * (d - mu).^2;");
    eval("phat2 = mle(x, 'logpdf', logpdf_fn, 'start', 0.5);");
    EXPECT_NEAR(evalScalar("phat2(1)"), 3.0, 1e-4);

    eval("nloglf_fn = @(p, d, cens, freq) 0.5 * sum((d - p(1)).^2);");
    eval("phat3 = mle(x, 'nloglf', nloglf_fn, 'start', 0.0);");
    EXPECT_NEAR(evalScalar("phat3(1)"), 3.0, 1e-4);
}
