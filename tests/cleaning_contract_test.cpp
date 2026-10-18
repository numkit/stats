// Cleaning-family type/behavior contracts, probed against the
// reference engine R2025b (probe batteries build/probe70{,b,c,d}.m):
// detrend / isoutlier / rmoutliers / fillmissing / rmmissing /
// standardizeMissing. One TEST_F per probed contract branch.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace {

class CleaningTest : public ::testing::Test {
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

// ── detrend ───────────────────────────────────────────────────────────

TEST_F(CleaningTest, DetrendFloatOnlyGate)
{
    EXPECT_EQ(idOf("detrend(int8(1:5))"), "numkit:detrend:invalidType");
    EXPECT_EQ(idOf("detrend(logical([1 0 1]))"), "numkit:detrend:invalidType");
    EXPECT_EQ(idOf("detrend('ab')"), "numkit:detrend:invalidType");
    // single is legal and stays single
    EXPECT_EQ(cls("detrend(single(1:5), 1)"), "single");
    // complex float is legal
    EXPECT_EQ(cls("detrend(complex(1:5, 1:5), 1)"), "double");
}

TEST_F(CleaningTest, DetrendDegreeForms)
{
    // 'cons'.. prefix matches constant; result = x - mean
    EXPECT_NEAR(num("detrend(1:5, 'cons')"), -2.0, 1e-12);
    // 'l' prefix matches linear; true is a logical degree (1)
    engine.eval("dl = detrend(1:5, 'l'); dt = detrend(1:5, true);");
    EXPECT_NEAR(num("dl(2)"), 0.0, 1e-12);
    EXPECT_NEAR(num("dt(2)"), 0.0, 1e-12);
    // fractional / negative degrees reject with the probed class
    EXPECT_EQ(idOf("detrend(1:5, 2.5)"), "numkit:detrend:invalidTrendInput");
    EXPECT_EQ(idOf("detrend(1:5, -1)"), "numkit:detrend:invalidTrendInput");
    // 'c' matches neither prefix at 4 chars -> NV-name -> odd count
    EXPECT_EQ(idOf("detrend(1:5, 'c')"), "numkit:detrend:keyWithoutValue");
}

TEST_F(CleaningTest, DetrendBreakpointsAndNaN)
{
    // order 0 + bp: subtract the mean of the first segment (bp 1-based)
    EXPECT_NEAR(num("detrend(1:5, 0, [2 4])"), -0.5, 1e-12);
    engine.eval("dbp = detrend(1:5, 0, [2 4]);");
    EXPECT_NEAR(num("dbp(5)"), 3.5, 1e-12);
    // fractional breakpoints are legal (reference does not round)
    EXPECT_NEAR(num("detrend(1:5, 1, [2.5])"), 0.0, 1e-12);
    // default NaN policy poisons the run; omitnan keeps the fit
    EXPECT_TRUE(std::isnan(num("detrend([1 2 NaN 4 5])")));
    engine.eval("don = detrend([1 2 NaN 4 5], 'omitnan');");
    EXPECT_NEAR(num("don(2)"), 0.0, 1e-12);
    EXPECT_TRUE(std::isnan(num("don(3)")));
    // Continuous=false: segment-wise fit
    EXPECT_NEAR(num("detrend(1:5, 1, [2 4], 'Continuous', false)(1)"), 0.0, 1e-12);
    // per-column operation
    engine.eval("dc = detrend([1:5]', 1);");
    EXPECT_NEAR(num("dc(3)"), 0.0, 1e-12);
}

TEST_F(CleaningTest, DetrendDirectCppPathUnified)
{
    // Direct C++ crosses the same float gate as the interpreter.
    numkit::Value x = engine.eval("int8([1 2 3])");
    try {
        numkit::stats::detrend_of(x, 1, nullptr);
        FAIL() << "expected numkit::Error";
    } catch (const numkit::Error &e) {
        EXPECT_EQ(std::string(e.identifier()), "numkit:detrend:invalidType");
    }
}

// ── isoutlier ─────────────────────────────────────────────────────────

TEST_F(CleaningTest, IsoutlierTypeGate)
{
    EXPECT_EQ(idOf("isoutlier(int8([1 2 3]))"), "numkit:isoutlier:invalidType");
    EXPECT_EQ(idOf("isoutlier(logical([1 0 1]))"), "numkit:isoutlier:invalidType");
    EXPECT_EQ(idOf("isoutlier(complex([1 2 3]))"), "numkit:isoutlier:expectedReal");
    EXPECT_EQ(cls("isoutlier(single([1 2 100 3]))"), "logical");
}

TEST_F(CleaningTest, IsoutlierMethodsAndPrefixes)
{
    EXPECT_EQ(cls("isoutlier([1 2 100 3], 'med')"), "logical");
    EXPECT_EQ(num("sum(isoutlier([1 2 100 3], 'med'))"), 1.0);
    // percentiles method (vector must be a sorted 1x2 row in [0,100])
    EXPECT_EQ(num("sum(isoutlier([1 2 100 3], 'percentiles', [10 90]))"), 0.0);
    EXPECT_EQ(idOf("isoutlier([1 2 3], 'percentiles', [10])"),
              "numkit:isoutlier:badPercentiles");
    EXPECT_EQ(idOf("isoutlier([1 2 3], 'percentiles', [90 10])"),
              "numkit:isoutlier:badPercentiles");
    // moving methods
    EXPECT_EQ(num("sum(isoutlier([1 2 100 3], 'movmedian', 5))"), 1.0);
    EXPECT_EQ(idOf("isoutlier([1 2 100 3], 'movmedian')"),
              "numkit:isoutlier:missingWindow");
    // dim form: per-row detection
    engine.eval("M = [1 2 9; 10 20 30; 3 4 50]; "
                "tf2 = isoutlier(M, 2);");
    EXPECT_EQ(num("sum(tf2(:))"), 2.0);   // rows 1 and 3 flag (probed)
}

TEST_F(CleaningTest, IsoutlierThresholdsAndCenter)
{
    // [tf, lo, hi, center] on [1 2 100 3]: median 2.5, scaled MAD 1.4826
    engine.eval("[tf, lo, hi, c] = isoutlier([1 2 100 3]);");
    EXPECT_NEAR(num("c"), 2.5, 1e-12);
    EXPECT_NEAR(num("hi"), 6.94780665551681, 1e-9);
    EXPECT_NEAR(num("lo"), -1.94780665551681, 1e-9);
    EXPECT_EQ(num("sum(tf)"), 1.0);
    // single thresholds stay single
    engine.eval("[~, ~, ~, c2] = isoutlier(single([1 2 100 3]));");
    EXPECT_EQ(cls("c2"), "single");
}

TEST_F(CleaningTest, IsoutlierOptionValidation)
{
    EXPECT_EQ(idOf("isoutlier([1 2 3], 'grubbs', 'ThresholdFactor', 5)"),
              "numkit:isoutlier:alphaRange");
    EXPECT_EQ(idOf("isoutlier([1 2 3], 'ThresholdFactor', -1)"),
              "numkit:isoutlier:tfNonneg");
    EXPECT_EQ(idOf("isoutlier([1 2 3], 'ThresholdFactor', NaN)"),
              "numkit:isoutlier:tfNaN");
    EXPECT_EQ(idOf("isoutlier([1 2 3], 'MaxNumOutliers', 3)"),
              "numkit:isoutlier:maxNumGesdOnly");
}

TEST_F(CleaningTest, IsoutlierDirectCppGate)
{
    numkit::Value x = engine.eval("int8([1 2 3])");
    try {
        numkit::stats::isoutlier_of(x, nullptr);
        FAIL() << "expected numkit::Error";
    } catch (const numkit::Error &e) {
        EXPECT_EQ(std::string(e.identifier()), "numkit:isoutlier:invalidType");
    }
}

// ── rmoutliers ────────────────────────────────────────────────────────

TEST_F(CleaningTest, RmoutliersMatrixRowsAndColumns)
{
    engine.eval("A = [1 2 9; 10 20 30; 3 4 50];");
    EXPECT_EQ(num("size(rmoutliers(A), 1)"), 2.0);       // row removed
    engine.eval("RB = rmoutliers(A);");
    EXPECT_EQ(num("RB(2, 2)"), 4.0);
    EXPECT_EQ(num("size(rmoutliers(A, 2), 2)"), 2.0);    // column removed
    // all reference methods available
    EXPECT_EQ(num("numel(rmoutliers([1 2 100 3], 'movmedian', 5))"), 3.0);
    EXPECT_EQ(num("numel(rmoutliers([1 2 100 3], 'gesd'))"), 3.0);
    EXPECT_EQ(num("numel(rmoutliers([1 2 100 3], 'grubbs'))"), 3.0);
    // single contagion
    EXPECT_EQ(cls("rmoutliers(single([1 2 100 3]))"), "single");
    EXPECT_EQ(idOf("rmoutliers(int8([1 2 100 3]))"), "numkit:rmoutliers:invalidType");
}

TEST_F(CleaningTest, RmoutliersMinNumAndLocations)
{
    engine.eval("A = [1 100 9; 10 20 1000; 3 4 50];");
    EXPECT_EQ(num("size(rmoutliers(A, 'MinNumOutliers', 2), 1)"), 3.0);
    EXPECT_EQ(num("size(rmoutliers(A, 'MinNumOutliers', 1), 1)"), 1.0);
    EXPECT_EQ(idOf("rmoutliers([1 2 3], 'MinNumOutliers', 1.5)"),
              "numkit:rmoutliers:badMinNum");
    // OutlierLocations override
    engine.eval("[B, ind] = rmoutliers([1 100 3], 'OutlierLocations', "
                "[false true false]);");
    EXPECT_EQ(num("numel(B)"), 2.0);
    EXPECT_EQ(num("ind(2)"), 1.0);
}

// ── rmmissing ─────────────────────────────────────────────────────────

TEST_F(CleaningTest, RmmissingRowsColumnsAndVector)
{
    engine.eval("A = [1 NaN 3; NaN 4 5; 6 7 8];");
    EXPECT_EQ(num("rmmissing(A)(1, 1)"), 6.0);           // rows removed
    EXPECT_EQ(num("size(rmmissing(A), 1)"), 1.0);
    EXPECT_EQ(num("size(rmmissing(A, 2), 2)"), 1.0);     // columns removed
    EXPECT_EQ(num("numel(rmmissing([1 NaN 3]))"), 2.0);
    EXPECT_EQ(num("numel(rmmissing([1; NaN; 3]))"), 2.0);
    // 2nd output: removed-rows mask, column for dim 1 / row for dim 2
    engine.eval("[B, I] = rmmissing(A);");
    EXPECT_EQ(num("size(I, 2)"), 1.0);
    EXPECT_EQ(num("sum(I)"), 2.0);
    // MinNumMissing
    EXPECT_EQ(num("size(rmmissing([1 NaN 3; NaN 4 5; 6 7 NaN], "
                  "'MinNumMissing', 2), 1)"), 3.0);
    // single / int8 / char contracts
    EXPECT_EQ(cls("rmmissing(single([1 NaN 3]))"), "single");
    EXPECT_EQ(cls("rmmissing(int8([1 2 3]))"), "int8");
    EXPECT_EQ(cls("rmmissing('ab')"), "char");
    // complex works
    EXPECT_EQ(num("numel(rmmissing(complex([1 NaN 3])))"), 2.0);
    // rejection classes
    EXPECT_EQ(idOf("rmmissing(ones(2,2,2))"), "numkit:rmmissing:ndArrays");
    EXPECT_EQ(idOf("rmmissing([1 2 3], 0)"), "numkit:rmmissing:badDim");
    EXPECT_EQ(idOf("rmmissing([1 2 3], 1.5)"), "numkit:rmmissing:badDim");
}

// ── fillmissing ───────────────────────────────────────────────────────

TEST_F(CleaningTest, FillmissingConstantContract)
{
    // 'constant' REQUIRES its value
    EXPECT_EQ(idOf("fillmissing([1 NaN 3], 'constant')"),
              "numkit:fillmissing:constantInput");
    // prefix matching
    EXPECT_EQ(num("fillmissing([1 NaN 3], 'const', 9)(2)"), 9.0);
    // ambiguity / unknown reject
    EXPECT_EQ(idOf("fillmissing([1 NaN 3], 'ne')"),
              "numkit:fillmissing:methodInvalid");
    EXPECT_EQ(idOf("fillmissing([1 NaN 3], 'p')"),
              "numkit:fillmissing:methodInvalid");
    // spline / pchip / makima interpolation
    EXPECT_NEAR(num("fillmissing([1 NaN 9 16], 'spline')(2)"), 4.0, 1e-12);
    EXPECT_NEAR(num("fillmissing([1 NaN 3 4], 'pchip')(2)"), 2.0, 1e-12);
    EXPECT_NEAR(num("fillmissing([1 NaN 3 4], 'makima')(2)"), 2.0, 1e-12);
    // complex fill constant on real input promotes output to complex
    engine.eval("zc = fillmissing([1 NaN 3], 'constant', 2 + 3i);");
    EXPECT_NEAR(num("real(zc(2))"), 2.0, 1e-12);
    EXPECT_NEAR(num("imag(zc(2))"), 3.0, 1e-12);
    // EndValues is legal with constant
    EXPECT_EQ(num("fillmissing([1 NaN 3], 'constant', 9, "
                  "'EndValues', 'none')(2)"), 9.0);
    // per-slice constant vector (3 columns -> 3 constants)
    engine.eval("B = fillmissing([1 NaN 3; 4 NaN 6], 'constant', [9 8 7]);");
    EXPECT_EQ(num("B(1, 2)"), 8.0);
    EXPECT_EQ(num("B(2, 2)"), 8.0);
    EXPECT_EQ(idOf("fillmissing([1 NaN 3; 4 NaN 6], 'constant', [9 8])"),
              "numkit:fillmissing:SizeConstant");
}

TEST_F(CleaningTest, FillmissingMovingAndEndValues)
{
    // moving fills come from the NaN-omitted statistic of the ORIGINAL
    // data (probed: [NaN 1 NaN NaN 4 NaN] win 3 movmean = [1 1 1 4 4 4])
    engine.eval("y = fillmissing([NaN 1 NaN NaN 4 NaN], 'movmean', 3);");
    EXPECT_NEAR(num("y(1)"), 1.0, 1e-12);
    EXPECT_NEAR(num("y(4)"), 4.0, 1e-12);
    EXPECT_NEAR(num("y(5)"), 4.0, 1e-12);
    engine.eval("yn = fillmissing([NaN 1 NaN NaN 4 NaN], 'movmean', 3, "
                "'EndValues', 'none');");
    EXPECT_TRUE(std::isnan(num("yn(1)")));
    EXPECT_TRUE(std::isnan(num("yn(6)")));
    // EndValues previous/next: only the edge a direction can supply
    engine.eval("p1 = fillmissing([NaN 1 3 NaN]', 'previous', "
                "'EndValues', 'previous');");
    EXPECT_TRUE(std::isnan(num("p1(1)")));
    EXPECT_NEAR(num("p1(4)"), 3.0, 1e-12);
    engine.eval("p2 = fillmissing([NaN 1 3 NaN]', 'previous', "
                "'EndValues', 'next');");
    EXPECT_NEAR(num("p2(1)"), 1.0, 1e-12);
    EXPECT_TRUE(std::isnan(num("p2(4)")));
    // numeric EndValues constant fills everything
    engine.eval("p3 = fillmissing([NaN NaN NaN]', 'previous', "
                "'EndValues', 7);");
    EXPECT_NEAR(num("p3(2)"), 7.0, 1e-12);
    // 2nd output: filled-location map
    engine.eval("[B, FB] = fillmissing([1 NaN 3], 'constant', 9);");
    EXPECT_EQ(num("sum(FB)"), 1.0);
    // single / int8 / complex classes
    EXPECT_EQ(cls("fillmissing(single([1 NaN 3]), 'constant', 9)"), "single");
    EXPECT_EQ(cls("fillmissing(int8([1 2 3]), 'constant', 9)"), "int8");
    engine.eval("z = fillmissing(complex([1 NaN 3]), 'constant', 1i);");
    EXPECT_EQ(num("imag(z(2))"), 1.0);
    // dim positional (per-row fill for dim 2)
    engine.eval("r = fillmissing([1 NaN 3; 4 NaN 6], 'previous', 2);");
    EXPECT_NEAR(num("r(1, 2)"), 1.0, 1e-12);
}

TEST_F(CleaningTest, SamplePointsAndAdvancedFillmissing)
{
    // detrend with SamplePoints
    engine.eval("s = [0 1 3 6 10]; x = 2 * s + 5; d = detrend(x, 1, 'SamplePoints', s);");
    EXPECT_NEAR(num("max(abs(d))"), 0.0, 1e-11);
    EXPECT_EQ(idOf("detrend([1 2 3], 1, 'SamplePoints', [1 3 2])"),
              "numkit:detrend:samplePointsSorted");

    // isoutlier and rmoutliers with SamplePoints
    engine.eval("tf_sp = isoutlier([1 2 100 3 4], 'movmedian', 4, 'SamplePoints', [0 1 2 3 10]);");
    EXPECT_EQ(num("tf_sp(3)"), 1.0);
    EXPECT_EQ(num("numel(rmoutliers([1 2 100 3 4], 'movmedian', 4, 'SamplePoints', [0 1 2 3 10]))"), 4.0);

    // fillmissing with SamplePoints and MaxGap
    engine.eval("y_sp = fillmissing([0 NaN 6], 'linear', 'SamplePoints', [0 1 3]);");
    EXPECT_NEAR(num("y_sp(2)"), 2.0, 1e-12);
    engine.eval("y_mg = fillmissing([1 NaN NaN 4 5 NaN 7], 'linear', 'MaxGap', 2);");
    EXPECT_TRUE(std::isnan(num("y_mg(2)")));
    EXPECT_TRUE(std::isnan(num("y_mg(3)")));
    EXPECT_NEAR(num("y_mg(6)"), 6.0, 1e-12);

    // fillmissing with 'knn'
    engine.eval("A_knn = [1 10; 1.1 NaN; 5 50; 5.1 52]; B_knn = fillmissing(A_knn, 'knn', 1);");
    EXPECT_NEAR(num("B_knn(2, 2)"), 10.0, 1e-12);
    engine.eval("B_knn_s = fillmissing(A_knn, 'knn', 1, 'Distance', 'seuclidean');");
    EXPECT_NEAR(num("B_knn_s(2, 2)"), 10.0, 1e-12);

    // fillmissing with custom function handle
    engine.eval("fh = @(xs, ts, tq) repmat(mean(xs), numel(tq), 1); "
                "y_fh = fillmissing([2 NaN 6 10], fh, 3);");
    EXPECT_NEAR(num("y_fh(2)"), 4.0, 1e-12);
}

// ── standardizeMissing ────────────────────────────────────────────────

TEST_F(CleaningTest, StandardizeMissingIndicatorVector)
{
    engine.eval("s1 = standardizeMissing([1 2 3], [2 3]); "
                "s2 = standardizeMissing([1 NaN 3], [1 NaN]);");
    EXPECT_EQ(num("s1(1)"), 1.0);
    EXPECT_TRUE(std::isnan(num("s1(2)")));
    // NaN indicators also match NaN
    EXPECT_TRUE(std::isnan(num("s2(2)")));
    // integer / logical / char arrays flow through unchanged
    EXPECT_EQ(cls("standardizeMissing(int8([1 2 3]), 2)"), "int8");
    engine.eval("si = standardizeMissing(int8([1 2 3]), 2);");
    EXPECT_EQ(num("double(si(2))"), 2.0);
    EXPECT_EQ(cls("standardizeMissing(logical([1 0 1]), 1)"), "logical");
    EXPECT_EQ(cls("standardizeMissing('abc', 'c')"), "char");
    // single stays single
    EXPECT_EQ(cls("standardizeMissing(single([1 2]), 1)"), "single");
    // indicator compatibility classes
    EXPECT_EQ(idOf("standardizeMissing([1 2], 'a')"),
              "numkit:standardizeMissing:indicatorsDouble");
    EXPECT_EQ(idOf("standardizeMissing(int8([1 2 3]), int16(2))"),
              "numkit:standardizeMissing:indicatorsIntegerLogical");
    // empty indicators: no-op
    engine.eval("se = standardizeMissing([1 2 3], []);");
    EXPECT_EQ(num("se(2)"), 2.0);
}

} // namespace
