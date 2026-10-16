// toolboxes/stats/tests/argcheck_stats_test.cpp
//
// *rnd size-input contract (numkit::argcheck migration, 2026-09-17). The
// contract is the array-creation family's, reference-probed against
// R2025b on 2026-09-17:
//   * scalar n → n×n; (m, n) → m×n; [m n] vector → m×n;
//   * fractional / NaN / Inf / complex size inputs are each their own
//     nk-voiced error class (sizeNotInteger / sizeNotFinite / sizeNotReal);
//   * a NEGATIVE size and an EMPTY size argument both mean a zero
//     dimension (geornd(p, -1) → 0×0, geornd(p, [], 3) → 0×3);
//   * more than two size inputs rejects — an nk CAP, not the reference
//     contract (the reference accepts an unbounded size tail;
//     bugs/opened/stats/rnd-size-vector-nd.md). This test pins the
//     current nk behaviour and flips with the fix.
// Covered through both migration shapes: the shared parse_rng_size
// (normrnd / wblrnd — 14 adapter families) and the previously-inlining
// adapters (geornd / hygernd — 9 families), plus the multivariate
// count inputs (mvnrnd / mvtrnd / mnrnd).

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/value/error.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class ArgcheckStatsRnd : public ::testing::Test
{
public:
    StandardEngine engine;

    int evalInt(const std::string &c) { return static_cast<int>(engine.eval(c).toScalar()); }

    // size(...) of an expression as "RxC"
    std::pair<int, int> sizeOf(const std::string &expr)
    {
        engine.eval("__sz = size(" + expr + ");");
        auto *v = engine.getVariable("__sz");
        return {static_cast<int>(v->doubleData()[0]),
                static_cast<int>(v->doubleData()[1])};
    }

    void expectError(const std::string &code, const std::string &id)
    {
        try {
            engine.eval(code);
            FAIL() << "expected error not thrown: " << code;
        } catch (const Error &e) {
            EXPECT_EQ(e.identifier(), id) << code << "\n  got: " << e.what();
        }
    }
};

TEST_F(ArgcheckStatsRnd, ScalarForms)
{
    EXPECT_EQ(sizeOf("geornd(0.5)"), (std::pair<int, int>{1, 1}));
    EXPECT_EQ(sizeOf("geornd(0.5, 3)"), (std::pair<int, int>{3, 3}));
    EXPECT_EQ(sizeOf("geornd(0.5, 2, 3)"), (std::pair<int, int>{2, 3}));
    // Shared-helper families with different arity offsets.
    EXPECT_EQ(sizeOf("normrnd(0, 1, 3)"), (std::pair<int, int>{3, 3}));
    EXPECT_EQ(sizeOf("wblrnd(1, 1, 2, 3)"), (std::pair<int, int>{2, 3}));
    EXPECT_EQ(sizeOf("poissrnd(2, 3)"), (std::pair<int, int>{3, 3}));
}

TEST_F(ArgcheckStatsRnd, VectorForm)
{
    EXPECT_EQ(sizeOf("geornd(0.5, [2 3])"), (std::pair<int, int>{2, 3}));
    EXPECT_EQ(sizeOf("normrnd(0, 1, [2 3])"), (std::pair<int, int>{2, 3}));
    EXPECT_EQ(sizeOf("hygernd(10, 5, 4, [2 3])"), (std::pair<int, int>{2, 3}));
}

TEST_F(ArgcheckStatsRnd, NegativeClampsToEmpty)
{
    EXPECT_EQ(sizeOf("geornd(0.5, -1)"), (std::pair<int, int>{0, 0}));
    EXPECT_EQ(sizeOf("normrnd(0, 1, -2)"), (std::pair<int, int>{0, 0}));
    EXPECT_EQ(sizeOf("geornd(0.5, 2, -3)"), (std::pair<int, int>{2, 0}));
    EXPECT_EQ(sizeOf("mvtrnd(eye(2), 5, -1)"), (std::pair<int, int>{0, 2}));
}

TEST_F(ArgcheckStatsRnd, EmptyMeansZeroDimension)
{
    EXPECT_EQ(sizeOf("geornd(0.5, [])"), (std::pair<int, int>{0, 0}));
    EXPECT_EQ(sizeOf("geornd(0.5, [], 3)"), (std::pair<int, int>{0, 3}));
    EXPECT_EQ(sizeOf("geornd(0.5, 2, [])"), (std::pair<int, int>{2, 0}));
}

TEST_F(ArgcheckStatsRnd, FractionalSizeErrors)
{
    expectError("geornd(0.5, 2.7)", "numkit:geornd:sizeNotInteger");          // inlined family
    expectError("normrnd(0, 1, 2.5)", "numkit:normrnd:sizeNotInteger");      // helper family
    expectError("geornd(0.5, 2, 3.5)", "numkit:geornd:sizeNotInteger");
    expectError("geornd(0.5, [2.5 3])", "numkit:geornd:sizeNotInteger");     // vector form
    expectError("hygernd(10, 5, 4, 2.7)", "numkit:hygernd:sizeNotInteger");
}

TEST_F(ArgcheckStatsRnd, NonFiniteSizeErrors)
{
    expectError("geornd(0.5, NaN)", "numkit:geornd:sizeNotFinite");
    expectError("normrnd(0, 1, Inf)", "numkit:normrnd:sizeNotFinite");
    expectError("geornd(0.5, [2 Inf])", "numkit:geornd:sizeNotFinite");
}

TEST_F(ArgcheckStatsRnd, ComplexSizeErrors)
{
    expectError("geornd(0.5, 2i)", "numkit:geornd:sizeNotReal");
    expectError("normrnd(0, 1, [2 3i])", "numkit:normrnd:sizeNotReal");
}

TEST_F(ArgcheckStatsRnd, TooManySizeInputs)
{
    expectError("geornd(0.5, 2, 3, 4)", "numkit:geornd:maxnargin");
    expectError("normrnd(0, 1, 2, 3, 4)", "numkit:normrnd:maxnargin");
}

TEST_F(ArgcheckStatsRnd, MultivariateCountValidation)
{
    expectError("mvnrnd(0, 1, 3.5)", "numkit:mvnrnd:badCases");
    expectError("mvtrnd(eye(2), 5, 2.5)", "numkit:mvtrnd:badCases");
    expectError("mnrnd(3, [0.2 0.8], 2.5)", "numkit:mnrnd:sizeNotInteger");
    // Happy path stays intact.
    EXPECT_EQ(sizeOf("mvtrnd(eye(2), 5, 3)"), (std::pair<int, int>{3, 2}));
    EXPECT_EQ(sizeOf("mnrnd(3, [0.2 0.8], 2)"), (std::pair<int, int>{2, 2}));
}

// ── Dimension-argument contract (R2025b-probed 2026-09-17): dim must be
// a positive integer scalar; vecdim entries carry the same per-element
// rule plus the range check. ─────────────────────────────────────────
TEST_F(ArgcheckStatsRnd, DimArgumentValidation)
{
    expectError("bounds([1 2 3; 4 5 6], 1.5)", "numkit:bounds:dimNotInteger");
    expectError("bounds([1 2 3; 4 5 6], 0)", "numkit:bounds:dimNotPositive");
    expectError("bounds([1 2 3; 4 5 6], -1)", "numkit:bounds:dimNotPositive");
    expectError("bounds([1 2 3; 4 5 6], NaN)", "numkit:bounds:dimNotFinite");
    expectError("bounds([1 2 3; 4 5 6], [1.5 2])", "numkit:bounds:dimNotInteger");
    expectError("median(1:4, 2.5)", "numkit:median:dimNotInteger");
    expectError("mode(1:4, 0)", "numkit:mode:dimNotPositive");
    expectError("iqr([1 2 3; 4 5 6], 1.5)", "numkit:iqr:dimNotInteger");
    expectError("range([1 2 3;4 5 6], -2)", "numkit:range:dimNotPositive");
    expectError("geomean([1 2 3;4 5 6], 1.5)", "numkit:geomean:dimNotInteger");
    expectError("harmmean([1 2 3;4 5 6], 1.5)", "numkit:harmmean:dimNotInteger");
    expectError("mad([1 2 3;4 5 6], 0, 1.5)", "numkit:mad:dimNotInteger");
    expectError("moment(1:5, 2, 1.5)", "numkit:moment:dimNotInteger");
    expectError("trimmean(1:10, 20, 1.5)", "numkit:trimmean:dimNotInteger");
    expectError("zscore([1 2 3;4 5 6], 0, 1.5)", "numkit:zscore:dimNotInteger");
    // Happy paths: integer dims and the vecdim flatten form stay intact.
    EXPECT_EQ(sizeOf("bounds([1 2 3; 4 5 6], 1)"), (std::pair<int, int>{1, 3}));
    EXPECT_EQ(sizeOf("bounds([1 2 3; 4 5 6], [1 2])"), (std::pair<int, int>{1, 1}));
    EXPECT_NEAR(evalInt("numel(median([1 2 3; 4 5 6], 2))"), 2, 0);
}

// mink/maxk k: non-negative integer — 0 is legal and yields an empty
// result (R2025b-probed).
TEST_F(ArgcheckStatsRnd, TopKCountValidation)
{
    expectError("mink(1:5, 2.5)", "numkit:mink:kNotInteger");
    expectError("mink(1:5, -1)", "numkit:mink:kNotNegative");
    expectError("maxk(1:5, 2.5)", "numkit:maxk:kNotInteger");
    expectError("maxk(1:5, -1)", "numkit:maxk:kNotNegative");
    expectError("mink(1:5, 2, 1.5)", "numkit:mink:dimNotInteger");
    EXPECT_EQ(evalInt("numel(mink(1:5, 2))"), 2);
    EXPECT_EQ(evalInt("numel(maxk(1:5, 3))"), 3);
    EXPECT_EQ(evalInt("numel(mink(1:5, 0))"), 0);   // k = 0 → empty
}

// detrend order / moment order: non-negative integer; string modes
// ('constant'/'linear') untouched.
TEST_F(ArgcheckStatsRnd, OrderValidation)
{
    expectError("detrend(1:10, 1.5)", "numkit:detrend:invalidTrendInput");
    expectError("detrend(1:10, -1)", "numkit:detrend:invalidTrendInput");
    expectError("moment(1:5, 2.5)", "numkit:moment:orderNotInteger");
    EXPECT_NEAR(evalInt("numel(detrend(1:10, 3))"), 10, 0);
    EXPECT_NEAR(evalInt("numel(detrend(1:10, 'linear'))"), 10, 0);
}

// 0/1 flags: mad / zscore / cov — fractional is its own class (the
// integer check fires first), integer-but-not-0/1 another.
TEST_F(ArgcheckStatsRnd, Flag01Validation)
{
    expectError("mad([1 2 3;4 5 6], 1.5)", "numkit:mad:flagNotInteger");
    expectError("mad([1 2 3;4 5 6], 2)", "numkit:mad:flagNot01");
    expectError("zscore([1 2 3;4 5 6], 1.5)", "numkit:zscore:flagNotInteger");
    expectError("zscore([1 2 3;4 5 6], 2)", "numkit:zscore:flagNot01");
    expectError("cov([1 2 3], [4 5 6], 1.5)", "numkit:cov:wNotInteger");
    expectError("cov([1 2 3], [4 5 6], 2)", "numkit:cov:wNot01");
    EXPECT_NEAR(evalInt("numel(cov([1 2 3], [4 5 6], 1))"), 4, 0);
    EXPECT_NEAR(evalInt("numel(mad([1 2 3;4 5 6], 1))"), 3, 0);
}

// parcorr/crosscorr NumLags: positive integer, both the name-value and
// positional forms.
TEST_F(ArgcheckStatsRnd, NumLagsValidation)
{
    expectError("parcorr(1:10, 'NumLags', 2.5)", "numkit:parcorr:NumLagsNotInteger");
    expectError("parcorr(1:10, 'NumLags', -1)", "numkit:parcorr:NumLagsNotPositive");
    expectError("parcorr(1:10, 2.5)", "numkit:parcorr:NumLagsNotInteger");
    expectError("crosscorr(1:10, 1:10, 2.5)", "numkit:crosscorr:NumLagsNotInteger");
    EXPECT_EQ(evalInt("numel(parcorr(1:10, 'NumLags', 4))"), 5);
    EXPECT_EQ(evalInt("numel(crosscorr(1:10, 1:10, 3))"), 7);
}

// nan-aware reductions / moments flags & dims / moving windows /
// spline orders / design sizes / fold counts (R2025b-probed 2026-09-17;
// fnder order and lhsdesign 'Iterations' are reference-TOLERANT and
// deliberately left unvalidated).
TEST_F(ArgcheckStatsRnd, BatchThreeValidation)
{
    expectError("nansum([1 2 3], 1.5)", "numkit:nansum:dimNotInteger");
    expectError("nanmax([1 2 3], [], 0)", "numkit:nanmax:dimNotPositive");
    expectError("nanvar([1 2 3], 1.5)", "numkit:nanvar:wNotInteger");
    expectError("nanvar([1 2 3], 0, 1.5)", "numkit:nanvar:dimNotInteger");
    expectError("nanstd([1 2 3], 2)", "numkit:nanstd:wNot01");
    expectError("nancov([1 2 3], 1.5)", "numkit:nancov:wNotInteger");
    expectError("skewness([1 2 3], 1.5)", "numkit:skewness:flagNotInteger");
    expectError("kurtosis([1 2 3], 2)", "numkit:kurtosis:flagNot01");
    expectError("smoothdata(1:10, 2.5)", "numkit:smoothdata:windowNotInteger");
    expectError("smoothdata(1:10, 0)", "numkit:smoothdata:windowNotPositive");
    expectError("smoothdata(1:10, 'movmean', 2.5)", "numkit:smoothdata:windowNotInteger");
    expectError("hampel(1:10, 1.5)", "numkit:hampel:kNotInteger");
    expectError("hampel(1:10, 0)", "numkit:hampel:kNotPositive");
    expectError("lhsdesign(2.5, 3)", "numkit:lhsdesign:sizeNotInteger");
    expectError("lhsdesign(2, 2.5)", "numkit:lhsdesign:sizeNotInteger");
    expectError("lhsnorm([0 0], eye(2), 2.5)", "numkit:lhsnorm:sizeNotInteger");
    expectError("haltonset(2.5)", "numkit:haltonset:dNotInteger");
    expectError("haltonset(0)", "numkit:haltonset:dNotPositive");
    expectError("crossval(@sum, rand(8,1), rand(8,1), 'KFold', 2.5)", "numkit:crossval:KFoldNotInteger");
    expectError("pdist2([1 2 3]', [1 2 3]', 'Smallest', 2.5)", "numkit:pdist2:badK");
    // Happy paths stay intact.
    EXPECT_EQ(evalInt("numel(nansum([1 2 NaN 4], 2))"), 1);
    EXPECT_EQ(evalInt("numel(smoothdata(1:10, 3))"), 10);
    EXPECT_EQ(evalInt("numel(hampel(1:10, 2))"), 10);
    EXPECT_EQ(sizeOf("lhsdesign(2, 3)"), (std::pair<int, int>{2, 3}));
    EXPECT_EQ(sizeOf("lhsdesign(0, 3)"), (std::pair<int, int>{0, 3}));  // n=0 → empty
    EXPECT_EQ(evalInt("numel(haltonset(2))"), 1);
    EXPECT_NEAR(evalInt("numel(skewness([1 2 3; 4 5 8], 1, 2))"), 2, 0);
}
