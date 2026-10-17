// toolboxes/stats/tests/known_bugs_test.cpp
//
// One DISABLED_ test per OPEN bug in bugs/stats/*.md. Disabled until fixed
// (don't break the green baseline); remove `DISABLED_` to turn into a live
// regression guard. Asserts MATLAB R2025b-correct behaviour.

#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/core/standard_engine.hpp>
#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class StatsKnownBug : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override {}
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

// NOTE: anova1 matrix-input bug FIXED — see toolboxes/stats/tests/anova1_test.cpp.

// bugs/stats/movfun-typeclass.md — FIXED (movsum/movprod/movmean promote
// integer/logical input to double). Full guard in movfun_typeclass_test.cpp.
TEST_F(StatsKnownBug, MovfunTypeClass)
{
    eval("y = movsum(int8([3 1 2 5 4]), 3);");
    EXPECT_DOUBLE_EQ(evalScalar("isa(y,'double')"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("y(4)"), 11.0);
    eval("m = movmean(int8([3 1 2 5 4]), 3);");
    EXPECT_NEAR(evalScalar("m(5)"), 4.5, 1e-12);
    eval("l = movsum(logical([1 0 1 1 0]), 3);");
    EXPECT_DOUBLE_EQ(evalScalar("l(2)"), 2.0);
}

// bugs/stats/movfun-order-stats.md — FIXED (movmax/movmin preserve class;
// movmedian rounds int half-away, logical->double). Full guard in
// movfun_order_stats_test.cpp.
TEST_F(StatsKnownBug, MovfunOrderStats)
{
    eval("a = movmax(int8([3 1 2 5 4]), 3);");
    EXPECT_DOUBLE_EQ(evalScalar("isa(a,'int8')"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("a(3)"), 5.0);
    eval("m = movmedian(int8([3 1 2 5 4]), 2);");
    EXPECT_DOUBLE_EQ(evalScalar("isa(m,'int8')"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("m(5)"), 5.0);   // 4.5 -> 5
    eval("lg = movmax(logical([1 0 1 1 0]), 3);");
    EXPECT_DOUBLE_EQ(evalScalar("islogical(lg)"), 1.0);
}

// bugs/stats/mle-output.md — 2nd output pci (confidence intervals). FIXED
// 2026-06-05 (deep coverage in toolboxes/stats/tests/mle_pci_test.cpp).
TEST_F(StatsKnownBug, MleConfidenceIntervals)
{
    eval("[phat, pci] = mle([2 3 4 5 6 4 3]);");
    EXPECT_NEAR(evalScalar("pci(1,1)"), 2.613054, 1e-5);
}

// bugs/stats/distribution-dispatchers.md — cdf/pdf/icdf/random.
TEST_F(StatsKnownBug, DistributionDispatchers)
{
    EXPECT_NEAR(evalScalar("cdf('Normal', 1, 0, 1)"),       0.841344746, 1e-7);
    EXPECT_NEAR(evalScalar("pdf('Poisson', 2, 3)"),         0.224041808, 1e-7);
    EXPECT_NEAR(evalScalar("icdf('Normal', 0.975, 0, 1)"),  1.959963985, 1e-7);
    EXPECT_EQ(static_cast<int>(evalScalar("numel(random('Normal',0,1,1,3))")), 3);
}

// bugs/stats/isoutlier-gesd.md — generalized ESD test. FIXED 2026-06-05
// (deep coverage in toolboxes/stats/tests/isoutlier_gesd_test.cpp).
TEST_F(StatsKnownBug, IsoutlierGesd)
{
    eval("m = isoutlier([1 2 3 4 5 6 7 8 9 50], 'gesd');");
    EXPECT_DOUBLE_EQ(evalScalar("sum(double(m))"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("double(m(10))"), 1.0);
}

// bugs/stats/smoothdata-methods.md — sgolay FIXED (explicit window exact).
TEST_F(StatsKnownBug, SmoothdataSgolay)
{
    // Default-window form runs + preserves length. (MATLAB's default window is a
    // data-dependent heuristic that nk's shared default only approximates, so
    // here we check shape, not values, for the default form.)
    eval("y = smoothdata([1 5 2 8 3 9 4 7 2 8 3], 'sgolay');");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(y)")), 11);
    // Explicit odd window matches MATLAB R2025b exactly (degree-2 Savitzky-Golay).
    eval("z = smoothdata([1 5 2 8 3 9 4 7 2 6 1 8], 'sgolay', 5);");
    EXPECT_NEAR(evalScalar("z(1)"),  1.114286, 1e-5);   // leading edge
    EXPECT_NEAR(evalScalar("z(4)"),  4.4,      1e-5);
    EXPECT_NEAR(evalScalar("z(7)"),  7.0,      1e-5);    // interior
    EXPECT_NEAR(evalScalar("z(12)"), 7.142857, 1e-5);   // trailing edge
}

// bugs/stats/smoothdata-methods.md — lowess / loess local regression (FIXED).
TEST_F(StatsKnownBug, SmoothdataLowessLoess)
{
    eval("x = [1 5 2 8 3 9 4 7 2 6 1 8];");
    eval("yl = smoothdata(x, 'lowess', 5);");           // local linear
    EXPECT_NEAR(evalScalar("yl(1)"),  1.711260, 1e-5);
    EXPECT_NEAR(evalScalar("yl(3)"),  4.576790, 1e-5);
    EXPECT_NEAR(evalScalar("yl(12)"), 6.194210, 1e-5);
    eval("yo = smoothdata(x, 'loess', 5);");            // local quadratic
    EXPECT_NEAR(evalScalar("yo(1)"),  1.550870, 1e-5);
    EXPECT_NEAR(evalScalar("yo(3)"),  2.0,      1e-5);   // interior identity (3-pt quad fit)
    EXPECT_NEAR(evalScalar("yo(11)"), 3.731,    1e-3);
}

// bugs/stats/kstest-pvalue.md — exact small-n Kolmogorov distribution. FIXED
// 2026-06-05 (deep coverage in toolboxes/stats/tests/kstest_exact_test.cpp).
TEST_F(StatsKnownBug, KstestPValue)
{
    eval("[h, p, ks, cv] = kstest([-1 0 1 2 -0.5 0.5]);");
    EXPECT_NEAR(evalScalar("ks"), 0.19146246, 1e-6);   // statistic
    EXPECT_NEAR(evalScalar("p"),  0.94998410, 1e-5);   // was nk 0.9804
    EXPECT_NEAR(evalScalar("cv"), 0.51926000, 1e-4);   // was nk 0.5544
}

// bugs/stats/friedman.md — Friedman's nonparametric two-way ANOVA missing.
TEST_F(StatsKnownBug, Friedman)
{
    eval("p = friedman([1 2 3; 2 3 4; 3 4 5; 1 3 5], 1);");
    EXPECT_NEAR(evalScalar("p"), 0.018315639, 1e-6);
}

// bugs/stats/corr-pvalue.md — [r,p]=corr 2nd output. FIXED 2026-06-05
// (deep coverage in toolboxes/stats/tests/corr_pvalue_test.cpp).
TEST_F(StatsKnownBug, CorrPValue)
{
    eval("x=[1 2 3 4 5]'; y=[2 1 4 3 6]'; [r,p]=corr(x,y,'type','Kendall');");
    EXPECT_NEAR(evalScalar("r"), 0.6,      1e-6);
    EXPECT_NEAR(evalScalar("p"), 0.233333, 1e-5);
}

// bugs/stats/kstest-pvalue.md — kstest2 p-value (Stephens corrected). FIXED
// 2026-06-05 (deep coverage in toolboxes/stats/tests/kstest_exact_test.cpp).
TEST_F(StatsKnownBug, Kstest2PValue)
{
    eval("[h,p,k]=kstest2([1 2 3 4 5],[2 3 4 5 6 7]);");
    EXPECT_NEAR(evalScalar("k"), 0.3333333, 1e-6);   // statistic
    EXPECT_NEAR(evalScalar("p"), 0.8470543, 1e-5);   // was nk 0.9223
}

// bugs/stats/dwtest-pvalue.md — exact DW p-value (Imhof). FIXED 2026-06-05
// (deep coverage in toolboxes/stats/tests/dwtest_exact_test.cpp).
TEST_F(StatsKnownBug, DwtestPValue)
{
    eval("[p,dw]=dwtest([1 2 1 3 2 4]', [ones(6,1) (1:6)']);");
    EXPECT_NEAR(evalScalar("dw"), 0.3142857, 1e-6);  // statistic
    EXPECT_LT(evalScalar("p"), 1e-4);                // MATLAB ~0 (was nk ~0.017)
}

// bugs/closed/stats/mahal-singular.md (FIXED; live guard) — the QR
// formulation no longer throws on collinear X. Full-rank exactness in
// RankDeficientNoThrow (mahal_test.cpp); MATLAB's rank-deficient VALUES
// are dust-amplified (R(2,2)~-4.3e-16) — not matchable without
// bit-identical LAPACK, see the bug file.
TEST_F(StatsKnownBug, MahalSingularNoThrow)
{
    EXPECT_NO_THROW(eval("d = mahal([1 1; 2 2], [0 0; 1 1; 2 2; 3 3]);"));
}

// bugs/stats/pdist-metrics.md — 'seuclidean'/'spearman' + cosine zero-vector.
// FIXED 2026-06-05 (deep coverage in toolboxes/stats/tests/pdist_metrics_test.cpp).
TEST_F(StatsKnownBug, PdistMetrics)
{
    eval("A = [1 2 3; 4 5 7; 1 0 2];");
    // seuclidean: was "unknown metric"; MATLAB max dist 3.2433.
    EXPECT_NEAR(evalScalar("max(pdist(A,'seuclidean'))"), 3.2433, 1e-3);
    EXPECT_NO_THROW(eval("pdist(A,'spearman');"));
    // cosine distance with a zero-norm row should be NaN (MATLAB), not 1.
    EXPECT_TRUE(eval("isnan(pdist([0 0;3 4],'cosine'))").toBool());
}

// bugs/stats/distribution-array-params.md — broadcast array distribution params.
// FIXED 2026-06-05 (cycles 29-38): all 16 *pdf/*cdf/*inv families broadcast
// ARRAY parameters. Deep per-family coverage in
// toolboxes/stats/tests/dist_broadcast_test.cpp (36 TEST_F) + the corr.json-style
// parity spec tools/parity/specs/dist_broadcast.json.
TEST_F(StatsKnownBug, DistributionArrayParams)
{
    eval("y = normpdf(0, 0, [1 2 4]);");          // MATLAB [0.3989 0.1995 0.0997]
    EXPECT_NEAR(evalScalar("y(2)"), 0.19947114, 1e-6);
    eval("b = binopdf(2, [4 5 6], 0.5);");        // MATLAB [0.375 0.3125 0.234375]
    EXPECT_NEAR(evalScalar("b(3)"), 0.234375, 1e-6);
    eval("g = gampdf(1, [1 2 3], 1);");           // MATLAB [0.3679 0.3679 0.1839]
    EXPECT_NEAR(evalScalar("g(3)"), 0.18393972, 1e-6);
}

// bugs/stats/autocorr.md — autocorr/parcorr/crosscorr (Econometrics) missing.
TEST_F(StatsKnownBug, Autocorr)
{
    eval("ac = autocorr([1 2 3 2 1 2 3 2 1]);");   // default NumLags=min(20,N-1)=8
    EXPECT_EQ(static_cast<int>(evalScalar("numel(ac)")), 9);
    EXPECT_NEAR(evalScalar("ac(1)"), 1.0, 1e-12);                // lag 0 == 1
    EXPECT_NEAR(evalScalar("ac(2)"), 0.0202020202020202, 1e-9); // lag 1
    eval("pc = parcorr([1 2 3 2 1 2 3 2 1]);");
    EXPECT_NEAR(evalScalar("pc(1)"), 1.0, 1e-12);   // PACF lag 0 == 1
    // crosscorr zero-lag of x and reversed x == -1 (perfectly anti-correlated)
    eval("xc = crosscorr([1 2 3 4],[4 3 2 1],'NumLags',2);");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(xc)")), 5);
    EXPECT_NEAR(evalScalar("xc(3)"), -1.0, 1e-9);
}

// NOTE: combnk scalar-set bug FIXED — see toolboxes/stats/tests/combnk_test.cpp.

// bugs/stats/jackknife.md — FIXED: jackknife_reg now does the leave-one-out
// loop inline (via callFunctionHandle), like bootstrp_reg, instead of
// delegating to a dead C++ stub. jackknife(@mean,[1 2 3 4 5]) is the 5x1
// column of leave-one-out means [3.5; 3.25; 3; 2.75; 2.5].
TEST_F(StatsKnownBug, JackknifeMean)
{
    // Vector: 5 leave-one-out means of [1..5].
    eval("j = jackknife(@mean, [1 2 3 4 5]);");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(j)")), 5);
    EXPECT_NEAR(evalScalar("j(1)"), 3.5, 1e-12);    // mean without obs 1
    EXPECT_NEAR(evalScalar("j(2)"), 3.25, 1e-12);
    EXPECT_NEAR(evalScalar("j(3)"), 3.0, 1e-12);
    EXPECT_NEAR(evalScalar("j(5)"), 2.5, 1e-12);    // mean without obs 5
    // Matrix: rows are observations -> n x K output.
    eval("J = jackknife(@(x) mean(x), [1 1; 2 2; 3 3]);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(J,1)")), 3);
    EXPECT_EQ(static_cast<int>(evalScalar("size(J,2)")), 2);
    EXPECT_NEAR(evalScalar("J(1,1)"), 2.5, 1e-12);  // drop row 1 -> mean(2,3)
    EXPECT_NEAR(evalScalar("J(3,1)"), 1.5, 1e-12);  // drop row 3 -> mean(1,2)
}

// bugs/closed/stats/rnd-size-vector-nd.md — 3+ element size vectors must
// produce N-D output; nk flattens to first × product(rest).
TEST_F(StatsKnownBug, RndSizeVectorND)
{
    eval("s = size(geornd(0.5, [2 3 4]));");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(s)")), 3);
    EXPECT_EQ(static_cast<int>(evalScalar("s(1)")), 2);
    EXPECT_EQ(static_cast<int>(evalScalar("s(2)")), 3);
    EXPECT_EQ(static_cast<int>(evalScalar("s(3)")), 4);
}

// bugs/opened/stats/mvnrnd-zero-n.md — an explicit zero/negative n must
// give a 0×d empty; the derive-from-mu sentinel intercepts it.
TEST_F(StatsKnownBug, MvnrndZeroN)
{
    eval("s = size(mvnrnd(0, 1, -2));");
    EXPECT_EQ(static_cast<int>(evalScalar("s(1)")), 0);
    EXPECT_EQ(static_cast<int>(evalScalar("s(2)")), 1);
}

// bugs/closed/stats/vecdim-rank-model.md — vecdim entries legal in the
// reference (implicit rank 2 for vectors; rank-4+ arrays) are rejected.
TEST_F(StatsKnownBug, MedianVecdimVector)
{
    eval("m = median(1:5, [1 2]);");
    EXPECT_DOUBLE_EQ(evalScalar("m"), 3.0);
}

TEST_F(StatsKnownBug, MedianVecdimScalar)
{
    eval("m = median(3, [1 2]);");
    EXPECT_DOUBLE_EQ(evalScalar("m"), 3.0);
}

TEST_F(StatsKnownBug, MedianVecdimRank4)
{
    eval("m = median(ones(2,3,4,5), [1 2 3 4]);");
    EXPECT_DOUBLE_EQ(evalScalar("m"), 1.0);
}

TEST_F(StatsKnownBug, IqrVecdimVector)
{
    eval("r = iqr(1:5, [1 2]);");
    EXPECT_DOUBLE_EQ(evalScalar("r"), 2.5);
}

TEST_F(StatsKnownBug, BoundsVecdimVector)
{
    eval("[lo, hi] = bounds(1:5, [1 2]);");
    EXPECT_DOUBLE_EQ(evalScalar("lo;"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("hi;"), 5.0);
}

// bugs/closed/stats/vecdim-partial.md — partial vecdim reduction must
// keep the non-listed dimensions instead of throwing.
TEST_F(StatsKnownBug, MedianVecdimPartial)
{
    eval("m = median(ones(2,2,2), [1 3]);");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(m)")), 2);
    EXPECT_DOUBLE_EQ(evalScalar("m(1)"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("m(2)"), 1.0);
}

// bugs/closed/stats/trimmean-vecdim.md — the dim slot must accept vecdim.
TEST_F(StatsKnownBug, TrimmeanVecdim)
{
    eval("t = trimmean(1:5, 10, [1 2]);");
    EXPECT_DOUBLE_EQ(evalScalar("t"), 3.0);
}

// bugs/closed/stats/vecdim-residual.md — zscore, range, mad, geomean,
// harmmean, and nan* accept vecdim ([1 2], [1 3]) and 'all'.
TEST_F(StatsKnownBug, VecdimResidual)
{
    EXPECT_DOUBLE_EQ(evalScalar("range(1:5, [1 2])"), 4.0);
    EXPECT_DOUBLE_EQ(evalScalar("mad(1:5, 0, [1 2])"), 1.2);
    EXPECT_NEAR(evalScalar("geomean([1 2 4], [1 2])"), 2.0, 1e-12);
    EXPECT_NEAR(evalScalar("harmmean([1 2 4], [1 2])"), 12.0 / 7.0, 1e-12);
    eval("[z, mu, sig] = zscore([1 2 3; 4 5 6], 0, [1 2]);");
    EXPECT_NEAR(evalScalar("mu"), 3.5, 1e-12);
    EXPECT_EQ(static_cast<int>(evalScalar("numel(z)")), 6);
    EXPECT_DOUBLE_EQ(evalScalar("nansum([1 2 NaN 4], [1 2])"), 7.0);
    EXPECT_DOUBLE_EQ(evalScalar("nanmean([1 2 NaN 6], [1 2])"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("nanmedian([1 2 NaN 9], [1 2])"), 2.0);
    EXPECT_DOUBLE_EQ(evalScalar("nanmax([1 5 NaN 3], [], [1 2])"), 5.0);
    EXPECT_DOUBLE_EQ(evalScalar("nanmin([1 5 NaN 3], [], [1 2])"), 1.0);
    EXPECT_NEAR(evalScalar("nanvar([1 2 NaN 3], 0, [1 2])"), 1.0, 1e-12);
    EXPECT_NEAR(evalScalar("nanstd([1 2 NaN 3], 0, [1 2])"), 1.0, 1e-12);
}

// bugs/closed/stats/rnd-size-vector-nd.md — the reference accepts an
// unbounded trailing size tail (geornd(p,2,3,4) -> 2x3x4).
TEST_F(StatsKnownBug, RndNdSizeTails)
{
    eval("r = geornd(0.5, 2, 3, 4);");
    eval("s = size(r);");
    EXPECT_DOUBLE_EQ(evalScalar("numel(s)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("s(1)"), 2.0);
    EXPECT_DOUBLE_EQ(evalScalar("s(2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("s(3)"), 4.0);
}

// bugs/opened/stats/arity-min-forms.md — 0-argument forms legal in the
// reference (fitdist() lists distributions; haltonset() defaults d;
// randg() -> scalar draw) are rejected by nk's min-1 regs.
TEST_F(StatsKnownBug, DISABLED_ZeroArgForms)
{
    eval("c = fitdist();");
    EXPECT_DOUBLE_EQ(evalScalar("iscell(c)"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("numel(c) > 5"), 1.0);
    eval("h = haltonset();");
    EXPECT_DOUBLE_EQ(evalScalar("isa(h, 'double') + numel(h)"), 2.0);
    eval("g = randg();");
    EXPECT_DOUBLE_EQ(evalScalar("isscalar(g)"), 1.0);
}

// bugs/closed/stats/of-helpers-raw-doubledata.md — the *_of compute
// helpers now route through the *_full cleaning contracts (FIXED).
TEST_F(StatsKnownBug, DetrendCharRejected)
{
    eval("s = 'ab';");
    try {
        stats::detrend_of(*engine.getVariable("s"), 1, engine.resource());
        FAIL() << "C++ path should reject a char input";
    } catch (const numkit::Error &e) {
        EXPECT_EQ(e.identifier().rfind("numkit:detrend:", 0), 0u) << e.identifier() << ": " << e.what();
    }
}

// Structural phase (2026-09-18): rescale's compute-side type contract —
// the reference rejects non-numeric/non-logical first arguments.
TEST_F(StatsKnownBug, RescaleTypeContract)
{
    EXPECT_ANY_THROW(eval("rescale('x');"));
    EXPECT_ANY_THROW(eval("rescale([1i 2i]);"));
    EXPECT_NO_THROW(eval("rescale([1 2 3]);"));
    EXPECT_NO_THROW(eval("rescale(logical([1 0 1]));"));
}

// bugs/closed/stats/ppmaks-coefs-layout.md — nk's pp-struct coefs accept
// both (d·L)×K and d×(L·k) layouts; the reference reads the 2×4 coefs
// below as 4 pieces / order 1 / dim 2.
TEST_F(StatsKnownBug, SplinePpLayout)
{
    eval("br = 0:4; co = [1 0.5 0.2 1; -1 2 0 0.5];");
    eval("pp = ppmak(br, co);");
    EXPECT_NEAR(evalScalar("fnbrk(pp, 'pieces')"), 4, 0);
    EXPECT_NEAR(evalScalar("fnbrk(pp, 'order')"), 1, 0);
    EXPECT_NEAR(evalScalar("fnbrk(pp, 'dim')"), 2, 0);
    eval("y = fnval(pp, 2.5);");
    EXPECT_NEAR(evalScalar("y(1)"), 0.2, 1e-12);
    EXPECT_NEAR(evalScalar("y(2)"), 0.0, 1e-12);
}
