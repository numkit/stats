// toolboxes/stats/tests/arity_bounds_test.cpp
//
// Max-arity rejection guards for the stats domain (argcheck migration,
// 2026-09-17). Every bounded contract probed against R2025b gets a
// max+1 call that must fail with nk:<fn>:maxnargin; checkNargin is the
// first reg statement, so the dummy values never reach compute code.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/value/error.hpp>
#include <gtest/gtest.h>

#include <vector>

namespace {

struct Case
{
    const char *fn;
    int maxArgs;
};

const std::vector<Case> kBounds = {
    {"anova1", 4},
    {"anova2", 3},
    {"augknt", 3},
    {"aveknt", 2},
    {"betafit", 2},
    {"betainv", 3},
    {"betalike", 2},
    {"betapdf", 3},
    {"betastat", 2},
    {"binocdf", 4},
    {"binofit", 3},
    {"binoinv", 3},
    {"binopdf", 3},
    {"binostat", 2},
    {"bounds", 3},
    {"brk2knt", 2},
    {"canoncorr", 2},
    {"chi2cdf", 3},
    {"chi2inv", 2},
    {"chi2pdf", 2},
    {"chi2stat", 1},
    {"cholcov", 2},
    {"classify", 5},
    {"combnk", 2},
    {"cophenet", 2},
    {"corrcov", 2},
    {"cov", 4},
    {"csapi", 3},
    {"datastats", 2},
    {"dummyvar", 1},
    {"evcdf", 5},
    {"evfit", 5},
    {"evinv", 5},
    {"evlike", 4},
    {"evpdf", 3},
    {"evstat", 2},
    {"expcdf", 4},
    {"expfit", 4},
    {"expinv", 4},
    {"explike", 4},
    {"exppdf", 2},
    {"expstat", 1},
    {"fcdf", 4},
    {"finv", 3},
    {"fncmb", 4},
    {"fnder", 2},
    {"fnint", 2},
    {"fpdf", 3},
    {"friedman", 3},
    {"fstat", 2},
    {"gamcdf", 5},
    {"gamfit", 5},
    {"gaminv", 5},
    {"gamlike", 4},
    {"gampdf", 3},
    {"gamstat", 2},
    {"geocdf", 3},
    {"geoinv", 2},
    {"geopdf", 2},
    {"geostat", 1},
    {"gevcdf", 4},
    {"gevfit", 3},
    {"gevinv", 4},
    {"gevlike", 2},
    {"gevpdf", 4},
    {"gevstat", 3},
    {"gpcdf", 4},
    {"gpfit", 3},
    {"gpinv", 4},
    {"gplike", 2},
    {"gppdf", 4},
    {"gpstat", 3},
    {"grp2idx", 1},
    {"hampel", 3},
    {"hygecdf", 5},
    {"hygeinv", 4},
    {"hygepdf", 4},
    {"hygestat", 3},
    {"inconsistent", 2},
    {"iqr", 2},
    {"iwishrnd", 3},
    {"jbtest", 3},
    {"knt2brk", 1},
    {"kruskalwallis", 3},
    {"kurtosis", 3},
    {"lhsnorm", 4},
    {"logncdf", 5},
    {"lognfit", 5},
    {"logninv", 5},
    {"lognlike", 4},
    {"lognpdf", 3},
    {"lognstat", 2},
    {"lscov", 4},
    {"mad", 3},
    {"mahal", 2},
    {"mnpdf", 2},
    {"mnrnd", 3},
    {"mode", 2},
    {"moment", 3},
    {"mvnpdf", 3},
    {"mvnrnd", 4},
    {"mvtpdf", 3},
    {"mvtrnd", 3},
    {"nancov", 2},
    {"nanmean", 2},
    {"nanmedian", 2},
    {"nanstd", 3},
    {"nansum", 2},
    {"nanvar", 3},
    {"nbincdf", 4},
    {"nbinfit", 3},
    {"nbininv", 3},
    {"nbinpdf", 3},
    {"nbinstat", 2},
    {"ncfcdf", 5},
    {"ncfinv", 4},
    {"ncfpdf", 4},
    {"ncfstat", 3},
    {"nctcdf", 4},
    {"nctinv", 3},
    {"nctpdf", 3},
    {"nctstat", 2},
    {"ncx2cdf", 4},
    {"ncx2inv", 3},
    {"ncx2pdf", 3},
    {"ncx2stat", 2},
    {"normcdf", 5},
    {"normfit", 5},
    {"norminv", 5},
    {"normlike", 4},
    {"normpdf", 3},
    {"normstat", 2},
    {"pcacov", 1},
    {"pcares", 2},
    {"poisscdf", 3},
    {"poissfit", 2},
    {"poissinv", 2},
    {"poisspdf", 2},
    {"poisstat", 1},
    {"ppmak", 3},
    {"prepareCurveData", 3},
    {"prepareSurfaceData", 4},
    {"range", 2},
    {"raylcdf", 2},
    {"raylfit", 2},
    {"raylinv", 2},
    {"raylpdf", 2},
    {"raylstat", 1},
    {"regress", 3},
    {"ridge", 4},
    {"robustfit", 7},
    {"skewness", 3},
    {"squareform", 2},
    {"std", 4},
    {"tabulate", 1},
    {"tcdf", 3},
    {"tiedrank", 5},
    {"tinv", 2},
    {"tpdf", 2},
    {"trimmean", 4},
    {"tstat", 1},
    {"unidcdf", 3},
    {"unidinv", 2},
    {"unidpdf", 2},
    {"unidstat", 1},
    {"unifcdf", 3},
    {"unifinv", 3},
    {"unifit", 2},
    {"unifpdf", 3},
    {"unifstat", 2},
    {"var", 4},
    {"wblcdf", 5},
    {"wblfit", 5},
    {"wblinv", 5},
    {"wbllike", 4},
    {"wblpdf", 3},
    {"wblstat", 2},
    {"wishrnd", 3},
    {"zscore", 3},
};

std::string callExpr(const Case &c)
{
    std::string s = c.fn;
    s += '(';
    for (int i = 0; i <= c.maxArgs; ++i) {
        if (i)
            s += ", ";
        s += '1';
    }
    s += ')';
    return s;
}

} // namespace

class StatsArityBoundsTest : public ::testing::TestWithParam<Case>
{
public:
    numkit::StandardEngine engine;
};

TEST_P(StatsArityBoundsTest, RejectsMaxPlusOne)
{
    const Case &c = GetParam();
    try {
        engine.eval(callExpr(c));
        FAIL() << "expected maxnargin error: " << callExpr(c);
    } catch (const numkit::Error &e) {
        EXPECT_EQ(e.identifier(), std::string("numkit:") + c.fn + ":maxnargin")
            << callExpr(c) << "\n  got: " << e.what();
    }
}

INSTANTIATE_TEST_SUITE_P(StatsArityBounds, StatsArityBoundsTest,
                         ::testing::ValuesIn(kBounds));
