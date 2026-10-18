// toolboxes/stats/tests/descriptive_contract_test.cpp
//
// Probed R2025b contracts for the descriptive family (probe68, stats
// argcheck-drain portion 2026-09-18): var/std weight+flag forms and
// class gates, median class preservation + missing-value aliases,
// corrcoef name-value strictness, nan* 'all' flag and single contagion.

#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class DescriptiveContractTest : public ::testing::Test
{
public:
    StandardEngine engine;
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
    std::string evalClass(const std::string &c) { return eval(c).toString(); }
    bool throwsWith(const std::string &c, const std::string &id)
    {
        try {
            eval(c);
        } catch (const Error &e) {
            return e.identifier() == id;
        }
        return false;
    }
};

TEST_F(DescriptiveContractTest, VarStdWeightAndFlagForms)
{
    // integers reject (integerClass); char as W rejects (unknownFlag —
    // flags only live in the dim position); logical/complex/char X pass.
    EXPECT_TRUE(throwsWith("var(int8([1 2 3]))", "numkit:var:integerClass"));
    EXPECT_TRUE(throwsWith("std(int8([1 2 3]))", "numkit:std:integerClass"));
    EXPECT_TRUE(throwsWith("var([1 2 3], 'all')", "numkit:var:unknownFlag"));
    EXPECT_TRUE(throwsWith("var([1 2 3], 2)", "numkit:var:invalidWgts"));
    EXPECT_TRUE(throwsWith("var([1 2 3], [1 2])", "numkit:var:invalidSizeWgts"));
    // valid weight + weighted dim form (probed V24: 0.5556)
    EXPECT_NEAR(evalScalar("var([1 2 3; 4 5 6; 7 8 9], [1 2 3], 2)(1)"),
                0.555556, 1e-5);
    EXPECT_NEAR(evalScalar("var([1 2 NaN 4], 1, 'omitnan')"), 1.55556, 1e-5);
    // single contagion + char/logical widening
    EXPECT_EQ(evalClass("class(var(single([1 2 3])))"), "single");
    EXPECT_EQ(evalClass("class(std(single([1 2 3]), 1))"), "single");
    eval("ee = var('ab', 1);");
    EXPECT_NEAR(evalScalar("ee"), 0.25, 1e-12);
}

TEST_F(DescriptiveContractTest, MedianClassAndFlags)
{
    // integer + logical classes are PRESERVED (probed M05/M06)
    EXPECT_EQ(evalClass("class(median(int8([1 2 3; 4 5 6; 7 8 9])))"), "int8");
    EXPECT_EQ(evalClass("class(median(logical([1 0 1; 0 1 0; 1 1 0])))"),
              "logical");
    EXPECT_EQ(evalClass("class(median(single([1 2 3])))"), "single");
    // missing-value flag aliases work ('omitmissing' == 'omitnan')
    EXPECT_NEAR(evalScalar("median([1 NaN 3], 'omitmissing')"), 2.0, 1e-12);
    EXPECT_NEAR(evalScalar("median([1 NaN 3], 'omitnan')"), 2.0, 1e-12);
    // unknown option rejects with the reference's unknownOption class
    EXPECT_TRUE(throwsWith("median([1 2 3], 'type')", "numkit:median:unknownOption"));
    EXPECT_TRUE(std::isnan(evalScalar("median([])")));
    EXPECT_NEAR(evalScalar("median('ab')"), 97.5, 1e-12);
}

TEST_F(DescriptiveContractTest, CorrcoefNameValueStrictness)
{
    EXPECT_TRUE(throwsWith("corrcoef([1 2; 3 4], 'Pearson', 'Type')",
                           "numkit:corrcoef:invalidArgName"));
    EXPECT_TRUE(throwsWith("corrcoef([1 2; 3 4], 'Kendall')",
                           "numkit:corrcoef:unmatchedPVPair"));
    EXPECT_TRUE(throwsWith("corrcoef([1 2; 3 4], 'Rows', 'invalid')",
                           "numkit:corrcoef:invalidRowChoice"));
    EXPECT_TRUE(throwsWith("corrcoef(int8([1 2; 3 4]))",
                           "numkit:corrcoef:mixedClasses"));
    // Alpha bounds compute fine
    eval("r = corrcoef([1 2; 3 4], 'Alpha', 0.1);");
    // complex input: R alone computes Hermitian correlation; [R,P] rejects with ComplexInputs.
    eval("rc = corrcoef([1+1i 2-1i; 3+2i 4+3i; 5-1i 6+2i]);");
    EXPECT_NEAR(evalScalar("real(rc(1,1))"), 1.0, 1e-12);
    EXPECT_TRUE(throwsWith("[rc, pc] = corrcoef([1+1i 2-1i; 3+2i 4+3i])",
                           "numkit:corrcoef:ComplexInputs"));
}

TEST_F(DescriptiveContractTest, NanFamilyAllFlagAndSingle)
{
    // 'all' flag computes (was a dimNotScalar rejection)
    EXPECT_NEAR(evalScalar("nanmean([1 2; 3 4], 'all')"), 2.5, 1e-12);
    EXPECT_NEAR(evalScalar("nansum([1 2; 3 4], 'all')"), 10.0, 1e-12);
    // single contagion (probed N09)
    EXPECT_EQ(evalClass("class(nansum(single([1 NaN 3])))"), "single");
    EXPECT_EQ(evalClass("class(nanmean(single([1 NaN 3])))"), "single");
    // Language-level nanstd is backed by C++ stats::nanstdev (matching
    // stats::stdev for std, since std is a C++ reserved namespace).
    EXPECT_NEAR(evalScalar("nanstd([1 NaN 3])"), 1.41421356237, 1e-9);
    EXPECT_NEAR(evalScalar("std([1 NaN 3], 'omitnan')"), 1.41421356237, 1e-9);
}
