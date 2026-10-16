// Spline-family contract additions, probed against the reference
// engine R2025b (build/probe78.m): ppmak's strictly-increasing breaks
// gate and fnbrk's breaks fallback for unknown parts. The pp-struct
// coefs LAYOUT divergence (nk rows = d·pieces vs the reference
// L×(k·d)) is filed as bugs/opened/stats/ppmaks-coefs-layout.md with
// a DISABLED guard — these tests pin only the layout-independent
// contract branches.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <gtest/gtest.h>

#include <string>

namespace {

class SplineContractTest : public ::testing::Test {
public:
    numkit::StandardEngine engine;
    numkit::Value eval(const std::string &c) { return engine.eval(c); }
    double num(const std::string &c)
    {
        return eval(c).elemAsDouble(0);
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

TEST_F(SplineContractTest, PpmakBreaksGate)
{
    // Decreasing breaks reject (probe78 S04); repeated extremes
    // reject (S05) — the reference reaches the piece-count classes,
    // nk voices the monotonicity directly with the same rejection.
    EXPECT_EQ(idOf("ppmak([4 3 2 1], ones(3, 2))"),
              "numkit:ppmak:breaks");
    EXPECT_EQ(idOf("ppmak([0 1 1 2], ones(3, 2))"),
              "numkit:ppmak:breaks");
    // A valid strictly-increasing sequence with matching pieces works.
    engine.eval("pp = ppmak(0:4, ones(4, 2));");
    EXPECT_EQ(num("fnbrk(pp, 'pieces')"), 4.0);
    EXPECT_EQ(num("fnbrk(pp, 'order')"), 2.0);
}

TEST_F(SplineContractTest, FnbrkPartsAndFallback)
{
    engine.eval("pp = ppmak(0:4, ones(4, 2));");
    EXPECT_EQ(num("fnbrk(pp, 'pieces')"), 4.0);
    EXPECT_EQ(num("fnbrk(pp, 'order')"), 2.0);
    EXPECT_EQ(num("fnbrk(pp, 'dim')"), 1.0);
    EXPECT_EQ(num("numel(fnbrk(pp, 'breaks'))"), 5.0);
    EXPECT_EQ(num("numel(fnbrk(pp, 'coefs'))"), 8.0);
    EXPECT_EQ(eval("fnbrk(pp, 'form')").toString(), "pp");
    // Unknown part names fall back to the breaks (probe78 S13).
    engine.eval("fb = fnbrk(pp, 'bogus');");
    EXPECT_EQ(num("numel(fb)"), 5.0);
    EXPECT_NEAR(num("fb(3)"), 2.0, 1e-12);
}

TEST_F(SplineContractTest, FnvalAndFncmbForms)
{
    engine.eval("pp = ppmak(0:4, [1 2; 3 4; 5 6; 7 8]);");
    engine.eval("y = fnval(pp, [1.5 2.5]);");
    EXPECT_NEAR(num("y(1)"), 5.5, 1e-12);
    EXPECT_NEAR(num("y(2)"), 8.5, 1e-12);
    // fncmb scale and linear combination (nk layout).
    engine.eval("p2 = fncmb(pp, 2); y2 = fnval(p2, 1.5);");
    EXPECT_NEAR(num("y2"), 11.0, 1e-12);
    engine.eval("p3 = fncmb(pp, 1, pp, -1); y3 = fnval(p3, 2.5);");
    EXPECT_NEAR(num("y3"), 0.0, 1e-12);
    // Non-spline fncmb pairings reject.
    EXPECT_NE(idOf("fncmb(ones(2), ones(2))"), "<no-error>");
}

} // namespace
