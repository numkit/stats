// Clustering-family type/behavior contracts, probed against the
// reference engine R2025b (build/probe77.m + the full-precision
// recheck): pdist / pdist2 / mahal / linkage / cluster /
// inconsistent / silhouette. One TEST_F per probed contract branch.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/stats/cluster/distance.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace {

class ClusterContractTest : public ::testing::Test {
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
        engine.eval("X = [0 0; 1 0; 0 1; 1 1; 3 3];");
        engine.eval("Yp = pdist(X);");
    }
};

TEST_F(ClusterContractTest, PdistMetricsAndGates)
{
    engine.eval("d1 = pdist(X);");
    EXPECT_NEAR(num("sum(d1)"), 21.110597487539646, 1e-9);
    EXPECT_NEAR(num("sum(pdist(X, 'cityblock'))"), 28.0, 1e-9);
    engine.eval("d3 = pdist(X, 'minkowski', 3);");
    EXPECT_NEAR(at("d3", 0), 1.0, 1e-12);
    engine.eval("d4 = pdist(X, 'mahalanobis');");
    EXPECT_NEAR(at("d4", 0), 1.477097891751993, 1e-9);
    // Single-row input → 1×0 empty (probe P06); single stays single.
    EXPECT_EQ(num("numel(pdist(ones(1,2)))"), 0.0);
    EXPECT_EQ(cls("pdist(single(X))"), "single");
    // Unknown metric (probe P05).
    EXPECT_EQ(idOf("pdist(X, 'bogus')"),
              "numkit:pdist:distanceFunctionNotFound");
    // Prefix matching ('city' → cityblock).
    EXPECT_NEAR(num("sum(pdist(X, 'city'))"), 28.0, 1e-9);
}

TEST_F(ClusterContractTest, Pdist2MahalFormsAndGates)
{
    engine.eval("D8 = pdist2(X, X(1:2,:));");
    EXPECT_NEAR(num("sum(D8(:))"), 14.676619087329465, 1e-9);
    // The nk top-k extension (removed in the reference — filed) still
    // works with the metric in any position.
    engine.eval("[D9, I9] = pdist2(X, X(1:2,:), 'euclidean', 'Smallest', 2);");
    EXPECT_EQ(num("size(D9,1)"), 2.0);
    EXPECT_EQ(num("size(D9,2)"), 2.0);
    // Unknown distance (probe P10).
    EXPECT_EQ(idOf("pdist2(X, X(1:2,:), 'bogus')"),
              "numkit:pdist2:unrecognizedDistance");
    // mahal values (probe P11).
    engine.eval("md = mahal(X(2:5,:), X);");
    EXPECT_NEAR(at("md", 0), 2.1818181818181812, 1e-9);
    EXPECT_NEAR(at("md", 2), 0.0, 1e-12);
    EXPECT_NEAR(at("md", 3), 2.9090909090909087, 1e-9);
}

TEST_F(ClusterContractTest, LinkageClusterInconsistentSilhouette)
{
    engine.eval("Z1 = linkage(Yp);");
    EXPECT_NEAR(at("Z1", 0), 3.0, 1e-12);
    EXPECT_NEAR(at("Z1", 2), 1.0, 1e-12);
    EXPECT_NEAR(at("Z1", 11), 2.8284271247461903, 1e-9);
    // Raw-data form with method and metric (probes L02/L03).
    engine.eval("Z2 = linkage(X, 'complete');");
    EXPECT_NEAR(at("Z2", 10), 1.4142135623730951, 1e-9);
    engine.eval("Z3 = linkage(X, 'single', 'cityblock');");
    EXPECT_NEAR(at("Z3", 11), 4.0, 1e-12);
    // Method / metric / criterion gates (probes L04/L09 + P05-class).
    EXPECT_EQ(idOf("linkage(Yp, 'bogus')"), "numkit:linkage:badMethod");
    EXPECT_EQ(idOf("linkage(X, 'single', 'bogus')"),
              "numkit:linkage:distanceFunctionNotFound");
    EXPECT_EQ(idOf("cluster(Z1, 'criterion', 'bogus')"),
              "numkit:cluster:badCriterion");
    // inconsistent values + depth (probes L05/L06).
    engine.eval("I5 = inconsistent(Z1);");
    EXPECT_NEAR(at("I5", 0), 1.0, 1e-12);
    EXPECT_NEAR(at("I5", 3), 1.9142135623730951, 1e-9);
    engine.eval("I6 = inconsistent(Z1, 3);");
    EXPECT_NEAR(at("I6", 3), 1.6094757082487299, 1e-9);
    EXPECT_EQ(idOf("inconsistent(Z1, 0)"), "numkit:inconsistent:badDepth");
    // cluster maxclust / cutoff (probes L07/L08).
    // Cluster labels are arbitrary between engines — assert the
    // partition structure (4 + 1 split, probe L07/L08).
    engine.eval("c7 = cluster(Z1, 'maxclust', 2); "
                "u7 = unique(c7); n7 = histcounts(c7, [0.5 1.5 2.5]);");
    EXPECT_EQ(num("numel(u7)"), 2.0);
    EXPECT_NEAR(num("max(n7) - min(n7)"), 3.0, 1e-12);
    engine.eval("c8 = cluster(Z1, 'cutoff', 0.5); "
                "n8 = histcounts(c8, [0.5 1.5 2.5]);");
    EXPECT_NEAR(num("max(n8) - min(n8)"), 3.0, 1e-12);
    EXPECT_EQ(idOf("cluster(Z1)"), "numkit:cluster:needOption");
    // silhouette values (probe S01).
    engine.eval("s = silhouette(X, [1 1 2 2 2]);");
    EXPECT_NEAR(at("s", 0), 0.8571428571428571, 1e-9);
    EXPECT_NEAR(at("s", 2), -0.7857142857142857, 1e-9);
    EXPECT_NEAR(at("s", 4), 0.32258064516129031, 1e-9);
    EXPECT_EQ(idOf("silhouette(X, [1 1 2 2 2], 'bogus')"),
              "numkit:silhouette:distanceFunctionNotFound");
}

} // namespace
