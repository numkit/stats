// toolboxes/stats/tests/cvpartition_evalclusters_test.cpp
//
// Regression tests for cvpartition (training/test/repartition) and evalclusters.

#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace numkit;

class CvEvalClustersTest : public ::testing::Test
{
public:
    StandardEngine engine;
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

TEST_F(CvEvalClustersTest, HoldoutFractionAndCount)
{
    eval("rng(7); c = cvpartition(100, 'Holdout', 0.3); "
         "tr = training(c); te = test(c);");
    EXPECT_DOUBLE_EQ(evalScalar("c.NumObservations"), 100.0);
    EXPECT_DOUBLE_EQ(evalScalar("c.NumTestSets"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("c.TrainSize"), 70.0);
    EXPECT_DOUBLE_EQ(evalScalar("c.TestSize"), 30.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(tr)"), 70.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(te)"), 30.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(tr & te)"), 0.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(tr | te)"), 100.0);

    eval("c2 = cvpartition(50, 'Holdout', 12);");
    EXPECT_DOUBLE_EQ(evalScalar("c2.TestSize"), 12.0);
    EXPECT_DOUBLE_EQ(evalScalar("c2.TrainSize"), 38.0);
}

TEST_F(CvEvalClustersTest, KFoldLeaveoutResubstitutionAndStratified)
{
    eval("rng(11); c = cvpartition(25, 'KFold', 5); "
         "cov = zeros(25, 1); "
         "for k = 1:5, "
         "  tr = training(c, k); te = test(c, k); "
         "  assert(sum(tr & te) == 0 && sum(tr | te) == 25); "
         "  cov = cov + double(te); "
         "end");
    EXPECT_DOUBLE_EQ(evalScalar("all(cov == 1)"), 1.0);

    eval("cl = cvpartition(10, 'Leaveout'); "
         "cr = cvpartition(10, 'Resubstitution');");
    EXPECT_DOUBLE_EQ(evalScalar("cl.NumTestSets"), 10.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(test(cl, 4))"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(training(cl, 4))"), 9.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(training(cr))"), 10.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(test(cr))"), 10.0);

    // Stratified holdout preserves class proportions
    eval("rng(19); grp = [ones(60,1); 2*ones(40,1)]; "
         "cs = cvpartition(grp, 'Holdout', 0.25); "
         "te = test(cs); "
         "t1 = sum(te(1:60)); t2 = sum(te(61:100)); "
         "cs2 = repartition(cs);");
    EXPECT_DOUBLE_EQ(evalScalar("t1"), 15.0);
    EXPECT_DOUBLE_EQ(evalScalar("t2"), 10.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(test(cs2))"), 25.0);

    // Error branches for cvpartition / training / test / repartition
    EXPECT_THROW(eval("cvpartition(0, 'Holdout', 0.2)"), std::exception);
    EXPECT_THROW(eval("cvpartition(10, 'Holdout', 1.5)"), std::exception);
    EXPECT_THROW(eval("cvpartition(10, 'KFold', 11)"), std::exception);
    EXPECT_THROW(eval("cvpartition(10, 'UnknownMode')"), std::exception);
    EXPECT_THROW(eval("test(c, 0)"), std::exception);
    EXPECT_THROW(eval("training(c, 6)"), std::exception);
    EXPECT_THROW(eval("repartition(struct('a', 1))"), std::exception);
}

TEST_F(CvEvalClustersTest, EvalClustersAllThreeCriteriaAndLinkageAndMatrix)
{
    eval("rng(1); "
         "X = [randn(40,2); randn(40,2) + 12; randn(40,2) + [0 12]]; "
         "eCH  = evalclusters(X, 'kmeans', 'CalinskiHarabasz', 'KList', 1:5); "
         "eDB  = evalclusters(X, 'kmeans', 'DaviesBouldin', 'KList', 1:5); "
         "eSil = evalclusters(X, 'kmeans', 'Silhouette', 'KList', 1:5); "
         "eLnk = evalclusters(X, 'linkage', 'CalinskiHarabasz', 'KList', 1:5);");
    EXPECT_DOUBLE_EQ(evalScalar("eCH.OptimalK"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("eDB.OptimalK"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("eSil.OptimalK"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("eLnk.OptimalK"), 3.0);
    EXPECT_EQ(eval("eLnk.ClusteringFunction").toString(), "linkage");
    EXPECT_TRUE(std::isnan(evalScalar("eCH.CriterionValues(1)")));
    EXPECT_TRUE(std::isnan(evalScalar("eDB.CriterionValues(1)")));
    EXPECT_TRUE(std::isnan(evalScalar("eSil.CriterionValues(1)")));
    EXPECT_DOUBLE_EQ(evalScalar("numel(eCH.OptimalY)"), 120.0);

    // Precomputed cluster matrix form (columns = candidate partitions for k=2, 3, 4)
    eval("c2 = [ones(40,1); 2*ones(80,1)]; "
         "c3 = [ones(40,1); 2*ones(40,1); 3*ones(40,1)]; "
         "c4 = [ones(20,1); 4*ones(20,1); 2*ones(40,1); 3*ones(40,1)]; "
         "ePre = evalclusters(X, [c2 c3 c4], 'CalinskiHarabasz');");
    EXPECT_DOUBLE_EQ(evalScalar("ePre.OptimalK"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("ePre.OptimalIndex"), 2.0);
    EXPECT_EQ(eval("ePre.ClusteringFunction").toString(), "custom");

    // Error branches: unknown criterion, unknown clustering algorithm, missing KList, bad shape
    EXPECT_THROW(eval("evalclusters(X, 'kmeans', 'NoSuchCriterion', 'KList', 1:3)"), std::exception);
    EXPECT_THROW(eval("evalclusters(X, 'bogus_algo', 'CalinskiHarabasz', 'KList', 1:3)"), std::exception);
    EXPECT_THROW(eval("evalclusters(X, 'kmeans', 'CalinskiHarabasz')"), std::exception);
    EXPECT_THROW(eval("evalclusters(X, ones(5, 2), 'CalinskiHarabasz')"), std::exception);
}
