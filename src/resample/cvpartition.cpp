// toolboxes/stats/src/resample/cvpartition.cpp
//
// Engine-free compute implementation for cvpartition, training, test, and
// repartition (Phase 2b compute/register split).

#include "resample/resample_detail.hpp"

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/random.h>
#include <keel/string.h>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::stats {

namespace {

static void shuffleIndices(::numkit::ops::RngContext &rng, keel::Array<keel::usize> &idx)
{
    for (keel::usize i = idx.size(); i > 1; --i) {
        const keel::usize j = keel::as<keel::usize>(
            keel::random::uniform_i64(rng, 0, keel::as<keel::i64>(i - 1)));
        const keel::usize tmp = idx[i - 1];
        idx[i - 1] = idx[j];
        idx[j] = tmp;
    }
}

static Value buildCvPartitionStruct(
    keel::StringRef typeLower, keel::usize nObs, keel::usize numTestSets,
    const keel::Array<keel::i32> &foldIds, bool stratified,
    keel::Allocator *mr)
{
    Value foldVal = Value::matrix(nObs, 1, ValueType::DOUBLE, mr);
    keel::f64 *fd = foldVal.doubleDataMut();
    for (keel::usize i = 0; i < nObs; ++i)
        fd[i] = keel::as<keel::f64>(foldIds[i]);

    Value trainSz = Value::matrix(1, numTestSets, ValueType::DOUBLE, mr);
    Value testSz = Value::matrix(1, numTestSets, ValueType::DOUBLE, mr);
    keel::f64 *trd = trainSz.doubleDataMut();
    keel::f64 *tsd = testSz.doubleDataMut();
    if (typeLower == "resubstitution") {
        trd[0] = keel::as<keel::f64>(nObs);
        tsd[0] = keel::as<keel::f64>(nObs);
    } else if (typeLower == "holdout") {
        keel::usize nTest = 0;
        for (keel::usize i = 0; i < nObs; ++i)
            if (foldIds[i] == 1) ++nTest;
        trd[0] = keel::as<keel::f64>(nObs - nTest);
        tsd[0] = keel::as<keel::f64>(nTest);
    } else {
        for (keel::usize k = 0; k < numTestSets; ++k) {
            keel::usize nTest = 0;
            const keel::i32 target = keel::as<keel::i32>(k + 1);
            for (keel::usize i = 0; i < nObs; ++i)
                if (foldIds[i] == target) ++nTest;
            trd[k] = keel::as<keel::f64>(nObs - nTest);
            tsd[k] = keel::as<keel::f64>(nTest);
        }
    }

    Value c = Value::structure(mr);
    c.field("Type") = Value::fromString(typeLower, mr);
    c.field("NumObservations") = Value::scalar(keel::as<keel::f64>(nObs), mr);
    c.field("NumTestSets") = Value::scalar(keel::as<keel::f64>(numTestSets), mr);
    c.field("TrainSize") = keel::move(trainSz);
    c.field("TestSize") = keel::move(testSz);
    c.field("ImplFoldId_") = keel::move(foldVal);
    c.field("IsStratified_") = Value::logicalScalar(stratified, mr);
    return c;
}

} // namespace

Value cvpartition_compute(::numkit::ops::RngContext &rng,
                          const Value &nOrGroup,
                          keel::StringRef typeStr,
                          const Value *paramArg,
                          keel::Allocator *mr)
{
    keel::usize nObs = 0;
    bool stratified = false;
    keel::Array<keel::Array<keel::usize>> strata;
    if (nOrGroup.isNumeric() && nOrGroup.isScalar()) {
        const keel::f64 nv = nOrGroup.toScalar();
        if (!keel::is_finite(nv) || nv < 1.0 || nv != keel::floor(nv))
            throw Error("cvpartition: N must be a positive integer",
                        0, 0, "cvpartition", "", "numkit:cvpartition:badN");
        nObs = keel::as<keel::usize>(nv);
        keel::Array<keel::usize> all;
        all.reserve(nObs);
        for (keel::usize i = 0; i < nObs; ++i) all.push_back(i);
        strata.push_back(keel::move(all));
    } else {
        nObs = nOrGroup.numel();
        if (nObs < 1)
            throw Error("cvpartition: grouping vector must be nonempty",
                        0, 0, "cvpartition", "", "numkit:cvpartition:emptyGroup");
        stratified = true;
        keel::Array<keel::String> labels;
        for (keel::usize i = 0; i < nObs; ++i) {
            keel::String key;
            if (nOrGroup.isNumeric() || nOrGroup.isLogical()) {
                key = keel::format("{:.17g}", nOrGroup.elemAsDouble(i));
            } else if (nOrGroup.isString()) {
                key = keel::String::from(nOrGroup.stringElem(i));
            } else if (nOrGroup.isCell()) {
                key = nOrGroup.cellAt(i).toString();
            } else {
                key = nOrGroup.toString();
            }
            keel::usize found = labels.size();
            for (keel::usize j = 0; j < labels.size(); ++j) {
                if (labels[j] == key) { found = j; break; }
            }
            if (found == labels.size()) {
                labels.push_back(keel::move(key));
                strata.push_back(keel::Array<keel::usize>{});
            }
            strata[found].push_back(i);
        }
    }

    keel::String typeLower = keel::String::from(typeStr);
    typeLower.to_ascii_lower_in_place();

    keel::Array<keel::i32> foldIds(nObs, 0);
    keel::usize numTestSets = 1;

    if (typeLower == "holdout") {
        keel::f64 p = 0.1;
        if (paramArg && paramArg->isNumeric() && paramArg->isScalar())
            p = paramArg->toScalar();
        else if (paramArg)
            throw Error("cvpartition: Holdout parameter must be a real scalar",
                        0, 0, "cvpartition", "", "numkit:cvpartition:badHoldout");
        if (!keel::is_finite(p) || p <= 0.0 || p >= keel::as<keel::f64>(nObs))
            throw Error("cvpartition: Holdout value must be in (0, 1) or integer in [1, N-1]",
                        0, 0, "cvpartition", "", "numkit:cvpartition:badHoldout");
        const bool isFrac = (p < 1.0);
        if (!isFrac && p != keel::floor(p))
            throw Error("cvpartition: Holdout count >= 1 must be an integer",
                        0, 0, "cvpartition", "", "numkit:cvpartition:badHoldout");
        for (keel::usize s = 0; s < strata.size(); ++s) {
            shuffleIndices(rng, strata[s]);
            keel::usize nTest = isFrac
                ? keel::as<keel::usize>(keel::floor(p * keel::as<keel::f64>(strata[s].size()) + 0.5))
                : keel::as<keel::usize>(keel::floor(p * keel::as<keel::f64>(strata[s].size())
                                        / keel::as<keel::f64>(nObs) + 0.5));
            if (strata.size() == 1 && !isFrac) nTest = keel::as<keel::usize>(p);
            if (nTest >= strata[s].size() && strata[s].size() > 1) nTest = strata[s].size() - 1;
            for (keel::usize i = 0; i < nTest; ++i)
                foldIds[strata[s][i]] = 1;
        }
        numTestSets = 1;
    } else if (typeLower == "kfold") {
        keel::usize k = 10;
        if (paramArg) {
            if (!paramArg->isNumeric() || !paramArg->isScalar())
                throw Error("cvpartition: KFold K must be a positive integer scalar",
                            0, 0, "cvpartition", "", "numkit:cvpartition:badKFold");
            const keel::f64 kv = paramArg->toScalar();
            if (!keel::is_finite(kv) || kv < 2.0 || kv > keel::as<keel::f64>(nObs) || kv != keel::floor(kv))
                throw Error("cvpartition: KFold K must be an integer in [2, N]",
                            0, 0, "cvpartition", "", "numkit:cvpartition:badKFold");
            k = keel::as<keel::usize>(kv);
        } else if (k > nObs) {
            k = nObs;
        }
        keel::usize cursor = 0;
        for (keel::usize s = 0; s < strata.size(); ++s) {
            shuffleIndices(rng, strata[s]);
            for (keel::usize i = 0; i < strata[s].size(); ++i) {
                foldIds[strata[s][i]] = keel::as<keel::i32>((cursor % k) + 1);
                ++cursor;
            }
        }
        numTestSets = k;
    } else if (typeLower == "leaveout") {
        for (keel::usize i = 0; i < nObs; ++i)
            foldIds[i] = keel::as<keel::i32>(i + 1);
        numTestSets = nObs;
    } else if (typeLower == "resubstitution") {
        numTestSets = 1;
    } else {
        throw Error(keel::format("cvpartition: unknown partition type '{}'", typeLower),
                    0, 0, "cvpartition", "", "numkit:cvpartition:badType");
    }

    return buildCvPartitionStruct(typeLower.as_ref(), nObs, numTestSets,
                                  foldIds, stratified, mr);
}

Value cvpartition_mask(const Value &c, bool isTest, keel::i32 foldIdx,
                       const char *fnName, keel::Allocator *mr)
{
    if (!c.isStruct() || !c.hasField("ImplFoldId_") || !c.hasField("Type"))
        throw Error(keel::format("{}: first argument must be a cvpartition object", fnName),
                    0, 0, fnName, "", keel::format("numkit:{}:invalidPartition", fnName));
    const Value &foldVal = c.field("ImplFoldId_");
    const keel::usize nObs = foldVal.numel();
    const keel::usize numTestSets = keel::as<keel::usize>(c.field("NumTestSets").toScalar());
    if (foldIdx < 1 || keel::as<keel::usize>(foldIdx) > numTestSets)
        throw Error(keel::format("{}: fold index {} out of range [1, {}]", fnName, foldIdx, numTestSets),
                    0, 0, fnName, "", keel::format("numkit:{}:badFoldIndex", fnName));
    const keel::String type = c.field("Type").toString();
    Value out = Value::matrix(nObs, 1, ValueType::LOGICAL, mr);
    keel::u8 *od = out.logicalDataMut();
    const keel::f64 *fd = foldVal.doubleData();
    if (type == "resubstitution") {
        for (keel::usize i = 0; i < nObs; ++i) od[i] = 1u;
    } else if (type == "holdout") {
        for (keel::usize i = 0; i < nObs; ++i) {
            const bool inTest = (keel::as<keel::i32>(fd[i]) == 1);
            od[i] = (isTest ? inTest : !inTest) ? 1u : 0u;
        }
    } else {
        for (keel::usize i = 0; i < nObs; ++i) {
            const bool inTest = (keel::as<keel::i32>(fd[i]) == foldIdx);
            od[i] = (isTest ? inTest : !inTest) ? 1u : 0u;
        }
    }
    return out;
}

Value cvpartition_repartition(::numkit::ops::RngContext &rng,
                              const Value &c,
                              keel::Allocator *mr)
{
    if (!c.isStruct() || !c.hasField("ImplFoldId_") || !c.hasField("Type"))
        throw Error("repartition: argument must be a cvpartition object",
                    0, 0, "repartition", "", "numkit:repartition:invalidPartition");
    const keel::String type = c.field("Type").toString();
    const Value &oldFold = c.field("ImplFoldId_");
    const keel::usize nObs = oldFold.numel();
    const keel::usize numTestSets = keel::as<keel::usize>(c.field("NumTestSets").toScalar());
    const bool stratified = c.hasField("IsStratified_") && c.field("IsStratified_").toBool();
    keel::Array<keel::i32> foldIds(nObs, 0);
    for (keel::usize i = 0; i < nObs; ++i)
        foldIds[i] = keel::as<keel::i32>(oldFold.elemAsDouble(i));
    if (type == "holdout" || type == "kfold") {
        keel::Array<keel::usize> perm;
        perm.reserve(nObs);
        for (keel::usize i = 0; i < nObs; ++i) perm.push_back(i);
        shuffleIndices(rng, perm);
        keel::Array<keel::i32> shuffled(nObs, 0);
        for (keel::usize i = 0; i < nObs; ++i) shuffled[perm[i]] = foldIds[i];
        foldIds = keel::move(shuffled);
    }
    return buildCvPartitionStruct(type.as_ref(), nObs, numTestSets,
                                  foldIds, stratified, mr);
}

} // namespace numkit::stats
