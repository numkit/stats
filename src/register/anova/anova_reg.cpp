// toolboxes/signal/src/anova/anova_reg.cpp
//
// CallContext register half of anova/anova.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/array.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/anova/anova.hpp>
#include <numkit/stats/distributions/chi2.hpp>
#include <numkit/stats/distributions/fisher_f.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "anova/anova_detail.hpp"
#include <numkit/value/error.hpp>

#include <keel/number.h>
#include <keel/tuple.h>

namespace numkit::stats {

namespace detail {

void anova1_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "anova1");
    auto *mr = ctx.engine->allocator();

    // Two input forms:
    //   anova1(y, group)  — y vector, group is the grouping variable
    //   anova1(X)         — X a matrix (>=2 cols): each COLUMN is a group.
    // For the matrix form, stack the columns into (y, group) with group =
    // 1-based column index, then run the existing one-way ANOVA. NaNs are
    // dropped by bucket(), matching MATLAB's column-per-group handling.
    Value yStore, gStore;
    const Value *yp = nullptr, *gp = nullptr;
    if (args[0].dims().rows() > 1 && args[0].dims().cols() > 1) {
        const Value &X = args[0];
        const keel::usize R = X.dims().rows(), C = X.dims().cols();
        yStore = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        gStore = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        keel::f64 *yd = yStore.doubleDataMut();
        keel::f64 *gd = gStore.doubleDataMut();
        keel::usize k = 0;
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize r = 0; r < R; ++r) {
                yd[k] = X.elemAsDouble(c * R + r);
                gd[k] = keel::as<keel::f64>(c + 1);
                ++k;
            }
        yp = &yStore;
        gp = &gStore;
    } else {
        if (args.size() < 2)
            throw Error("anova1: requires (y, group[, 'off'])",
                        0, 0, "anova1", "", "numkit:anova1:nargin");
        yp = &args[0];
        gp = &args[1];
    }
    auto [p, F, dfB, dfW, ssB, ssW] = anova1(*yp, *gp, mr);
    outs[0] = Value::scalar(p, mr);
    if (nargout > 1) {
        // 4×6 cell table { 'Source','SS','df','MS','F','Prob>F'
        //                  'Groups',ssB,dfB,ssB/dfB,F,p
        //                  'Error', ssW,dfW,ssW/dfW,[],[]
        //                  'Total', ssB+ssW,dfB+dfW,[],[],[] }.
        Value tbl = Value::cell(4, 6, mr);
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        auto strV = [&](const char *s){ return Value::fromString(s, mr); };
        auto numV = [&](keel::f64 v){ return Value::scalar(v, mr); };
        auto setCell = [&](keel::usize r, keel::usize c, const Value &v) {
            // column-major linear index for 4×6 cell: r + c*4
            tbl.cellAt(r + c * 4) = v;
        };
        setCell(0, 0, strV("Source"));
        setCell(0, 1, strV("SS"));
        setCell(0, 2, strV("df"));
        setCell(0, 3, strV("MS"));
        setCell(0, 4, strV("F"));
        setCell(0, 5, strV("Prob>F"));
        setCell(1, 0, strV("Groups"));
        setCell(1, 1, numV(ssB));
        setCell(1, 2, numV(dfB));
        setCell(1, 3, numV(dfB > 0 ? ssB / dfB : nan));
        setCell(1, 4, numV(F));
        setCell(1, 5, numV(p));
        setCell(2, 0, strV("Error"));
        setCell(2, 1, numV(ssW));
        setCell(2, 2, numV(dfW));
        setCell(2, 3, numV(dfW > 0 ? ssW / dfW : nan));
        setCell(2, 4, numV(nan));
        setCell(2, 5, numV(nan));
        setCell(3, 0, strV("Total"));
        setCell(3, 1, numV(ssB + ssW));
        setCell(3, 2, numV(dfB + dfW));
        setCell(3, 3, numV(nan));
        setCell(3, 4, numV(nan));
        setCell(3, 5, numV(nan));
        outs[1] = keel::move(tbl);
    }
    if (nargout > 2) {
        // Re-bucket to populate the full stats struct expected by
        // post-hoc tools (multcompare, etc.). The data scan is cheap;
        // we only do it here, when the third output is actually
        // requested.
        auto buckets = bucket(*yp, *gp);
        const keel::usize K = buckets.size();
        // means + per-group sizes.
        Value meansV = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        Value nV     = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        Value gnamesV = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        keel::f64 *mD = meansV.doubleDataMut();
        keel::f64 *nD = nV.doubleDataMut();
        keel::f64 *gD = gnamesV.doubleDataMut();
        keel::f64 sse = 0.0;
        keel::usize total = 0;
        for (keel::usize k = 0; k < K; ++k) {
            const auto &g = buckets[k];
            const keel::usize ng = g.values.size();
            keel::f64 sum = 0.0;
            for (keel::f64 x : g.values) sum += x;
            const keel::f64 m = (ng > 0) ? sum / keel::as<keel::f64>(ng) : 0.0;
            mD[k] = m;
            nD[k] = keel::as<keel::f64>(ng);
            gD[k] = g.label;
            for (keel::f64 x : g.values) {
                const keel::f64 d = x - m;
                sse += d * d;
            }
            total += ng;
        }
        const keel::f64 sPooled = (total > K)
            ? keel::sqrt(sse / keel::as<keel::f64>(total - K))
            : keel::Limits<keel::f64>::quiet_nan();

        Value s = Value::structure(mr);
        s.field("F")      = Value::scalar(F, mr);
        s.field("df")     = Value::scalar(dfW, mr);
        s.field("means")  = keel::move(meansV);
        s.field("n")      = keel::move(nV);
        s.field("gnames") = keel::move(gnamesV);
        s.field("s")      = Value::scalar(sPooled, mr);
        s.field("source") = Value::fromString("anova1", mr);
        outs[2] = keel::move(s);
    }
}

void kruskalwallis_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "kruskalwallis");
    auto *mr = ctx.engine->allocator();

    // Matrix-only form: when group is omitted (or the 2nd arg is the
    // 'off'/'on' display flag string), infer groups from columns. Build
    // a flat y column and a same-length group index column.
    Value yArg = args[0];
    Value gArg;
    bool haveExplicitGroup = false;
    keel::usize flagPos = 1;
    if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()
        && !args[1].isEmpty()) {
        gArg = args[1];
        haveExplicitGroup = true;
        flagPos = 2;
    }
    if (!haveExplicitGroup && !args[0].dims().isVector() && !args[0].isScalar()) {
        const auto &dd = args[0].dims();
        const keel::usize R = dd.rows(), C = dd.cols();
        Value yFlat = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        Value gFlat = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        keel::f64 *yp = yFlat.doubleDataMut();
        keel::f64 *gp = gFlat.doubleDataMut();
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize r = 0; r < R; ++r) {
                yp[c * R + r] = args[0].elemAsDouble(c * R + r);
                gp[c * R + r] = keel::as<keel::f64>(c + 1);
            }
        yArg = keel::move(yFlat);
        gArg = keel::move(gFlat);
    } else if (!haveExplicitGroup) {
        throw Error("kruskalwallis: vector y requires explicit group argument",
                    0, 0, "kruskalwallis", "", "numkit:kruskalwallis:noGroup");
    }
    (void)flagPos;   // 'off'/'on' display flag accepted but ignored
    auto [p, H, df, sumR2] = kruskalwallis(yArg, gArg, mr);
    outs[0] = keel::move(p);
    if (nargout > 1) {
        // 4×6 cell table { Source, SS, df, MS, Chi-sq, Prob>Chi-sq } —
        // mirror MATLAB shape; SS / MS pieces aren't strictly defined for
        // the K-W rank statistic, but populate what we can.
        Value tbl = Value::cell(4, 6, mr);
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        auto strV = [&](const char *s){ return Value::fromString(s, mr); };
        auto numV = [&](keel::f64 v){ return Value::scalar(v, mr); };
        auto setCell = [&](keel::usize r, keel::usize c, const Value &v) {
            tbl.cellAt(r + c * 4) = v;
        };
        setCell(0, 0, strV("Source"));
        setCell(0, 1, strV("SS"));
        setCell(0, 2, strV("df"));
        setCell(0, 3, strV("MS"));
        setCell(0, 4, strV("Chi-sq"));
        setCell(0, 5, strV("Prob>Chi-sq"));
        setCell(1, 0, strV("Groups"));
        setCell(1, 1, sumR2);
        setCell(1, 2, df);
        setCell(1, 3, numV(nan));
        setCell(1, 4, H);
        setCell(1, 5, p);
        setCell(2, 0, strV("Error"));
        for (keel::usize c = 1; c < 6; ++c) setCell(2, c, numV(nan));
        setCell(3, 0, strV("Total"));
        for (keel::usize c = 1; c < 6; ++c) setCell(3, c, numV(nan));
        outs[1] = keel::move(tbl);
    }
    if (nargout > 2) {
        // Compute per-group n[] and meanRanks[] for the documented stats
        // struct shape {gnames, n, source, meanranks, sumt}.
        const keel::usize N = yArg.numel();
        struct Pair { keel::f64 v; keel::usize idx; keel::f64 g; };
        keel::Array<Pair> all;
        all.reserve(N);
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = yArg.elemAsDouble(i);
            const keel::f64 g = gArg.elemAsDouble(i);
            if (keel::is_nan(v) || keel::is_nan(g)) continue;
            all.push_back({v, i, g});
        }
        const keel::usize M = all.size();
        keel::Array<keel::usize> ord(M);
        for (keel::usize i = 0; i < M; ++i) ord[i] = i;
        keel::sort(ord.begin(), ord.end(),
                  [&](keel::usize a, keel::usize b){ return all[a].v < all[b].v; });
        keel::Array<keel::f64> rk(M);
        keel::Array<keel::usize> tieSizes;
        keel::usize i = 0;
        while (i < M) {
            keel::usize j = i + 1;
            while (j < M && all[ord[j]].v == all[ord[i]].v) ++j;
            const keel::f64 avg = keel::as<keel::f64>(i + j + 1) / 2.0;
            for (keel::usize k = i; k < j; ++k) rk[ord[k]] = avg;
            if (j - i > 1) tieSizes.push_back(j - i);
            i = j;
        }
        // Bucket per group label (sorted unique).
        keel::Array<keel::f64> uniq;
        uniq.reserve(M);
        for (const auto &item : all) uniq.push_back(item.g);
        keel::sort(uniq.begin(), uniq.end());
        uniq.resize(keel::unique(uniq.begin(), uniq.end()) - uniq.begin());
        const keel::usize K = uniq.size();
        keel::Array<keel::usize> n(K, 0);
        keel::Array<keel::f64> sumR(K, 0.0);
        for (keel::usize k = 0; k < M; ++k) {
            auto it = keel::lower_bound(uniq.begin(), uniq.end(), all[k].g);
            const keel::usize gi = keel::as<keel::usize>(it - uniq.begin());
            n[gi] += 1;
            sumR[gi] += rk[k];
        }
        keel::f64 sumt = 0.0;
        for (keel::usize t : tieSizes) {
            const keel::f64 td = keel::as<keel::f64>(t);
            sumt += td * td * td - td;
        }

        Value nVec = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        Value mrVec = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        Value gnames = Value::cell(K, 1, mr);
        for (keel::usize k = 0; k < K; ++k) {
            nVec.doubleDataMut()[k] = keel::as<keel::f64>(n[k]);
            mrVec.doubleDataMut()[k] = (n[k] > 0) ? sumR[k] / keel::as<keel::f64>(n[k]) : 0.0;
            // gnames as decimal string of the label (matches MATLAB)
            char buf[64];
            gnames.cellAt(k) = Value::fromString(keel::format_f64_general_to(buf, uniq[k], 6), mr);
        }
        Value s = Value::structure(mr);
        s.field("gnames")    = keel::move(gnames);
        s.field("n")         = keel::move(nVec);
        s.field("source")    = Value::fromString("kruskalwallis", mr);
        s.field("meanranks") = keel::move(mrVec);
        s.field("sumt")      = Value::scalar(sumt, mr);
        // Keep legacy fields for callers that already used them.
        s.field("chi2stat")  = H;
        s.field("df")        = df;
        outs[2] = keel::move(s);
    }
}

void dummyvar_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "dummyvar");
    outs[0] = dummyvar(args[0], ctx.engine->allocator());
}

void anova2_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "anova2");
    keel::i32 reps = 1;
    if (args.size() >= 2 && !args[1].isEmpty()
        && !args[1].isChar() && !args[1].isString()) {
        reps = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "reps", "anova2"), 2, "reps", "anova2");
    }
    auto *mr = ctx.engine->allocator();
    const Value &Y = args[0];
    if (Y.dims().ndim() != 2)
        throw Error("anova2: input must be a 2D matrix",
                    0, 0, "anova2", "", "numkit:anova2:notMatrix");
    const keel::usize M = Y.dims().rows();
    const keel::usize c = Y.dims().cols();
    const keel::usize R = keel::as<keel::usize>(reps);
    if (c < 2 || M < 2 * R || (M % R) != 0)
        throw Error("anova2: row count must be an integer multiple (>= 2) of reps and columns >= 2",
                    0, 0, "anova2", "", "numkit:anova2:tooSmall");
    const keel::usize r = M / R;
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();

    keel::f64 pCols = nan, pRows = nan, pInter = nan;
    keel::f64 Fc = nan, Fr = nan, Fi = nan;
    keel::f64 dfC = 0.0, dfR = 0.0, dfI = 0.0, dfE = 0.0;
    keel::f64 ssC = 0.0, ssR = 0.0, ssI = 0.0, ssE = 0.0;

    keel::Array<keel::f64> rowMean(r, 0.0), colMean(c, 0.0);
    keel::f64 grandSum = 0.0;
    for (keel::usize j = 0; j < c; ++j) {
        for (keel::usize i = 0; i < r; ++i) {
            for (keel::usize k = 0; k < R; ++k) {
                const keel::f64 v = Y.elemAsDouble((i * R + k) + j * M);
                rowMean[i] += v;
                colMean[j] += v;
                grandSum += v;
            }
        }
    }
    const keel::f64 N = keel::as<keel::f64>(r * c * R);
    const keel::f64 grandMean = grandSum / N;
    for (keel::usize i = 0; i < r; ++i)
        rowMean[i] /= keel::as<keel::f64>(c * R);
    for (keel::usize j = 0; j < c; ++j)
        colMean[j] /= keel::as<keel::f64>(r * R);

    if (R == 1) {
        keel::tie(pCols, pRows, Fc, Fr, dfC, dfR, dfE, ssC, ssR, ssE) =
            anova2(Y, mr);
    } else {
        keel::Array<keel::f64> cellMean(r * c, 0.0);
        for (keel::usize j = 0; j < c; ++j) {
            for (keel::usize i = 0; i < r; ++i) {
                keel::f64 s = 0.0;
                for (keel::usize k = 0; k < R; ++k)
                    s += Y.elemAsDouble((i * R + k) + j * M);
                cellMean[i + j * r] = s / keel::as<keel::f64>(R);
            }
        }
        for (keel::usize i = 0; i < r; ++i) {
            const keel::f64 d = rowMean[i] - grandMean;
            ssR += keel::as<keel::f64>(c * R) * d * d;
        }
        for (keel::usize j = 0; j < c; ++j) {
            const keel::f64 d = colMean[j] - grandMean;
            ssC += keel::as<keel::f64>(r * R) * d * d;
        }
        for (keel::usize j = 0; j < c; ++j) {
            for (keel::usize i = 0; i < r; ++i) {
                const keel::f64 cm = cellMean[i + j * r];
                const keel::f64 di = cm - rowMean[i] - colMean[j] + grandMean;
                ssI += keel::as<keel::f64>(R) * di * di;
                for (keel::usize k = 0; k < R; ++k) {
                    const keel::f64 de = Y.elemAsDouble((i * R + k) + j * M) - cm;
                    ssE += de * de;
                }
            }
        }
        dfC = keel::as<keel::f64>(c - 1);
        dfR = keel::as<keel::f64>(r - 1);
        dfI = keel::as<keel::f64>((r - 1) * (c - 1));
        dfE = keel::as<keel::f64>(r * c * (R - 1));
        const keel::f64 msE = (dfE > 0.0) ? (ssE / dfE) : nan;
        if (msE > 0.0) {
            Fc = (ssC / dfC) / msE;
            Fr = (ssR / dfR) / msE;
            Fi = (ssI / dfI) / msE;
            pCols  = keel::max(0.0, 1.0 - fcdf(Value::scalar(Fc, mr), dfC, dfE, mr).toScalar());
            pRows  = keel::max(0.0, 1.0 - fcdf(Value::scalar(Fr, mr), dfR, dfE, mr).toScalar());
            pInter = keel::max(0.0, 1.0 - fcdf(Value::scalar(Fi, mr), dfI, dfE, mr).toScalar());
        } else if (msE == 0.0) {
            const keel::f64 inf = keel::Limits<keel::f64>::infinity();
            Fc = Fr = Fi = inf;
            pCols = pRows = pInter = 0.0;
        }
    }

    // p output: 1x3 row [p_cols, p_rows, p_inter] (or 1x2 when reps==1 in some callers,
    // keep 1x3 with p_inter = NaN for reps=1, and real pInter for reps>1).
    auto pV = Value::matrix(1, 3, ValueType::DOUBLE, mr);
    keel::f64 *pd = pV.doubleDataMut();
    pd[0] = pCols;
    pd[1] = pRows;
    pd[2] = pInter;
    outs[0] = keel::move(pV);

    if (nargout > 1) {
        const keel::usize tblRows = (R > 1) ? 6 : 5;
        Value tbl = Value::cell(tblRows, 6, mr);
        auto strV = [&](const char *s){ return Value::fromString(s, mr); };
        auto numV = [&](keel::f64 v){ return Value::scalar(v, mr); };
        auto setCell = [&](keel::usize rr, keel::usize cc, const Value &v) {
            tbl.cellAt(rr + cc * tblRows) = v;
        };
        setCell(0, 0, strV("Source"));
        setCell(0, 1, strV("SS"));
        setCell(0, 2, strV("df"));
        setCell(0, 3, strV("MS"));
        setCell(0, 4, strV("F"));
        setCell(0, 5, strV("Prob>F"));
        setCell(1, 0, strV("Columns"));
        setCell(1, 1, numV(ssC));
        setCell(1, 2, numV(dfC));
        setCell(1, 3, numV(dfC > 0 ? ssC / dfC : nan));
        setCell(1, 4, numV(Fc));
        setCell(1, 5, numV(pCols));
        setCell(2, 0, strV("Rows"));
        setCell(2, 1, numV(ssR));
        setCell(2, 2, numV(dfR));
        setCell(2, 3, numV(dfR > 0 ? ssR / dfR : nan));
        setCell(2, 4, numV(Fr));
        setCell(2, 5, numV(pRows));
        keel::usize errRow = 3;
        if (R > 1) {
            setCell(3, 0, strV("Interaction"));
            setCell(3, 1, numV(ssI));
            setCell(3, 2, numV(dfI));
            setCell(3, 3, numV(dfI > 0 ? ssI / dfI : nan));
            setCell(3, 4, numV(Fi));
            setCell(3, 5, numV(pInter));
            errRow = 4;
        }
        setCell(errRow, 0, strV("Error"));
        setCell(errRow, 1, numV(ssE));
        setCell(errRow, 2, numV(dfE));
        setCell(errRow, 3, numV(dfE > 0 ? ssE / dfE : nan));
        setCell(errRow, 4, numV(nan));
        setCell(errRow, 5, numV(nan));
        setCell(errRow + 1, 0, strV("Total"));
        setCell(errRow + 1, 1, numV(ssC + ssR + ssI + ssE));
        setCell(errRow + 1, 2, numV(dfC + dfR + dfI + dfE));
        setCell(errRow + 1, 3, numV(nan));
        setCell(errRow + 1, 4, numV(nan));
        setCell(errRow + 1, 5, numV(nan));
        outs[1] = keel::move(tbl);
    }
    if (nargout > 2) {
        Value colMeansV = Value::matrix(1, c, ValueType::DOUBLE, mr);
        for (keel::usize j = 0; j < c; ++j)
            colMeansV.doubleDataMut()[j] = colMean[j];
        Value rowMeansV = Value::matrix(1, r, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < r; ++i)
            rowMeansV.doubleDataMut()[i] = rowMean[i];
        Value s = Value::structure(mr);
        s.field("source")   = Value::fromString("anova2", mr);
        s.field("sigmasq")  = Value::scalar(dfE > 0 ? ssE / dfE : nan, mr);
        s.field("colmeans") = keel::move(colMeansV);
        s.field("coln")     = Value::scalar(keel::as<keel::f64>(r * R), mr);
        s.field("rowmeans") = keel::move(rowMeansV);
        s.field("rown")     = Value::scalar(keel::as<keel::f64>(c * R), mr);
        s.field("inter")    = Value::logicalScalar(R > 1, mr);
        s.field("pval")     = Value::scalar(pInter, mr);
        s.field("df")       = Value::scalar(dfE, mr);
        outs[2] = keel::move(s);
    }
}

void friedman_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "friedman");
    auto *mr = ctx.engine->allocator();

    // reps: numeric 2nd arg (default 1). A char 2nd/3rd arg is the
    // 'off'/'on' display flag — accepted and ignored.
    keel::i32 reps = 1;
    if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()
        && !args[1].isEmpty())
        reps = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "reps", "friedman"), 2, "reps", "friedman");

    auto [p, Q, df] = friedman(args[0], reps, mr);
    outs[0] = keel::move(p);
    if (nargout > 1) outs[1] = keel::move(Q);    // chi-square statistic (not MATLAB's tbl)
    if (nargout > 2) outs[2] = keel::move(df);   // degrees of freedom (k-1)
}

} // namespace detail

} // namespace numkit::stats
