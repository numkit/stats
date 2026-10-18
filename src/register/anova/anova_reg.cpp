// toolboxes/signal/src/anova/anova_reg.cpp
//
// CallContext register half of anova/anova.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/anova/anova.hpp>
#include <numkit/stats/distributions/chi2.hpp>
#include <numkit/stats/distributions/fisher_f.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "anova/anova_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::stats {

namespace detail {

void anova1_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "anova1");
    auto *mr = ctx.engine->resource();

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
        const size_t R = X.dims().rows(), C = X.dims().cols();
        yStore = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        gStore = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        double *yd = yStore.doubleDataMut();
        double *gd = gStore.doubleDataMut();
        size_t k = 0;
        for (size_t c = 0; c < C; ++c)
            for (size_t r = 0; r < R; ++r) {
                yd[k] = X.elemAsDouble(c * R + r);
                gd[k] = static_cast<double>(c + 1);
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
        const double nan = std::numeric_limits<double>::quiet_NaN();
        auto strV = [&](const char *s){ return Value::fromString(s, mr); };
        auto numV = [&](double v){ return Value::scalar(v, mr); };
        auto setCell = [&](size_t r, size_t c, const Value &v) {
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
        outs[1] = std::move(tbl);
    }
    if (nargout > 2) {
        // Re-bucket to populate the full stats struct expected by
        // post-hoc tools (multcompare, etc.). The data scan is cheap;
        // we only do it here, when the third output is actually
        // requested.
        auto buckets = bucket(*yp, *gp);
        const std::size_t K = buckets.size();
        // means + per-group sizes.
        Value meansV = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        Value nV     = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        Value gnamesV = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        double *mD = meansV.doubleDataMut();
        double *nD = nV.doubleDataMut();
        double *gD = gnamesV.doubleDataMut();
        double sse = 0.0;
        std::size_t total = 0;
        for (std::size_t k = 0; k < K; ++k) {
            const auto &g = buckets[k];
            const std::size_t ng = g.values.size();
            double sum = 0.0;
            for (double x : g.values) sum += x;
            const double m = (ng > 0) ? sum / static_cast<double>(ng) : 0.0;
            mD[k] = m;
            nD[k] = static_cast<double>(ng);
            gD[k] = g.label;
            for (double x : g.values) {
                const double d = x - m;
                sse += d * d;
            }
            total += ng;
        }
        const double sPooled = (total > K)
            ? std::sqrt(sse / static_cast<double>(total - K))
            : std::numeric_limits<double>::quiet_NaN();

        Value s = Value::structure(mr);
        s.field("F")      = Value::scalar(F, mr);
        s.field("df")     = Value::scalar(dfW, mr);
        s.field("means")  = std::move(meansV);
        s.field("n")      = std::move(nV);
        s.field("gnames") = std::move(gnamesV);
        s.field("s")      = Value::scalar(sPooled, mr);
        s.field("source") = Value::fromString("anova1", mr);
        outs[2] = std::move(s);
    }
}

void kruskalwallis_reg(Span<const Value> args, size_t nargout,
                       Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "kruskalwallis");
    auto *mr = ctx.engine->resource();

    // Matrix-only form: when group is omitted (or the 2nd arg is the
    // 'off'/'on' display flag string), infer groups from columns. Build
    // a flat y column and a same-length group index column.
    Value yArg = args[0];
    Value gArg;
    bool haveExplicitGroup = false;
    size_t flagPos = 1;
    if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()
        && !args[1].isEmpty()) {
        gArg = args[1];
        haveExplicitGroup = true;
        flagPos = 2;
    }
    if (!haveExplicitGroup && !args[0].dims().isVector() && !args[0].isScalar()) {
        const auto &dd = args[0].dims();
        const size_t R = dd.rows(), C = dd.cols();
        Value yFlat = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        Value gFlat = Value::matrix(R * C, 1, ValueType::DOUBLE, mr);
        double *yp = yFlat.doubleDataMut();
        double *gp = gFlat.doubleDataMut();
        for (size_t c = 0; c < C; ++c)
            for (size_t r = 0; r < R; ++r) {
                yp[c * R + r] = args[0].elemAsDouble(c * R + r);
                gp[c * R + r] = static_cast<double>(c + 1);
            }
        yArg = std::move(yFlat);
        gArg = std::move(gFlat);
    } else if (!haveExplicitGroup) {
        throw Error("kruskalwallis: vector y requires explicit group argument",
                    0, 0, "kruskalwallis", "", "numkit:kruskalwallis:noGroup");
    }
    (void)flagPos;   // 'off'/'on' display flag accepted but ignored
    auto [p, H, df, sumR2] = kruskalwallis(yArg, gArg, mr);
    outs[0] = std::move(p);
    if (nargout > 1) {
        // 4×6 cell table { Source, SS, df, MS, Chi-sq, Prob>Chi-sq } —
        // mirror MATLAB shape; SS / MS pieces aren't strictly defined for
        // the K-W rank statistic, but populate what we can.
        Value tbl = Value::cell(4, 6, mr);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        auto strV = [&](const char *s){ return Value::fromString(s, mr); };
        auto numV = [&](double v){ return Value::scalar(v, mr); };
        auto setCell = [&](size_t r, size_t c, const Value &v) {
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
        for (size_t c = 1; c < 6; ++c) setCell(2, c, numV(nan));
        setCell(3, 0, strV("Total"));
        for (size_t c = 1; c < 6; ++c) setCell(3, c, numV(nan));
        outs[1] = std::move(tbl);
    }
    if (nargout > 2) {
        // Compute per-group n[] and meanRanks[] for the documented stats
        // struct shape {gnames, n, source, meanranks, sumt}.
        const size_t N = yArg.numel();
        struct Pair { double v; size_t idx; double g; };
        std::vector<Pair> all;
        all.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            const double v = yArg.elemAsDouble(i);
            const double g = gArg.elemAsDouble(i);
            if (std::isnan(v) || std::isnan(g)) continue;
            all.push_back({v, i, g});
        }
        const size_t M = all.size();
        std::vector<size_t> ord(M);
        for (size_t i = 0; i < M; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(),
                  [&](size_t a, size_t b){ return all[a].v < all[b].v; });
        std::vector<double> rk(M);
        std::vector<size_t> tieSizes;
        size_t i = 0;
        while (i < M) {
            size_t j = i + 1;
            while (j < M && all[ord[j]].v == all[ord[i]].v) ++j;
            const double avg = static_cast<double>(i + j + 1) / 2.0;
            for (size_t k = i; k < j; ++k) rk[ord[k]] = avg;
            if (j - i > 1) tieSizes.push_back(j - i);
            i = j;
        }
        // Bucket per group label (sorted unique).
        std::vector<double> uniq;
        uniq.reserve(M);
        for (auto &p : all) uniq.push_back(p.g);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        const size_t K = uniq.size();
        std::vector<size_t> n(K, 0);
        std::vector<double> sumR(K, 0.0);
        for (size_t k = 0; k < M; ++k) {
            auto it = std::lower_bound(uniq.begin(), uniq.end(), all[k].g);
            const size_t gi = static_cast<size_t>(it - uniq.begin());
            n[gi] += 1;
            sumR[gi] += rk[k];
        }
        double sumt = 0.0;
        for (size_t t : tieSizes) {
            const double td = static_cast<double>(t);
            sumt += td * td * td - td;
        }

        Value nVec = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        Value mrVec = Value::matrix(K, 1, ValueType::DOUBLE, mr);
        Value gnames = Value::cell(K, 1, mr);
        for (size_t k = 0; k < K; ++k) {
            nVec.doubleDataMut()[k] = static_cast<double>(n[k]);
            mrVec.doubleDataMut()[k] = (n[k] > 0) ? sumR[k] / static_cast<double>(n[k]) : 0.0;
            // gnames as decimal string of the label (matches MATLAB)
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%g", uniq[k]);
            gnames.cellAt(k) = Value::fromString(buf, mr);
        }
        Value s = Value::structure(mr);
        s.field("gnames")    = std::move(gnames);
        s.field("n")         = std::move(nVec);
        s.field("source")    = Value::fromString("kruskalwallis", mr);
        s.field("meanranks") = std::move(mrVec);
        s.field("sumt")      = Value::scalar(sumt, mr);
        // Keep legacy fields for callers that already used them.
        s.field("chi2stat")  = H;
        s.field("df")        = df;
        outs[2] = std::move(s);
    }
}

void dummyvar_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "dummyvar");
    outs[0] = dummyvar(args[0], ctx.engine->resource());
}

void anova2_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "anova2");
    int reps = 1;
    if (args.size() >= 2 && !args[1].isEmpty()
        && !args[1].isChar() && !args[1].isString()) {
        reps = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "reps", "anova2"), 2, "reps", "anova2");
    }
    auto *mr = ctx.engine->resource();
    const Value &Y = args[0];
    if (Y.dims().ndim() != 2)
        throw Error("anova2: input must be a 2D matrix",
                    0, 0, "anova2", "", "numkit:anova2:notMatrix");
    const std::size_t M = Y.dims().rows();
    const std::size_t c = Y.dims().cols();
    const std::size_t R = static_cast<std::size_t>(reps);
    if (c < 2 || M < 2 * R || (M % R) != 0)
        throw Error("anova2: row count must be an integer multiple (>= 2) of reps and columns >= 2",
                    0, 0, "anova2", "", "numkit:anova2:tooSmall");
    const std::size_t r = M / R;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    double pCols = nan, pRows = nan, pInter = nan;
    double Fc = nan, Fr = nan, Fi = nan;
    double dfC = 0.0, dfR = 0.0, dfI = 0.0, dfE = 0.0;
    double ssC = 0.0, ssR = 0.0, ssI = 0.0, ssE = 0.0;

    std::vector<double> rowMean(r, 0.0), colMean(c, 0.0);
    double grandSum = 0.0;
    for (std::size_t j = 0; j < c; ++j) {
        for (std::size_t i = 0; i < r; ++i) {
            for (std::size_t k = 0; k < R; ++k) {
                const double v = Y.elemAsDouble((i * R + k) + j * M);
                rowMean[i] += v;
                colMean[j] += v;
                grandSum += v;
            }
        }
    }
    const double N = static_cast<double>(r * c * R);
    const double grandMean = grandSum / N;
    for (std::size_t i = 0; i < r; ++i)
        rowMean[i] /= static_cast<double>(c * R);
    for (std::size_t j = 0; j < c; ++j)
        colMean[j] /= static_cast<double>(r * R);

    if (R == 1) {
        std::tie(pCols, pRows, Fc, Fr, dfC, dfR, dfE, ssC, ssR, ssE) =
            anova2(Y, mr);
    } else {
        std::vector<double> cellMean(r * c, 0.0);
        for (std::size_t j = 0; j < c; ++j) {
            for (std::size_t i = 0; i < r; ++i) {
                double s = 0.0;
                for (std::size_t k = 0; k < R; ++k)
                    s += Y.elemAsDouble((i * R + k) + j * M);
                cellMean[i + j * r] = s / static_cast<double>(R);
            }
        }
        for (std::size_t i = 0; i < r; ++i) {
            const double d = rowMean[i] - grandMean;
            ssR += static_cast<double>(c * R) * d * d;
        }
        for (std::size_t j = 0; j < c; ++j) {
            const double d = colMean[j] - grandMean;
            ssC += static_cast<double>(r * R) * d * d;
        }
        for (std::size_t j = 0; j < c; ++j) {
            for (std::size_t i = 0; i < r; ++i) {
                const double cm = cellMean[i + j * r];
                const double di = cm - rowMean[i] - colMean[j] + grandMean;
                ssI += static_cast<double>(R) * di * di;
                for (std::size_t k = 0; k < R; ++k) {
                    const double de = Y.elemAsDouble((i * R + k) + j * M) - cm;
                    ssE += de * de;
                }
            }
        }
        dfC = static_cast<double>(c - 1);
        dfR = static_cast<double>(r - 1);
        dfI = static_cast<double>((r - 1) * (c - 1));
        dfE = static_cast<double>(r * c * (R - 1));
        const double msE = (dfE > 0.0) ? (ssE / dfE) : nan;
        if (msE > 0.0) {
            Fc = (ssC / dfC) / msE;
            Fr = (ssR / dfR) / msE;
            Fi = (ssI / dfI) / msE;
            pCols  = std::max(0.0, 1.0 - fcdf(Value::scalar(Fc, mr), dfC, dfE, mr).toScalar());
            pRows  = std::max(0.0, 1.0 - fcdf(Value::scalar(Fr, mr), dfR, dfE, mr).toScalar());
            pInter = std::max(0.0, 1.0 - fcdf(Value::scalar(Fi, mr), dfI, dfE, mr).toScalar());
        } else if (msE == 0.0) {
            const double inf = std::numeric_limits<double>::infinity();
            Fc = Fr = Fi = inf;
            pCols = pRows = pInter = 0.0;
        }
    }

    // p output: 1x3 row [p_cols, p_rows, p_inter] (or 1x2 when reps==1 in some callers,
    // keep 1x3 with p_inter = NaN for reps=1, and real pInter for reps>1).
    auto pV = Value::matrix(1, 3, ValueType::DOUBLE, mr);
    double *pd = pV.doubleDataMut();
    pd[0] = pCols;
    pd[1] = pRows;
    pd[2] = pInter;
    outs[0] = std::move(pV);

    if (nargout > 1) {
        const std::size_t tblRows = (R > 1) ? 6 : 5;
        Value tbl = Value::cell(tblRows, 6, mr);
        auto strV = [&](const char *s){ return Value::fromString(s, mr); };
        auto numV = [&](double v){ return Value::scalar(v, mr); };
        auto setCell = [&](size_t rr, size_t cc, const Value &v) {
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
        std::size_t errRow = 3;
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
        outs[1] = std::move(tbl);
    }
    if (nargout > 2) {
        Value colMeansV = Value::matrix(1, c, ValueType::DOUBLE, mr);
        for (std::size_t j = 0; j < c; ++j)
            colMeansV.doubleDataMut()[j] = colMean[j];
        Value rowMeansV = Value::matrix(1, r, ValueType::DOUBLE, mr);
        for (std::size_t i = 0; i < r; ++i)
            rowMeansV.doubleDataMut()[i] = rowMean[i];
        Value s = Value::structure(mr);
        s.field("source")   = Value::fromString("anova2", mr);
        s.field("sigmasq")  = Value::scalar(dfE > 0 ? ssE / dfE : nan, mr);
        s.field("colmeans") = std::move(colMeansV);
        s.field("coln")     = Value::scalar(static_cast<double>(r * R), mr);
        s.field("rowmeans") = std::move(rowMeansV);
        s.field("rown")     = Value::scalar(static_cast<double>(c * R), mr);
        s.field("inter")    = Value::logicalScalar(R > 1, mr);
        s.field("pval")     = Value::scalar(pInter, mr);
        s.field("df")       = Value::scalar(dfE, mr);
        outs[2] = std::move(s);
    }
}

void friedman_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "friedman");
    auto *mr = ctx.engine->resource();

    // reps: numeric 2nd arg (default 1). A char 2nd/3rd arg is the
    // 'off'/'on' display flag — accepted and ignored.
    int reps = 1;
    if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()
        && !args[1].isEmpty())
        reps = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "reps", "friedman"), 2, "reps", "friedman");

    // reps>1 (replicates per cell) uses a two-way layout whose ranking does
    // NOT reduce to averaging-then-rank — nk doesn't match MATLAB there
    // yet, so reject rather than return a wrong p. bugs/stats/friedman.
    if (reps != 1)
        throw Error("friedman: reps > 1 is not yet supported (use reps = 1)",
                    0, 0, "friedman", "", "numkit:friedman:reps");

    auto [p, Q, df] = friedman(args[0], reps, mr);
    outs[0] = std::move(p);
    if (nargout > 1) outs[1] = std::move(Q);    // chi-square statistic (not MATLAB's tbl)
    if (nargout > 2) outs[2] = std::move(df);   // degrees of freedom (k-1)
}

} // namespace detail

} // namespace numkit::stats
