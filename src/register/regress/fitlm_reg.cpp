// stats/src/register/regress/fitlm_reg.cpp
//
// Linear regression model fitting (`fitlm`) and prediction (`predict`).
// Supports numeric matrices as well as `table` / `timetable` inputs via the
// L0 `numkit::TablePayload` interface.
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/sys/plugin.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/table_payload.hpp>
#include <numkit/value/value.hpp>

namespace numkit::stats::detail {

namespace {

keel::StringRef trimAscii(keel::StringRef s)
{
    keel::usize a = 0;
    keel::usize b = s.size();
    while (a < b && keel::is_ascii_space(s[a]))
        ++a;
    while (b > a && keel::is_ascii_space(s[b - 1]))
        --b;
    return s.substr(a, b - a);
}

} // namespace

void fitlm_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 8, "fitlm");
    auto *mr = ctx.engine->allocator();
    auto *alloc = mr ? mr : keel::default_allocator();
    keel::Array<keel::Array<keel::f64>> Xcols(alloc);
    keel::Array<keel::f64> yvec(alloc);
    keel::Array<keel::String> predNames(alloc);
    keel::usize N = 0;

    if (const TablePayload *tp = tablePayloadConst(args[0])) {
        N = tp->nrows;
        keel::i32 respIdx = keel::as<keel::i32>(tp->width()) - 1;
        keel::Array<keel::i32> pIdx(alloc);
        if (args.size() >= 2 && (args[1].isChar() || args[1].isString())) {
            keel::String f = args[1].toString();
            keel::usize tilde = f.find('~');
            if (tilde != keel::kInvalidIndex) {
                keel::StringRef rName = trimAscii(f.substr(0, tilde));
                keel::i32 ri = tp->findVar(rName);
                if (ri >= 0)
                    respIdx = ri;
                keel::StringRef rhs = f.substr(tilde + 1);
                keel::usize pos = 0;
                while (pos <= rhs.size()) {
                    keel::usize plus = rhs.find('+', pos);
                    keel::StringRef tok = trimAscii(rhs.substr(
                        pos, (plus == keel::kInvalidIndex) ? keel::kInvalidIndex : (plus - pos)));
                    if (!tok.empty() && tok != "1") {
                        keel::i32 pi = tp->findVar(tok);
                        if (pi >= 0)
                            pIdx.push_back(pi);
                    }
                    if (plus == keel::kInvalidIndex)
                        break;
                    pos = plus + 1;
                }
            }
        }
        if (pIdx.empty()) {
            for (keel::usize j = 0; j < tp->width(); ++j)
                if (keel::as<keel::i32>(j) != respIdx && tp->cols[j].isNumeric())
                    pIdx.push_back(keel::as<keel::i32>(j));
        }
        yvec.assign(N, 0.0);
        if (respIdx >= 0 && keel::as<keel::usize>(respIdx) < tp->width()) {
            const Value &yc = tp->cols[keel::as<keel::usize>(respIdx)];
            for (keel::usize i = 0; i < N && i < yc.numel(); ++i)
                yvec[i] = yc.elemAsDouble(i);
        }
        for (keel::i32 pi : pIdx) {
            predNames.push_back(tp->varNames[keel::as<keel::usize>(pi)].clone(alloc));
            const Value &xc = tp->cols[keel::as<keel::usize>(pi)];
            keel::Array<keel::f64> col(alloc);
            col.assign(N, 0.0);
            for (keel::usize i = 0; i < N && i < xc.numel(); ++i)
                col[i] = xc.elemAsDouble(i);
            Xcols.push_back(keel::move(col));
        }
    } else {
        argcheck::checkNargin(args.size(), 2, 8, "fitlm");
        const Value &X = args[0];
        const Value &Y = args[1];
        N = Y.numel();
        yvec.resize(N);
        for (keel::usize i = 0; i < N; ++i)
            yvec[i] = Y.elemAsDouble(i);
        keel::usize xr = X.dims().rows();
        keel::usize xc = X.dims().cols();
        if (xr == 1 && xc == N && N > 1) {
            keel::Array<keel::f64> col(alloc);
            col.resize(N);
            for (keel::usize i = 0; i < N; ++i)
                col[i] = X.elemAsDouble(i);
            Xcols.push_back(keel::move(col));
            predNames.push_back(keel::String::from("x1", alloc));
        } else {
            for (keel::usize j = 0; j < xc; ++j) {
                keel::Array<keel::f64> col(alloc);
                col.assign(N, 0.0);
                for (keel::usize i = 0; i < N && i < xr; ++i)
                    col[i] = X.elemAsDouble(j * xr + i);
                Xcols.push_back(keel::move(col));
                predNames.push_back(keel::format_alloc(alloc, "x{}", j + 1));
            }
        }
    }

    const keel::usize P = Xcols.size();
    const keel::usize K = P + 1;
    keel::Array<keel::f64> XtX(alloc);
    XtX.assign(K * K, 0.0);
    keel::Array<keel::f64> Xty(alloc);
    Xty.assign(K, 0.0);
    for (keel::usize i = 0; i < N; ++i) {
        keel::f64 yi = yvec[i];
        if (!keel::is_finite(yi))
            continue;
        bool ok = true;
        for (keel::usize j = 0; j < P; ++j)
            if (!keel::is_finite(Xcols[j][i])) {
                ok = false;
                break;
            }
        if (!ok)
            continue;
        for (keel::usize r = 0; r < K; ++r) {
            keel::f64 xr = (r == 0) ? 1.0 : Xcols[r - 1][i];
            Xty[r] += xr * yi;
            for (keel::usize c = 0; c < K; ++c) {
                keel::f64 xc = (c == 0) ? 1.0 : Xcols[c - 1][i];
                XtX[r * K + c] += xr * xc;
            }
        }
    }

    // Augment [XtX | I | Xty] to solve for beta and (XtX)^-1 simultaneously
    const keel::usize W = 2 * K + 1;
    keel::Array<keel::f64> aug(alloc);
    aug.assign(K * W, 0.0);
    for (keel::usize r = 0; r < K; ++r) {
        for (keel::usize c = 0; c < K; ++c)
            aug[r * W + c] = XtX[r * K + c];
        aug[r * W + (K + r)] = 1.0;
        aug[r * W + 2 * K] = Xty[r];
    }
    for (keel::usize p = 0; p < K; ++p) {
        keel::usize piv = p;
        keel::f64 maxA = keel::abs(aug[p * W + p]);
        for (keel::usize r = p + 1; r < K; ++r) {
            keel::f64 a = keel::abs(aug[r * W + p]);
            if (a > maxA) {
                maxA = a;
                piv = r;
            }
        }
        if (piv != p) {
            for (keel::usize c = 0; c < W; ++c)
                keel::swap(aug[p * W + c], aug[piv * W + c]);
        }
        keel::f64 diag = aug[p * W + p];
        if (keel::abs(diag) < 1e-15)
            continue;
        for (keel::usize c = p; c < W; ++c)
            aug[p * W + c] /= diag;
        for (keel::usize r = 0; r < K; ++r) {
            if (r == p)
                continue;
            keel::f64 f = aug[r * W + p];
            if (f == 0.0)
                continue;
            for (keel::usize c = p; c < W; ++c)
                aug[r * W + c] -= f * aug[p * W + c];
        }
    }
    keel::Array<keel::f64> beta(alloc);
    keel::Array<keel::f64> invDiag(alloc);
    beta.assign(K, 0.0);
    invDiag.assign(K, 0.0);
    for (keel::usize r = 0; r < K; ++r) {
        beta[r] = aug[r * W + 2 * K];
        invDiag[r] = keel::max(0.0, aug[r * W + K + r]);
    }

    Value fitted = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    Value rawRes = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *fitPtr = fitted.doubleDataMut();
    keel::f64 *resPtr = rawRes.doubleDataMut();
    keel::f64 ySum = 0.0;
    for (keel::usize i = 0; i < N; ++i)
        ySum += yvec[i];
    keel::f64 yMean = (N > 0) ? (ySum / keel::as<keel::f64>(N)) : 0.0;
    keel::f64 sse = 0.0;
    keel::f64 sst = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        keel::f64 yh = beta[0];
        for (keel::usize j = 0; j < P; ++j)
            yh += beta[j + 1] * Xcols[j][i];
        keel::f64 ri = yvec[i] - yh;
        fitPtr[i] = yh;
        resPtr[i] = ri;
        sse += ri * ri;
        keel::f64 dy = yvec[i] - yMean;
        sst += dy * dy;
    }
    keel::f64 dfe = (N > K) ? keel::as<keel::f64>(N - K) : 1.0;
    keel::f64 mse = sse / dfe;
    keel::f64 r2Ord = (sst > 0.0) ? (1.0 - sse / sst) : 1.0;
    keel::f64 r2Adj = (sst > 0.0 && N > K && N > 1)
                          ? (1.0 - (sse / dfe) / (sst / keel::as<keel::f64>(N - 1)))
                          : r2Ord;

    Value estV = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    Value seV  = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    Value tV   = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    Value pV   = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    for (keel::usize r = 0; r < K; ++r) {
        keel::f64 se = keel::sqrt(mse * invDiag[r]);
        keel::f64 ts = (se > 0.0) ? (beta[r] / se) : 0.0;
        keel::f64 pv = keel::erfc(keel::abs(ts) / keel::sqrt(2.0));
        estV.doubleDataMut()[r] = beta[r];
        seV.doubleDataMut()[r]  = se;
        tV.doubleDataMut()[r]   = ts;
        pV.doubleDataMut()[r]   = pv;
    }

    Value coefs = Value::structure(mr);
    coefs.field("Estimate") = keel::move(estV);
    coefs.field("SE")       = keel::move(seV);
    coefs.field("tStat")    = keel::move(tV);
    coefs.field("pValue")   = keel::move(pV);

    Value rsq = Value::structure(mr);
    rsq.field("Ordinary") = Value::scalar(r2Ord, mr);
    rsq.field("Adjusted") = Value::scalar(r2Adj, mr);

    Value resStruct = Value::structure(mr);
    resStruct.field("Raw") = keel::move(rawRes);

    Value pNamesCell = Value::cell(1, P, mr);
    for (keel::usize j = 0; j < P; ++j)
        pNamesCell.cellAt(j) = Value::fromString(predNames[j], mr);

    Value mdl = Value::structure(mr);
    mdl.field("Coefficients")    = keel::move(coefs);
    mdl.field("Rsquared")        = keel::move(rsq);
    mdl.field("Fitted")          = keel::move(fitted);
    mdl.field("Residuals")       = keel::move(resStruct);
    mdl.field("MSE")             = Value::scalar(mse, mr);
    mdl.field("RMSE")            = Value::scalar(keel::sqrt(mse), mr);
    mdl.field("DFE")             = Value::scalar(dfe, mr);
    mdl.field("NumObservations") = Value::scalar(keel::as<keel::f64>(N), mr);
    mdl.field("PredictorNames")  = keel::move(pNamesCell);
    outs[0] = keel::move(mdl);
}

void predict_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 6, "predict");
    auto *mr = ctx.engine->allocator();
    const Value &mdl = args[0];
    const Value &Xnew = args[1];
    if (!mdl.isStruct() || !mdl.hasField("Coefficients"))
        throw Error("predict: first argument must be a fitted model",
                    0, 0, "predict", "", "numkit:predict:invalidModel");
    const Value &coefs = mdl.field("Coefficients");
    if (!coefs.isStruct() || !coefs.hasField("Estimate"))
        throw Error("predict: model Coefficients must be a struct with an Estimate field",
                    0, 0, "predict", "", "numkit:predict:invalidModel");
    const Value &est = coefs.field("Estimate");
    if (!est.isNumeric())
        throw Error("predict: model Estimate must be numeric",
                    0, 0, "predict", "", "numkit:predict:invalidModel");
    const keel::usize K = est.numel();
    const keel::usize P = (K > 0) ? (K - 1) : 0;
    const keel::f64 b0 = (K > 0) ? est.elemAsDouble(0) : 0.0;

    if (const TablePayload *tp = tablePayloadConst(Xnew)) {
        const keel::usize M = tp->nrows;
        Value ypred = Value::matrix(M, 1, ValueType::DOUBLE, mr);
        keel::f64 *yp = ypred.doubleDataMut();
        for (keel::usize i = 0; i < M; ++i)
            yp[i] = b0;
        const Value *pNames = mdl.hasField("PredictorNames") ? &mdl.field("PredictorNames") : nullptr;
        for (keel::usize j = 0; j < P; ++j) {
            keel::i32 colIdx = -1;
            if (pNames && pNames->isCell() && j < pNames->numel())
                colIdx = tp->findVar(pNames->cellAt(j).toString());
            if (colIdx < 0 && j < tp->width())
                colIdx = keel::as<keel::i32>(j);
            if (colIdx >= 0 && keel::as<keel::usize>(colIdx) < tp->width()) {
                const Value &col = tp->cols[keel::as<keel::usize>(colIdx)];
                keel::f64 bj = est.elemAsDouble(j + 1);
                for (keel::usize i = 0; i < M && i < col.numel(); ++i)
                    yp[i] += bj * col.elemAsDouble(i);
            }
        }
        outs[0] = keel::move(ypred);
        return;
    }

    if (!Xnew.isNumeric())
        throw Error("predict: Xnew must be a numeric array or table",
                    0, 0, "predict", "", "numkit:predict:invalidX");

    if (P <= 1 && (Xnew.dims().rows() == 1 || Xnew.dims().cols() == 1)) {
        Value ypred = Value::matrix(Xnew.dims().rows(), Xnew.dims().cols(), ValueType::DOUBLE, mr);
        keel::f64 *yp = ypred.doubleDataMut();
        keel::f64 b1 = (P == 1) ? est.elemAsDouble(1) : 0.0;
        for (keel::usize i = 0; i < Xnew.numel(); ++i)
            yp[i] = b0 + b1 * Xnew.elemAsDouble(i);
        outs[0] = keel::move(ypred);
        return;
    }

    const keel::usize M = Xnew.dims().rows();
    const keel::usize xc = Xnew.dims().cols();
    Value ypred = Value::matrix(M, 1, ValueType::DOUBLE, mr);
    keel::f64 *yp = ypred.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) {
        keel::f64 s = b0;
        for (keel::usize j = 0; j < P && j < xc; ++j)
            s += est.elemAsDouble(j + 1) * Xnew.elemAsDouble(j * M + i);
        yp[i] = s;
    }
    outs[0] = keel::move(ypred);
}

} // namespace numkit::stats::detail
