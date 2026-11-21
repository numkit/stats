// toolboxes/stats/src/anova/anova.cpp
//
// anova1, anova2, kruskalwallis, friedman, dummyvar delegating numerical
// kernels to keel::stats.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/anova/anova.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>

#include "anova_detail.hpp"

namespace numkit::stats {

std::tuple<keel::f64, keel::f64, keel::f64, keel::f64, keel::f64, keel::f64>
anova1(const Value &y, const Value &group, keel::Allocator *mr)
{
    auto buckets = bucket(y, group);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> vals(scratch.allocator());
    keel::Array<keel::usize> offsets(scratch.allocator());
    offsets.push_back(0u);
    for (keel::usize k = 0; k < buckets.size(); ++k) {
        for (keel::f64 v : buckets[k].values) vals.push_back(v);
        offsets.push_back(vals.size());
    }
    keel::stats::Anova1Result res;
    (void)keel::stats::anova1_grouped(vals.as_ref(), offsets.as_ref(), res);
    return std::make_tuple(res.p, res.f, res.df_between, res.df_within, res.ss_between, res.ss_within);
}

std::tuple<keel::f64, keel::f64, keel::f64, keel::f64,
           keel::f64, keel::f64, keel::f64,
           keel::f64, keel::f64, keel::f64>
anova2(const Value &Y, keel::Allocator *mr)
{
    if (Y.dims().ndim() != 2)
        throw Error("anova2: input must be a 2D matrix", 0, 0, "anova2", "", "numkit:anova2:notMatrix");
    const keel::usize m = keel::as<keel::usize>(Y.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(Y.dims().dim(1));
    if (m < 2 || n < 2)
        throw Error("anova2: input must be at least 2x2", 0, 0, "anova2", "", "numkit:anova2:tooSmall");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> yv(m * n, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < m * n; ++i) yv[i] = Y.elemAsDouble(i);

    keel::stats::Anova2Result res;
    (void)keel::stats::anova2(yv.as_ref(), m, n, res, scratch.allocator());
    return std::make_tuple(res.p_cols, res.p_rows, res.f_cols, res.f_rows,
                           res.df_cols, res.df_rows, res.df_err,
                           res.ss_cols, res.ss_rows, res.ss_err);
}

std::tuple<Value, Value, Value, Value>
kruskalwallis(const Value &y, const Value &group, keel::Allocator *mr)
{
    auto buckets = bucket(y, group);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> vals(scratch.allocator());
    keel::Array<keel::usize> offsets(scratch.allocator());
    offsets.push_back(0u);
    for (keel::usize k = 0; k < buckets.size(); ++k) {
        for (keel::f64 v : buckets[k].values) vals.push_back(v);
        offsets.push_back(vals.size());
    }
    keel::stats::KruskalResult res;
    (void)keel::stats::kruskal_wallis(vals.as_ref(), offsets.as_ref(), res, scratch.allocator());
    return std::make_tuple(Value::scalar(res.p, mr), Value::scalar(res.h, mr),
                           Value::scalar(res.df, mr), Value::scalar(res.sum_r2_n, mr));
}

std::tuple<Value, Value, Value>
friedman(const Value &x, keel::i32 reps, keel::Allocator *mr)
{
    if (reps < 1)
        throw Error("friedman: reps must be a positive integer", 0, 0, "friedman", "", "numkit:friedman:reps");

    const keel::usize nrows = x.dims().rows();
    const keel::usize ncols = x.dims().cols();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(nrows * ncols, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < nrows * ncols; ++i) xv[i] = x.elemAsDouble(i);

    keel::stats::FriedmanResult res;
    (void)keel::stats::friedman(xv.as_ref(), nrows, ncols, reps, res, scratch.allocator());
    return std::make_tuple(Value::scalar(res.p, mr), Value::scalar(res.q, mr),
                           Value::scalar(res.df, mr));
}

Value dummyvar(const Value &group, keel::Allocator *mr)
{
    const keel::usize N = group.numel();
    if (N == 0) return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    keel::Array<keel::f64> uniq;
    uniq.reserve(N);
    for (keel::usize i = 0; i < N; ++i) uniq.push_back(group.elemAsDouble(i));
    std::sort(uniq.begin(), uniq.end());
    uniq.resize(keel::as<keel::usize>(std::unique(uniq.begin(), uniq.end()) - uniq.begin()));
    const keel::usize K = uniq.size();

    Value out = Value::matrix(N, K, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N * K; ++i) od[i] = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 li = group.elemAsDouble(i);
        for (keel::usize k = 0; k < K; ++k) {
            if (uniq[k] == li) {
                od[i + k * N] = 1.0;
                break;
            }
        }
    }
    return out;
}

} // namespace numkit::stats
