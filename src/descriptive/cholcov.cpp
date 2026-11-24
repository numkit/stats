// toolboxes/stats/src/descriptive/cholcov.cpp
// Thin Value wrapper over keel::stats::cholcov.

#include <keel/algo.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/descriptive/descriptive.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::stats {

namespace {

void cholcov_argcheck(const Value &SIGMA)
{
    if (SIGMA.dims().rows() != SIGMA.dims().cols())
        throw Error("cholcov: SIGMA must be square",
                    0, 0, "cholcov", "", "numkit:cholcov:NotSquare");
}

} // namespace

keel::Pair<Value, Value>
cholcov(const Value &SIGMA, keel::Allocator *mr)
{
    cholcov_argcheck(SIGMA);
    const keel::usize R = SIGMA.dims().rows();
    if (R == 0) {
        Value T = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return {keel::move(T), Value::scalar(0.0, mr)};
    }

    keel::stats::CholCovResult res(mr);
    const auto st = keel::stats::cholcov(keel::ArrayRef<keel::f64>(SIGMA.doubleData(), R * R),
                                         R, res, mr);
    if (st != keel::stats::StatsStatus::kOk)
        throw Error("cholcov: invalid matrix", 0, 0, "cholcov", "", "numkit:cholcov:Invalid");

    if (res.non_pd_count > 0) {
        Value T = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return {keel::move(T), Value::scalar(keel::as<keel::f64>(res.non_pd_count), mr)};
    }

    Value T = Value::matrix(res.rows, res.cols, ValueType::DOUBLE, mr);
    if (!res.t_col_major.empty())
        keel::copy_n(T.doubleDataMut(), res.t_col_major.data(), res.rows * res.cols);
    return {keel::move(T), Value::scalar(0.0, mr)};
}

} // namespace numkit::stats
