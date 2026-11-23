// toolboxes/stats/src/lda/lda.cpp
//
// Discriminant analysis (classify), delegating numerical kernels to keel::stats.

#include <keel/arena.h>
#include <keel/tuple.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/stats.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/stats/lda/lda.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::stats {

namespace {

keel::stats::DiscrimType classify_argcheck(const Value &sample, const Value &training,
                                           const Value &group, keel::StringRef type_in)
{
    const keel::usize d   = training.dims().cols();
    const keel::usize Ntr = training.dims().rows();
    if (sample.dims().cols() != d)
        throw Error("classify: sample and training must have the same number of columns",
                    0, 0, "classify", "", "numkit:classify:cols");
    if (group.numel() != Ntr)
        throw Error("classify: group must have one entry per training row",
                    0, 0, "classify", "", "numkit:classify:group");

    keel::String type = keel::String::from(type_in);
    type.to_ascii_lower_in_place();
    if (type.empty() || type == "linear") return keel::stats::DiscrimType::kLinear;
    if (type == "diaglinear")             return keel::stats::DiscrimType::kDiagLinear;
    if (type == "quadratic")              return keel::stats::DiscrimType::kQuadratic;
    if (type == "diagquadratic")          return keel::stats::DiscrimType::kDiagQuadratic;
    throw Error("classify: unsupported type (linear|diaglinear|quadratic|diagquadratic)",
                0, 0, "classify", "", "numkit:classify:type");
}

} // namespace

keel::Tuple<Value, Value, Value, Value>
classify(const Value &sample, const Value &training, const Value &group,
         keel::StringRef type_in, keel::Allocator *mr)
{
    const auto dtype = classify_argcheck(sample, training, group, type_in);
    const keel::usize d     = training.dims().cols();
    const keel::usize Ntr   = training.dims().rows();
    const keel::usize Nsamp = sample.dims().rows();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> sv(Nsamp * d, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < Nsamp * d; ++i) sv[i] = sample.elemAsDouble(i);
    keel::Array<keel::f64> tv(Ntr * d, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < Ntr * d; ++i) tv[i] = training.elemAsDouble(i);
    keel::Array<keel::f64> gv(Ntr, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < Ntr; ++i) gv[i] = group.elemAsDouble(i);

    keel::stats::LdaResult res(scratch.allocator());
    const auto st = keel::stats::lda_classify(
        sv.as_ref(), Nsamp, tv.as_ref(), Ntr, d, gv.as_ref(), dtype, res, scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("classify: covariance is not positive definite",
                    0, 0, "classify", "", "numkit:classify:psd");

    Value cV    = Value::matrix(Nsamp, 1, ValueType::DOUBLE, mr);
    Value postV = Value::matrix(Nsamp, res.num_classes, ValueType::DOUBLE, mr);
    Value logpV = Value::matrix(Nsamp, 1, ValueType::DOUBLE, mr);
    if (Nsamp > 0) {
        keel::copy_n(cV.doubleDataMut(), res.pred.data(), Nsamp);
        keel::copy_n(postV.doubleDataMut(), res.posterior.data(), Nsamp * res.num_classes);
        keel::copy_n(logpV.doubleDataMut(), res.logp.data(), Nsamp);
    }
    return {keel::move(cV), Value::scalar(res.train_error, mr), keel::move(postV), keel::move(logpV)};
}

} // namespace numkit::stats
