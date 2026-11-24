// toolboxes/stats/src/anova/multcompare.cpp
// Thin Value wrapper over keel::stats::multcompare.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/stats.h>
#include <numkit/stats/anova/anova.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::stats {

namespace {

void multcompare_argcheck(const Value &stats, keel::f64 alpha)
{
    if (!stats.isStruct())
        throw Error("multcompare: stats must be a struct (from anova1)",
                    0, 0, "multcompare", "", "numkit:multcompare:notStruct");
    if (!stats.hasField("means") || !stats.hasField("n")
        || !stats.hasField("s") || !stats.hasField("df"))
        throw Error("multcompare: stats must have fields {means, n, s, df}",
                    0, 0, "multcompare", "", "numkit:multcompare:missingField");
    if (alpha <= 0.0 || alpha >= 1.0 || keel::is_nan(alpha))
        throw Error("multcompare: alpha must be in (0, 1)",
                    0, 0, "multcompare", "", "numkit:multcompare:badAlpha");
    if (stats.field("means").numel() < 2)
        throw Error("multcompare: need at least 2 groups",
                    0, 0, "multcompare", "", "numkit:multcompare:tooFewGroups");
}

keel::stats::McCorrection toKeelCorrection(McCorrection ctype) noexcept
{
    switch (ctype) {
        case McCorrection::TukeyKramer: return keel::stats::McCorrection::kTukeyKramer;
        case McCorrection::Bonferroni:  return keel::stats::McCorrection::kBonferroni;
        case McCorrection::LSD:         return keel::stats::McCorrection::kLsd;
    }
    return keel::stats::McCorrection::kTukeyKramer;
}

} // namespace

Value multcompare(const Value &stats, keel::f64 alpha, McCorrection ctype,
                   keel::Allocator *mr)
{
    multcompare_argcheck(stats, alpha);

    const Value &means = stats.field("means");
    const Value &nv    = stats.field("n");
    const keel::f64 s  = stats.field("s").toScalar();
    const keel::f64 df = stats.field("df").toScalar();
    const keel::usize K = means.numel();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> m(K, 0.0, scratch.allocator());
    keel::Array<keel::f64> n(K, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < K; ++i) {
        m[i] = means.elemAsDouble(i);
        n[i] = nv.elemAsDouble(i);
    }

    const keel::usize nPairs = K * (K - 1) / 2;
    auto out = Value::matrix(nPairs, 6, ValueType::DOUBLE, mr);
    (void)keel::stats::multcompare(m.as_ref(), n.as_ref(), s, df, alpha,
                                   toKeelCorrection(ctype),
                                   keel::MutArrayRef<keel::f64>(out.doubleDataMut(), nPairs * 6));
    return out;
}

} // namespace numkit::stats
