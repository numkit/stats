// toolboxes/signal/src/copulas/copulas_reg.cpp
//
// CallContext register half of copulas/copulas.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/copulas/copulas.hpp>
#include <numkit/stats/distributions/multivariate.hpp>
#include <numkit/stats/distributions/normal.hpp>
#include <numkit/stats/distributions/students_t.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/argcheck.hpp>

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

namespace {
std::string family_lower(const Value &v)
{
    std::string s = v.toString();
    for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
}

void copulapdf_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "copulapdf");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    const std::string fam = family_lower(args[0]);
    if (fam == "gaussian") {
        outs[0] = copulapdf_gaussian(args[1], args[2], mr);
    } else if (fam == "t") {
        if (args.size() < 4)
            throw Error("copulapdf 't': requires nu (4th arg)",
                        0, 0, "copulapdf", "", "numkit:copulapdf:nuMissing");
        outs[0] = copulapdf_t(args[1], args[2], args[3].toScalar(), mr);
    } else if (fam == "clayton") {
        outs[0] = copulapdf_clayton(args[1], args[2].toScalar(), mr);
    } else if (fam == "frank") {
        outs[0] = copulapdf_frank(args[1], args[2].toScalar(), mr);
    } else if (fam == "gumbel") {
        outs[0] = copulapdf_gumbel(args[1], args[2].toScalar(), mr);
    } else {
        throw Error("copulapdf: unknown family '" + fam + "'",
                    0, 0, "copulapdf", "", "numkit:copulapdf:badFamily");
    }
}

void copulacdf_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "copulacdf");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->resource();
    const std::string fam = family_lower(args[0]);
    if (fam == "gaussian") {
        outs[0] = copulacdf_gaussian(args[1], args[2], mr);
    } else if (fam == "t") {
        if (args.size() < 4)
            throw Error("copulacdf 't': requires nu (4th arg)",
                        0, 0, "copulacdf", "", "numkit:copulacdf:nuMissing");
        outs[0] = copulacdf_t(args[1], args[2], args[3].toScalar(), mr);
    } else if (fam == "clayton") {
        outs[0] = copulacdf_clayton(args[1], args[2].toScalar(), mr);
    } else if (fam == "frank") {
        outs[0] = copulacdf_frank(args[1], args[2].toScalar(), mr);
    } else if (fam == "gumbel") {
        outs[0] = copulacdf_gumbel(args[1], args[2].toScalar(), mr);
    } else {
        throw Error("copulacdf: unknown family '" + fam + "'",
                    0, 0, "copulacdf", "", "numkit:copulacdf:badFamily");
    }
}

} // namespace detail

} // namespace numkit::stats
