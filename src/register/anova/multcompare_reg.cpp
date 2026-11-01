// toolboxes/signal/src/anova/multcompare_reg.cpp
//
// CallContext register half of anova/multcompare.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/anova/anova.hpp>
#include <numkit/stats/distributions/students_t.hpp>   // tinv, tcdf
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
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

McCorrection parse_ctype(const std::string &raw)
{
    std::string s = raw;
    for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "bonferroni")                 return McCorrection::Bonferroni;
    if (s == "lsd")                        return McCorrection::LSD;
    if (s == "tukey-kramer" || s == "hsd") return McCorrection::TukeyKramer;
    throw Error("multcompare: unknown ctype '" + raw + "'",
                0, 0, "multcompare", "", "numkit:multcompare:badCtype");
}

} // namespace

void multcompare_reg(Span<const Value> args, size_t /*nargout*/,
                     Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "multcompare");  // no max: name-value tail in the reference
    double alpha = 0.05;
    McCorrection ctype = McCorrection::Bonferroni;

    size_t i = 1;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()) {
        if (!args[i].isEmpty())
            alpha = args[i].toScalar();
        ++i;
        if (i < args.size() && (args[i].isChar() || args[i].isString()) && i + 1 == args.size()) {
            ctype = parse_ctype(args[i].toString());
            ++i;
        }
    } else if (i < args.size() && (args[i].isChar() || args[i].isString()) && i + 1 == args.size()) {
        ctype = parse_ctype(args[i].toString());
        ++i;
    }

    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("multcompare: expected parameter name",
                        0, 0, "multcompare", "", "numkit:multcompare:badParam");
        std::string key = args[i].toString();
        for (auto &c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const Value &val = args[i + 1];
        if (key == "alpha") {
            alpha = val.toScalar();
        } else if (key == "ctype" || key == "criticalvaluetype") {
            ctype = parse_ctype(val.toString());
        } else if (key == "display" || key == "dimension" || key == "estimate") {
            // Optional display/dimension flags accepted for compatibility.
        } else {
            throw Error("multcompare: unknown parameter '" + args[i].toString() + "'",
                        0, 0, "multcompare", "", "numkit:multcompare:badParam");
        }
        i += 2;
    }
    outs[0] = multcompare(args[0], alpha, ctype, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::stats
