// toolboxes/signal/src/cluster/knnsearch_reg.cpp
//
// CallContext register half of cluster/knnsearch.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/cluster/distance.hpp>
#include <numkit/stats/cluster/knnsearch.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "cluster/knnsearch_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>


namespace numkit::stats {

namespace detail {

void knnsearch_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "knnsearch");  // no max: name-value tail in the reference
    KnnOpts o = parse_knn_args(args, "knnsearch");
    auto [Idx, D] = knnsearch(args[0], args[1], o.K, o.metric, o.p, ctx.engine->allocator());
    outs[0] = keel::move(Idx);
    if (nargout > 1) outs[1] = keel::move(D);
}

void rangesearch_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "rangesearch");  // no max: name-value tail in the reference
    const keel::f64 r = args[2].toScalar();
    KnnOpts o;
    // parse remaining name-value
    if (args.size() > 3) {
        // shift args >= 3 into a temp slice for parser by reusing logic
        // (pdist2 args 0/1 are the X,Y placeholders for parse_knn_args).
        // Simpler: hand-parse here.
        keel::usize i = 3;
        while (i < args.size()) {
            if (!args[i].isChar() && !args[i].isString())
                throw Error("rangesearch: expected name-value pair",
                            0, 0, "rangesearch", "", "numkit:rangesearch:nvpair");
            keel::String name = args[i].toString();
            name.to_ascii_lower_in_place();
            if (i + 1 >= args.size())
                throw Error(keel::format("rangesearch: '{}' missing value", name),
                            0, 0, "rangesearch", "", "numkit:rangesearch:nvval");
            const Value &v = args[i + 1];
            if (name == "distance") {
                if (!v.isChar() && !v.isString())
                    throw Error("rangesearch: Distance must be a string",
                                0, 0, "rangesearch", "", "numkit:rangesearch:dist");
                o.metric = v.toString();
            } else if (name == "p" || name == "minkowskiexponent") {
                o.p = v.toScalar();
            }
            i += 2;
        }
    }
    auto [Idx, D] = rangesearch(args[0], args[1], r, o.metric, o.p, ctx.engine->allocator());
    outs[0] = keel::move(Idx);
    if (nargout > 1) outs[1] = keel::move(D);
}

} // namespace detail

} // namespace numkit::stats
