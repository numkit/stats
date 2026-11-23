// toolboxes/signal/src/descriptive/crosstab_reg.cpp
//
// CallContext register half of descriptive/crosstab.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/array.h>
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/stats/distributions/chi2.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "descriptive/crosstab_detail.hpp"
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>


namespace numkit::stats {

namespace detail {

void crosstab_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "crosstab");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    const Value &y_opt = (args.size() >= 2) ? args[1] : Value::Empty;
    auto [T, chi2, p] = crosstab(args[0], y_opt, mr);
    outs[0] = keel::move(T);
    if (nargout > 1) outs[1] = Value::scalar(chi2, mr);
    if (nargout > 2) outs[2] = Value::scalar(p, mr);
    if (nargout > 3) {
        // labels: (maxCategories × numVars) cell. Column j = num2str of
        // each sorted-unique value of variable j (char row); rows beyond
        // a variable's category count are padded with [] (empty double),
        // matching MATLAB R2025b.
        keel::Array<keel::Array<keel::f64>> uniques;
        uniques.push_back(sortedUniqueNoNaN(args[0]));
        if (!y_opt.isEmpty())
            uniques.push_back(sortedUniqueNoNaN(y_opt));
        keel::usize maxR = 0;
        for (const auto &u : uniques) maxR = keel::max(maxR, u.size());
        const keel::usize numVars = uniques.size();
        Value labels = Value::cell(maxR, numVars, mr);
        for (keel::usize j = 0; j < numVars; ++j) {
            const auto &u = uniques[j];
            for (keel::usize i = 0; i < maxR; ++i) {
                Value &cell = labels.cellAt(i + j * maxR);
                cell = (i < u.size())
                           ? Value::fromString(labelString(u[i]), mr)
                           : Value::Empty;
            }
        }
        outs[3] = keel::move(labels);
    }
}

} // namespace detail

} // namespace numkit::stats
