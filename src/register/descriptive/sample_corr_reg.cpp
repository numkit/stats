// toolboxes/stats/src/descriptive/sample_corr_reg.cpp
//
// CallContext register half of descriptive/sample_corr.cpp (autocorr / crosscorr).
// Engine-coupled glue: parses the MATLAB call forms (positional NumLags or the
// 'NumLags'/'NumSTD' name-value pairs) and threads the 3 outputs by nargout.
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/stats/descriptive/descriptive.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <tuple>
#include <utility>

namespace numkit::stats {

namespace detail {

namespace {

// Parse optional NumLags / NumSTD from args starting at `start`. A numeric arg
// at `start` is positional NumLags; 'NumLags'/'NumSTD' name-value pairs (any
// case) override. Leaves numLags = -1 (→ default) when unspecified.
void parseCorrOpts(keel::ArrayRef<Value> args, keel::usize start, int &numLags, double &numSTD,
                       const char *fn)
{
    for (keel::usize i = start; i < args.size(); ++i) {
        if (args[i].isChar()) {
            keel::String opt = args[i].toString();
            for (char &c : opt) c = keel::to_ascii_lower(c);
            if (i + 1 < args.size()) {
                if (opt == "numlags") {
                    numLags = argcheck::expectedInteger(
                        argcheck::expectedReal(args[i + 1], int(i) + 2,
                                               "NumLags", fn),
                        int(i) + 2, "NumLags", fn);
                    argcheck::expectedPositive(double(numLags), int(i) + 2,
                                               "NumLags", fn);
                }
                else if (opt == "numstd")  numSTD  = args[i + 1].toScalar();
                ++i;  // consume the value
            }
        } else if (i == start) {
            numLags = argcheck::expectedInteger(
                argcheck::expectedReal(args[i], int(i) + 1, "NumLags", fn),
                int(i) + 1, "NumLags", fn);
            argcheck::expectedPositive(double(numLags), int(i) + 1, "NumLags", fn);
        }
    }
}

} // namespace

void autocorr_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "autocorr");  // no max: name-value tail in the reference
    int    numLags = -1;
    double numSTD  = 2.0;
    parseCorrOpts(args, 1, numLags, numSTD, "autocorr");

    auto [acf, lags, bounds] = autocorr(args[0], numLags, numSTD, ctx.engine->allocator());
    outs[0] = std::move(acf);
    if (nargout > 1) outs[1] = std::move(lags);
    if (nargout > 2) outs[2] = std::move(bounds);
}

void crosscorr_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "crosscorr");  // no max: name-value tail in the reference
    int    numLags = -1;
    double numSTD  = 2.0;
    parseCorrOpts(args, 2, numLags, numSTD, "crosscorr");

    auto [xcf, lags, bounds] =
        crosscorr(args[0], args[1], numLags, numSTD, ctx.engine->allocator());
    outs[0] = std::move(xcf);
    if (nargout > 1) outs[1] = std::move(lags);
    if (nargout > 2) outs[2] = std::move(bounds);
}

void parcorr_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "parcorr");  // no max: name-value tail in the reference
    int    numLags = -1;
    double numSTD  = 2.0;
    parseCorrOpts(args, 1, numLags, numSTD, "parcorr");

    auto [pacf, lags, bounds] = parcorr(args[0], numLags, numSTD, ctx.engine->allocator());
    outs[0] = std::move(pacf);
    if (nargout > 1) outs[1] = std::move(lags);
    if (nargout > 2) outs[2] = std::move(bounds);
}

} // namespace detail

} // namespace numkit::stats
