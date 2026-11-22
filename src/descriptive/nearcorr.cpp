// toolboxes/stats/src/descriptive/nearcorr.cpp
//
// Nearest correlation matrix (Higham 2002 alternating projections with
// Dykstra's correction), delegating numerical kernels to keel::stats::nearcorr.

#include <keel/arena.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/stats.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/stats/descriptive/descriptive.hpp>
#include "descriptive/descriptive_extras_detail.hpp"

#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::stats {

namespace {

using argcheck::ClsDbl;
using argcheck::ClsSgl;
using argcheck::classMaskOf;

inline constexpr argcheck::ClassMask kFloatReal =
    classMaskOf(ClsDbl) | classMaskOf(ClsSgl);

void nearcorr_argcheck(const Value &A)
{
    const auto rejectA = [&](keel::StringRef why) {
        throw Error(keel::format("nearcorr: invalid value of 'A'. Expected A to be a "
                    "symmetric matrix with unit diagonal and entries in "
                    "[-1,1] of double or single class ({}).", why),
                    0, 0, "nearcorr", "", "numkit:nearcorr:invalidInputA");
    };
    if (!A.isNumeric() || A.numel() == 0 || A.dims().ndim() > 2)
        rejectA(keel::format("got a {}{}", mtypeName(A.type()),
                    A.numel() == 0 ? " array, which is empty or not a matrix"
                                   : " array, which is not a matrix"));
    if (!argcheck::maskAccepts(kFloatReal, argcheck::classBitOf(A.type())))
        rejectA("integer classes are not accepted; pass double or single");
    if (A.isComplex())
        rejectA("got a complex array");
    const keel::usize N = A.dims().rows();
    if (A.dims().cols() != N)
        rejectA("the matrix is not square");
    for (keel::usize j = 0; j < N; ++j)
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = A.elemAsDouble(i + j * N);
            if (v < -1.0 || v > 1.0)
                rejectA(keel::format("entry ({},{}) is outside [-1,1]", i + 1, j + 1));
            if (i < j && v != A.elemAsDouble(j + i * N))
                rejectA("the matrix is not symmetric");
        }
    for (keel::usize i = 0; i < N; ++i)
        if (A.elemAsDouble(i + i * N) != 1.0)
            rejectA(keel::format("diagonal entry ({},{}) is not 1", i + 1, i + 1));
}

} // namespace

Value nearcorr_full(const Value &A, keel::ArrayRef<Value> nv, keel::Allocator *mr)
{
    nearcorr_argcheck(A);
    const keel::usize N = A.dims().rows();
    const bool wantSingle = (A.type() == ValueType::SINGLE);

    keel::f64 tol = 1e-6;
    keel::f64 maxIterD = 200.0;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> wScale(scratch.allocator());

    static constexpr keel::StringRef kNv[] = {"tolerance", "maxiterations", "method", "weights"};
    const auto badOpt = [&](keel::StringRef why) {
        throw Error(keel::format("nearcorr: error in optional input arguments ({}).", why),
                    0, 0, "nearcorr", "", "numkit:nearcorr:optionalInputError");
    };
    argcheck::forEachNvPair(
        nv, 0, keel::ArrayRef<keel::StringRef>(kNv), "nearcorr",
        "numkit:nearcorr:optionalInputError",
        [&](keel::usize hit, const Value &v) {
            switch (hit) {
            case 0: {
                if (v.type() != ValueType::DOUBLE || !v.isScalar())
                    badOpt("'Tolerance' must be a double scalar");
                const keel::f64 t = v.toScalar();
                if (!(t > 0.0))
                    badOpt(keel::format("'Tolerance' must be positive, got {}", t));
                tol = t;
                break;
            }
            case 1: {
                if (v.type() != ValueType::DOUBLE || !v.isScalar())
                    badOpt("'MaxIterations' must be a double scalar");
                const keel::f64 m = v.toScalar();
                if (!(m > 0.0) || m != keel::floor(m))
                    badOpt(keel::format("'MaxIterations' must be a positive integer, got {}", m));
                maxIterD = m;
                break;
            }
            case 2: {
                keel::String rawMv = v.toString();
                keel::String mv = rawMv.clone();
                mv.to_ascii_lower_in_place();
                if (mv != "newton" && mv != "projection")
                    badOpt(keel::format("'Method' must be 'newton' or 'projection', got '{}'", rawMv));
                break;
            }
            case 3: {
                if (!v.isNumeric() || v.isComplex() || v.numel() == 0
                    || v.dims().ndim() > 2
                    || !argcheck::maskAccepts(kFloatReal, argcheck::classBitOf(v.type())))
                    badOpt("'Weights' must be a real float vector or symmetric matrix");
                const bool isVec = v.dims().isVector() && v.numel() == N;
                const bool isMat = (v.dims().rows() == N && v.dims().cols() == N);
                if (!isVec && !isMat)
                    badOpt("'Weights' must be a length-N vector or N-by-N matrix");
                for (keel::usize k = 0; k < v.numel(); ++k) {
                    const keel::f64 wk = v.elemAsDouble(k);
                    if (!(wk > 0.0) || !keel::is_finite(wk))
                        badOpt("'Weights' entries must be finite and positive");
                }
                wScale.resize(N * N);
                if (isVec) {
                    for (keel::usize j = 0; j < N; ++j)
                        for (keel::usize i = 0; i < N; ++i)
                            wScale[i + j * N] = keel::sqrt(v.elemAsDouble(i) * v.elemAsDouble(j));
                } else {
                    for (keel::usize j = 0; j < N; ++j)
                        for (keel::usize i = 0; i < N; ++i) {
                            if (i < j && v.elemAsDouble(i + j * N) != v.elemAsDouble(j + i * N))
                                badOpt("'Weights' matrix must be symmetric");
                            wScale[i + j * N] = keel::sqrt(v.elemAsDouble(i + j * N));
                        }
                }
                break;
            }
            default: break;
            }
        });

    if (N == 1)
        return wantSingle ? Value::scalar(keel::as<keel::f32>(1.0)) : Value::scalar(1.0);

    keel::Array<keel::f64> av(N * N, 0.0, scratch.allocator());
    for (keel::usize k = 0; k < N * N; ++k) av[k] = A.elemAsDouble(k);

    keel::Array<keel::f64> res(N * N, 0.0, scratch.allocator());
    (void)keel::stats::nearcorr(av.as_ref(), N, wScale.as_ref(), tol,
                                keel::as<keel::i64>(maxIterD), res.as_mut_ref(),
                                scratch.allocator());

    Value out = wantSingle ? Value::matrix(N, N, ValueType::SINGLE, mr)
                           : Value::matrix(N, N, ValueType::DOUBLE, mr);
    if (wantSingle) {
        keel::f32 *od = out.singleDataMut();
        for (keel::usize k = 0; k < N * N; ++k) od[k] = keel::as<keel::f32>(res[k]);
    } else {
        keel::copy_n(out.doubleDataMut(), res.data(), N * N);
    }
    return out;
}

Value nearcorr(const Value &A, keel::Allocator *mr)
{
    return nearcorr_full(A, keel::ArrayRef<Value>(), mr);
}

} // namespace numkit::stats
