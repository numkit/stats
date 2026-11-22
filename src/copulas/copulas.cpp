// toolboxes/stats/src/copulas/copulas.cpp
//
// Five copula families (Gaussian, t, Clayton, Frank, Gumbel) — pdf + cdf.

#include <keel/arena.h>
#include <keel/math.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/stats.h>
#include <numkit/stats/copulas/copulas.hpp>
#include <numkit/stats/distributions/multivariate.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <utility>

namespace numkit::stats {

namespace {

std::pair<keel::usize, keel::usize>
copula_U_argcheck(const Value &U, keel::usize d_expected, const char *fn)
{
    keel::usize n = 0, d = 0;
    if (U.isScalar()) {
        if (d_expected != 1)
            throw Error(keel::format("{}: U has wrong shape", fn),
                        0, 0, fn, "", keel::format("numkit:{}:shapeU", fn));
        n = 1; d = 1;
    } else if (U.dims().rows() == 1) {
        n = 1; d = U.dims().cols();
    } else {
        n = U.dims().rows(); d = U.dims().cols();
    }
    if (d != d_expected)
        throw Error(keel::format("{}: U has {} columns, expected {}", fn, d, d_expected),
                    0, 0, fn, "", keel::format("numkit:{}:dimU", fn));
    for (keel::usize i = 0; i < n * d; ++i) {
        const keel::f64 v = U.elemAsDouble(i);
        if (!(v > 0.0 && v < 1.0))
            throw Error(keel::format("{}: U entries must lie in (0, 1)", fn),
                        0, 0, fn, "", keel::format("numkit:{}:rangeU", fn));
    }
    return {n, d};
}

keel::f64 extract_rho_2x2(const Value &R, const char *fn)
{
    if (R.dims().rows() != 2 || R.dims().cols() != 2)
        throw Error(keel::format("{}: expected 2 × 2 R", fn),
                    0, 0, fn, "", keel::format("numkit:{}:dimR", fn));
    const keel::f64 r00 = R.elemAsDouble(0);
    const keel::f64 r10 = R.elemAsDouble(1);
    const keel::f64 r01 = R.elemAsDouble(2);
    const keel::f64 r11 = R.elemAsDouble(3);
    if (keel::abs(r00 - 1.0) > 1e-9 || keel::abs(r11 - 1.0) > 1e-9)
        throw Error(keel::format("{}: R must have unit diagonal", fn),
                    0, 0, fn, "", keel::format("numkit:{}:Rdiag", fn));
    if (keel::abs(r01 - r10) > 1e-9)
        throw Error(keel::format("{}: R must be symmetric", fn),
                    0, 0, fn, "", keel::format("numkit:{}:Rsym", fn));
    const keel::f64 rho = 0.5 * (r01 + r10);
    if (!(rho > -1.0 && rho < 1.0))
        throw Error(keel::format("{}: ρ must lie strictly in (-1, 1)", fn),
                    0, 0, fn, "", keel::format("numkit:{}:Rrange", fn));
    return rho;
}

keel::usize check_R_dim(const Value &R, const char *fn)
{
    if (R.dims().rows() != R.dims().cols() || R.dims().rows() < 2)
        throw Error(keel::format("{}: R must be a square d × d matrix with d ≥ 2", fn),
                    0, 0, fn, "", keel::format("numkit:{}:dimR", fn));
    const keel::usize d = R.dims().rows();
    for (keel::usize i = 0; i < d; ++i) {
        if (keel::abs(R.elemAsDouble(i * d + i) - 1.0) > 1e-9)
            throw Error(keel::format("{}: R must have unit diagonal", fn),
                        0, 0, fn, "", keel::format("numkit:{}:Rdiag", fn));
        for (keel::usize j = i + 1; j < d; ++j) {
            if (keel::abs(R.elemAsDouble(j * d + i) - R.elemAsDouble(i * d + j)) > 1e-9)
                throw Error(keel::format("{}: R must be symmetric", fn),
                            0, 0, fn, "", keel::format("numkit:{}:Rsym", fn));
        }
    }
    return d;
}

void unpack_U(const Value &U, keel::usize n, keel::usize d, keel::Array<keel::f64> &buf)
{
    buf.resize(n * d);
    if (U.isScalar()) {
        buf[0] = U.toScalar();
        return;
    }
    for (keel::usize i = 0; i < n * d; ++i)
        buf[i] = U.elemAsDouble(i);
}

void unpack_R(const Value &R, keel::usize d, keel::Array<keel::f64> &buf)
{
    buf.resize(d * d);
    for (keel::usize i = 0; i < d * d; ++i)
        buf[i] = R.elemAsDouble(i);
}

keel::f64 bivtcdf(keel::f64 a, keel::f64 b, keel::f64 rho, keel::f64 nu, keel::Allocator *mr)
{
    Value Cval = Value::matrix(2, 2, ValueType::DOUBLE, mr);
    Cval.doubleDataMut()[0] = 1.0;
    Cval.doubleDataMut()[1] = rho;
    Cval.doubleDataMut()[2] = rho;
    Cval.doubleDataMut()[3] = 1.0;
    Value Xval = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    Xval.doubleDataMut()[0] = a;
    Xval.doubleDataMut()[1] = b;
    Value Pv = mvtcdf(Xval, Cval, nu, 0.001, mr);
    return Pv.elemAsDouble(0);
}

} // anonymous

Value copulapdf_gaussian(const Value &U, const Value &R, keel::Allocator *mr)
{
    const keel::usize d = check_R_dim(R, "copulapdf");
    auto [n, dU] = copula_U_argcheck(U, d, "copulapdf");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    keel::Array<keel::f64> r_buf(scratch.allocator());
    unpack_U(U, n, d, u_buf);
    unpack_R(R, d, r_buf);

    const auto st = keel::stats::copula_pdf_gaussian(
        u_buf.as_ref(), n, d, r_buf.as_ref(),
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n), scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("copulapdf: R must be positive definite",
                    0, 0, "copulapdf", "", "numkit:copulapdf:notPD");
    return out;
}

Value copulacdf_gaussian(const Value &U, const Value &R, keel::Allocator *mr)
{
    const keel::usize d = check_R_dim(R, "copulacdf");
    auto [n, dU] = copula_U_argcheck(U, d, "copulacdf");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    unpack_U(U, n, d, u_buf);

    if (d == 2) {
        const keel::f64 rho = extract_rho_2x2(R, "copulacdf");
        (void)keel::stats::copula_cdf_gaussian_2d(
            u_buf.as_ref(), n, rho, keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n));
        return out;
    }

    Value Xz = Value::matrix(n, d, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n * d; ++i)
        Xz.doubleDataMut()[i] = keel::stats::norm_inv(u_buf[i], 0.0, 1.0);
    Value mu0 = Value::matrix(1, d, ValueType::DOUBLE, mr);
    for (keel::usize j = 0; j < d; ++j) mu0.doubleDataMut()[j] = 0.0;
    Value Pv = mvncdf(Xz, mu0, R, mr);
    for (keel::usize i = 0; i < n; ++i) out.doubleDataMut()[i] = Pv.elemAsDouble(i);
    return out;
}

Value copulapdf_t(const Value &U, const Value &R, keel::f64 nu, keel::Allocator *mr)
{
    const keel::usize d = check_R_dim(R, "copulapdf");
    auto [n, dU] = copula_U_argcheck(U, d, "copulapdf");
    if (!(nu > 0.0))
        throw Error("copulapdf: nu must be positive",
                    0, 0, "copulapdf", "", "numkit:copulapdf:badNu");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    keel::Array<keel::f64> r_buf(scratch.allocator());
    unpack_U(U, n, d, u_buf);
    unpack_R(R, d, r_buf);

    const auto st = keel::stats::copula_pdf_t(
        u_buf.as_ref(), n, d, r_buf.as_ref(), nu,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n), scratch.allocator());
    if (st == keel::stats::StatsStatus::kSingularMatrix)
        throw Error("copulapdf: R must be positive definite",
                    0, 0, "copulapdf", "", "numkit:copulapdf:notPD");
    return out;
}

Value copulacdf_t(const Value &U, const Value &R, keel::f64 nu, keel::Allocator *mr)
{
    const keel::usize d = check_R_dim(R, "copulacdf");
    auto [n, dU] = copula_U_argcheck(U, d, "copulacdf");
    if (!(nu > 0.0))
        throw Error("copulacdf: nu must be positive",
                    0, 0, "copulacdf", "", "numkit:copulacdf:badNu");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    unpack_U(U, n, d, u_buf);

    if (d == 2) {
        const keel::f64 rho = extract_rho_2x2(R, "copulacdf");
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 z1 = keel::stats::t_inv(u_buf[i], nu);
            const keel::f64 z2 = keel::stats::t_inv(u_buf[n + i], nu);
            out.doubleDataMut()[i] = bivtcdf(z1, z2, rho, nu, mr);
        }
        return out;
    }

    Value Xz = Value::matrix(n, d, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n * d; ++i)
        Xz.doubleDataMut()[i] = keel::stats::t_inv(u_buf[i], nu);
    Value Pv = mvtcdf(Xz, R, nu, 0.005, mr);
    for (keel::usize i = 0; i < n; ++i) out.doubleDataMut()[i] = Pv.elemAsDouble(i);
    return out;
}

Value copulapdf_clayton(const Value &U, keel::f64 alpha, keel::Allocator *mr)
{
    auto [n, d] = copula_U_argcheck(U, 2, "copulapdf");
    if (!(alpha > 0.0))
        throw Error("copulapdf: Clayton alpha must be > 0",
                    0, 0, "copulapdf", "", "numkit:copulapdf:badAlpha");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    unpack_U(U, n, 2, u_buf);
    (void)keel::stats::copula_pdf_clayton(u_buf.as_ref(), n, alpha,
                                          keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n));
    return out;
}

Value copulacdf_clayton(const Value &U, keel::f64 alpha, keel::Allocator *mr)
{
    auto [n, d] = copula_U_argcheck(U, 2, "copulacdf");
    if (!(alpha > 0.0))
        throw Error("copulacdf: Clayton alpha must be > 0",
                    0, 0, "copulacdf", "", "numkit:copulacdf:badAlpha");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    unpack_U(U, n, 2, u_buf);
    (void)keel::stats::copula_cdf_clayton(u_buf.as_ref(), n, alpha,
                                          keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n));
    return out;
}

Value copulapdf_frank(const Value &U, keel::f64 alpha, keel::Allocator *mr)
{
    auto [n, d] = copula_U_argcheck(U, 2, "copulapdf");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    unpack_U(U, n, 2, u_buf);
    (void)keel::stats::copula_pdf_frank(u_buf.as_ref(), n, alpha,
                                        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n));
    return out;
}

Value copulacdf_frank(const Value &U, keel::f64 alpha, keel::Allocator *mr)
{
    auto [n, d] = copula_U_argcheck(U, 2, "copulacdf");
    if (alpha == 0.0)
        throw Error("copulacdf: Frank alpha must be non-zero "
                    "(use independence for α = 0)",
                    0, 0, "copulacdf", "", "numkit:copulacdf:badAlpha");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    unpack_U(U, n, 2, u_buf);
    (void)keel::stats::copula_cdf_frank(u_buf.as_ref(), n, alpha,
                                        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n));
    return out;
}

Value copulapdf_gumbel(const Value &U, keel::f64 alpha, keel::Allocator *mr)
{
    auto [n, d] = copula_U_argcheck(U, 2, "copulapdf");
    if (!(alpha >= 1.0))
        throw Error("copulapdf: Gumbel alpha must be ≥ 1",
                    0, 0, "copulapdf", "", "numkit:copulapdf:badAlpha");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    unpack_U(U, n, 2, u_buf);
    (void)keel::stats::copula_pdf_gumbel(u_buf.as_ref(), n, alpha,
                                         keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n));
    return out;
}

Value copulacdf_gumbel(const Value &U, keel::f64 alpha, keel::Allocator *mr)
{
    auto [n, d] = copula_U_argcheck(U, 2, "copulacdf");
    if (!(alpha >= 1.0))
        throw Error("copulacdf: Gumbel alpha must be ≥ 1",
                    0, 0, "copulacdf", "", "numkit:copulacdf:badAlpha");
    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> u_buf(scratch.allocator());
    unpack_U(U, n, 2, u_buf);
    (void)keel::stats::copula_cdf_gumbel(u_buf.as_ref(), n, alpha,
                                         keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n));
    return out;
}

} // namespace numkit::stats
