// toolboxes/stats/src/qmc/qmc.cpp

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/stats/qmc/qmc.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::stats {

namespace {

// First d primes via simple sieve / trial division.
keel::Array<keel::i32> first_primes(keel::usize d)
{
    keel::Array<keel::i32> ps;
    ps.reserve(d);
    keel::i32 x = 2;
    while (ps.size() < d) {
        bool p = true;
        for (keel::i32 q : ps) {
            if (q * q > x) break;
            if (x % q == 0) { p = false; break; }
        }
        if (p) ps.push_back(x);
        ++x;
    }
    return ps;
}

// Reverse-Radix-2 permutation pi_p of {0, ..., p-1}:
// take smallest power of 2 >= p, bit-reverse j = 0 .. 2^k - 1 across k bits,
// and keep values < p in order.
keel::Array<keel::i32> rr2_permutation(keel::i32 p)
{
    keel::i32 k = 1;
    while ((1 << k) < p) ++k;
    keel::i32 m = (1 << k);
    keel::Array<keel::i32> perm;
    perm.reserve(p);
    for (keel::i32 j = 0; j < m; ++j) {
        keel::i32 rev = 0;
        for (keel::i32 b = 0; b < k; ++b) {
            if ((j >> b) & 1) rev |= (1 << (k - 1 - b));
        }
        if (rev < p) perm.push_back(rev);
    }
    return perm;
}

keel::f64 radical_inverse(keel::i64 i, keel::i32 p, const keel::Array<keel::i32> *perm = nullptr)
{
    keel::f64 r = 0.0;
    keel::f64 f = 1.0 / keel::as<keel::f64>(p);
    while (i > 0) {
        keel::i32 d = keel::as<keel::i32>(i % p);
        if (perm) d = (*perm)[keel::as<keel::usize>(d)];
        r += f * keel::as<keel::f64>(d);
        i /= p;
        f /= keel::as<keel::f64>(p);
    }
    return r;
}

} // anonymous

namespace {
void haltonset_argcheck(keel::i32 d)
{
    if (d < 1)
        throw Error("haltonset: dim must be >= 1",
                    0, 0, "haltonset", "", "numkit:haltonset:dim");
}
} // namespace

Value haltonset(keel::i32 d, keel::i64 skip, keel::i64 leap, keel::Allocator *mr)
{
    haltonset_argcheck(d);
    Value s = Value::structure(mr);
    s.field("kind") = Value::fromString("halton", mr);
    s.field("dim")  = Value::scalar(keel::as<keel::f64>(d), mr);
    s.field("skip") = Value::scalar(keel::as<keel::f64>(skip), mr);
    s.field("leap") = Value::scalar(keel::as<keel::f64>(leap), mr);
    s.field("scramble") = Value::fromString("", mr);
    return s;
}

Value scramble(const Value &stream, keel::StringRef type, keel::Allocator *mr)
{
    if (!stream.isStruct() || !stream.hasField("kind"))
        throw Error("scramble: input must be a quasi-random stream struct",
                    0, 0, "scramble", "", "numkit:scramble:struct");
    Value out = stream;
    out.field("scramble") = Value::fromString(type, mr);
    return out;
}

Value net(const Value &stream, keel::i64 n, keel::Allocator *mr)
{
    if (!stream.isStruct() || !stream.hasField("kind"))
        throw Error("net: input must be a quasi-random stream struct",
                    0, 0, "net", "", "numkit:net:struct");
    const keel::String kind = stream.field("kind").toString();
    if (kind != "halton")
        throw Error("net: only halton streams supported in this release",
                    0, 0, "net", "", "numkit:net:kind");
    if (n < 0)
        throw Error("net: n must be >= 0",
                    0, 0, "net", "", "numkit:net:n");
    const keel::usize d    = keel::as<keel::usize>(stream.field("dim").toScalar());
    const keel::i64 skip = keel::as<keel::i64>(stream.field("skip").toScalar());
    const keel::i64 leap = keel::as<keel::i64>(stream.field("leap").toScalar());
    keel::String scram;
    if (stream.hasField("scramble"))
        scram = stream.field("scramble").toString();
    bool useRR2 = (scram == "RR2" || scram == "rr2");
    const keel::Array<keel::i32> primes = first_primes(d);
    keel::Array<keel::Array<keel::i32>> perms;
    if (useRR2) {
        perms.reserve(d);
        for (keel::usize k = 0; k < d; ++k)
            perms.push_back(rr2_permutation(primes[k]));
    }
    const keel::i64 step = leap + 1;
    Value X = Value::matrix(keel::as<keel::usize>(n), d, ValueType::DOUBLE, mr);
    if (n == 0) return X;
    keel::f64 *xd = X.doubleDataMut();
    for (keel::i64 row = 0; row < n; ++row) {
        const keel::i64 idx = skip + row * step;
        for (keel::usize k = 0; k < d; ++k)
            xd[keel::as<keel::usize>(row) + k * keel::as<keel::usize>(n)]
                = radical_inverse(idx, primes[k], useRR2 ? &perms[k] : nullptr);
    }
    return X;
}

} // namespace numkit::stats
