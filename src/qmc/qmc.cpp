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
keel::Array<int> first_primes(keel::usize d)
{
    keel::Array<int> ps;
    ps.reserve(d);
    int x = 2;
    while (ps.size() < d) {
        bool p = true;
        for (int q : ps) {
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
keel::Array<int> rr2_permutation(int p)
{
    int k = 1;
    while ((1 << k) < p) ++k;
    int m = (1 << k);
    keel::Array<int> perm;
    perm.reserve(p);
    for (int j = 0; j < m; ++j) {
        int rev = 0;
        for (int b = 0; b < k; ++b) {
            if ((j >> b) & 1) rev |= (1 << (k - 1 - b));
        }
        if (rev < p) perm.push_back(rev);
    }
    return perm;
}

double radical_inverse(long long i, int p, const keel::Array<int> *perm = nullptr)
{
    double r = 0.0;
    double f = 1.0 / keel::as<double>(p);
    while (i > 0) {
        int d = keel::as<int>(i % p);
        if (perm) d = (*perm)[keel::as<keel::usize>(d)];
        r += f * keel::as<double>(d);
        i /= p;
        f /= keel::as<double>(p);
    }
    return r;
}

} // anonymous

namespace {
void haltonset_argcheck(int d)
{
    if (d < 1)
        throw Error("haltonset: dim must be >= 1",
                    0, 0, "haltonset", "", "numkit:haltonset:dim");
}
} // namespace

Value haltonset(int d, long long skip, long long leap, keel::Allocator *mr)
{
    haltonset_argcheck(d);
    Value s = Value::structure(mr);
    s.field("kind") = Value::fromString("halton", mr);
    s.field("dim")  = Value::scalar(keel::as<double>(d), mr);
    s.field("skip") = Value::scalar(keel::as<double>(skip), mr);
    s.field("leap") = Value::scalar(keel::as<double>(leap), mr);
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

Value net(const Value &stream, long long n, keel::Allocator *mr)
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
    const long long skip = keel::as<long long>(stream.field("skip").toScalar());
    const long long leap = keel::as<long long>(stream.field("leap").toScalar());
    keel::String scram;
    if (stream.hasField("scramble"))
        scram = stream.field("scramble").toString();
    bool useRR2 = (scram == "RR2" || scram == "rr2");
    const keel::Array<int> primes = first_primes(d);
    keel::Array<keel::Array<int>> perms;
    if (useRR2) {
        perms.reserve(d);
        for (keel::usize k = 0; k < d; ++k)
            perms.push_back(rr2_permutation(primes[k]));
    }
    const long long step = leap + 1;
    Value X = Value::matrix(keel::as<keel::usize>(n), d, ValueType::DOUBLE, mr);
    if (n == 0) return X;
    double *xd = X.doubleDataMut();
    for (long long row = 0; row < n; ++row) {
        const long long idx = skip + row * step;
        for (keel::usize k = 0; k < d; ++k)
            xd[keel::as<keel::usize>(row) + k * keel::as<keel::usize>(n)]
                = radical_inverse(idx, primes[k], useRR2 ? &perms[k] : nullptr);
    }
    return X;
}

} // namespace numkit::stats
