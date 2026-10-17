// toolboxes/stats/src/qmc/qmc.cpp

#include <numkit/stats/qmc/qmc.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <vector>

namespace numkit::stats {

namespace {

// First d primes via simple sieve / trial division.
std::vector<int> first_primes(size_t d)
{
    std::vector<int> ps;
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
std::vector<int> rr2_permutation(int p)
{
    int k = 1;
    while ((1 << k) < p) ++k;
    int m = (1 << k);
    std::vector<int> perm;
    perm.reserve(static_cast<size_t>(p));
    for (int j = 0; j < m; ++j) {
        int rev = 0;
        for (int b = 0; b < k; ++b) {
            if ((j >> b) & 1) rev |= (1 << (k - 1 - b));
        }
        if (rev < p) perm.push_back(rev);
    }
    return perm;
}

double radical_inverse(long long i, int p, const std::vector<int> *perm = nullptr)
{
    double r = 0.0;
    double f = 1.0 / static_cast<double>(p);
    while (i > 0) {
        int d = static_cast<int>(i % p);
        if (perm) d = (*perm)[static_cast<size_t>(d)];
        r += f * static_cast<double>(d);
        i /= p;
        f /= static_cast<double>(p);
    }
    return r;
}

} // anonymous

Value haltonset(int d, long long skip, long long leap, std::pmr::memory_resource *mr)
{
    if (d < 1)
        throw Error("haltonset: dim must be >= 1",
                    0, 0, "haltonset", "", "numkit:haltonset:dim");
    Value s = Value::structure(mr);
    s.field("kind") = Value::fromString("halton", mr);
    s.field("dim")  = Value::scalar(static_cast<double>(d), mr);
    s.field("skip") = Value::scalar(static_cast<double>(skip), mr);
    s.field("leap") = Value::scalar(static_cast<double>(leap), mr);
    s.field("scramble") = Value::fromString("", mr);
    return s;
}

Value scramble(const Value &stream, const std::string &type, std::pmr::memory_resource *mr)
{
    if (!stream.isStruct() || !stream.hasField("kind"))
        throw Error("scramble: input must be a quasi-random stream struct",
                    0, 0, "scramble", "", "numkit:scramble:struct");
    Value out = stream;
    out.field("scramble") = Value::fromString(type, mr);
    return out;
}

Value net(const Value &stream, long long n, std::pmr::memory_resource *mr)
{
    if (!stream.isStruct() || !stream.hasField("kind"))
        throw Error("net: input must be a quasi-random stream struct",
                    0, 0, "net", "", "numkit:net:struct");
    const std::string kind = stream.field("kind").toString();
    if (kind != "halton")
        throw Error("net: only halton streams supported in this release",
                    0, 0, "net", "", "numkit:net:kind");
    if (n < 0)
        throw Error("net: n must be >= 0",
                    0, 0, "net", "", "numkit:net:n");
    const size_t d    = static_cast<size_t>(stream.field("dim").toScalar());
    const long long skip = static_cast<long long>(stream.field("skip").toScalar());
    const long long leap = static_cast<long long>(stream.field("leap").toScalar());
    std::string scram;
    if (stream.hasField("scramble"))
        scram = stream.field("scramble").toString();
    bool useRR2 = (scram == "RR2" || scram == "rr2");
    const std::vector<int> primes = first_primes(d);
    std::vector<std::vector<int>> perms;
    if (useRR2) {
        perms.resize(d);
        for (size_t k = 0; k < d; ++k)
            perms[k] = rr2_permutation(primes[k]);
    }
    const long long step = leap + 1;
    Value X = Value::matrix(static_cast<size_t>(n), d, ValueType::DOUBLE, mr);
    if (n == 0) return X;
    double *xd = X.doubleDataMut();
    for (long long row = 0; row < n; ++row) {
        const long long idx = skip + row * step;
        for (size_t k = 0; k < d; ++k)
            xd[static_cast<size_t>(row) + k * static_cast<size_t>(n)]
                = radical_inverse(idx, primes[k], useRR2 ? &perms[k] : nullptr);
    }
    return X;
}

} // namespace numkit::stats
