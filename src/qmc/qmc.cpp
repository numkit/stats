// toolboxes/stats/src/qmc/qmc.cpp
// Thin Value wrapper over keel::stats::halton_net.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/stats.h>
#include <keel/string.h>
#include <numkit/stats/qmc/qmc.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::stats {

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
    const keel::i64 skip   = keel::as<keel::i64>(stream.field("skip").toScalar());
    const keel::i64 leap   = keel::as<keel::i64>(stream.field("leap").toScalar());
    keel::String scram;
    if (stream.hasField("scramble"))
        scram = stream.field("scramble").toString();
    const bool useRR2 = scram.equals_ignore_ascii_case("rr2");

    const keel::usize nu = keel::as<keel::usize>(n);
    Value X = Value::matrix(nu, d, ValueType::DOUBLE, mr);
    if (nu == 0) return X;
    (void)keel::stats::halton_net(nu, d, skip, leap, useRR2,
                                  keel::MutArrayRef<keel::f64>(X.doubleDataMut(), nu * d), mr);
    return X;
}

} // namespace numkit::stats
