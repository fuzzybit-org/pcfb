#include "pcfb.h"
#include "fixture_circuits.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

TEST_CASE("builder computes thresholds and always_true", "[builder]") {
    pcfb::builder b;
    REQUIRE(b.add_sample(0.0, "zero", 0));
    REQUIRE(b.add_sample(1.0, "one", 1));
    REQUIRE(b.add_sample(std::ldexp(1.0, -64), "ulp", 2));
    auto z = b.add_op(pcfb::OpCode::BITWISE_ONES, {});
    REQUIRE(z);
    REQUIRE(b.add_query(*z));
    auto bytes = b.finish();
    REQUIRE(bytes);
    auto circuit = pcfb::io::from_bytes(*bytes);
    REQUIRE(circuit);
    const auto *ops = circuit->subcircuits()->Get(0)->operators();

    const auto *zero = ops->Get(0)->args_as_SampleArgs();
    REQUIRE(zero->probability() == 0.0);
    REQUIRE(zero->threshold() == 0);
    REQUIRE_FALSE(zero->always_true());

    const auto *one = ops->Get(1)->args_as_SampleArgs();
    REQUIRE(one->probability() == 1.0);
    REQUIRE(one->always_true());
    REQUIRE(one->threshold() == pcfb::kThresholdAlwaysTrue);

    const auto *ulp = ops->Get(2)->args_as_SampleArgs();
    REQUIRE(ulp->threshold() == 1);
    REQUIRE_FALSE(ulp->always_true());
}

TEST_CASE("builder rejects topological violations and reserved opcodes", "[builder]") {
    pcfb::builder b;
    const std::array<std::uint32_t, 1> future{0};
    auto bad = b.add_op(pcfb::OpCode::BITWISE_NOT, future);
    REQUIRE_FALSE(bad);
    REQUIRE(bad.error().code == pcfb::Error::Code::BuilderError);

    auto s = b.add_sample(0.5, "A", 0);
    REQUIRE(s);
    auto reserved = b.add_op(pcfb::OpCode::TALLY, std::array<std::uint32_t, 1>{*s});
    REQUIRE_FALSE(reserved);

    auto empty_and = b.add_op(pcfb::OpCode::BITWISE_AND, {});
    REQUIRE_FALSE(empty_and);

    auto dup = b.add_sample(0.1, "B", 0);
    REQUIRE_FALSE(dup);
}

TEST_CASE("builder requires a query", "[builder]") {
    pcfb::builder b;
    REQUIRE(b.add_sample(0.5, "A", 0));
    auto finished = b.finish();
    REQUIRE_FALSE(finished);
}

TEST_CASE("explicit threshold add_sample", "[builder]") {
    pcfb::builder b;
    auto t = b.add_sample(std::uint64_t{1} << 63, false, "half", 0);
    REQUIRE(t);
    REQUIRE(b.add_query(*t));
    auto bytes = b.finish();
    REQUIRE(bytes);
    auto circuit = pcfb::io::from_bytes(*bytes);
    const auto *sa = circuit->subcircuits()->Get(0)->operators()->Get(0)->args_as_SampleArgs();
    REQUIRE(sa->threshold() == (std::uint64_t{1} << 63));
    REQUIRE(sa->probability() == 0.5);
}

TEST_CASE("builder writes the lowest schema minor that covers the query kinds", "[builder][hull]") {
    auto build = [](bool with_hull) {
        pcfb::builder b;
        auto a = b.add_sample(0.5, "A", 0);
        REQUIRE(a);
        auto n = b.add_op(pcfb::OpCode::BITWISE_NOT, std::array<std::uint32_t, 1>{*a});
        REQUIRE(n);
        REQUIRE(b.add_query(*n, pcfb::QueryKind::PROBABILITY, "boolean"));
        if (with_hull) {
            REQUIRE(b.add_query(*n, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "hull"));
        }
        auto bytes = b.finish();
        REQUIRE(bytes);
        return *bytes;
    };

    const auto v11 = build(false);
    auto c11 = pcfb::io::from_bytes(v11);
    REQUIRE(c11);
    REQUIRE(c11->schema_version() == pcfb::kSchemaVersionV11);
    REQUIRE(pcfb::validate(*c11).empty());

    const auto v12 = build(true);
    auto c12 = pcfb::io::from_bytes(v12);
    REQUIRE(c12);
    REQUIRE(c12->schema_version() == pcfb::kSchemaVersionV12);
    REQUIRE(pcfb::validate(*c12).empty());
    const auto *q = c12->subcircuits()->Get(0)->queries();
    REQUIRE(q->size() == 2);
    REQUIRE(q->Get(1)->kind() == pcfb::QueryKind::MONOTONE_HULL_PROBABILITY);
}

TEST_CASE("builder rejects query kinds it does not define", "[builder][hull]") {
    pcfb::builder b;
    auto a = b.add_sample(0.5, "A", 0);
    REQUIRE(a);
    auto bad = b.add_query(*a, static_cast<pcfb::QueryKind>(7));
    REQUIRE_FALSE(bad);
    REQUIRE(bad.error().code == pcfb::Error::Code::BuilderError);
}

TEST_CASE("literal_atoms.pcfb: all three modes in one v1.2 file", "[fixtures][hull][literal]") {
    auto bytes = pcfb::fixtures::literal_atoms();
    REQUIRE(bytes);
    auto circuit = pcfb::io::from_bytes(*bytes);
    REQUIRE(circuit);
    REQUIRE(circuit->schema_version() == pcfb::kSchemaVersionV12);
    REQUIRE(pcfb::validate(*circuit).empty());

    const auto *queries = circuit->subcircuits()->Get(0)->queries();
    REQUIRE(queries->size() == 4);
    // The PROBABILITY query has neither field, so it reads back as the defaults.
    REQUIRE(queries->Get(0)->literal_mode() == pcfb::LiteralMode::ATOM_SPLIT);
    REQUIRE(queries->Get(0)->literal_samples() == nullptr);

    const auto atom_split = pcfb::read_literals(*queries->Get(1));
    REQUIRE(atom_split.mode == pcfb::LiteralMode::ATOM_SPLIT);
    REQUIRE(atom_split.samples.size() == 1);

    const auto omega_l = pcfb::read_literals(*queries->Get(2));
    REQUIRE(omega_l.mode == pcfb::LiteralMode::OMEGA_L);
    REQUIRE(omega_l.samples == atom_split.samples);

    const auto negation = pcfb::read_literals(*queries->Get(3));
    REQUIRE(negation.mode == pcfb::LiteralMode::NEGATION);
    REQUIRE(negation.samples.empty());
}

TEST_CASE("builder writes the literal fields only when non-default", "[builder][literal]") {
    // A hull query with the default mode and an empty list must serialise exactly as a hull query did before the
    // amendment: no literal_samples vector, and literal_mode left at its default (absent on the wire).
    auto build = [](bool with_default_literals) {
        pcfb::builder b;
        auto a = b.add_sample(0.5, "A", 0);
        REQUIRE(a);
        auto n = b.add_op(pcfb::OpCode::BITWISE_NOT, std::array<std::uint32_t, 1>{*a});
        REQUIRE(n);
        REQUIRE(b.add_query(*n, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "hull", {},
                            with_default_literals ? pcfb::query_literals{pcfb::LiteralMode::ATOM_SPLIT, {}}
                                                  : pcfb::query_literals{}));
        auto bytes = b.finish();
        REQUIRE(bytes);
        return *bytes;
    };
    REQUIRE(build(true) == build(false));

    // A non-default mode with no samples writes the mode alone.
    pcfb::builder b;
    auto a = b.add_sample(0.5, "A", 0);
    REQUIRE(a);
    auto n = b.add_op(pcfb::OpCode::BITWISE_NOT, std::array<std::uint32_t, 1>{*a});
    REQUIRE(n);
    REQUIRE(b.add_query(*n, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "hull", {},
                        pcfb::query_literals{pcfb::LiteralMode::NEGATION, {}}));
    auto bytes = b.finish();
    REQUIRE(bytes);
    auto circuit = pcfb::io::from_bytes(*bytes);
    REQUIRE(circuit);
    REQUIRE(pcfb::validate(*circuit).empty());
    const auto *q = circuit->subcircuits()->Get(0)->queries()->Get(0);
    REQUIRE(q->literal_mode() == pcfb::LiteralMode::NEGATION);
    REQUIRE(q->literal_samples() == nullptr);
}

TEST_CASE("builder rejects malformed literal fields", "[builder][literal]") {
    pcfb::builder b;
    auto a = b.add_sample(0.5, "A", 0);
    auto c = b.add_sample(0.25, "B", 1);
    REQUIRE(a);
    REQUIRE(c);
    const std::array<std::uint32_t, 2> ins{*a, *c};
    auto g = b.add_op(pcfb::OpCode::BITWISE_AND, ins);
    REQUIRE(g);

    SECTION("an unknown mode") {
        auto r = b.add_query(*g, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "q", {},
                             pcfb::query_literals{static_cast<pcfb::LiteralMode>(9), {}});
        REQUIRE_FALSE(r);
        REQUIRE(r.error().code == pcfb::Error::Code::BuilderError);
    }
    SECTION("an index that is not a SAMPLE") {
        auto r = b.add_query(*g, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "q", {},
                             pcfb::query_literals{pcfb::LiteralMode::ATOM_SPLIT, {*g}});
        REQUIRE_FALSE(r);
        REQUIRE(r.error().code == pcfb::Error::Code::BuilderError);
    }
    SECTION("an out-of-range index") {
        auto r = b.add_query(*g, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "q", {},
                             pcfb::query_literals{pcfb::LiteralMode::ATOM_SPLIT, {99}});
        REQUIRE_FALSE(r);
    }
    SECTION("a descending list") {
        auto r = b.add_query(*g, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "q", {},
                             pcfb::query_literals{pcfb::LiteralMode::OMEGA_L, {*c, *a}});
        REQUIRE_FALSE(r);
        REQUIRE(r.error().code == pcfb::Error::Code::BuilderError);
    }
    SECTION("a duplicate index") {
        auto r = b.add_query(*g, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "q", {},
                             pcfb::query_literals{pcfb::LiteralMode::OMEGA_L, {*a, *a}});
        REQUIRE_FALSE(r);
    }
    SECTION("literal samples on a PROBABILITY query") {
        auto r = b.add_query(*g, pcfb::QueryKind::PROBABILITY, "q", {},
                             pcfb::query_literals{pcfb::LiteralMode::ATOM_SPLIT, {*a}});
        REQUIRE_FALSE(r);
    }
    SECTION("a non-default mode on a PROBABILITY query") {
        auto r = b.add_query(*g, pcfb::QueryKind::PROBABILITY, "q", {},
                             pcfb::query_literals{pcfb::LiteralMode::NEGATION, {}});
        REQUIRE_FALSE(r);
    }
}

TEST_CASE("k-of-n k > n is rejected", "[builder]") {
    pcfb::builder b;
    auto a = b.add_sample(0.5, "A", 0);
    REQUIRE(a);
    auto op = b.add_op(pcfb::OpCode::BITWISE_ATLEAST, std::array<std::uint32_t, 1>{*a}, {}, 2u);
    REQUIRE_FALSE(op);
}

TEST_CASE("builder names tensors from their ops on large circuits", "[builder]") {
    constexpr std::uint32_t n = 200000;
    pcfb::builder b;
    REQUIRE(b.add_sample(0.5, "x", 1));
    std::uint32_t prev = 0;
    for (std::uint32_t i = 1; i < n; ++i) {
        const std::array<std::uint32_t, 1> in{prev};
        auto t = b.add_op(pcfb::OpCode::BITWISE_NOT, in, {}, std::nullopt, i % 2 == 0 ? "even" : "");
        REQUIRE(t);
        prev = *t;
    }
    REQUIRE(b.add_query(prev));
    auto bytes = b.finish();
    REQUIRE(bytes);
    auto circuit = pcfb::io::from_bytes(*bytes);
    REQUIRE(circuit);
    const auto *tensors = circuit->subcircuits()->Get(0)->tensors();
    REQUIRE(tensors->size() == n);
    REQUIRE(tensors->Get(0)->name()->str() == "x");
    REQUIRE(tensors->Get(2)->name()->str() == "even");
    REQUIRE(tensors->Get(3)->name() == nullptr);
}
