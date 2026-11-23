#include "pcfb.h"
#include "exact_eval.h"
#include "fixture_circuits.h"
#include "portable_random.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using Catch::Matchers::WithinAbs;

/// A built vector: the serialised bytes plus the tensor index of the literal sample.
struct BuiltVector {
    std::vector<std::byte> bytes;
    std::uint32_t listed{};
};

/// Vector 1: g = a ∧ ¬b with the inverted edge directly into the SAMPLE b; p_a = 0.5, p_b = 0.25.
BuiltVector build_a_and_not_b() {
    pcfb::builder b;
    auto a = b.add_sample(0.5, "A", 0);
    auto ev_b = b.add_sample(0.25, "B", 1);
    REQUIRE(a);
    REQUIRE(ev_b);
    const std::array<std::uint32_t, 1> pos{*a};
    const std::array<std::uint32_t, 1> inv{*ev_b};
    auto g = b.add_op(pcfb::OpCode::BITWISE_AND, pos, inv);
    REQUIRE(g);
    REQUIRE(b.add_query(*g, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "hull"));
    auto bytes = b.finish();
    REQUIRE(bytes);
    return {*bytes, *ev_b};
}

/// Vector 2: g = c ∧ ¬(¬b). The inner ¬b is a single-input gate with an inverted input edge into the SAMPLE b;
/// the outer ¬ is an inverted edge into that gate, i.e. a gate complement that stays a Boolean negation.
/// p_b = 0.25, p_c = 0.5.
BuiltVector build_c_and_not_not_b() {
    pcfb::builder b;
    auto ev_b = b.add_sample(0.25, "B", 0);
    auto ev_c = b.add_sample(0.5, "C", 1);
    REQUIRE(ev_b);
    REQUIRE(ev_c);
    const std::array<std::uint32_t, 1> b_inv{*ev_b};
    auto not_b = b.add_op(pcfb::OpCode::BITWISE_NO_OP, {}, b_inv);
    REQUIRE(not_b);
    const std::array<std::uint32_t, 1> c_in{*ev_c};
    const std::array<std::uint32_t, 1> not_b_in{*not_b};
    auto g = b.add_op(pcfb::OpCode::BITWISE_AND, c_in, not_b_in);
    REQUIRE(g);
    REQUIRE(b.add_query(*g, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "hull"));
    auto bytes = b.finish();
    REQUIRE(bytes);
    return {*bytes, *ev_b};
}

double literal_hull(const pcfb::ir::Circuit &circuit, std::uint32_t query_index, pcfb::LiteralMode mode,
                    std::span<const std::uint32_t> listed) {
    return pcfb::test::exact_literal_hull_probability(circuit, query_index, mode, listed);
}

} // namespace

TEST_CASE("literal hull vector 1: g = a ∧ ¬b with b listed", "[exact_eval][literal]") {
    // g̃ = a ∧ b̄ is already monotone over the atoms {a, b, b̄}, so ↑g̃ = g̃ and h(x) = x_a ∧ (1 − x_b).
    const auto v = build_a_and_not_b();
    auto circuit = pcfb::io::from_bytes(v.bytes);
    REQUIRE(circuit);
    const std::array<std::uint32_t, 1> listed{v.listed};

    REQUIRE(pcfb::test::exact_probability(*circuit) == 0.375); // P(a)(1 − P(b))
    REQUIRE_THAT(literal_hull(*circuit, 0, pcfb::LiteralMode::ATOM_SPLIT, listed), WithinAbs(0.375, 1e-15));
    REQUIRE_THAT(literal_hull(*circuit, 0, pcfb::LiteralMode::OMEGA_L, listed), WithinAbs(0.375, 1e-15));
    REQUIRE_THAT(literal_hull(*circuit, 0, pcfb::LiteralMode::NEGATION, listed), WithinAbs(0.5, 1e-15));
    // ↑g = a: with b dropped the witness is a alone.
    REQUIRE(pcfb::test::exact_hull_probability(*circuit) == 0.5);
}

TEST_CASE("literal hull vector 2: g = c ∧ ¬(¬b) with b listed", "[exact_eval][literal]") {
    // g̃ = c ∧ ¬b̄ is not monotone over the atoms; ↑g̃ = c, because the witness sets c and drops b̄.
    const auto v = build_c_and_not_not_b();
    auto circuit = pcfb::io::from_bytes(v.bytes);
    REQUIRE(circuit);
    const std::array<std::uint32_t, 1> listed{v.listed};

    REQUIRE(pcfb::test::exact_probability(*circuit) == 0.125); // P(b)P(c)
    REQUIRE_THAT(literal_hull(*circuit, 0, pcfb::LiteralMode::ATOM_SPLIT, listed), WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(literal_hull(*circuit, 0, pcfb::LiteralMode::OMEGA_L, listed), WithinAbs(0.125, 1e-15));
    REQUIRE_THAT(literal_hull(*circuit, 0, pcfb::LiteralMode::NEGATION, listed), WithinAbs(0.125, 1e-15));
    // The plain hull is ↑(b ∧ c) = b ∧ c.
    REQUIRE(pcfb::test::exact_hull_probability(*circuit) == 0.125);
}

TEST_CASE("literal hull of the conformance fixture", "[exact_eval][literal][fixtures]") {
    auto bytes = pcfb::fixtures::literal_atoms();
    REQUIRE(bytes);
    auto circuit = pcfb::io::from_bytes(*bytes);
    REQUIRE(circuit);
    REQUIRE(pcfb::validate(*circuit).empty());
    const auto *queries = circuit->subcircuits()->Get(0)->queries();
    REQUIRE(queries->size() == 4);
    REQUIRE(queries->Get(0)->kind() == pcfb::QueryKind::PROBABILITY);
    for (flatbuffers::uoffset_t i = 1; i < 4; ++i) {
        REQUIRE(queries->Get(i)->kind() == pcfb::QueryKind::MONOTONE_HULL_PROBABILITY);
    }
    REQUIRE(queries->Get(1)->literal_mode() == pcfb::LiteralMode::ATOM_SPLIT);
    REQUIRE(queries->Get(2)->literal_mode() == pcfb::LiteralMode::OMEGA_L);
    REQUIRE(queries->Get(3)->literal_mode() == pcfb::LiteralMode::NEGATION);

    REQUIRE(pcfb::test::exact_probability(*circuit, 0) == pcfb::fixtures::kLiteralAtomsProbability);
    REQUIRE_THAT(pcfb::test::exact_literal_hull_probability(*circuit, 1),
                 WithinAbs(pcfb::fixtures::kLiteralAtomsAtomSplit, 1e-15));
    REQUIRE_THAT(pcfb::test::exact_literal_hull_probability(*circuit, 2),
                 WithinAbs(pcfb::fixtures::kLiteralAtomsOmegaL, 1e-15));
    REQUIRE_THAT(pcfb::test::exact_literal_hull_probability(*circuit, 3),
                 WithinAbs(pcfb::fixtures::kLiteralAtomsNegation, 1e-15));
}

namespace {

/// One random circuit over `n` SAMPLEs plus its target, for the cross-check below.
struct RandomCircuit {
    std::vector<std::byte> bytes;
    std::vector<std::uint32_t> samples;
    std::uint32_t target{};
};

/// Relative tolerance for the cross-check. Both sides sum at most 2^8 = 256 terms of magnitude at most 1, so the
/// accumulated rounding of one sum is below 256 · 2^-53 ≈ 2.8e-14; the two sides also add their terms in different
/// orders (the plain-hull evaluator substitutes p ∈ {0, 1} samples, the mode evaluators keep them as coordinates),
/// so 1e-12 covers the difference with three orders of magnitude to spare. The hand-computed vectors above use
/// exact binary values and are compared with ==.
constexpr double kCrossCheckTolerance = 1e-12;

/// Seeded random circuit: n ≤ 8 SAMPLEs (p ∈ {0, 0.5, 1} and random p), then a DAG of
/// AND / OR / XOR / NAND / NOR / XNOR / NOT / NO_OP / ATLEAST / ATMOST gates with random inverted inputs,
/// shared subtrees and an occasional constant.
RandomCircuit random_circuit(std::mt19937_64 &rng) {
    pcfb::builder b;
    const std::uint32_t n = static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, 2, 8));
    std::vector<std::uint32_t> samples;
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto pick = pcfb_test_random::uniform(rng, 0, 3);
        const double p = pick == 0 ? 0.0 : pick == 1 ? 1.0 : pick == 2 ? 0.5 : pcfb_test_random::unit(rng);
        auto t = b.add_sample(p, "e" + std::to_string(i), i);
        REQUIRE(t);
        samples.push_back(*t);
    }

    std::vector<std::uint32_t> nodes = samples;
    if (pcfb_test_random::chance(rng, 0.25)) {
        auto k = pcfb_test_random::chance(rng, 0.5) ? b.add_op(pcfb::OpCode::BITWISE_ZEROS, {})
                                                    : b.add_op(pcfb::OpCode::BITWISE_ONES, {});
        REQUIRE(k);
        nodes.push_back(*k);
    }
    auto pick_child = [&] {
        return nodes[static_cast<std::size_t>(pcfb_test_random::uniform(rng, 0, nodes.size() - 1))];
    };
    const std::uint32_t n_gates = static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, 1, 6));
    for (std::uint32_t gi = 0; gi < n_gates; ++gi) {
        const auto opcode = static_cast<pcfb::OpCode>(pcfb_test_random::uniform(rng, 0, 8));
        const bool unary = opcode == pcfb::OpCode::BITWISE_NO_OP || opcode == pcfb::OpCode::BITWISE_NOT;
        const std::uint32_t arity = unary ? 1u : static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, 1, 3));
        std::vector<std::uint32_t> pos;
        std::vector<std::uint32_t> inv;
        for (std::uint32_t a = 0; a < arity; ++a) {
            const bool inverted = pcfb_test_random::chance(rng, 0.4);
            (inverted ? inv : pos).push_back(pick_child());
        }
        // A unary op takes exactly one input, possibly inverted; the arity-1 draw above can produce either.
        if (unary && inv.size() > 1) {
            pos.push_back(inv.back());
            inv.pop_back();
        }
        const bool kofn = opcode == pcfb::OpCode::BITWISE_ATLEAST || opcode == pcfb::OpCode::BITWISE_ATMOST ||
                          opcode == pcfb::OpCode::BITWISE_EXACTLY;
        const std::uint32_t n_in = static_cast<std::uint32_t>(pos.size() + inv.size());
        const std::optional<std::uint32_t> arg =
            kofn ? std::optional<std::uint32_t>(static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, 0, n_in)))
                 : std::nullopt;
        auto t = b.add_op(opcode, pos, inv, arg);
        REQUIRE(t);
        nodes.push_back(*t);
    }
    const auto target = nodes.back();
    REQUIRE(b.add_query(target));
    auto bytes = b.finish();
    REQUIRE(bytes);
    return {*bytes, samples, target};
}

} // namespace

namespace {

/// RFC §8.11 `complement_under_negation`, target `P1`: `P1 + G1 G2`, `G1 * A -G`, `G2 * D -B`,
/// `G + B E`. `-B` is a direct complement (a marked literal of B); `-G` is a gate complement.
/// With p_A = 0.1, p_B = 0.2, p_D = 0.3, p_E = 0.4 the RFC's table gives
/// Boolean 0.2736, atom-split 0.316, ω_L 0.296, plain hull 0.37.
struct RfcP1 {
    std::vector<std::byte> bytes;
    std::uint32_t b{};
};

RfcP1 build_rfc_p1() {
    pcfb::builder b;
    auto a = b.add_sample(0.1, "A", 0);
    auto ev_b = b.add_sample(0.2, "B", 1);
    auto d = b.add_sample(0.3, "D", 2);
    auto e = b.add_sample(0.4, "E", 3);
    REQUIRE(a);
    REQUIRE(ev_b);
    REQUIRE(d);
    REQUIRE(e);
    auto g = b.add_op(pcfb::OpCode::BITWISE_OR, std::array<std::uint32_t, 2>{*ev_b, *e}, {}, std::nullopt, "G");
    REQUIRE(g);
    const std::array<std::uint32_t, 1> a_in{*a};
    const std::array<std::uint32_t, 1> g_inv{*g};
    auto g1 = b.add_op(pcfb::OpCode::BITWISE_AND, a_in, g_inv, std::nullopt, "G1");
    REQUIRE(g1);
    const std::array<std::uint32_t, 1> d_in{*d};
    const std::array<std::uint32_t, 1> b_inv{*ev_b};
    auto g2 = b.add_op(pcfb::OpCode::BITWISE_AND, d_in, b_inv, std::nullopt, "G2");
    REQUIRE(g2);
    const std::array<std::uint32_t, 2> top_in{*g1, *g2};
    auto top = b.add_op(pcfb::OpCode::BITWISE_OR, top_in, {}, std::nullopt, "P1");
    REQUIRE(top);
    REQUIRE(b.add_query(*top, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "P1"));
    auto bytes = b.finish();
    REQUIRE(bytes);
    return {*bytes, *ev_b};
}

/// RFC §8.11 target `P4`: `P4 * C -H`, `H * C -F`. `-F` is a direct complement of F inside the negated gate H;
/// `-H` is a gate complement. p_C = 0.5, p_F = 0.6. Boolean 0.3, atom-split 0.5, ω_L 0.3, plain hull 0.3.
struct RfcP4 {
    std::vector<std::byte> bytes;
    std::uint32_t f{};
};

RfcP4 build_rfc_p4() {
    pcfb::builder b;
    auto c = b.add_sample(0.5, "C", 0);
    auto f = b.add_sample(0.6, "F", 1);
    REQUIRE(c);
    REQUIRE(f);
    const std::array<std::uint32_t, 1> c_in{*c};
    const std::array<std::uint32_t, 1> f_inv{*f};
    auto h = b.add_op(pcfb::OpCode::BITWISE_AND, c_in, f_inv, std::nullopt, "H");
    REQUIRE(h);
    const std::array<std::uint32_t, 1> h_inv{*h};
    auto p4 = b.add_op(pcfb::OpCode::BITWISE_AND, c_in, h_inv, std::nullopt, "P4");
    REQUIRE(p4);
    REQUIRE(b.add_query(*p4, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, "P4"));
    auto bytes = b.finish();
    REQUIRE(bytes);
    return {*bytes, *f};
}

} // namespace

TEST_CASE("literal hull reproduces the RFC §8.11 worked examples", "[exact_eval][literal][rfc]") {
    // These four values per target are the RFC's own (hand-computed and cross-checked by exhaustive enumeration
    // on the reference tools), so they test this evaluator against an independent implementation.
    SECTION("P1: a marked event that also occurs under a negated gate separates atom_split from omega_l") {
        const auto v = build_rfc_p1();
        auto circuit = pcfb::io::from_bytes(v.bytes);
        REQUIRE(circuit);
        const std::array<std::uint32_t, 1> listed{v.b};
        REQUIRE_THAT(pcfb::test::exact_probability(*circuit), WithinAbs(0.2736, 1e-12));
        REQUIRE_THAT(pcfb::test::exact_literal_hull_probability(*circuit, 0, pcfb::LiteralMode::ATOM_SPLIT, listed),
                     WithinAbs(0.316, 1e-12));
        REQUIRE_THAT(pcfb::test::exact_literal_hull_probability(*circuit, 0, pcfb::LiteralMode::OMEGA_L, listed),
                     WithinAbs(0.296, 1e-12));
        REQUIRE_THAT(pcfb::test::exact_hull_probability(*circuit), WithinAbs(0.37, 1e-12));
    }
    SECTION("P4: the atom-split hull is above the plain hull (Theorem 10(d))") {
        const auto v = build_rfc_p4();
        auto circuit = pcfb::io::from_bytes(v.bytes);
        REQUIRE(circuit);
        const std::array<std::uint32_t, 1> listed{v.f};
        REQUIRE_THAT(pcfb::test::exact_probability(*circuit), WithinAbs(0.3, 1e-12));
        REQUIRE_THAT(pcfb::test::exact_literal_hull_probability(*circuit, 0, pcfb::LiteralMode::ATOM_SPLIT, listed),
                     WithinAbs(0.5, 1e-12));
        REQUIRE_THAT(pcfb::test::exact_literal_hull_probability(*circuit, 0, pcfb::LiteralMode::OMEGA_L, listed),
                     WithinAbs(0.3, 1e-12));
        REQUIRE_THAT(pcfb::test::exact_hull_probability(*circuit), WithinAbs(0.3, 1e-12));
    }
}

TEST_CASE("literal hull modes against the plain hull on 500 seeded circuits", "[exact_eval][literal]") {
    std::mt19937_64 rng(0x1f2e3d4c5b6a7980ULL);
    for (int round = 0; round < 500; ++round) {
        const auto rc = random_circuit(rng);
        auto circuit = pcfb::io::from_bytes(rc.bytes);
        REQUIRE(circuit);
        const double boolean = pcfb::test::exact_probability(*circuit);
        const double plain = pcfb::test::exact_hull_probability(*circuit);

        // Δ = ∅: every mode is the plain v1.2 hull.
        for (auto mode : {pcfb::LiteralMode::ATOM_SPLIT, pcfb::LiteralMode::OMEGA_L, pcfb::LiteralMode::NEGATION}) {
            REQUIRE_THAT(literal_hull(*circuit, 0, mode, {}), WithinAbs(plain, kCrossCheckTolerance));
        }

        // A random non-empty Δ ⊆ samples, sorted.
        std::vector<std::uint32_t> delta;
        for (auto t : rc.samples) {
            if (pcfb_test_random::chance(rng, 0.4)) {
                delta.push_back(t);
            }
        }
        const double split = literal_hull(*circuit, 0, pcfb::LiteralMode::ATOM_SPLIT, delta);
        const double omega = literal_hull(*circuit, 0, pcfb::LiteralMode::OMEGA_L, delta);
        const double negation = literal_hull(*circuit, 0, pcfb::LiteralMode::NEGATION, delta);

        // NEGATION ignores the list, and fixes a value for every mode.
        REQUIRE_THAT(negation, WithinAbs(plain, kCrossCheckTolerance));
        REQUIRE_THAT(literal_hull(*circuit, 0, pcfb::LiteralMode::NEGATION, rc.samples),
                     WithinAbs(plain, kCrossCheckTolerance));

        // OMEGA_L with Δ = every sample fixes every event, i.e. the Boolean probability.
        REQUIRE_THAT(literal_hull(*circuit, 0, pcfb::LiteralMode::OMEGA_L, rc.samples),
                     WithinAbs(boolean, kCrossCheckTolerance));

        // g ≤ ω_L g ≤ ↑g pointwise, hence on probabilities (Theorem 1(a, d); the atom list only adds fixed events).
        REQUIRE(boolean <= omega + kCrossCheckTolerance);
        REQUIRE(omega <= negation + kCrossCheckTolerance);
        // No order between atom_split and the plain hull (Theorem 10(d)); only g ≤ h holds (Theorem 10(b)).
        REQUIRE(boolean <= split + kCrossCheckTolerance);
    }
}
