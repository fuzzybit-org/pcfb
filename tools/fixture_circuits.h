#pragma once

#include "pcfb.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace pcfb::fixtures {

/// AND of p=0.5 and p=0.25. Exact query probability: 0.125.
inline void pin_fixture_metadata(builder &b) {
    // Pin generator identity so committed .pcfb bytes do not change with
    // `project(VERSION 0.0.${BUILD_NUMBER})`.
    (void)b.set("general.generator", "pcfb-fixtures");
    (void)b.set("general.generator_version", "1.1.0");
}

/// AND of p=0.5 and p=0.25. Exact query probability: 0.125.
inline Result<std::vector<std::byte>> and2() {
    builder b;
    b.name("and2");
    pin_fixture_metadata(b);
    auto a = b.add_sample(0.5, "A", 0);
    auto c = b.add_sample(0.25, "B", 1);
    if (!a || !c) {
        return std::unexpected(a ? c.error() : a.error());
    }
    const std::array<std::uint32_t, 2> ins{*a, *c};
    auto z = b.add_op(OpCode::BITWISE_AND, ins);
    if (!z) {
        return std::unexpected(z.error());
    }
    if (auto q = b.add_query(*z, QueryKind::PROBABILITY, "top"); !q) {
        return std::unexpected(q.error());
    }
    return b.finish();
}

/// ATLEAST(2 of 3) independent p=0.5 events. Exact query probability: 0.5.
inline Result<std::vector<std::byte>> kofn() {
    builder b;
    b.name("kofn");
    pin_fixture_metadata(b);
    auto x = b.add_sample(0.5, "X", 0);
    auto y = b.add_sample(0.5, "Y", 1);
    auto z = b.add_sample(0.5, "Z", 2);
    if (!x || !y || !z) {
        return std::unexpected((!x ? x : !y ? y : z).error());
    }
    const std::array<std::uint32_t, 3> ins{*x, *y, *z};
    auto out = b.add_op(OpCode::BITWISE_ATLEAST, ins, {}, 2u, "atleast2");
    if (!out) {
        return std::unexpected(out.error());
    }
    if (auto q = b.add_query(*out, QueryKind::PROBABILITY, "top"); !q) {
        return std::unexpected(q.error());
    }
    return b.finish();
}

/// NOR(A, ¬B) with A=0.5, B=0.25. Equivalent to ¬A ∧ B. Exact P = 0.125.
inline Result<std::vector<std::byte>> inverted() {
    builder b;
    b.name("inverted");
    pin_fixture_metadata(b);
    auto a = b.add_sample(0.5, "A", 0);
    auto c = b.add_sample(0.25, "B", 1);
    if (!a || !c) {
        return std::unexpected(a ? c.error() : a.error());
    }
    const std::array<std::uint32_t, 1> ins{*a};
    const std::array<std::uint32_t, 1> inv{*c};
    auto z = b.add_op(OpCode::BITWISE_NOR, ins, inv, std::nullopt, "nor_inv");
    if (!z) {
        return std::unexpected(z.error());
    }
    if (auto q = b.add_query(*z, QueryKind::PROBABILITY, "top"); !q) {
        return std::unexpected(q.error());
    }
    return b.finish();
}

/// Delete-term sequence SEQ = (A ∨ B) ∧ ¬(X ∧ Y) with A=0.25, B=0.5 and two
/// p = 1 flags X, Y. Query 0 (PROBABILITY) is exactly 0: X ∧ Y holds in every
/// sample. Query 1 (MONOTONE_HULL_PROBABILITY, v1.2) is P(A ∨ B) = 0.625: the
/// flags may be dropped, so the hull is the union of the cut sets of A ∨ B that
/// do not contain {X, Y}.
inline Result<std::vector<std::byte>> delete_term() {
    builder b;
    b.name("delete_term");
    pin_fixture_metadata(b);
    auto a = b.add_sample(0.25, "A", 0);
    auto c = b.add_sample(0.5, "B", 1);
    auto x = b.add_sample(1.0, "X", 2);
    auto y = b.add_sample(1.0, "Y", 3);
    if (!a || !c || !x || !y) {
        return std::unexpected((!a ? a : !c ? c : !x ? x : y).error());
    }
    auto f = b.add_op(OpCode::BITWISE_OR, std::array<std::uint32_t, 2>{*a, *c}, {}, std::nullopt, "F");
    auto del = b.add_op(OpCode::BITWISE_AND, std::array<std::uint32_t, 2>{*x, *y}, {}, std::nullopt, "DEL");
    if (!f || !del) {
        return std::unexpected((!f ? f : del).error());
    }
    auto seq = b.add_op(OpCode::BITWISE_AND, std::array<std::uint32_t, 1>{*f}, std::array<std::uint32_t, 1>{*del},
                        std::nullopt, "SEQ");
    if (!seq) {
        return std::unexpected(seq.error());
    }
    if (auto q = b.add_query(*seq, QueryKind::PROBABILITY, "seq_boolean"); !q) {
        return std::unexpected(q.error());
    }
    if (auto q = b.add_query(*seq, QueryKind::MONOTONE_HULL_PROBABILITY, "seq_hull"); !q) {
        return std::unexpected(q.error());
    }
    return b.finish();
}

inline constexpr double kAnd2Probability = 0.125;
inline constexpr double kKofnProbability = 0.5;
inline constexpr double kInvertedProbability = 0.125;
inline constexpr double kDeleteTermProbability = 0.0;
inline constexpr double kDeleteTermHullProbability = 0.625;

/// Literal-atom fixture (v1.2 amendment, spec §5.1): the circuit of RFC §7.3's
/// second vector, `g = c ∧ ¬(¬b)` with `¬b` a single-input gate over `b` and the
/// outer `¬` a gate complement, `b` the only literal sample.
///
/// `p_b = 0.25`, `p_c = 0.5`. Boolean: `g = b ∧ c`, so P = 0.125.
/// - `ATOM_SPLIT`: g̃ = c ∧ ¬b̄, whose hull drops b̄, so P = P(c) = 0.5.
/// - `OMEGA_L`: b is fixed and g is monotone in c, so ω_L g = g and P = P(b)P(c) = 0.125.
/// - `NEGATION`: the list is ignored, ↑g = b ∧ c, so P = P(b)P(c) = 0.125.
inline Result<std::vector<std::byte>> literal_atoms() {
    builder b;
    b.name("literal_atoms");
    pin_fixture_metadata(b);
    auto ev_b = b.add_sample(0.25, "B", 0);
    auto ev_c = b.add_sample(0.5, "C", 1);
    if (!ev_b || !ev_c) {
        return std::unexpected(ev_b ? ev_c.error() : ev_b.error());
    }
    // ¬b as a single-input gate (a gate complement, not a direct complement): still an inverted edge into B.
    auto not_b = b.add_op(OpCode::BITWISE_NO_OP, {}, std::array<std::uint32_t, 1>{*ev_b}, std::nullopt, "not_b");
    if (!not_b) {
        return std::unexpected(not_b.error());
    }
    // g = c ∧ ¬(¬b): the outer ¬ is an inverted input edge into the gate `not_b`.
    auto g = b.add_op(OpCode::BITWISE_AND, std::array<std::uint32_t, 1>{*ev_c}, std::array<std::uint32_t, 1>{*not_b},
                      std::nullopt, "top");
    if (!g) {
        return std::unexpected(g.error());
    }
    const std::vector<std::uint32_t> literal_b{*ev_b};
    if (auto q = b.add_query(*g, QueryKind::PROBABILITY, "top_boolean"); !q) {
        return std::unexpected(q.error());
    }
    if (auto q = b.add_query(*g, QueryKind::MONOTONE_HULL_PROBABILITY, "top_atom_split", {},
                             query_literals{LiteralMode::ATOM_SPLIT, literal_b});
        !q) {
        return std::unexpected(q.error());
    }
    if (auto q = b.add_query(*g, QueryKind::MONOTONE_HULL_PROBABILITY, "top_omega_l", {},
                             query_literals{LiteralMode::OMEGA_L, literal_b});
        !q) {
        return std::unexpected(q.error());
    }
    // NEGATION with an empty list: a non-default mode and no samples, so the file carries the mode alone.
    if (auto q = b.add_query(*g, QueryKind::MONOTONE_HULL_PROBABILITY, "top_negation", {},
                             query_literals{LiteralMode::NEGATION, {}});
        !q) {
        return std::unexpected(q.error());
    }
    return b.finish();
}

inline constexpr double kLiteralAtomsProbability = 0.125;
inline constexpr double kLiteralAtomsAtomSplit = 0.5;
inline constexpr double kLiteralAtomsOmegaL = 0.125;
inline constexpr double kLiteralAtomsNegation = 0.125;

} // namespace pcfb::fixtures
