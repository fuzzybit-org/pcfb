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

inline constexpr double kAnd2Probability = 0.125;
inline constexpr double kKofnProbability = 0.5;
inline constexpr double kInvertedProbability = 0.125;

} // namespace pcfb::fixtures
