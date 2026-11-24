#pragma once

#include "pcfb/ir/circuit.h"
#include "pcfb/literal.h"

#include <cstdint>
#include <span>

namespace pcfb::test {

/// Exhaustive exact P(query) for a core-1.1 circuit with at most 24 non-degenerate
/// SAMPLE events. Independent Bernoullis; constants (p ∈ {0,1} / always_true)
/// are substituted rather than enumerated. Ignores `Query.kind`.
double exact_probability(const ir::Circuit &circuit, std::uint32_t query_index = 0);

/// Exhaustive exact monotone-hull probability P(↑g) of the query target g, where
/// ↑g(x) = 1 iff g(y) = 1 for some y ≤ x (spec v1.2 §5). SAMPLEs with p = 1 stay
/// variables (they may be dropped by y); SAMPLEs with p = 0 are substituted by 0,
/// which leaves P(↑g) unchanged. At most 24 such variables. Ignores `Query.kind`
/// and the query's literal fields.
double exact_hull_probability(const ir::Circuit &circuit, std::uint32_t query_index = 0);

/// Exhaustive exact value of a hull query whose `literal_mode` and
/// `literal_samples` are read from the query (spec §5.1). Per mode:
///
/// - `ATOM_SPLIT`: tabulate g̃ over the event atoms plus one complement atom per
///   listed sample, close the table under the subset-OR transform over every
///   atom, and weight the event assignment x by its mass at the coupled
///   valuation κ(x) (the complement atom of s is 1 − x_s, so an atom pair with
///   both atoms true has mass 0).
/// - `OMEGA_L`: tabulate g, close it under subset-OR along every event that is
///   not listed (a listed event is fixed by ω_L, so its axis is not closed),
///   and weight by the event mass.
/// - `NEGATION`: as `OMEGA_L` with the list ignored.
///
/// Events with p ∈ {0, 1} stay variables rather than being substituted: the
/// complement atom of a p = 0 event is 1 almost surely and droppable
/// (Theorem 10(g)). The work is 2^(|E| + |listed|) table entries, so the
/// evaluator requires |E| + |listed| ≤ 20 and throws std::runtime_error above it.
double exact_literal_hull_probability(const ir::Circuit &circuit, std::uint32_t query_index = 0);

/// As above, with the mode and the listed sample tensor indices given
/// explicitly (`literal_samples` need not be sorted).
double exact_literal_hull_probability(const ir::Circuit &circuit, std::uint32_t query_index, LiteralMode mode,
                                      std::span<const std::uint32_t> literal_samples);

} // namespace pcfb::test
