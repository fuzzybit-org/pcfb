#pragma once

#include "pcfb/ir/circuit.h"

namespace pcfb::test {

/// Exhaustive exact P(query) for a core-1.1 circuit with at most 24 non-degenerate
/// SAMPLE events. Independent Bernoullis; constants (p ∈ {0,1} / always_true)
/// are substituted rather than enumerated. Ignores `Query.kind`.
double exact_probability(const ir::Circuit &circuit, std::uint32_t query_index = 0);

/// Exhaustive exact monotone-hull probability P(↑g) of the query target g, where
/// ↑g(x) = 1 iff g(y) = 1 for some y ≤ x (spec v1.2 §5). SAMPLEs with p = 1 stay
/// variables (they may be dropped by y); SAMPLEs with p = 0 are substituted by 0,
/// which leaves P(↑g) unchanged. At most 24 such variables. Ignores `Query.kind`.
double exact_hull_probability(const ir::Circuit &circuit, std::uint32_t query_index = 0);

} // namespace pcfb::test
