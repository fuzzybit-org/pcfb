#pragma once

#include "pcfb/ir/circuit.h"

namespace pcfb::test {

/// Exhaustive exact P(query) for a core-1.1 circuit with at most 24 non-degenerate
/// SAMPLE events. Independent Bernoullis; constants (p ∈ {0,1} / always_true)
/// are substituted rather than enumerated.
double exact_probability(const ir::Circuit &circuit, std::uint32_t query_index = 0);

} // namespace pcfb::test
