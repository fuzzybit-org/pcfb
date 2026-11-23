/**
 * @file literal.h
 * @brief Literal atoms of a hull query (v1.2 amendment to spec §5.1).
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pcfb_generated.h"

namespace pcfb {

/// Literal atoms of a hull query: the mode plus the tensor indices of the
/// `SAMPLE` outputs in the query's Δ (spec §5.1).
///
/// `samples` is sorted and duplicate-free. Complemented edges into a listed
/// sample, inside the query target's cone, are read according to `mode`;
/// complemented edges into unlisted samples are ordinary Boolean complements
/// in every mode. An empty list makes every mode the plain v1.2 hull, which is
/// also what a file written without these fields means.
struct query_literals {
    LiteralMode mode{LiteralMode::ATOM_SPLIT};
    std::vector<std::uint32_t> samples;

    [[nodiscard]] bool is_default() const noexcept {
        return mode == LiteralMode::ATOM_SPLIT && samples.empty();
    }
};

/// Reads the literal fields of `query`. A query written before the amendment
/// has neither field and yields the defaults.
/// @param query Query whose `literal_mode` / `literal_samples` are read.
[[nodiscard]] inline query_literals read_literals(const Query &query) {
    query_literals out;
    out.mode = query.literal_mode();
    if (const auto *vec = query.literal_samples()) {
        out.samples.assign(vec->begin(), vec->end());
    }
    return out;
}

} // namespace pcfb
