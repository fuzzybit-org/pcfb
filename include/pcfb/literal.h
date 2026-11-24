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

/// Whether answering `query` requires a runtime to implement its `literal_mode` (spec §8, runtimes).
///
/// False when `literal_samples` is absent or empty: such a query is the plain hull whatever its mode, so a runtime
/// that implements the plain hull MUST answer it and MUST NOT reject it for a mode it does not implement. That is
/// what lets a plain-hull runtime read pre-amendment hull files, and a query whose non-default mode was written with
/// no list. True otherwise: the mode changes the answer, so a runtime that does not implement it MUST reject the
/// query before sampling.
///
/// A runtime MAY accept a query whose listed samples all lie outside the target's cone, because every mode then
/// equals the plain hull (spec §5.2); deciding that needs the cone, so this helper does not look at it.
[[nodiscard]] inline bool literals_require_mode(const Query &query) {
    const auto *samples = query.literal_samples();
    return samples != nullptr && samples->size() != 0;
}

} // namespace pcfb
