/**
 * @file builder.h
 * @brief In-memory construction of conforming PCFB v1.2 (`core-1.1` opset) files.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pcfb/error.h"
#include "pcfb/literal.h"
#include "pcfb_generated.h"

namespace pcfb {

/// One term of a query group: `weight` times the result of the query added `query`-th (0-based) by `add_query`.
struct query_term {
    std::uint32_t query{};
    double weight{1.0};
};

/// Incremental writer for a single core-1.1 subcircuit.
///
/// Tensor ids are assigned in SSA order: each `add_sample` / `add_op` appends
/// one tensor that is the unique output of that operator. Inputs must refer to
/// already-created tensors, which is exactly spec §3.2 (topological order +
/// single assignment). `finish()` writes required metadata, an empty
/// `buffers[0]`, and the lowest `schema_version` that covers the query kinds
/// used: `0x00010001` unless a v1.2 feature (a `MONOTONE_HULL_PROBABILITY`
/// query or a query group) is present, then `0x00010002`. Files without v1.2
/// features are therefore byte-identical to v1.1 builder output.
class builder {
public:
    using tensor_id = std::uint32_t;

    builder();
    builder(const builder &) = delete;
    builder &operator=(const builder &) = delete;
    builder(builder &&) noexcept;
    builder &operator=(builder &&) noexcept;
    ~builder();

    builder &name(std::string_view circuit_name);
    builder &description(std::string_view circuit_description);

    /// SAMPLE from a probability in `[0, 1]`. Threshold is
    /// `probability_to_threshold(p)`; `always_true` is set iff `p == 1`.
    Result<tensor_id> add_sample(double probability, std::string_view name, std::uint32_t event_id,
                                 SampleSource source = SampleSource::BASIC_EVENT);

    /// SAMPLE from an explicit 64-bit threshold. `probability` is stored as
    /// `threshold / 2^64` (or 1 if `always_true`). Prefer the probability
    /// overload: it is exact for every binary64.
    Result<tensor_id> add_sample(std::uint64_t threshold, bool always_true, std::string_view name,
                                 std::uint32_t event_id, SampleSource source = SampleSource::BASIC_EVENT);

    /// Gate / constant. `args` is `KofNArgs.k` for ATLEAST / ATMOST / EXACTLY
    /// and must be absent for every other core-1.1 opcode. SAMPLE is rejected
    /// (use `add_sample`).
    Result<tensor_id> add_op(OpCode opcode, std::span<const std::uint32_t> inputs,
                             std::span<const std::uint32_t> inverted_inputs = {},
                             std::optional<std::uint32_t> args = std::nullopt, std::string_view name = {});

    /// Rejects kinds this library does not define. Several queries may share a
    /// target (e.g. PROBABILITY and MONOTONE_HULL_PROBABILITY of one sequence).
    ///
    /// `literals` is written only when it differs from the defaults (mode
    /// `ATOM_SPLIT`, no samples), so files without literal atoms are
    /// byte-identical to v1.2 files produced before this field existed. A
    /// non-default mode or a non-empty list on a `PROBABILITY` query is
    /// rejected, as is an unsorted list or an index that is not a `SAMPLE`
    /// output.
    Result<void> add_query(std::uint32_t target, QueryKind kind = QueryKind::PROBABILITY, std::string_view name = {},
                           std::span<const double> confidence = {}, query_literals literals = {});

    /// A named weighted sum of query results (spec §5.4), such as an initiating-event frequency
    /// `sum_i f_i * P(target | initiator i)`. `terms` index the queries added so far, in `add_query` order. Rejected:
    /// no terms, an index that is not an added query, a query listed twice, queries of different kinds, and a weight
    /// that is negative or not finite. Written only when at least one group exists; a file with groups declares minor
    /// >= 2. `unit` labels the summed value (empty: dimensionless).
    Result<void> add_query_group(std::string_view name, std::string_view unit, std::span<const query_term> terms);

    Result<void> set(std::string_view key, std::string_view value);
    Result<void> set(std::string_view key, std::int64_t value);
    Result<void> set(std::string_view key, double value);
    Result<void> set(std::string_view key, std::span<const std::byte> value);

    /// Serialise a conforming buffer. Fails if there is no query, a
    /// duplicate `event_id`, or a previous `add_*` error left the builder dirty.
    Result<std::vector<std::byte>> finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pcfb
