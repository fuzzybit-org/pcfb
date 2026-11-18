/**
 * @file validate.h
 * @brief Normative PCFB v1.1 validation (spec §8).
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pcfb/ir/circuit.h"

namespace pcfb {

/// One diagnostic produced by `pcfb::validate`. `code` is stable; `location` is
/// a dotted path into the file (`circuit.subcircuits[0].operators[3]`).
struct Diagnostic {
    enum class Code : std::uint16_t {
        Ok = 0,
        // §8.1
        BadIdentifier,
        BadMajorVersion,
        UnknownOpset,
        MissingRequiredMetadata,
        // §8.2
        MissingEmptyBuffer,
        BufferIndexOutOfRange,
        // §8.3
        TensorIndexOutOfRange,
        // §8.4
        NotTopological,
        MultipleAssignment,
        // §8.5
        BadArity,
        BadArgs,
        ReservedOpcode,
        BadSample,
        InconsistentThreshold,
        // §8.6
        DuplicateEventId,
        MissingQuery,
        // extras that still violate the core-1.1 profile
        ExtraSubcircuit,
        InvalidFile,
    };

    Code code{Code::Ok};
    std::string message;
    std::string location;
};

/// Validates every numbered rule in spec §8 against an already-verified
/// (`pcfb::io::{read,from_bytes}`) circuit view. Returns an empty vector iff
/// the circuit is a conforming core-1.1 file.
[[nodiscard]] std::vector<Diagnostic> validate(const ir::Circuit &circuit);

} // namespace pcfb
