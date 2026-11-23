/**
 * @file validate.h
 * @brief Normative PCFB v1.2 validation (spec §8).
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
        // §8.7 (v1.2); appended so earlier codes keep their values
        UnknownQueryKind,
        SchemaVersionTooOld,
        // §8.8 (v1.2 literal-atom amendment); appended so earlier codes keep their values
        UnknownLiteralMode,
        LiteralSampleNotSample,
        LiteralSamplesNotAscending,
        LiteralFieldsOnNonHullQuery,
    };

    Code code{Code::Ok};
    std::string message;
    std::string location;
};

/// Validates every numbered rule in spec §8 against an already-verified
/// (`pcfb::io::{read,from_bytes}`) circuit view. Returns an empty vector iff
/// the circuit is a conforming v1.2 file (`core-1.1` opset). A query kind newer
/// than this library is reported as `UnknownQueryKind`: a runtime cannot
/// answer it and must reject the file (spec §8).
[[nodiscard]] std::vector<Diagnostic> validate(const ir::Circuit &circuit);

} // namespace pcfb
