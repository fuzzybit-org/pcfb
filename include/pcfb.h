/**
 * @file pcfb.h
 * @brief Main header file for the PCFB (Probabilistic Circuit FlatBuffer) library.
 * 
 * This is the primary include file for the PCFB library that provides a single source of truth
 * for representing probabilistic circuits as tensors using FlatBuffers serialization format.
 * 
 * The library provides:
 * - High-level C++ API for working with probabilistic circuits
 * - Serialization/deserialization to/from .pcfb files
 * - Intermediate representation (IR) for circuit manipulation
 * - Error handling utilities
 * 
 * @example
 * ```cpp
 * #include "pcfb.h"
 * using namespace pcfb;
 * 
 * // Read a circuit from file
 * auto result = io::read("circuit.pcfb");
 * if (result) {
 *     auto circuit = *result;
 *     std::cout << "PCFB version: " << pcfb::version() << std::endl;
 *     // Work with the circuit...
 * }
 * ```
 */
#pragma once

// The generated FlatBuffer schema header.
// This file is produced at configure/build time and will reside next to this
// header in the install tree.
#include "pcfb_v1_generated.h"
#include "pcfb/io.h"
#include "pcfb/error.h"
#include "pcfb/ir/circuit.h"

/**
 * @brief Root namespace for the PCFB (Probabilistic Circuit FlatBuffer) library.
 * 
 * The pcfb namespace contains all public APIs for working with probabilistic circuits
 * serialized using FlatBuffers. It provides high-level interfaces for:
 * - Circuit serialization and deserialization (pcfb::io)
 * - Intermediate representation for circuit manipulation (pcfb::ir)
 * - Error handling and result types (pcfb::Error, pcfb::Result)
 * 
 * @namespace pcfb
 */
namespace pcfb {

/**
 * @brief Retrieves the run-time version string of the PCFB library.
 * 
 * This function returns a compile-time constant string that matches the CMake
 * project version. The version string follows semantic versioning conventions.
 * 
 * @return A null-terminated string containing the library version.
 * @retval "dev" During development builds
 * @retval "x.y.z" For released versions (e.g., "1.0.0")
 * 
 * @note This function is constexpr and noexcept, making it suitable for use
 *       in constexpr contexts and guaranteed not to throw exceptions.
 * 
 * @example
 * ```cpp
 * std::cout << "Using PCFB version: " << pcfb::version() << std::endl;
 * ```
 */
inline constexpr const char *version() noexcept {
    return "dev";
}

} // namespace pcfb 