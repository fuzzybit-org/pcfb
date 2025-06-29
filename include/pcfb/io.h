/**
 * @file io.h
 * @brief Public serialization and deserialization (SerDe) helpers for PCFB files.
 * 
 * This file provides the primary I/O interface for the PCFB library, enabling
 * efficient reading and writing of probabilistic circuit data to/from the
 * FlatBuffer-based .pcfb file format. The API is designed for both performance
 * and safety, using memory-mapped I/O for reading and explicit error handling.
 * 
 * Key features:
 * - Memory-mapped file reading for zero-copy access to large circuit files
 * - Automatic FlatBuffer verification for data integrity
 * - RAII-based resource management with automatic cleanup
 * - Support for both file-based and in-memory operations
 * - Comprehensive error handling with detailed error messages
 * 
 * @example
 * ```cpp
 * #include "pcfb/io.h"
 * using namespace pcfb;
 * 
 * // Read a circuit from disk
 * auto circuit_result = io::read("large_circuit.pcfb");
 * if (circuit_result) {
 *     auto circuit = *circuit_result;
 *     // Work with the circuit (zero-copy access)...
 *     
 *     // Write modified circuit back to disk
 *     auto write_result = io::write(circuit, "output.pcfb");
 *     if (!write_result) {
 *         std::cerr << "Write failed: " << write_result.error().message << std::endl;
 *     }
 * }
 * ```
 */
#pragma once

#include <filesystem>
#include <span>
#include <vector>

#include "pcfb/error.h"
#include "pcfb/ir/circuit.h"

/**
 * @brief Namespace containing all I/O operations for PCFB circuit serialization.
 * 
 * The pcfb::io namespace provides high-level functions for reading and writing
 * probabilistic circuits to/from the .pcfb file format. All functions use the
 * Result<T> pattern for explicit error handling and return detailed error
 * information when operations fail.
 * 
 * The I/O operations are designed to be:
 * - **Efficient**: Use memory mapping for zero-copy file access
 * - **Safe**: Comprehensive buffer verification and error checking
 * - **Convenient**: Simple function-based API with clear semantics
 * - **Robust**: Detailed error reporting for debugging and monitoring
 * 
 * @namespace pcfb::io
 */
namespace pcfb::io {

/**
 * @brief Reads a .pcfb file using memory mapping and returns an owning Circuit view.
 * 
 * This function efficiently reads a probabilistic circuit from a .pcfb file using
 * memory mapping (mmap) for zero-copy access. The returned Circuit object owns
 * the memory mapping and will automatically clean up resources when destroyed.
 * 
 * The function performs comprehensive validation:
 * - Verifies the file exists and is readable
 * - Checks that the file is not empty
 * - Validates the FlatBuffer structure for data integrity
 * - Ensures proper memory mapping setup
 * 
 * @param file_path Path to the .pcfb file to read. Can be relative or absolute.
 * 
 * @return Result<ir::Circuit> containing either:
 *         - Success: A Circuit object with owning reference to the mapped file data
 *         - Error: Detailed error information if the operation failed
 * 
 * @retval Error::Code::IoError File cannot be opened or read
 * @retval Error::Code::StatFailed Cannot retrieve file metadata
 * @retval Error::Code::MapFailed Memory mapping operation failed
 * @retval Error::Code::VerificationFailed FlatBuffer verification failed
 * 
 * @note The returned Circuit object keeps the file mapped in memory until it
 *       is destroyed, enabling efficient random access to circuit data.
 * 
 * @warning The file must remain unchanged while the Circuit object exists,
 *          as it directly references the mapped file contents.
 * 
 * @example
 * ```cpp
 * auto result = pcfb::io::read("/path/to/circuit.pcfb");
 * if (result) {
 *     auto circuit = *result;
 *     std::cout << "Circuit loaded, size: " << circuit.size() << " bytes" << std::endl;
 *     // Circuit automatically unmaps file when it goes out of scope
 * } else {
 *     std::cerr << "Failed to read circuit: " << result.error().message << std::endl;
 * }
 * ```
 */
Result<ir::Circuit> read(const std::filesystem::path &file_path);

/**
 * @brief Writes a Circuit to disk as a .pcfb file.
 * 
 * This function serializes a probabilistic circuit and writes it to the specified
 * file path in the .pcfb format. The function creates all necessary parent
 * directories if they don't exist and overwrites any existing file at the target path.
 * 
 * The write operation:
 * - Validates that the input circuit is valid (not null/empty)
 * - Serializes the circuit data into a byte vector
 * - Atomically writes the data to the target file
 * - Ensures data integrity through proper file handling
 * 
 * @param circuit The Circuit object to serialize and write. Must be valid.
 * @param file_path Target path for the output .pcfb file. Parent directories
 *                  will be created if they don't exist.
 * 
 * @return Result<void> indicating success or failure:
 *         - Success: void (empty result)
 *         - Error: Detailed error information describing the failure
 * 
 * @retval Error::Code::InvalidBuffer Input circuit is invalid or null
 * @retval Error::Code::IoError File cannot be created, opened, or written to
 * 
 * @note This function performs a complete copy of the circuit data, so it may
 *       use significant memory for large circuits during the write operation.
 * 
 * @example
 * ```cpp
 * // Assume we have a valid circuit
 * ir::Circuit circuit = ...;
 * 
 * auto result = pcfb::io::write(circuit, "output/saved_circuit.pcfb");
 * if (result) {
 *     std::cout << "Circuit successfully saved!" << std::endl;
 * } else {
 *     std::cerr << "Write failed: " << result.error().message << std::endl;
 * }
 * ```
 */
Result<void> write(const ir::Circuit &circuit, const std::filesystem::path &file_path);

/**
 * @brief Creates a non-owning Circuit view from a raw byte span.
 * 
 * This function creates a Circuit object that provides a view into an existing
 * byte buffer containing .pcfb data. The caller is responsible for keeping the
 * source buffer alive for the lifetime of the returned Circuit object.
 * 
 * This function is useful for:
 * - Processing circuit data received over a network
 * - Working with circuit data embedded in other data structures
 * - Creating temporary circuit views without file I/O
 * - Testing and validation scenarios
 * 
 * @param buffer A span of bytes containing valid .pcfb format data.
 *               The buffer must remain valid for the lifetime of the returned Circuit.
 * 
 * @return Result<ir::Circuit> containing either:
 *         - Success: A non-owning Circuit view into the provided buffer
 *         - Error: Detailed error information if the buffer is invalid
 * 
 * @retval Error::Code::InvalidBuffer Buffer is empty or null
 * @retval Error::Code::VerificationFailed Buffer contains invalid FlatBuffer data
 * 
 * @warning The caller must ensure the source buffer remains valid and unchanged
 *          for the entire lifetime of the returned Circuit object. The Circuit
 *          does not take ownership of the buffer data.
 * 
 * @example
 * ```cpp
 * std::vector<std::byte> circuit_data = load_from_network();
 * 
 * auto result = pcfb::io::from_bytes(circuit_data);
 * if (result) {
 *     auto circuit = *result;
 *     // Work with circuit view...
 *     // circuit_data must remain valid while circuit is in use
 * } else {
 *     std::cerr << "Invalid circuit data: " << result.error().message << std::endl;
 * }
 * ```
 */
Result<ir::Circuit> from_bytes(std::span<const std::byte> buffer);

/**
 * @brief Serializes a Circuit into a fresh byte vector.
 * 
 * This function creates a complete copy of a Circuit's data as a standalone
 * byte vector in the .pcfb format. The returned vector owns its data and can
 * be safely used independently of the source Circuit object.
 * 
 * This function is useful for:
 * - Preparing circuit data for network transmission
 * - Creating persistent copies of circuit data
 * - Serializing circuits for storage in databases or archives
 * - Creating independent copies for parallel processing
 * 
 * @param circuit The Circuit object to serialize. Must be valid and non-null.
 * 
 * @return Result<std::vector<std::byte>> containing either:
 *         - Success: A byte vector containing the complete serialized circuit
 *         - Error: Detailed error information if serialization failed
 * 
 * @retval Error::Code::InvalidBuffer Input circuit is invalid or null
 * 
 * @note The returned vector is completely independent of the source circuit
 *       and can be safely used after the source circuit is destroyed.
 * 
 * @note This function allocates memory for the entire circuit data, which
 *       may be significant for large circuits.
 * 
 * @example
 * ```cpp
 * ir::Circuit circuit = ...;  // Assume valid circuit
 * 
 * auto result = pcfb::io::to_bytes(circuit);
 * if (result) {
 *     auto serialized_data = *result;
 *     std::cout << "Serialized " << serialized_data.size() << " bytes" << std::endl;
 *     
 *     // Can now use serialized_data independently
 *     send_over_network(serialized_data);
 * } else {
 *     std::cerr << "Serialization failed: " << result.error().message << std::endl;
 * }
 * ```
 */
Result<std::vector<std::byte>> to_bytes(const ir::Circuit &circuit);

} // namespace pcfb::io