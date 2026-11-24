/**
 * @file circuit.h
 * @brief Lightweight immutable view wrapper around a PCFB Circuit FlatBuffer.
 * 
 * This file defines the Circuit class, which provides a high-level C++ interface
 * for working with probabilistic circuit data stored in FlatBuffer format. The
 * Circuit class is designed as an efficient, immutable view that can either
 * own its underlying data (via memory mapping) or provide a non-owning view
 * into existing buffer data.
 * 
 * Key design principles:
 * - **Zero-copy access**: Direct access to FlatBuffer data without deserialization
 * - **Memory safety**: RAII-based lifetime management with automatic cleanup
 * - **Immutable semantics**: Read-only access to circuit data
 * - **Flexible ownership**: Support for both owning and non-owning views
 * - **Type safety**: Strong typing with compile-time guarantees
 * 
 * The Circuit class serves as the primary interface for accessing probabilistic
 * circuit data in the PCFB library, providing a clean abstraction over the
 * underlying FlatBuffer representation.
 * 
 * @example
 * ```cpp
 * #include "pcfb/ir/circuit.h"
 * using namespace pcfb::ir;
 * 
 * // Create from file (owning view)
 * auto circuit_result = pcfb::io::read("circuit.pcfb");
 * if (circuit_result) {
 *     Circuit circuit = *circuit_result;
 *     
 *     if (circuit.valid()) {
 *         std::cout << "Circuit size: " << circuit.size() << " bytes" << std::endl;
 *         // Access FlatBuffer data directly
 *         const auto* fb_circuit = circuit.get();
 *         // Work with circuit data...
 *     }
 * }
 * ```
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "pcfb_generated.h"

#include "pcfb/error.h"
#include "pcfb/literal.h"

namespace pcfb { namespace io { struct _circuit_owner_accessor; } }

/**
 * @brief Namespace containing intermediate representation (IR) classes for PCFB.
 * 
 * The pcfb::ir namespace provides high-level wrapper classes around the raw
 * FlatBuffer data structures. These classes offer a more convenient and type-safe
 * interface for working with probabilistic circuit data while maintaining the
 * performance benefits of the underlying FlatBuffer representation.
 * 
 * @namespace pcfb::ir
 */
namespace pcfb::ir {

/**
 * @brief Lightweight immutable view wrapper around a PCFB Circuit FlatBuffer.
 * 
 * The Circuit class provides a high-level C++ interface for accessing probabilistic
 * circuit data stored in FlatBuffer format. It supports both owning and non-owning
 * semantics, making it suitable for various use cases from file I/O to in-memory
 * processing.
 * 
 * ## Key Features
 * 
 * - **Zero-copy access**: Direct pointer access to FlatBuffer data without copying
 * - **Automatic lifetime management**: RAII-based cleanup of resources
 * - **Flexible ownership**: Can own data (via memory mapping) or provide views
 * - **Type safety**: Compile-time guarantees about data validity
 * - **Immutable semantics**: Read-only access ensures data consistency
 * 
 * ## Usage Patterns
 * 
 * ### Owning View (from file)
 * ```cpp
 * auto result = pcfb::io::read("circuit.pcfb");
 * if (result) {
 *     Circuit circuit = *result;  // Owns the memory-mapped file
 *     // circuit keeps file mapped until destroyed
 * }
 * ```
 * 
 * ### Non-owning View (from buffer)
 * ```cpp
 * std::vector<std::byte> data = get_circuit_data();
 * auto result = pcfb::io::from_bytes(data);
 * if (result) {
 *     Circuit circuit = *result;  // Views into data vector
 *     // data must remain valid while circuit is used
 * }
 * ```
 * 
 * @class Circuit
 */
class Circuit {
public:
    /**
     * @brief Default constructor creating an invalid Circuit.
     * 
     * Creates a Circuit object in an invalid state. The resulting object
     * will return false from valid() and should not be used for data access
     * until properly initialized.
     * 
     * @note This constructor is primarily used for default initialization
     *       and container storage. Always check valid() before use.
     */
    Circuit() = default;

    /**
     * @brief Constructs a non-owning Circuit view from FlatBuffer data.
     * 
     * Creates a Circuit object that provides a view into existing FlatBuffer
     * data. The caller is responsible for ensuring the data remains valid
     * for the lifetime of this Circuit object.
     * 
     * @param root Pointer to the root Circuit FlatBuffer object. Must not be null.
     * @param base Pointer to the start of the buffer containing the FlatBuffer data.
     * @param size_bytes Size of the buffer in bytes.
     * 
     * @note This constructor is typically used internally by I/O functions.
     *       Use pcfb::io::from_bytes() for safe construction from user data.
     * 
     * @warning The provided pointers must remain valid for the lifetime of
     *          this Circuit object. This constructor does not take ownership.
     * 
     * @example
     * ```cpp
     * // Internal usage (typically via pcfb::io functions)
     * const auto* root = flatbuffers::GetRoot<pcfb::Circuit>(buffer);
     * Circuit circuit(root, buffer, buffer_size);
     * ```
     */
    Circuit(const pcfb::Circuit *root,
            const std::byte       *base,
            std::size_t            size_bytes) noexcept
        : root_(root), base_(base), size_(size_bytes) {}

    /**
     * @brief Provides pointer-like access to the underlying FlatBuffer Circuit.
     * 
     * This operator allows the Circuit object to be used with pointer-like
     * syntax for accessing methods and fields of the underlying FlatBuffer
     * Circuit object.
     * 
     * @return Const pointer to the FlatBuffer Circuit object, or nullptr if invalid.
     * 
     * @note Always check valid() before using this operator to ensure the
     *       returned pointer is not null.
     * 
     * @example
     * ```cpp
     * Circuit circuit = ...;
     * if (circuit.valid()) {
     *     // Use pointer-like syntax to access FlatBuffer methods
     *     auto node_count = circuit->nodes()->size();
     *     auto edges = circuit->edges();
     * }
     * ```
     */
    [[nodiscard]] const pcfb::Circuit *operator->() const noexcept { return root_; }
    
    /**
     * @brief Returns a pointer to the underlying FlatBuffer Circuit object.
     * 
     * Provides direct access to the FlatBuffer Circuit object for cases where
     * explicit pointer access is preferred over the -> operator.
     * 
     * @return Const pointer to the FlatBuffer Circuit object, or nullptr if invalid.
     * 
     * @note This method provides the same functionality as operator->() but
     *       with explicit method call syntax.
     * 
     * @example
     * ```cpp
     * Circuit circuit = ...;
     * const auto* fb_circuit = circuit.get();
     * if (fb_circuit != nullptr) {
     *     // Work with FlatBuffer object directly
     *     process_circuit_data(fb_circuit);
     * }
     * ```
     */
    [[nodiscard]] const pcfb::Circuit *get() const noexcept { return root_; }

    /**
     * @brief Checks if the Circuit object contains valid data.
     * 
     * Returns true if the Circuit object has been properly initialized with
     * valid FlatBuffer data and can be safely used for data access operations.
     * 
     * @return true if the Circuit contains valid data, false otherwise.
     * 
     * @note Always check this method before accessing circuit data to avoid
     *       null pointer dereferences.
     * 
     * @example
     * ```cpp
     * Circuit circuit = ...;
     * if (circuit.valid()) {
     *     // Safe to access circuit data
     *     auto size = circuit.size();
     *     const auto* data = circuit.get();
     * } else {
     *     std::cerr << "Invalid circuit data" << std::endl;
     * }
     * ```
     */
    [[nodiscard]] bool valid() const noexcept { return root_ != nullptr; }
    
    /**
     * @brief Returns the size of the underlying buffer in bytes.
     * 
     * This method returns the total size of the FlatBuffer data, including
     * all headers, metadata, and circuit data. This is useful for memory
     * usage tracking and buffer operations.
     * 
     * @return Size of the buffer in bytes, or 0 if the Circuit is invalid.
     * 
     * @note The returned size represents the entire FlatBuffer, not just the
     *       logical circuit data size.
     * 
     * @example
     * ```cpp
     * Circuit circuit = ...;
     * if (circuit.valid()) {
     *     std::cout << "Circuit buffer size: " << circuit.size() << " bytes" << std::endl;
     *     
     *     // Useful for memory tracking
     *     track_memory_usage(circuit.size());
     * }
     * ```
     */
    [[nodiscard]] size_t size() const noexcept { return size_; }
    
    /**
     * @brief Returns a pointer to the raw buffer data.
     * 
     * Provides direct access to the underlying byte buffer containing the
     * FlatBuffer data. This is primarily used for serialization operations
     * and low-level buffer manipulation.
     * 
     * @return Const pointer to the buffer data, or nullptr if invalid.
     * 
     * @warning This method provides raw access to the underlying buffer.
     *          Use with caution and prefer higher-level interfaces when possible.
     * 
     * @example
     * ```cpp
     * Circuit circuit = ...;
     * if (circuit.valid()) {
     *     const std::byte* raw_data = circuit.data();
     *     size_t data_size = circuit.size();
     *     
     *     // Example: Copy data for network transmission
     *     std::vector<std::byte> copy(raw_data, raw_data + data_size);
     * }
     * ```
     */
    [[nodiscard]] const std::byte *data() const noexcept { return base_; }

    /// v1.1 field (v1 typo `subciruits` kept the same slot). Null if invalid.
    [[nodiscard]] const flatbuffers::Vector<flatbuffers::Offset<pcfb::SubCircuit>> *subcircuits() const noexcept {
        return root_ ? root_->subcircuits() : nullptr;
    }

    [[nodiscard]] std::uint32_t schema_version() const noexcept { return root_ ? root_->schema_version() : 0; }

private:
    // Allow IO helpers to attach lifetime owners.
    friend struct pcfb::io::_circuit_owner_accessor;

    /**
     * @brief Internal method to attach an ownership object for lifetime management.
     * 
     * This method is used internally by the I/O subsystem to attach ownership
     * objects (such as memory mappings) to ensure proper resource cleanup.
     * 
     * @param owner Shared pointer to an object that should be kept alive for
     *              the lifetime of this Circuit.
     * 
     * @note This method is private and only accessible to friend classes.
     *       It is used internally for RAII-based resource management.
     */
    void _set_owner(std::shared_ptr<const void> owner) noexcept { owner_ = std::move(owner); }

    const pcfb::Circuit            *root_  {nullptr}; ///< Pointer to the FlatBuffer Circuit root
    const std::byte                *base_  {nullptr}; ///< Pointer to the start of buffer data
    std::size_t                     size_  {0};       ///< Size of the buffer in bytes
    std::shared_ptr<const void>     owner_ {};        ///< Ownership object for lifetime management
};

} // namespace pcfb::ir 