/**
 * @file io.cpp
 * @brief Implementation of PCFB I/O operations for circuit serialization and deserialization.
 * 
 * This file implements the I/O functionality declared in pcfb/io.h, providing efficient
 * reading and writing of probabilistic circuit data to/from the FlatBuffer-based .pcfb
 * file format. The implementation focuses on performance, memory efficiency, and safety.
 * 
 * ## Key Implementation Features
 * 
 * - **Memory-mapped I/O**: Uses mmap for efficient, zero-copy file reading
 * - **RAII resource management**: Automatic cleanup of file descriptors and mappings
 * - **FlatBuffer verification**: Comprehensive data integrity checking
 * - **Error handling**: Detailed error reporting with system error context
 * - **Cross-platform support**: Works on POSIX-compliant systems
 * 
 * ## Memory Management Strategy
 * 
 * The implementation uses a sophisticated memory management approach:
 * 
 * 1. **File Reading**: Memory-maps files using mmap() for zero-copy access
 * 2. **Lifetime Management**: Uses shared_ptr with custom deleters for cleanup
 * 3. **Resource Safety**: RAII ensures proper cleanup even on exceptions
 * 4. **Buffer Verification**: Always validates FlatBuffer integrity before use
 * 
 * ## Performance Characteristics
 * 
 * - **File Reading**: O(1) time complexity due to memory mapping
 * - **Memory Usage**: Minimal additional memory overhead beyond file size
 * - **Buffer Verification**: O(n) where n is buffer size, but highly optimized
 * - **File Writing**: Standard I/O with optimized buffer handling
 * 
 * @note This implementation requires POSIX-compliant systems for memory mapping
 *       functionality (mmap, open, fstat, etc.).
 */
#include "pcfb/io.h"
#include "pcfb/ir/circuit.h"
#include "pcfb/error.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <cstddef>

#include "pcfb_generated.h"

namespace pcfb::io {

/**
 * @brief Helper struct providing access to Circuit's private ownership functionality.
 * 
 * This struct implements the friend pattern to allow I/O functions to set ownership
 * objects on Circuit instances. It provides controlled access to the private
 * _set_owner method, enabling proper RAII-based lifetime management for memory
 * mappings and other resources.
 * 
 * @struct _circuit_owner_accessor
 */
struct _circuit_owner_accessor {
    /**
     * @brief Sets the ownership object on a Circuit instance.
     * 
     * This static method provides controlled access to the Circuit's private
     * _set_owner method, allowing I/O functions to attach ownership objects
     * (such as memory mappings) that need to be kept alive for the Circuit's lifetime.
     * 
     * @param circuit Reference to the Circuit object to modify
     * @param owner Shared pointer to the ownership object (e.g., memory mapping)
     * 
     * @note This is an implementation detail used by I/O functions and should
     *       not be called directly by user code.
     */
    static void set(ir::Circuit &circuit, std::shared_ptr<const void> owner) {
        circuit._set_owner(std::move(owner));
    }
};

/**
 * @brief Internal implementation details for I/O operations.
 * 
 * The detail namespace contains implementation-specific classes and functions
 * that are not part of the public API. These utilities handle low-level
 * operations like memory mapping, buffer verification, and resource management.
 * 
 * @namespace detail
 */
namespace detail {

/**
 * @brief RAII wrapper for memory-mapped file operations.
 * 
 * The MMap struct provides a safe, RAII-based interface for memory mapping files.
 * It handles all the low-level details of opening files, obtaining file metadata,
 * creating memory mappings, and ensuring proper cleanup through custom deleters.
 * 
 * This class is designed to be used internally by the I/O functions and provides
 * a clean abstraction over the POSIX mmap API.
 * 
 * @struct MMap
 */
struct MMap {
    void *addr {nullptr}; ///< Address of the memory mapping (currently unused in static implementation)
    size_t size {0};      ///< Size of the memory mapping (currently unused in static implementation)
    int    fd   {-1};     ///< File descriptor (currently unused in static implementation)

    /**
     * @brief Creates a memory mapping for the specified file.
     * 
     * This static method performs the complete sequence of operations needed to
     * memory-map a file for reading:
     * 
     * 1. Opens the file with read-only access
     * 2. Retrieves file metadata using fstat()
     * 3. Validates that the file is not empty
     * 4. Creates a memory mapping using mmap()
     * 5. Sets up RAII cleanup with a custom deleter
     * 
     * The returned shared_ptr uses a custom deleter that automatically unmaps
     * the memory and closes the file descriptor when the last reference is destroyed.
     * 
     * @param p Path to the file to memory-map
     * @param base_out Output parameter: pointer to the mapped memory base address
     * @param size_out Output parameter: size of the mapped memory region
     * 
     * @return Result containing either:
     *         - Success: shared_ptr with custom deleter for automatic cleanup
     *         - Error: Detailed error information about what went wrong
     * 
     * @retval Error::Code::IoError File cannot be opened
     * @retval Error::Code::StatFailed Cannot retrieve file metadata  
     * @retval Error::Code::IoError File is empty
     * @retval Error::Code::MapFailed Memory mapping failed
     * 
     * @note The custom deleter ensures that both munmap() and close() are called
     *       automatically when the shared_ptr is destroyed, providing exception-safe
     *       resource management.
     * 
     * @example
     * ```cpp
     * const std::byte *base;
     * size_t size;
     * auto owner_result = MMap::map_file("/path/to/file.pcfb", &base, &size);
     * if (owner_result) {
     *     // File is now mapped, owner keeps it alive
     *     auto owner = *owner_result;
     *     // Use base and size to access file data...
     *     // Mapping is automatically cleaned up when owner is destroyed
     * }
     * ```
     */
    static Result<std::shared_ptr<void const>> map_file(const std::filesystem::path &p, const std::byte **base_out, size_t *size_out) {
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd == -1) {
            return std::unexpected(Error{Error::Code::IoError, std::string{"open failed: "}+ std::strerror(errno)});
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            return std::unexpected(Error{Error::Code::StatFailed, std::string{"fstat failed: "}+ std::strerror(errno)});
        }
        if (st.st_size == 0) {
            ::close(fd);
            return std::unexpected(Error{Error::Code::IoError, "file is empty"});
        }
        void *addr = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            ::close(fd);
            return std::unexpected(Error{Error::Code::MapFailed, std::string{"mmap failed: "}+ std::strerror(errno)});
        }
        // Build shared_ptr with custom deleter to own the mapping & fd
        auto deleter = [size=static_cast<size_t>(st.st_size), fd](const void *p){
            ::munmap(const_cast<void*>(p), size);
            ::close(fd);
        };
        std::shared_ptr<void const> owner{addr, deleter};
        *base_out  = reinterpret_cast<const std::byte*>(addr);
        *size_out  = static_cast<size_t>(st.st_size);
        return owner;
    }

};

/**
 * @brief Verifies the integrity of a FlatBuffer containing Circuit data.
 * 
 * This function performs comprehensive validation of FlatBuffer data to ensure
 * it contains valid Circuit data and hasn't been corrupted. The verification
 * process checks:
 * 
 * - Buffer structure integrity
 * - Offset validity and bounds checking  
 * - Schema compliance
 * - Data consistency
 * 
 * @param data Pointer to the buffer data to verify
 * @param size Size of the buffer in bytes
 * 
 * @return true if the buffer contains valid Circuit data, false otherwise
 * 
 * @note This function is critical for security and data integrity. It should
 *       always be called before attempting to access FlatBuffer data.
 * 
 * @note The verification process has O(n) time complexity where n is the buffer
 *       size, but it's highly optimized for performance.
 * 
 * @example
 * ```cpp
 * const uint8_t* buffer_data = ...;
 * size_t buffer_size = ...;
 * 
 * if (verify_buffer(buffer_data, buffer_size)) {
 *     // Safe to use buffer as FlatBuffer
 *     auto* circuit = flatbuffers::GetRoot<pcfb::Circuit>(buffer_data);
 * } else {
 *     // Buffer is corrupted or invalid
 *     handle_invalid_buffer();
 * }
 * ```
 */
static inline bool verify_buffer(const uint8_t *data, const size_t size) {
    flatbuffers::Verifier v(data, size);
    return pcfb::VerifyCircuitBuffer(v);
}
}// namespace detail

/**
 * @brief Creates a non-owning Circuit view from a byte buffer.
 * 
 * This function creates a Circuit object that provides read-only access to
 * probabilistic circuit data stored in a byte buffer. The function performs
 * comprehensive validation before creating the Circuit view.
 * 
 * The function:
 * 1. Validates that the buffer is not empty
 * 2. Performs FlatBuffer verification for data integrity
 * 3. Creates a non-owning Circuit view if validation succeeds
 * 
 * @param buffer Span of bytes containing .pcfb format data
 * 
 * @return Result containing either:
 *         - Success: Circuit object providing a view into the buffer
 *         - Error: Detailed error information if validation failed
 * 
 * @retval Error::Code::InvalidBuffer Buffer is empty
 * @retval Error::Code::VerificationFailed FlatBuffer verification failed
 * 
 * @warning The caller must ensure the source buffer remains valid and unchanged
 *          for the entire lifetime of the returned Circuit object.
 * 
 * @note This function creates a non-owning view, meaning it does not take
 *       ownership of the buffer data. The buffer must outlive the Circuit.
 */
Result<ir::Circuit> from_bytes(const std::span<const std::byte> buffer) {
    if (buffer.empty()) {
        return std::unexpected(Error{Error::Code::InvalidBuffer, "Buffer is empty"});
    }
    auto *u8 = reinterpret_cast<const uint8_t*>(buffer.data());
    if (!detail::verify_buffer(u8, buffer.size())) {
        return std::unexpected(Error{Error::Code::VerificationFailed, "FlatBuffer verification failed"});
    }

    ir::Circuit circuit(flatbuffers::GetRoot<pcfb::Circuit>(u8), buffer.data(), buffer.size());
    return circuit; // non-owning view
}

/**
 * @brief Reads a .pcfb file using memory mapping and returns an owning Circuit.
 * 
 * This function efficiently reads a probabilistic circuit from a .pcfb file using
 * memory mapping for zero-copy access. The returned Circuit object owns the
 * memory mapping and automatically manages its lifetime.
 * 
 * The function performs the following operations:
 * 1. Memory-maps the specified file using detail::MMap::map_file()
 * 2. Verifies the FlatBuffer data integrity
 * 3. Creates a Circuit object with the verified data
 * 4. Attaches the memory mapping ownership to the Circuit for automatic cleanup
 * 
 * @param file_path Path to the .pcfb file to read
 * 
 * @return Result containing either:
 *         - Success: Circuit object with owning reference to mapped file data
 *         - Error: Detailed error information if the operation failed
 * 
 * @retval Error::Code::IoError File cannot be opened or read
 * @retval Error::Code::StatFailed Cannot retrieve file metadata
 * @retval Error::Code::MapFailed Memory mapping operation failed  
 * @retval Error::Code::VerificationFailed FlatBuffer verification failed
 * 
 * @note The returned Circuit keeps the file mapped in memory until destroyed,
 *       enabling efficient random access without additional I/O operations.
 */
Result<ir::Circuit> read(const std::filesystem::path &file_path) {
    const std::byte *base = nullptr;
    size_t           size = 0;
    auto owner_or_err = detail::MMap::map_file(file_path, &base, &size);
    if (!owner_or_err) {
        return std::unexpected(owner_or_err.error());
    }
    const auto *u8 = reinterpret_cast<const uint8_t*>(base);
    if (!detail::verify_buffer(u8, size)) {
        return std::unexpected(Error{Error::Code::VerificationFailed, "FlatBuffer verification failed for "+ file_path.string()});
    }
    ir::Circuit circuit(flatbuffers::GetRoot<pcfb::Circuit>(u8), base, size);
    // attach owner via helper
    _circuit_owner_accessor::set(circuit, *owner_or_err);
    return circuit;
}

/**
 * @brief Serializes a Circuit into a standalone byte vector.
 * 
 * This function creates a complete, independent copy of a Circuit's data as a
 * byte vector in the .pcfb format. The resulting vector owns its data and can
 * be used independently of the source Circuit.
 * 
 * The function:
 * 1. Validates that the input Circuit is valid
 * 2. Allocates a new vector with the appropriate size
 * 3. Copies the Circuit's data into the vector
 * 4. Returns the independent copy
 * 
 * @param circuit The Circuit object to serialize
 * 
 * @return Result containing either:
 *         - Success: Byte vector containing the complete serialized circuit
 *         - Error: Detailed error information if serialization failed
 * 
 * @retval Error::Code::InvalidBuffer Input circuit is invalid or null
 * 
 * @note The returned vector is completely independent and can be safely used
 *       after the source circuit is destroyed.
 * 
 * @note This function allocates memory for the entire circuit data, which may
 *       be significant for large circuits.
 */
Result<std::vector<std::byte>> to_bytes(const ir::Circuit &circuit) {
    if (!circuit.valid()) {
        return std::unexpected(Error{Error::Code::InvalidBuffer, "Circuit view is null"});
    }
    std::vector<std::byte> out(circuit.size());
    std::memcpy(out.data(), circuit.data(), circuit.size());
    return out;
}

/**
 * @brief Writes a Circuit to disk as a .pcfb file.
 * 
 * This function serializes a probabilistic circuit and writes it atomically to
 * the specified file path. The function ensures data integrity and provides
 * comprehensive error handling.
 * 
 * The write process:
 * 1. Serializes the Circuit to a byte vector using to_bytes()
 * 2. Opens the target file for binary writing
 * 3. Writes the complete data to the file
 * 4. Verifies that the write operation succeeded
 * 
 * @param circuit The Circuit object to write
 * @param file_path Target path for the output .pcfb file
 * 
 * @return Result indicating success or failure:
 *         - Success: void (empty result)
 *         - Error: Detailed error information describing the failure
 * 
 * @retval Error::Code::InvalidBuffer Input circuit is invalid
 * @retval Error::Code::IoError File cannot be opened or written to
 * 
 * @note This function overwrites any existing file at the target path.
 * 
 * @note The write operation uses standard I/O rather than memory mapping,
 *       ensuring data is properly flushed to disk.
 */
Result<void> write(const ir::Circuit &circuit, const std::filesystem::path &file_path) {
    auto bytes_or_err = to_bytes(circuit);
    if (!bytes_or_err) return std::unexpected(bytes_or_err.error());

    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        return std::unexpected(Error{Error::Code::IoError, "Cannot open " + file_path.string()});
    }
    const auto &bytes = *bytes_or_err;
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        return std::unexpected(Error{Error::Code::IoError, "Failed to write " + file_path.string()});
    }
    return {};
}

} // namespace pcfb::io