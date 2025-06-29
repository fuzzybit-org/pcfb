/** 
@file pcfb/io.h
@brief defines high-level SerDe public interface for the PCFB library.  
All entities live in the `::pcfb` namespace tree so that downstream users simply
need `#include <pcfb/io.h>` to access everything equired for loading/saving 
`.pcfb` files or in-memory buffers.
**/

#pragma once

#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <fstream>
#include <memory>

#include "pcfb_v1_generated.h" // FlatBuffer generated accessors.

namespace pcfb {

//------------------------------------------------------------------------------
// Error & Result helpers
//------------------------------------------------------------------------------

///\brief A minimal error wrapper used by all public IO helpers.
struct Error {
    enum class Code {
        Ok = 0,           ///< Success (should not be wrapped in std::unexpected).
        IoError,          ///< Could not read / write the requested file.
        VerificationFailed, ///< FlatBuffers verifier failed.
        InvalidBuffer,    ///< Buffer nullptr, size 0, or incorrect identifier.
        Unknown           ///< Catch-all.
    } code{Code::Unknown};

    std::string message{};

    [[nodiscard]] constexpr bool ok() const noexcept { return code == Code::Ok; }
};

/// Convenience alias for the upcoming C++23 std::expected.
template <typename T>
using Result = std::expected<T, Error>;

//------------------------------------------------------------------------------
// IR view types (thin wrappers over generated FlatBuffer accessors)
//------------------------------------------------------------------------------
namespace ir {

///\brief Immutable view of a Circuit FlatBuffer.
///
/// This wrapper keeps a pointer to the root as well as a std::shared_ptr owning
/// the underlying buffer when it was created via IO routines.  Users who build
/// circuits programmatically can construct the wrapper from an explicit root
/// pointer that they own.
class Circuit {
public:
    /// Construct an empty / null view.
    Circuit() = default;

    /// Construct from a raw root pointer (caller must ensure lifetime of the
    /// underlying memory).
    explicit Circuit(const pcfb::Circuit *root) noexcept : root_(root) {}

    /// Construct by taking shared ownership of a backing buffer.
    explicit Circuit(std::shared_ptr<const std::vector<std::byte>> buffer) noexcept
        : root_(buffer ? flatbuffers::GetRoot<pcfb::Circuit>(
                            reinterpret_cast<const uint8_t*>(buffer->data()))
                       : nullptr),
          owned_buffer_(std::move(buffer)) {}

    /// Access underlying generated accessor – convenience.
    [[nodiscard]] const pcfb::Circuit *operator->() const noexcept { return root_; }
    [[nodiscard]] const pcfb::Circuit *get() const noexcept { return root_; }

    /// True iff the view points at a valid root.
    [[nodiscard]] bool valid() const noexcept { return root_ != nullptr; }

    /// Return the shared backing buffer if the view owns one, else nullptr.
    [[nodiscard]] std::shared_ptr<const std::vector<std::byte>> buffer() const noexcept {
        return owned_buffer_;
    }

private:
    const pcfb::Circuit *root_{};
    // Owns the buffer for views created by IO helpers (optional, may be null).
    std::shared_ptr<const std::vector<std::byte>> owned_buffer_{};

    // Note: IO helpers live in a sibling namespace and manipulate the private
    // ownership fields via Circuit's public API or by declaring suitable
    // friendship at the helper-function level in a future implementation.
};

} // namespace ir

//------------------------------------------------------------------------------
// IO namespace – public SerDe helpers
//------------------------------------------------------------------------------
namespace io {

///\brief Read a `.pcfb` file from disk and return an immutable IR view.
Result<ir::Circuit> read(const std::filesystem::path &file_path);

///\brief Write a circuit back to disk as a `.pcfb` file.
Result<void> write(const ir::Circuit &circuit, const std::filesystem::path &file_path);

///\brief Create an IR view from an in-memory buffer (must outlive the view).
Result<ir::Circuit> from_bytes(std::span<const std::byte> buffer);

///\brief Serialize a circuit into a fresh binary buffer.
Result<std::vector<std::byte>> to_bytes(const ir::Circuit &circuit);

} // namespace io

} // namespace pcfb::io

//------------------------------------------------------------------------------
// Inline implementation details (header-only definitions)
//------------------------------------------------------------------------------

namespace pcfb::io {

namespace detail {

inline bool verify_buffer(const uint8_t *data, size_t size) {
    flatbuffers::Verifier verifier(data, size);
    return pcfb::VerifyCircuitBuffer(verifier);
}

} // namespace detail

inline Result<ir::Circuit> from_bytes(std::span<const std::byte> buffer) {
    if (buffer.empty()) {
        return std::unexpected(Error{Error::Code::InvalidBuffer, "Buffer is empty"});
    }

    const auto *data = reinterpret_cast<const uint8_t *>(buffer.data());
    const size_t size = buffer.size();

    if (!detail::verify_buffer(data, size)) {
        return std::unexpected(Error{Error::Code::VerificationFailed, "FlatBuffer verification failed"});
    }

    // Copy into an owning buffer so returned Circuit manages lifetime.
    auto owned_vec = std::make_shared<std::vector<std::byte>>(buffer.begin(), buffer.end());

    return ir::Circuit(std::const_pointer_cast<const std::vector<std::byte>>(owned_vec));
}

inline Result<ir::Circuit> read(const std::filesystem::path &file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::unexpected(Error{Error::Code::IoError, "Cannot open file: " + file_path.string()});
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return std::unexpected(Error{Error::Code::IoError, "File is empty: " + file_path.string()});
    }
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
        return std::unexpected(Error{Error::Code::IoError, "Failed to read file: " + file_path.string()});
    }

    auto owned_vec = std::make_shared<std::vector<std::byte>>(std::move(buffer));

    const uint8_t *data = reinterpret_cast<const uint8_t *>(owned_vec->data());
    if (!detail::verify_buffer(data, owned_vec->size())) {
        return std::unexpected(Error{Error::Code::VerificationFailed, "FlatBuffer verification failed for " + file_path.string()});
    }

    return ir::Circuit(owned_vec);
}

inline Result<std::vector<std::byte>> to_bytes(const ir::Circuit &circuit) {
    if (!circuit.valid()) {
        return std::unexpected(Error{Error::Code::InvalidBuffer, "Circuit view is null"});
    }

    auto buf_ptr = circuit.buffer();
    if (!buf_ptr) {
        return std::unexpected(Error{Error::Code::InvalidBuffer, "Circuit does not own a backing buffer; rebuilding not implemented"});
    }

    return *buf_ptr; // copy
}

inline Result<void> write(const ir::Circuit &circuit, const std::filesystem::path &file_path) {
    auto bytes_or_err = to_bytes(circuit);
    if (!bytes_or_err) {
        return std::unexpected(bytes_or_err.error());
    }

    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        return std::unexpected(Error{Error::Code::IoError, "Cannot open file for writing: " + file_path.string()});
    }

    const auto &bytes = *bytes_or_err;
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        return std::unexpected(Error{Error::Code::IoError, "Failed to write file: " + file_path.string()});
    }

    return {};
}

} // namespace pcfb::io