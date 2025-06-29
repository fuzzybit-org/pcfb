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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <utility>

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
        StatFailed,       ///< fstat failed when mmapping a file
        MapFailed,        ///< mmap failed
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
    explicit Circuit(const pcfb::Circuit *root) noexcept : root_(root), base_(reinterpret_cast<const std::byte*>(root)) {}

    /// Constructor from base ptr, root ptr, size
    Circuit(const std::byte* base, const pcfb::Circuit *root, size_t sz) noexcept : root_(root), base_(base), size_(sz) {}

    /// Access underlying generated accessor – convenience.
    [[nodiscard]] const pcfb::Circuit *operator->() const noexcept { return root_; }
    [[nodiscard]] const pcfb::Circuit *get() const noexcept { return root_; }

    /// True iff the view points at a valid root.
    [[nodiscard]] bool valid() const noexcept { return root_ != nullptr; }

    /// Size in bytes of the underlying buffer (needed for serialization).
    [[nodiscard]] size_t size() const noexcept { return size_; }

    /// Raw pointer to underlying bytes (read-only).
    [[nodiscard]] const std::byte *data() const noexcept {
        return base_;
    }

    void _adopt_map(std::unique_ptr<pcfb::io::detail::MMap> mmap) noexcept {
        size_ = mmap ? mmap->size() : size_;
        base_ = mmap ? reinterpret_cast<const std::byte*>(mmap->data()) : base_;
        map_ = std::move(mmap);
    }

private:
    const pcfb::Circuit *root_{};
    const std::byte *base_{}; // pointer to beginning of buffer
    size_t size_{}; // total bytes in buffer
    // mmap ownership (null if view is non-owning)
    std::unique_ptr<pcfb::io::detail::MMap> map_{};

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

// RAII wrapper for POSIX mmap'ed read-only file region
class MMap {
public:
    static std::expected<std::unique_ptr<MMap>, Error> map_file(const std::filesystem::path &p) {
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd == -1) {
            return std::unexpected(Error{Error::Code::IoError, std::string("open failed: ")+ std::strerror(errno)});
        }

        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            return std::unexpected(Error{Error::Code::StatFailed, std::string("fstat failed: ")+ std::strerror(errno)});
        }

        size_t size = static_cast<size_t>(st.st_size);
        if (size == 0) {
            ::close(fd);
            return std::unexpected(Error{Error::Code::IoError, "file is empty"});
        }

        void *addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            ::close(fd);
            return std::unexpected(Error{Error::Code::MapFailed, std::string("mmap failed: ")+ std::strerror(errno)});
        }

        return std::unique_ptr<MMap>(new MMap(fd, addr, size));
    }

    ~MMap() {
        if (addr_) {
            ::munmap(addr_, size_);
        }
        if (fd_ != -1) {
            ::close(fd_);
        }
    }

    MMap(const MMap&) = delete;
    MMap &operator=(const MMap&) = delete;
    MMap(MMap&& other) noexcept : fd_(other.fd_), addr_(other.addr_), size_(other.size_) {
        other.fd_ = -1; other.addr_ = nullptr; other.size_ = 0;
    }
    MMap& operator=(MMap&& other) noexcept {
        if (this!=&other) {
            this->~MMap();
            fd_=other.fd_; addr_=other.addr_; size_=other.size_;
            other.fd_=-1; other.addr_=nullptr; other.size_=0;
        }
        return *this;
    }

    [[nodiscard]] const void *data() const noexcept { return addr_; }
    [[nodiscard]] size_t      size() const noexcept { return size_; }

private:
    MMap(int fd, void *addr, size_t s) : fd_(fd), addr_(addr), size_(s) {}
    int fd_{-1};
    void *addr_{};
    size_t size_{};
};

// Small helper: verifies that the supplied buffer contains a valid PCFB Circuit
// flatbuffer.
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

    return ir::Circuit(reinterpret_cast<const std::byte*>(data), flatbuffers::GetRoot<pcfb::Circuit>(data), size);
}

inline Result<ir::Circuit> read(const std::filesystem::path &file_path) {
    auto mmap_or_err = detail::MMap::map_file(file_path);
    if (!mmap_or_err) {
        return std::unexpected(mmap_or_err.error());
    }

    std::unique_ptr<detail::MMap> map = std::move(*mmap_or_err);

    const uint8_t *data = reinterpret_cast<const uint8_t *>(map->data());
    size_t size = map->size();
    if (!detail::verify_buffer(data, size)) {
        return std::unexpected(Error{Error::Code::VerificationFailed, "FlatBuffer verification failed for " + file_path.string()});
    }

    ir::Circuit circuit(reinterpret_cast<const std::byte*>(data), flatbuffers::GetRoot<pcfb::Circuit>(data), size);
    circuit._adopt_map(std::move(map));
    return circuit;
}

inline Result<std::vector<std::byte>> to_bytes(const ir::Circuit &circuit) {
    const std::byte *ptr = circuit.data();
    if (ptr == nullptr || circuit.size()==0) {
        return std::unexpected(Error{Error::Code::InvalidBuffer, "Circuit has no backing buffer"});
    }

    std::vector<std::byte> out(circuit.size());
    std::memcpy(out.data(), ptr, circuit.size());
    return out;
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