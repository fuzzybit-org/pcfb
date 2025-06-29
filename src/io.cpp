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

#include "pcfb_v1_generated.h"

namespace pcfb::io {
namespace detail {

// Simple RAII for a read-only mmap
struct MMap {
    void *addr {nullptr};
    size_t size {0};
    int    fd   {-1};

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

} // namespace detail

static inline bool verify_buffer(const uint8_t *data, size_t size) {
    flatbuffers::Verifier v(data, size);
    return pcfb::VerifyCircuitBuffer(v);
}

Result<ir::Circuit> from_bytes(std::span<const std::byte> buffer) {
    if (buffer.empty()) {
        return std::unexpected(Error{Error::Code::InvalidBuffer, "Buffer is empty"});
    }
    auto *u8 = reinterpret_cast<const uint8_t*>(buffer.data());
    if (!verify_buffer(u8, buffer.size())) {
        return std::unexpected(Error{Error::Code::VerificationFailed, "FlatBuffer verification failed"});
    }

    ir::Circuit circuit(flatbuffers::GetRoot<pcfb::Circuit>(u8), buffer.data(), buffer.size());
    return circuit; // non-owning view
}

Result<ir::Circuit> read(const std::filesystem::path &path) {
    const std::byte *base = nullptr;
    size_t           size = 0;
    auto owner_or_err = detail::MMap::map_file(path, &base, &size);
    if (!owner_or_err) {
        return std::unexpected(owner_or_err.error());
    }
    const uint8_t *u8 = reinterpret_cast<const uint8_t*>(base);
    if (!verify_buffer(u8, size)) {
        return std::unexpected(Error{Error::Code::VerificationFailed, "FlatBuffer verification failed for "+ path.string()});
    }
    ir::Circuit circuit(flatbuffers::GetRoot<pcfb::Circuit>(u8), base, size);
    // attach owner
    struct _circuit_owner_accessor { static void set(ir::Circuit &c, std::shared_ptr<const void> o){ c._set_owner(std::move(o)); } };
    _circuit_owner_accessor::set(circuit, *owner_or_err);
    return circuit;
}

Result<std::vector<std::byte>> to_bytes(const ir::Circuit &c) {
    if (!c.valid()) {
        return std::unexpected(Error{Error::Code::InvalidBuffer, "Circuit view is null"});
    }
    std::vector<std::byte> out(c.size());
    std::memcpy(out.data(), c.data(), c.size());
    return out;
}

Result<void> write(const ir::Circuit &c, const std::filesystem::path &path) {
    auto bytes_or_err = to_bytes(c);
    if (!bytes_or_err) return std::unexpected(bytes_or_err.error());

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected(Error{Error::Code::IoError, "Cannot open " + path.string()});
    }
    const auto &bytes = *bytes_or_err;
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        return std::unexpected(Error{Error::Code::IoError, "Failed to write " + path.string()});
    }
    return {};
}

} // namespace pcfb::io 