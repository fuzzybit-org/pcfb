/**
 * @file circuit.h
 * @brief Lightweight immutable view wrapper around a PCFB Circuit flatbuffer.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "pcfb_v1_generated.h"

#include "pcfb/error.h"

namespace pcfb { namespace io { struct _circuit_owner_accessor; } }

namespace pcfb::ir {

class Circuit {
public:
    Circuit() = default;

    // Non-owning view constructor.
    Circuit(const pcfb::Circuit *root,
            const std::byte       *base,
            std::size_t            size_bytes) noexcept
        : root_(root), base_(base), size_(size_bytes) {}

    [[nodiscard]] const pcfb::Circuit *operator->() const noexcept { return root_; }
    [[nodiscard]] const pcfb::Circuit *get()         const noexcept { return root_; }

    [[nodiscard]] bool   valid() const noexcept { return root_ != nullptr; }
    [[nodiscard]] size_t size()  const noexcept { return size_; }
    [[nodiscard]] const std::byte *data() const noexcept { return base_; }

private:
    // Allow IO helpers to attach lifetime owners.
    friend struct pcfb::io::_circuit_owner_accessor;

    void _set_owner(std::shared_ptr<const void> owner) noexcept { owner_ = std::move(owner); }

    const pcfb::Circuit            *root_  {nullptr};
    const std::byte                *base_  {nullptr};
    std::size_t                     size_  {0};
    std::shared_ptr<const void>     owner_ {}; // keeps the buffer alive if present
};

} // namespace pcfb::ir 