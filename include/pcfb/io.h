/**
 * @file io.h
 * @brief Public SerDe helpers for PCFB.
 */
#pragma once

#include <filesystem>
#include <span>
#include <vector>

#include "pcfb/error.h"
#include "pcfb/ir/circuit.h"

namespace pcfb::io {

// Read a .pcfb file via mmap and return an owning Circuit view.
Result<ir::Circuit> read(const std::filesystem::path &file_path);

// Write a Circuit back to disk (.pcfb).
Result<void> write(const ir::Circuit &circuit, const std::filesystem::path &file_path);

// Create a non-owning Circuit view from a raw byte span (caller keeps the data alive).
Result<ir::Circuit> from_bytes(std::span<const std::byte> buffer);

// Serialize a Circuit into a fresh byte vector.
Result<std::vector<std::byte>> to_bytes(const ir::Circuit &circuit);

} // namespace pcfb::io