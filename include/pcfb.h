/**
 * @file pcfb.h
 * @brief Main header for the PCFB (Probabilistic Circuit FlatBuffer) library.
 */
#pragma once

#include "pcfb_generated.h"
#include "pcfb/builder.h"
#include "pcfb/error.h"
#include "pcfb/io.h"
#include "pcfb/ir/circuit.h"
#include "pcfb/threshold.h"
#include "pcfb/validate.h"
#include "pcfb/version.h"

namespace pcfb {

/// Library version from `project(VERSION …)` (`configure_file` of `pcfb/version.h`).
[[nodiscard]] inline constexpr const char *version() noexcept { return PCFB_VERSION; }

inline constexpr std::uint32_t kSchemaVersionV11 = 0x00010001u;
inline constexpr const char *kOpsetCore11 = "core-1.1";

} // namespace pcfb
