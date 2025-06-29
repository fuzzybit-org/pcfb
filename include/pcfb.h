#pragma once

// The generated FlatBuffer schema header.
// This file is produced at configure/build time and will reside next to this
// header in the install tree.
#include "pcfb_v1_generated.h"
#include "pcfb/io.h"

namespace pcfb {

// Simple helper to retrieve a run-time version string that matches the CMake
// project version.
inline constexpr const char *version() noexcept {
    return "dev";
}

} // namespace pcfb 