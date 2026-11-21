#pragma once

#include "pcfb_generated.h"

#include <cstdint>

namespace pcfb::detail {

constexpr std::uint32_t schema_version(std::uint32_t minor) noexcept { return (1u << 16U) | minor; }

/// Minor version (of major 1) that introduced `kind`, or 0 if this library does not define it.
constexpr std::uint32_t query_kind_minor(QueryKind kind) noexcept {
    switch (kind) {
    case QueryKind::PROBABILITY:
        return 1;
    case QueryKind::MONOTONE_HULL_PROBABILITY:
        return 2;
    }
    return 0;
}

} // namespace pcfb::detail
