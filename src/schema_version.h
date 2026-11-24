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

/// Minor version (of major 1) that introduced `mode`, or 0 if this library does not define it. Every mode is part of
/// the v1.2 amendment, so a file that uses any of them (and therefore sets `literal_samples` or a
/// non-default mode) must declare minor >= 2.
constexpr std::uint32_t literal_mode_minor(LiteralMode mode) noexcept {
    switch (mode) {
    case LiteralMode::ATOM_SPLIT:
    case LiteralMode::OMEGA_L:
    case LiteralMode::NEGATION:
        return 2;
    }
    return 0;
}

} // namespace pcfb::detail
