/**
 * @file error.h
 * @brief Error and Result helpers for the PCFB API.
 */
#pragma once

#include <expected>
#include <string>

namespace pcfb {

/// A minimal error wrapper used by all public API helpers.
struct Error {
    enum class Code {
        Ok = 0,
        IoError,
        VerificationFailed,
        InvalidBuffer,
        StatFailed,
        MapFailed,
        Unknown
    } code{Code::Unknown};

    std::string message{};

    [[nodiscard]] constexpr bool ok() const noexcept { return code == Code::Ok; }
};

/// Convenience alias for the upcoming C++23 std::expected.
template <typename T>
using Result = std::expected<T, Error>;

} // namespace pcfb 