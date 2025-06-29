/**
 * @file error.h
 * @brief Error handling and Result type definitions for the PCFB API.
 * 
 * This file provides comprehensive error handling utilities for the PCFB library,
 * including error codes, error messages, and a Result type that follows modern
 * C++ error handling patterns using std::expected-like semantics.
 * 
 * The error handling system is designed to:
 * - Provide clear, descriptive error messages
 * - Use type-safe error codes for programmatic handling
 * - Support functional-style error propagation with Result<T>
 * - Minimize exception usage in favor of explicit error handling
 * 
 * @example
 * ```cpp
 * #include "pcfb/error.h"
 * using namespace pcfb;
 * 
 * Result<int> divide(int a, int b) {
 *     if (b == 0) {
 *         return std::unexpected(Error{Error::Code::InvalidBuffer, "Division by zero"});
 *     }
 *     return a / b;
 * }
 * 
 * auto result = divide(10, 2);
 * if (result) {
 *     std::cout << "Result: " << *result << std::endl;
 * } else {
 *     std::cerr << "Error: " << result.error().message << std::endl;
 * }
 * ```
 */
#pragma once

#include <expected>
#include <string>

namespace pcfb {

/**
 * @brief A comprehensive error wrapper used by all public PCFB API functions.
 * 
 * The Error struct encapsulates both an error code (for programmatic handling)
 * and a human-readable error message (for logging and debugging). This design
 * allows for both type-safe error handling and informative error reporting.
 * 
 * @struct Error
 */
struct Error {
    /**
     * @brief Enumeration of all possible error codes in the PCFB library.
     * 
     * Each error code represents a specific category of error that can occur
     * during PCFB operations. The codes are designed to be stable across
     * library versions to ensure consistent error handling.
     * 
     * @enum Code
     */
    enum class Code {
        /**
         * @brief No error occurred - operation was successful.
         * 
         * This code should generally not appear in Error instances, as successful
         * operations return values directly rather than errors.
         */
        Ok = 0,
        
        /**
         * @brief General I/O operation failed.
         * 
         * This error occurs when file operations (read, write, open) fail due to
         * filesystem issues, permissions, or other I/O-related problems.
         * 
         * @example File not found, permission denied, disk full
         */
        IoError,
        
        /**
         * @brief FlatBuffer verification failed.
         * 
         * This error occurs when the FlatBuffer verification process detects
         * corrupted, invalid, or malformed buffer data. This is a critical
         * security and data integrity check.
         * 
         * @example Corrupted .pcfb file, truncated buffer, invalid schema
         */
        VerificationFailed,
        
        /**
         * @brief Buffer is invalid or in an unexpected state.
         * 
         * This error occurs when a buffer is null, empty when expected to contain data,
         * or otherwise in an invalid state for the requested operation.
         * 
         * @example Empty buffer passed to parser, null pointer dereference
         */
        InvalidBuffer,
        
        /**
         * @brief File stat operation failed.
         * 
         * This error occurs when the system cannot retrieve file metadata
         * (size, permissions, etc.) required for memory mapping operations.
         * 
         * @example File doesn't exist, insufficient permissions
         */
        StatFailed,
        
        /**
         * @brief Memory mapping operation failed.
         * 
         * This error occurs when mmap() fails to create a memory mapping of a file,
         * typically due to system resource constraints or file system limitations.
         * 
         * @example Out of virtual memory, file too large, unsupported file system
         */
        MapFailed,
        
        /**
         * @brief Unknown or unspecified error.
         * 
         * This is a catch-all error code for unexpected or unclassified errors.
         * It should be used sparingly and typically indicates a bug or unexpected
         * system state.
         */
        Unknown
    } code{Code::Unknown}; ///< The specific error code identifying the error type

    std::string message{}; ///< Human-readable error message providing additional context

    /**
     * @brief Checks if the error represents a successful operation.
     * 
     * This method provides a convenient way to check if an Error instance
     * represents success (Code::Ok) rather than an actual error condition.
     * 
     * @return true if the error code is Code::Ok, false otherwise
     * 
     * @note This method is constexpr and noexcept for optimal performance
     * 
     * @example
     * ```cpp
     * Error err{Error::Code::Ok, "Success"};
     * if (err.ok()) {
     *     std::cout << "Operation succeeded" << std::endl;
     * }
     * ```
     */
    [[nodiscard]] constexpr bool ok() const noexcept { return code == Code::Ok; }
};

/**
 * @brief Convenience alias for std::expected-like error handling.
 * 
 * Result<T> provides a modern, functional approach to error handling that
 * eliminates the need for exceptions in most cases. It can hold either a
 * successful value of type T or an Error describing what went wrong.
 * 
 * This follows the same pattern as the upcoming C++23 std::expected and
 * similar error handling mechanisms in other modern languages.
 * 
 * @tparam T The type of the successful result value
 * 
 * @example
 * ```cpp
 * Result<std::string> readFile(const std::string& path) {
 *     // ... implementation ...
 *     if (success) {
 *         return content;
 *     } else {
 *         return std::unexpected(Error{Error::Code::IoError, "Failed to read file"});
 *     }
 * }
 * 
 * auto result = readFile("config.txt");
 * if (result) {
 *     std::cout << "File content: " << *result << std::endl;
 * } else {
 *     std::cerr << "Error: " << result.error().message << std::endl;
 * }
 * ```
 */
template <typename T>
using Result = std::expected<T, Error>;

} // namespace pcfb 