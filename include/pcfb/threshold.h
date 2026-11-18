/**
 * @file threshold.h
 * @brief Exact conversion from a Bernoulli probability to a 64-bit sampling threshold.
 */
#pragma once

#include <bit>
#include <cstdint>
#include <limits>

namespace pcfb {

/// Sentinel written when `p >= 1`. Runtimes must honour `SampleArgs.always_true`
/// rather than comparing against this value (which is one ULP short of 2^64).
inline constexpr std::uint64_t kThresholdAlwaysTrue = std::numeric_limits<std::uint64_t>::max();

/**
 * Exact `min(floor(p · 2^64), 2^64 − 1)` for an IEEE-754 binary64 `p`.
 *
 * Method (integer arithmetic on the IEEE encoding; no floating multiply):
 *  - `p <= 0` or NaN → 0.
 *  - `p >= 1` → `2^64 − 1` (callers set `always_true` when `p == 1`).
 *  - Subnormals: `p · 2^64 < 2^{-958}` → 0.
 *  - Normals: `p = M · 2^{e-52}` with 53-bit significand `M` and unbiased
 *    exponent `e`, so `floor(p · 2^64) = floor(M · 2^{e+12})`. For `p < 1`
 *    this quantity is at most `2^64 − 2^{11}` and fits in `uint64_t`.
 *
 * This is exact for every finite binary64, including `p = 2^{-64}`, `p = 0.5`,
 * `p = 1 − 2^{-53}`, and every subnormal. A 32-bit threshold (`floor(p · 2^32)`)
 * is not conforming: it maps every `p < 2^{-32}` to 0.
 */
[[nodiscard]] constexpr std::uint64_t probability_to_threshold(double p) noexcept {
    if (!(p > 0.0)) {
        return 0;
    }
    if (p >= 1.0) {
        return kThresholdAlwaysTrue;
    }

    const auto bits = std::bit_cast<std::uint64_t>(p);
    const auto exp_bits = (bits >> 52U) & 0x7FFU;
    const auto frac = bits & 0xFFFFFFFFFFFFFULL;
    if (exp_bits == 0) {
        return 0; // subnormal
    }

    const int unbiased = static_cast<int>(exp_bits) - 1023;
    const auto significand = (std::uint64_t{1} << 52U) | frac;
    const int shift = unbiased + 12; // p * 2^64 = significand * 2^shift
    if (shift <= -64) {
        return 0;
    }
    if (shift < 0) {
        return significand >> static_cast<unsigned>(-shift);
    }
    if (shift >= 64) {
        return kThresholdAlwaysTrue;
    }
    // p < 1 ⇒ unbiased <= -1 ⇒ shift <= 11 ⇒ significand << shift < 2^64.
    return significand << static_cast<unsigned>(shift);
}

/// Inverse used only for explicit-threshold `add_sample`: `threshold / 2^64`,
/// saturating at 1. Not the inverse of `probability_to_threshold` (binary64
/// cannot represent every 64-bit dyadic).
[[nodiscard]] constexpr double threshold_to_probability(std::uint64_t threshold, bool always_true) noexcept {
    if (always_true) {
        return 1.0;
    }
    if (threshold == 0) {
        return 0.0;
    }
    // ldexp(threshold, -64) without <cmath>, exact for thresholds with ≤ 53
    // significant bits and correctly rounded otherwise.
    return std::bit_cast<double>((static_cast<std::uint64_t>(1023 - 64) << 52U)) * static_cast<double>(threshold);
}

} // namespace pcfb
