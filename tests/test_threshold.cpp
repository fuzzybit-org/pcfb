#include "pcfb/threshold.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

using pcfb::probability_to_threshold;

TEST_CASE("probability_to_threshold exact cases", "[threshold]") {
    SECTION("p = 0") {
        REQUIRE(probability_to_threshold(0.0) == 0);
        REQUIRE(probability_to_threshold(-0.0) == 0);
        REQUIRE(probability_to_threshold(-1.0) == 0);
    }

    SECTION("p = 1 saturates; always_true is the runtime path") {
        REQUIRE(probability_to_threshold(1.0) == pcfb::kThresholdAlwaysTrue);
        REQUIRE(probability_to_threshold(2.0) == pcfb::kThresholdAlwaysTrue);
    }

    SECTION("p = 2^-64") {
        REQUIRE(probability_to_threshold(std::ldexp(1.0, -64)) == 1);
    }

    SECTION("p = 0.5") {
        REQUIRE(probability_to_threshold(0.5) == (std::uint64_t{1} << 63));
    }

    SECTION("p = 1 - 2^-53 (largest binary64 below 1)") {
        const double p = std::nextafter(1.0, 0.0);
        REQUIRE(p == 1.0 - std::ldexp(1.0, -53));
        REQUIRE(probability_to_threshold(p) == (std::uint64_t{0} - (std::uint64_t{1} << 11)));
    }

    SECTION("subnormals map to 0") {
        REQUIRE(probability_to_threshold(std::numeric_limits<double>::denorm_min()) == 0);
        REQUIRE(probability_to_threshold(std::numeric_limits<double>::min() / 2.0) == 0);
    }

    SECTION("NaN maps to 0") {
        REQUIRE(probability_to_threshold(std::numeric_limits<double>::quiet_NaN()) == 0);
    }

    SECTION("powers of two") {
        REQUIRE(probability_to_threshold(std::ldexp(1.0, -1)) == (std::uint64_t{1} << 63));
        REQUIRE(probability_to_threshold(std::ldexp(1.0, -2)) == (std::uint64_t{1} << 62));
        REQUIRE(probability_to_threshold(std::ldexp(1.0, -63)) == 2);
        REQUIRE(probability_to_threshold(std::ldexp(1.0, -65)) == 0);
    }
}
