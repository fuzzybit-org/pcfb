#include "pcfb.h"
#include "exact_eval.h"
#include "fixture_circuits.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

namespace {

std::vector<std::byte> read_all(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in);
    in.seekg(0, std::ios::end);
    const auto n = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    std::vector<std::byte> buf(n);
    in.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(n));
    REQUIRE(in);
    return buf;
}

void check_fixture(const char *name, const pcfb::Result<std::vector<std::byte>> &generated, double expected_p) {
    REQUIRE(generated);
    const auto committed = read_all(std::filesystem::path(PCFB_FIXTURES_DIR) / name);
    REQUIRE(committed == *generated);

    auto circuit = pcfb::io::from_bytes(*generated);
    REQUIRE(circuit);
    REQUIRE(pcfb::validate(*circuit).empty());
    REQUIRE(pcfb::test::exact_probability(*circuit) == expected_p);
}

} // namespace

TEST_CASE("conformance fixtures regenerate and match committed bytes", "[fixtures]") {
    check_fixture("and2.pcfb", pcfb::fixtures::and2(), pcfb::fixtures::kAnd2Probability);
    check_fixture("kofn.pcfb", pcfb::fixtures::kofn(), pcfb::fixtures::kKofnProbability);
    check_fixture("inverted.pcfb", pcfb::fixtures::inverted(), pcfb::fixtures::kInvertedProbability);
}

TEST_CASE("chinese.pcfb is deferred pending fuzzybits#43", "[fixtures][deferred]") {
    REQUIRE_FALSE(std::filesystem::exists(std::filesystem::path(PCFB_FIXTURES_DIR) / "chinese.pcfb"));
}
