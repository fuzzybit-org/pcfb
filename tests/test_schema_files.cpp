#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

/// File contents after the first line (the header comment, which names the copy's role).
std::string body(const char *relative) {
    std::ifstream in(std::filesystem::path(PCFB_SCHEMA_DIR) / relative);
    REQUIRE(in);
    std::string first;
    std::getline(in, first);
    std::ostringstream rest;
    rest << in.rdbuf();
    return rest.str();
}

} // namespace

TEST_CASE("schema copies stay identical to their sources", "[schema]") {
    REQUIRE(body("drafts/pcfb_v1_2.fbs") == body("pcfb.fbs"));
    REQUIRE(body("drafts/pcfb_v1_1.fbs") == body("history/pcfb_v1_1.fbs"));
}
