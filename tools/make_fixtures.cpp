#include "fixture_circuits.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

namespace {

int write_file(const std::filesystem::path &path, const pcfb::Result<std::vector<std::byte>> &bytes) {
    if (!bytes) {
        std::cerr << path << ": " << bytes.error().message << '\n';
        return 1;
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "cannot open " << path << '\n';
        return 1;
    }
    out.write(reinterpret_cast<const char *>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
    if (!out) {
        std::cerr << "write failed: " << path << '\n';
        return 1;
    }
    std::cout << "wrote " << path << " (" << bytes->size() << " bytes)\n";
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    const std::filesystem::path dir = argc > 1 ? argv[1] : std::filesystem::path{"tests/fixtures"};
    int rc = 0;
    rc |= write_file(dir / "and2.pcfb", pcfb::fixtures::and2());
    rc |= write_file(dir / "kofn.pcfb", pcfb::fixtures::kofn());
    rc |= write_file(dir / "inverted.pcfb", pcfb::fixtures::inverted());
    rc |= write_file(dir / "delete_term.pcfb", pcfb::fixtures::delete_term());
    rc |= write_file(dir / "literal_atoms.pcfb", pcfb::fixtures::literal_atoms());
    return rc;
}
