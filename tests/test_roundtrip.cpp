#include "pcfb.h"
#include "fixture_circuits.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string_view>

TEST_CASE("builder → reader round trip", "[roundtrip]") {
    pcfb::builder b;
    b.name("rt").description("round-trip");
    REQUIRE(b.set("general.model_name", "rt"));
    REQUIRE(b.set("compile.collapsed_modules", std::int64_t{0}));
    REQUIRE(b.set("compile.exact_root", 0.125));
    const std::array<std::byte, 4> hash{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    REQUIRE(b.set("general.source_sha256", hash));

    auto a = b.add_sample(0.5, "A", 7, pcfb::SampleSource::BASIC_EVENT);
    auto c = b.add_sample(0.25, "B", 8, pcfb::SampleSource::COLLAPSED_MODULE);
    REQUIRE(a);
    REQUIRE(c);
    const std::array<std::uint32_t, 2> ins{*a, *c};
    auto z = b.add_op(pcfb::OpCode::BITWISE_AND, ins, {}, std::nullopt, "and");
    REQUIRE(z);
    const double conf[] = {0.95, 0.99};
    REQUIRE(b.add_query(*z, pcfb::QueryKind::PROBABILITY, "top", conf));

    auto bytes = b.finish();
    REQUIRE(bytes);

    auto circuit = pcfb::io::from_bytes(*bytes);
    REQUIRE(circuit);
    REQUIRE(circuit->valid());
    REQUIRE(circuit->subcircuits() != nullptr);
    REQUIRE(circuit->subcircuits()->size() == 1);
    REQUIRE(circuit->schema_version() == pcfb::kSchemaVersionV11);

    const auto *root = circuit->get();
    REQUIRE(flatbuffers::BufferHasIdentifier(circuit->data(), pcfb::CircuitIdentifier()));
    REQUIRE(root->name() != nullptr);
    REQUIRE(std::string_view(root->name()->c_str()) == "rt");

    const auto *sc = circuit->subcircuits()->Get(0);
    REQUIRE(sc->operators()->size() == 3);
    REQUIRE(sc->queries()->size() == 1);
    REQUIRE(sc->queries()->Get(0)->target() == *z);
    REQUIRE(sc->tensors()->Get(0)->type() == pcfb::TensorType::BIT);

    const auto *sample0 = sc->operators()->Get(0)->args_as_SampleArgs();
    REQUIRE(sample0 != nullptr);
    REQUIRE(sample0->probability() == 0.5);
    REQUIRE(sample0->threshold() == pcfb::probability_to_threshold(0.5));
    REQUIRE_FALSE(sample0->always_true());
    REQUIRE(sample0->event_id() == 7);
    REQUIRE(sample0->source() == pcfb::SampleSource::BASIC_EVENT);

    const auto *md = root->metadata();
    REQUIRE(md->LookupByKey("pcfb.opset") != nullptr);
    REQUIRE(std::string_view(md->LookupByKey("pcfb.opset")->value_as_StringValue()->value()->c_str()) ==
            pcfb::kOpsetCore11);
    REQUIRE(std::string_view(md->LookupByKey("general.endianness")->value_as_StringValue()->value()->c_str()) ==
            "little");
    REQUIRE(md->LookupByKey("compile.collapsed_modules")->value_as_IntValue()->value() == 0);
    REQUIRE(md->LookupByKey("compile.exact_root")->value_as_FloatValue()->value() == 0.125);

    auto diags = pcfb::validate(*circuit);
    REQUIRE(diags.empty());

    auto copy = pcfb::io::to_bytes(*circuit);
    REQUIRE(copy);
    REQUIRE(*copy == *bytes);
}

TEST_CASE("pcfb::version matches CMake project version", "[version]") {
    REQUIRE(std::string_view(pcfb::version()) == PCFB_VERSION);
    REQUIRE(std::string_view(pcfb::version()) != "dev");
    REQUIRE_FALSE(std::string_view(pcfb::version()).empty());
}
