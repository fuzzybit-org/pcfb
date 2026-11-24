#include "pcfb.h"
#include "fixture_circuits.h"

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<std::byte> pack(pcfb::CircuitT &model, const char *identifier = pcfb::CircuitIdentifier()) {
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(pcfb::Circuit::Pack(fbb, &model), identifier);
    const auto *p = fbb.GetBufferPointer();
    return {reinterpret_cast<const std::byte *>(p), reinterpret_cast<const std::byte *>(p) + fbb.GetSize()};
}

pcfb::CircuitT valid_skeleton() {
    pcfb::CircuitT m;
    m.schema_version = pcfb::kSchemaVersionV11;
    m.buffers.push_back(std::make_unique<pcfb::BufferT>());
    auto sc = std::make_unique<pcfb::SubCircuitT>();

    auto tensor = [](const char *name) {
        auto t = std::make_unique<pcfb::TensorT>();
        t->type = pcfb::TensorType::BIT;
        t->name = name;
        t->buffer_idx = 0;
        return t;
    };
    sc->tensors.push_back(tensor("A"));
    sc->tensors.push_back(tensor("top"));

    auto sample = std::make_unique<pcfb::OpT>();
    sample->opcode = pcfb::OpCode::SAMPLE;
    sample->output = 0;
    auto sa = std::make_unique<pcfb::SampleArgsT>();
    sa->probability = 0.5;
    sa->threshold = pcfb::probability_to_threshold(0.5);
    sa->event_id = 0;
    sa->source = pcfb::SampleSource::BASIC_EVENT;
    sa->name = "A";
    sample->args.Set(*sa);
    sc->operators.push_back(std::move(sample));

    auto ones = std::make_unique<pcfb::OpT>();
    ones->opcode = pcfb::OpCode::BITWISE_NO_OP;
    ones->inputs = {0};
    ones->output = 1;
    sc->operators.push_back(std::move(ones));

    auto q = std::make_unique<pcfb::QueryT>();
    q->target = 1;
    q->kind = pcfb::QueryKind::PROBABILITY;
    q->name = "top";
    sc->queries.push_back(std::move(q));
    sc->outputs = {1};

    m.subcircuits.push_back(std::move(sc));

    auto put = [&](const char *key, const char *value) {
        auto kv = std::make_unique<pcfb::KeyValueT>();
        kv->key = key;
        pcfb::StringValueT sv;
        sv.value = value;
        kv->value.Set(sv);
        m.metadata.push_back(std::move(kv));
    };
    put("general.endianness", "little");
    put("general.generator", "test");
    put("general.generator_version", "0");
    put("pcfb.opset", pcfb::kOpsetCore11);
    return m;
}

bool has_code(const std::vector<pcfb::Diagnostic> &diags, pcfb::Diagnostic::Code code) {
    for (const auto &d : diags) {
        if (d.code == code) {
            return true;
        }
    }
    return false;
}

std::vector<pcfb::Diagnostic> validate_bytes(const std::vector<std::byte> &bytes) {
    auto circuit = pcfb::io::from_bytes(bytes);
    REQUIRE(circuit);
    return pcfb::validate(*circuit);
}

} // namespace

TEST_CASE("validate accepts a positive core-1.1 circuit", "[validate]") {
    auto bytes = pcfb::fixtures::and2();
    REQUIRE(bytes);
    auto circuit = pcfb::io::from_bytes(*bytes);
    REQUIRE(circuit);
    const auto diags = pcfb::validate(*circuit);
    REQUIRE(diags.empty());
}

TEST_CASE("§8.1 identifier / major / opset", "[validate]") {
    SECTION("wrong identifier") {
        auto m = valid_skeleton();
        auto bytes = pack(m, "XXXX");
        // Verifier requires PCF1, so from_bytes should fail before validate.
        auto circuit = pcfb::io::from_bytes(bytes);
        REQUIRE_FALSE(circuit);
        REQUIRE(circuit.error().code == pcfb::Error::Code::VerificationFailed);
    }

    SECTION("major version 2") {
        auto m = valid_skeleton();
        m.schema_version = 0x00020000u;
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::BadMajorVersion));
    }

    SECTION("unknown opset") {
        auto m = valid_skeleton();
        for (auto &kv : m.metadata) {
            if (kv->key == "pcfb.opset") {
                pcfb::StringValueT sv;
                sv.value = "core-9.9";
                kv->value.Set(sv);
            }
        }
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::UnknownOpset));
    }
}

TEST_CASE("§8.2 buffers[0] empty and buffer_idx in range", "[validate]") {
    auto m = valid_skeleton();
    m.buffers[0]->data = {1, 2, 3};
    REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::MissingEmptyBuffer));
}

TEST_CASE("§8.3 tensor indices in range", "[validate]") {
    auto m = valid_skeleton();
    m.subcircuits[0]->queries[0]->target = 99;
    REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::TensorIndexOutOfRange));
}

TEST_CASE("§8.4 topological order and single assignment", "[validate]") {
    SECTION("use before def") {
        auto m = valid_skeleton();
        m.subcircuits[0]->operators[1]->inputs = {1}; // tensor 1 is this op's own output, not yet assigned
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::NotTopological));
    }
    SECTION("two writers") {
        auto m = valid_skeleton();
        m.subcircuits[0]->operators[1]->output = 0;
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::MultipleAssignment));
    }
}

TEST_CASE("§8.5 arity, args, SAMPLE, threshold", "[validate]") {
    SECTION("AND with zero inputs") {
        auto m = valid_skeleton();
        m.subcircuits[0]->operators[1]->opcode = pcfb::OpCode::BITWISE_AND;
        m.subcircuits[0]->operators[1]->inputs.clear();
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::BadArity));
    }
    SECTION("inconsistent threshold") {
        auto m = valid_skeleton();
        auto *sa = m.subcircuits[0]->operators[0]->args.AsSampleArgs();
        REQUIRE(sa != nullptr);
        sa->threshold = 1; // 0.5 should be 2^63
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::InconsistentThreshold));
    }
}

TEST_CASE("§8.6 unique event_id and at least one query", "[validate]") {
    SECTION("duplicate event_id") {
        auto m = valid_skeleton();
        auto t = std::make_unique<pcfb::TensorT>();
        t->type = pcfb::TensorType::BIT;
        m.subcircuits[0]->tensors.push_back(std::move(t));
        auto sample = std::make_unique<pcfb::OpT>();
        sample->opcode = pcfb::OpCode::SAMPLE;
        sample->output = 2;
        auto sa = std::make_unique<pcfb::SampleArgsT>();
        sa->probability = 0.25;
        sa->threshold = pcfb::probability_to_threshold(0.25);
        sa->event_id = 0; // duplicate
        sample->args.Set(*sa);
        m.subcircuits[0]->operators.push_back(std::move(sample));
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::DuplicateEventId));
    }
    SECTION("no queries") {
        auto m = valid_skeleton();
        m.subcircuits[0]->queries.clear();
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::MissingQuery));
    }
}

TEST_CASE("§8.8 literal-atom fields (v1.2 amendment)", "[validate][literal]") {
    // valid_skeleton() gives sample A (tensor 0) and NO_OP(A) (tensor 1). This adds sample B (tensor 2), the hull
    // target gate AND(A, ¬B) (tensor 3) and an out-of-cone sample C (tensor 4).
    auto hull_skeleton = [] {
        auto m = valid_skeleton();
        m.schema_version = pcfb::kSchemaVersionV12;
        auto add_sample = [&m](std::uint32_t output, std::uint32_t event_id, double p) {
            auto sa = std::make_unique<pcfb::SampleArgsT>();
            sa->probability = p;
            sa->threshold = pcfb::probability_to_threshold(p);
            sa->event_id = event_id;
            sa->source = pcfb::SampleSource::BASIC_EVENT;
            auto op = std::make_unique<pcfb::OpT>();
            op->opcode = pcfb::OpCode::SAMPLE;
            op->output = output;
            op->args.Set(*sa);
            m.subcircuits[0]->operators.push_back(std::move(op));
        };
        add_sample(2, 1, 0.25);
        auto gate = std::make_unique<pcfb::OpT>();
        gate->opcode = pcfb::OpCode::BITWISE_AND;
        gate->inputs = {0};
        gate->inverted_inputs = {2};
        gate->output = 3;
        m.subcircuits[0]->operators.push_back(std::move(gate));
        add_sample(4, 2, 0.75);
        for (int i = 0; i < 3; ++i) { // valid_skeleton() already declares tensors 0 and 1
            auto t = std::make_unique<pcfb::TensorT>();
            t->type = pcfb::TensorType::BIT;
            t->buffer_idx = 0;
            m.subcircuits[0]->tensors.push_back(std::move(t));
        }
        m.subcircuits[0]->queries[0]->target = 3;
        m.subcircuits[0]->queries[0]->kind = pcfb::QueryKind::MONOTONE_HULL_PROBABILITY;
        return m;
    };
    auto with_literals = [](pcfb::CircuitT &m, pcfb::LiteralMode mode, std::vector<std::uint32_t> samples) {
        auto *q = m.subcircuits[0]->queries[0].get();
        q->literal_mode = mode;
        q->literal_samples = std::move(samples);
    };

    SECTION("A hull query with sorted SAMPLE indices is valid") {
        auto m = hull_skeleton();
        with_literals(m, pcfb::LiteralMode::ATOM_SPLIT, {0, 2}); // samples A and B, both in the cone
        REQUIRE(validate_bytes(pack(m)).empty());
    }
    SECTION("Every mode on a hull query with an empty list is valid") {
        for (auto mode : {pcfb::LiteralMode::ATOM_SPLIT, pcfb::LiteralMode::OMEGA_L, pcfb::LiteralMode::NEGATION}) {
            auto m = hull_skeleton();
            with_literals(m, mode, {});
            REQUIRE(validate_bytes(pack(m)).empty());
        }
    }
    SECTION("Indices outside the target cone are valid and have no effect") {
        auto m = hull_skeleton();
        with_literals(m, pcfb::LiteralMode::ATOM_SPLIT, {2, 4}); // 4 = sample C, outside the cone of tensor 3
        REQUIRE(validate_bytes(pack(m)).empty());
    }
    SECTION("An unknown literal mode is rejected") {
        auto m = hull_skeleton();
        with_literals(m, static_cast<pcfb::LiteralMode>(9), {2});
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::UnknownLiteralMode));
    }
    SECTION("An unknown literal mode is rejected even without samples") {
        auto m = hull_skeleton();
        with_literals(m, static_cast<pcfb::LiteralMode>(9), {});
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::UnknownLiteralMode));
    }
    SECTION("An index that is not a SAMPLE output is rejected") {
        auto m = hull_skeleton();
        with_literals(m, pcfb::LiteralMode::ATOM_SPLIT, {3}); // the AND gate
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::LiteralSampleNotSample));
    }
    SECTION("An out-of-range index is rejected") {
        auto m = hull_skeleton();
        with_literals(m, pcfb::LiteralMode::ATOM_SPLIT, {77});
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::TensorIndexOutOfRange));
    }
    SECTION("Descending indices are rejected") {
        auto m = hull_skeleton();
        with_literals(m, pcfb::LiteralMode::ATOM_SPLIT, {4, 2});
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::LiteralSamplesNotAscending));
    }
    SECTION("Duplicate indices are rejected") {
        auto m = hull_skeleton();
        with_literals(m, pcfb::LiteralMode::OMEGA_L, {2, 2});
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::LiteralSamplesNotAscending));
    }
    SECTION("Literal samples on a PROBABILITY query are rejected") {
        auto m = hull_skeleton();
        m.subcircuits[0]->queries[0]->kind = pcfb::QueryKind::PROBABILITY;
        m.subcircuits[0]->queries[0]->literal_samples = {2};
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::LiteralFieldsOnNonHullQuery));
    }
    SECTION("A non-default literal mode on a PROBABILITY query is rejected") {
        auto m = hull_skeleton();
        m.subcircuits[0]->queries[0]->kind = pcfb::QueryKind::PROBABILITY;
        m.subcircuits[0]->queries[0]->literal_mode = pcfb::LiteralMode::OMEGA_L;
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::LiteralFieldsOnNonHullQuery));
    }
    SECTION("Literal fields need minor >= 2") {
        auto m = hull_skeleton();
        m.schema_version = pcfb::kSchemaVersionV11;
        with_literals(m, pcfb::LiteralMode::ATOM_SPLIT, {2});
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::SchemaVersionTooOld));
    }
}

TEST_CASE("§8.7 query kinds and schema minor", "[validate][hull]") {
    auto add_query = [](pcfb::CircuitT &m, std::uint32_t target, pcfb::QueryKind kind) {
        auto q = std::make_unique<pcfb::QueryT>();
        q->target = target;
        q->kind = kind;
        m.subcircuits[0]->queries.push_back(std::move(q));
    };

    SECTION("MONOTONE_HULL_PROBABILITY in a v1.2 file, on a target with an inverted input") {
        auto m = valid_skeleton();
        m.schema_version = pcfb::kSchemaVersionV12;
        m.subcircuits[0]->operators[1]->opcode = pcfb::OpCode::BITWISE_AND;
        m.subcircuits[0]->operators[1]->inputs.clear();
        m.subcircuits[0]->operators[1]->inverted_inputs = {0};
        add_query(m, 1, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY);
        REQUIRE(validate_bytes(pack(m)).empty());
    }
    SECTION("MONOTONE_HULL_PROBABILITY on a SAMPLE target") {
        auto m = valid_skeleton();
        m.schema_version = pcfb::kSchemaVersionV12;
        add_query(m, 0, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY);
        REQUIRE(validate_bytes(pack(m)).empty());
    }
    SECTION("MONOTONE_HULL_PROBABILITY with schema_version 1.1") {
        auto m = valid_skeleton();
        add_query(m, 1, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY);
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::SchemaVersionTooOld));
    }
    SECTION("MONOTONE_HULL_PROBABILITY without schema_version (v1.0)") {
        auto m = valid_skeleton();
        m.schema_version = 0;
        add_query(m, 1, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY);
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::SchemaVersionTooOld));
    }
    SECTION("a later minor is accepted") {
        auto m = valid_skeleton();
        m.schema_version = 0x00010009u;
        add_query(m, 1, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY);
        REQUIRE(validate_bytes(pack(m)).empty());
    }
    SECTION("unknown kind") {
        auto m = valid_skeleton();
        m.schema_version = 0x00010009u;
        add_query(m, 1, static_cast<pcfb::QueryKind>(200));
        REQUIRE(has_code(validate_bytes(pack(m)), pcfb::Diagnostic::Code::UnknownQueryKind));
    }
}
