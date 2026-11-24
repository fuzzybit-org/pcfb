#include "pcfb/builder.h"

#include "pcfb/threshold.h"
#include "pcfb/version.h"
#include "schema_version.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>

namespace pcfb {
namespace {

constexpr const char *kOpset = "core-1.1";

Error builder_error(std::string message) {
    return Error{Error::Code::BuilderError, std::move(message)};
}

bool is_core_opcode(OpCode op) {
    switch (op) {
    case OpCode::BITWISE_NO_OP:
    case OpCode::BITWISE_NOT:
    case OpCode::BITWISE_AND:
    case OpCode::BITWISE_OR:
    case OpCode::BITWISE_XOR:
    case OpCode::BITWISE_NAND:
    case OpCode::BITWISE_NOR:
    case OpCode::BITWISE_XNOR:
    case OpCode::BITWISE_ATLEAST:
    case OpCode::BITWISE_ATMOST:
    case OpCode::BITWISE_EXACTLY:
    case OpCode::BITWISE_ZEROS:
    case OpCode::BITWISE_ONES:
    case OpCode::SAMPLE:
        return true;
    default:
        return false;
    }
}

bool is_kofn(OpCode op) {
    return op == OpCode::BITWISE_ATLEAST || op == OpCode::BITWISE_ATMOST || op == OpCode::BITWISE_EXACTLY;
}

bool is_unary(OpCode op) { return op == OpCode::BITWISE_NO_OP || op == OpCode::BITWISE_NOT; }

bool is_nary_fold(OpCode op) {
    switch (op) {
    case OpCode::BITWISE_AND:
    case OpCode::BITWISE_OR:
    case OpCode::BITWISE_XOR:
    case OpCode::BITWISE_NAND:
    case OpCode::BITWISE_NOR:
    case OpCode::BITWISE_XNOR:
        return true;
    default:
        return false;
    }
}

bool is_nullary(OpCode op) { return op == OpCode::BITWISE_ZEROS || op == OpCode::BITWISE_ONES; }

bool valid_metadata_key(std::string_view key) {
    if (key.empty()) {
        return false;
    }
    for (char c : key) {
        if (!(c == '.' || c == '_' || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) {
            return false;
        }
    }
    return key.front() != '.' && key.back() != '.';
}

} // namespace

struct builder::Impl {
    struct Sample {
        double probability{};
        std::uint64_t threshold{};
        bool always_true{};
        std::uint32_t event_id{};
        SampleSource source{SampleSource::BASIC_EVENT};
        std::string name;
    };

    struct Op {
        OpCode opcode{OpCode::BITWISE_NO_OP};
        std::vector<std::uint32_t> inputs;
        std::vector<std::uint32_t> inverted_inputs;
        std::optional<std::uint32_t> k;
        std::optional<Sample> sample;
        std::uint32_t output{};
        std::string name;
    };

    struct Query {
        std::uint32_t target{};
        QueryKind kind{QueryKind::PROBABILITY};
        std::string name;
        std::vector<double> confidence;
        query_literals literals;
    };

    struct QueryGroup {
        std::string name;
        std::string unit;
        std::vector<query_term> terms;
    };

    using MetaValue = std::variant<std::string, std::int64_t, double, std::vector<std::byte>>;

    std::string circuit_name;
    std::string circuit_description;
    std::uint32_t next_tensor{0};
    std::vector<Op> ops;
    std::vector<Query> queries;
    std::vector<QueryGroup> query_groups;
    std::map<std::string, MetaValue> metadata;
    std::unordered_set<std::uint32_t> event_ids;
    std::vector<bool> is_sample; // per tensor: producer is a SAMPLE op
    bool finished{false};

    Result<tensor_id> check_inputs(std::span<const std::uint32_t> inputs,
                                   std::span<const std::uint32_t> inverted) const {
        auto in_range = [&](std::uint32_t id) -> Result<void> {
            if (id >= next_tensor) {
                return std::unexpected(builder_error("tensor index " + std::to_string(id) +
                                                     " is not yet defined (need topological order)"));
            }
            return {};
        };
        for (auto id : inputs) {
            if (auto r = in_range(id); !r) {
                return std::unexpected(r.error());
            }
        }
        for (auto id : inverted) {
            if (auto r = in_range(id); !r) {
                return std::unexpected(r.error());
            }
        }
        return 0;
    }

    Result<tensor_id> emit(Op op) {
        if (finished) {
            return std::unexpected(builder_error("builder::finish already called"));
        }
        op.output = next_tensor;
        const bool sample = op.opcode == OpCode::SAMPLE;
        ops.push_back(std::move(op));
        is_sample.push_back(sample);
        return next_tensor++;
    }

    Result<tensor_id> add_sample_impl(double probability, std::uint64_t threshold, bool always_true,
                                      std::string_view name, std::uint32_t event_id, SampleSource source) {
        if (!event_ids.insert(event_id).second) {
            return std::unexpected(builder_error("duplicate event_id " + std::to_string(event_id)));
        }
        Sample sample;
        sample.probability = probability;
        sample.always_true = always_true;
        sample.threshold = always_true ? kThresholdAlwaysTrue : threshold;
        sample.event_id = event_id;
        sample.source = source;
        sample.name = std::string(name);

        Op op;
        op.opcode = OpCode::SAMPLE;
        op.sample = std::move(sample);
        op.name = std::string(name);
        return emit(std::move(op));
    }
};

builder::builder() : impl_(std::make_unique<Impl>()) {}
builder::builder(builder &&) noexcept = default;
builder &builder::operator=(builder &&) noexcept = default;
builder::~builder() = default;

builder &builder::name(std::string_view circuit_name) {
    impl_->circuit_name = std::string(circuit_name);
    return *this;
}

builder &builder::description(std::string_view circuit_description) {
    impl_->circuit_description = std::string(circuit_description);
    return *this;
}

Result<builder::tensor_id> builder::add_sample(double probability, std::string_view name, std::uint32_t event_id,
                                               SampleSource source) {
    if (probability < 0.0 || probability > 1.0 || !std::isfinite(probability)) {
        return std::unexpected(builder_error("SAMPLE probability must be in [0, 1]"));
    }
    const bool always_true = probability == 1.0;
    return impl_->add_sample_impl(probability, probability_to_threshold(probability), always_true, name, event_id,
                                  source);
}

Result<builder::tensor_id> builder::add_sample(std::uint64_t threshold, bool always_true, std::string_view name,
                                               std::uint32_t event_id, SampleSource source) {
    return impl_->add_sample_impl(threshold_to_probability(threshold, always_true), threshold, always_true, name,
                                  event_id, source);
}

Result<builder::tensor_id> builder::add_op(OpCode opcode, std::span<const std::uint32_t> inputs,
                                           std::span<const std::uint32_t> inverted_inputs,
                                           std::optional<std::uint32_t> args, std::string_view name) {
    if (!is_core_opcode(opcode) || opcode == OpCode::SAMPLE) {
        return std::unexpected(builder_error(
            opcode == OpCode::SAMPLE ? "SAMPLE must be added via add_sample" : "opcode is reserved in core-1.1"));
    }
    if (auto r = impl_->check_inputs(inputs, inverted_inputs); !r) {
        return std::unexpected(r.error());
    }
    const auto n = inputs.size() + inverted_inputs.size();
    if (is_nullary(opcode)) {
        if (n != 0 || args) {
            return std::unexpected(builder_error("constant ops take no inputs and no args"));
        }
    } else if (is_unary(opcode)) {
        if (n != 1 || args) {
            return std::unexpected(builder_error("unary ops take exactly one input and no args"));
        }
    } else if (is_nary_fold(opcode)) {
        if (n < 1 || args) {
            return std::unexpected(builder_error("n-ary folds require ≥1 input and no args"));
        }
    } else if (is_kofn(opcode)) {
        if (n < 1 || !args) {
            return std::unexpected(builder_error("k-of-n ops require ≥1 input and args = k"));
        }
        if (*args > n) {
            return std::unexpected(builder_error("k-of-n requires 0 ≤ k ≤ n"));
        }
    } else {
        return std::unexpected(builder_error("unsupported opcode"));
    }

    Impl::Op op;
    op.opcode = opcode;
    op.inputs.assign(inputs.begin(), inputs.end());
    op.inverted_inputs.assign(inverted_inputs.begin(), inverted_inputs.end());
    op.k = args;
    op.name = std::string(name);
    return impl_->emit(std::move(op));
}

Result<void> builder::add_query(std::uint32_t target, QueryKind kind, std::string_view name,
                                std::span<const double> confidence, query_literals literals) {
    if (target >= impl_->next_tensor) {
        return std::unexpected(builder_error("query target is not a defined tensor"));
    }
    if (detail::query_kind_minor(kind) == 0) {
        return std::unexpected(builder_error("unknown query kind " + std::to_string(static_cast<unsigned>(kind))));
    }
    for (double c : confidence) {
        if (!(c > 0.0 && c < 1.0) || !std::isfinite(c)) {
            return std::unexpected(builder_error("query confidence levels must be in (0, 1)"));
        }
    }
    if (detail::literal_mode_minor(literals.mode) == 0) {
        return std::unexpected(builder_error("unknown literal mode " +
                                             std::to_string(static_cast<unsigned>(literals.mode))));
    }
    for (std::size_t i = 0; i < literals.samples.size(); ++i) {
        const auto id = literals.samples[i];
        if (id >= impl_->next_tensor || !impl_->is_sample[id]) {
            return std::unexpected(builder_error("literal sample " + std::to_string(id) +
                                                 " is not the output of a SAMPLE operator"));
        }
        if (i > 0 && literals.samples[i - 1] >= id) {
            return std::unexpected(builder_error("literal samples must be strictly ascending"));
        }
        if (kind != QueryKind::MONOTONE_HULL_PROBABILITY) {
            return std::unexpected(builder_error("literal samples are only meaningful on a "
                                                 "MONOTONE_HULL_PROBABILITY query"));
        }
    }
    if (kind != QueryKind::MONOTONE_HULL_PROBABILITY && literals.mode != LiteralMode::ATOM_SPLIT) {
        return std::unexpected(builder_error("a non-default literal mode is only meaningful on a "
                                             "MONOTONE_HULL_PROBABILITY query"));
    }
    impl_->queries.push_back(Impl::Query{target, kind, std::string(name), {confidence.begin(), confidence.end()},
                                         std::move(literals)});
    return {};
}

Result<void> builder::add_query_group(std::string_view name, std::string_view unit,
                                      std::span<const query_term> terms) {
    if (terms.empty()) {
        return std::unexpected(builder_error("a query group needs at least one term"));
    }
    std::unordered_set<std::uint32_t> seen;
    for (const auto &term : terms) {
        if (term.query >= impl_->queries.size()) {
            return std::unexpected(builder_error("query group term " + std::to_string(term.query) +
                                                 " is not an added query"));
        }
        if (!seen.insert(term.query).second) {
            return std::unexpected(builder_error("query group lists query " + std::to_string(term.query) + " twice"));
        }
        if (!std::isfinite(term.weight) || term.weight < 0.0) {
            return std::unexpected(builder_error("query group weights must be finite and >= 0"));
        }
        if (impl_->queries[term.query].kind != impl_->queries[terms.front().query].kind) {
            return std::unexpected(builder_error("the queries of a group must share one kind"));
        }
    }
    impl_->query_groups.push_back(
        Impl::QueryGroup{std::string(name), std::string(unit), {terms.begin(), terms.end()}});
    return {};
}

Result<void> builder::set(std::string_view key, std::string_view value) {
    if (!valid_metadata_key(key)) {
        return std::unexpected(builder_error("metadata keys must be lowercase and dot-separated"));
    }
    impl_->metadata[std::string(key)] = std::string(value);
    return {};
}

Result<void> builder::set(std::string_view key, std::int64_t value) {
    if (!valid_metadata_key(key)) {
        return std::unexpected(builder_error("metadata keys must be lowercase and dot-separated"));
    }
    impl_->metadata[std::string(key)] = value;
    return {};
}

Result<void> builder::set(std::string_view key, double value) {
    if (!valid_metadata_key(key)) {
        return std::unexpected(builder_error("metadata keys must be lowercase and dot-separated"));
    }
    impl_->metadata[std::string(key)] = value;
    return {};
}

Result<void> builder::set(std::string_view key, std::span<const std::byte> value) {
    if (!valid_metadata_key(key)) {
        return std::unexpected(builder_error("metadata keys must be lowercase and dot-separated"));
    }
    impl_->metadata[std::string(key)] = std::vector<std::byte>(value.begin(), value.end());
    return {};
}

Result<std::vector<std::byte>> builder::finish() {
    auto &st = *impl_;
    if (st.finished) {
        return std::unexpected(builder_error("builder::finish already called"));
    }
    if (st.queries.empty()) {
        return std::unexpected(builder_error("core-1.1 requires at least one query"));
    }
    if (st.next_tensor == 0) {
        return std::unexpected(builder_error("circuit has no tensors"));
    }

    if (!st.metadata.contains("general.generator")) {
        st.metadata["general.generator"] = std::string{"pcfb"};
    }
    if (!st.metadata.contains("general.generator_version")) {
        st.metadata["general.generator_version"] = std::string{PCFB_VERSION};
    }
    st.metadata["general.endianness"] = std::string{"little"};
    st.metadata["pcfb.opset"] = std::string{kOpset};

    flatbuffers::FlatBufferBuilder fbb;

    std::vector<const std::string *> tensor_names(st.next_tensor, nullptr);
    for (const auto &op : st.ops) {
        if (op.output < st.next_tensor && !op.name.empty() && tensor_names[op.output] == nullptr) {
            tensor_names[op.output] = &op.name;
        }
    }
    std::vector<flatbuffers::Offset<Tensor>> tensors;
    tensors.reserve(st.next_tensor);
    for (std::uint32_t i = 0; i < st.next_tensor; ++i) {
        tensors.push_back(CreateTensorDirect(fbb, nullptr, TensorType::BIT, 0,
                                             tensor_names[i] == nullptr ? nullptr : tensor_names[i]->c_str()));
    }

    std::vector<flatbuffers::Offset<Op>> operators;
    operators.reserve(st.ops.size());
    for (const auto &op : st.ops) {
        flatbuffers::Offset<void> args;
        OpArgs args_type = OpArgs::NONE;
        if (op.sample) {
            const auto &s = *op.sample;
            args_type = OpArgs::SampleArgs;
            args = CreateSampleArgsDirect(fbb, s.probability, s.threshold, s.always_true, s.event_id, s.source,
                                          s.name.empty() ? nullptr : s.name.c_str())
                       .Union();
        } else if (op.k) {
            args_type = OpArgs::KofNArgs;
            args = CreateKofNArgs(fbb, *op.k).Union();
        }
        operators.push_back(CreateOpDirect(fbb, op.opcode, args_type, args, &op.inputs, &op.inverted_inputs, op.output,
                                           op.name.empty() ? nullptr : op.name.c_str()));
    }

    std::vector<flatbuffers::Offset<Query>> queries;
    queries.reserve(st.queries.size());
    std::vector<std::uint32_t> outputs;
    outputs.reserve(st.queries.size());
    std::uint32_t schema_minor = 1;
    for (const auto &q : st.queries) {
        // Written only when non-default: a query without literal atoms must serialise to the same bytes as before.
        const auto *literal_samples =
            q.literals.samples.empty() ? nullptr : &q.literals.samples;
        queries.push_back(CreateQueryDirect(fbb, q.target, q.kind, q.name.empty() ? nullptr : q.name.c_str(),
                                            q.confidence.empty() ? nullptr : &q.confidence, q.literals.mode,
                                            literal_samples));
        outputs.push_back(q.target);
        schema_minor = std::max(schema_minor, detail::query_kind_minor(q.kind));
    }

    // Written only when non-empty: a file without groups serialises to the same bytes as before the amendment.
    std::vector<flatbuffers::Offset<QueryGroup>> groups;
    groups.reserve(st.query_groups.size());
    for (const auto &g : st.query_groups) {
        std::vector<QueryTerm> terms;
        terms.reserve(g.terms.size());
        for (const auto &t : g.terms) {
            terms.emplace_back(t.query, t.weight);
        }
        groups.push_back(CreateQueryGroupDirect(fbb, g.name.empty() ? nullptr : g.name.c_str(),
                                                g.unit.empty() ? nullptr : g.unit.c_str(), &terms));
        schema_minor = std::max(schema_minor, detail::query_groups_minor);
    }
    auto sub = CreateSubCircuitDirect(fbb, &tensors, nullptr, &outputs, &operators,
                                      st.circuit_name.empty() ? nullptr : st.circuit_name.c_str(), &queries,
                                      groups.empty() ? nullptr : &groups);

    std::vector<flatbuffers::Offset<SubCircuit>> subcircuits{sub};

    const std::vector<std::uint8_t> empty_data;
    auto empty_buf = CreateBufferDirect(fbb, &empty_data);
    std::vector<flatbuffers::Offset<Buffer>> buffers{empty_buf};

    std::vector<flatbuffers::Offset<KeyValue>> metadata;
    metadata.reserve(st.metadata.size());
    for (const auto &[key, value] : st.metadata) {
        MetadataValue value_type = MetadataValue::NONE;
        flatbuffers::Offset<void> packed;
        if (const auto *s = std::get_if<std::string>(&value)) {
            value_type = MetadataValue::StringValue;
            packed = CreateStringValueDirect(fbb, s->c_str()).Union();
        } else if (const auto *i = std::get_if<std::int64_t>(&value)) {
            value_type = MetadataValue::IntValue;
            packed = CreateIntValue(fbb, *i).Union();
        } else if (const auto *d = std::get_if<double>(&value)) {
            value_type = MetadataValue::FloatValue;
            packed = CreateFloatValue(fbb, *d).Union();
        } else if (const auto *b = std::get_if<std::vector<std::byte>>(&value)) {
            value_type = MetadataValue::BytesValue;
            std::vector<std::uint8_t> raw(b->size());
            std::transform(b->begin(), b->end(), raw.begin(),
                           [](std::byte x) { return static_cast<std::uint8_t>(x); });
            packed = CreateBytesValueDirect(fbb, &raw).Union();
        }
        metadata.push_back(CreateKeyValueDirect(fbb, key.c_str(), value_type, packed));
    }

    auto circuit = CreateCircuitDirect(fbb, &subcircuits, &buffers,
                                       st.circuit_name.empty() ? nullptr : st.circuit_name.c_str(),
                                       st.circuit_description.empty() ? nullptr : st.circuit_description.c_str(),
                                       detail::schema_version(schema_minor), &metadata);
    fbb.Finish(circuit, CircuitIdentifier());

    st.finished = true;
    const auto *buf = fbb.GetBufferPointer();
    const auto n = fbb.GetSize();
    std::vector<std::byte> out(n);
    std::memcpy(out.data(), buf, n);
    return out;
}

} // namespace pcfb
