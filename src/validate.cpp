#include "pcfb/validate.h"

#include "pcfb/threshold.h"
#include "pcfb_generated.h"
#include "schema_version.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace pcfb {
namespace {

constexpr const char *kOpset = "core-1.1";

std::string path_sub(std::uint32_t i) { return "circuit.subcircuits[" + std::to_string(i) + "]"; }

std::string path_op(std::uint32_t s, std::uint32_t i) {
    return path_sub(s) + ".operators[" + std::to_string(i) + "]";
}

bool is_reserved(OpCode op) {
    switch (op) {
    case OpCode::BITWISE_EVENS:
    case OpCode::BITWISE_ODDS:
    case OpCode::BITWISE_CONV_ND:
    case OpCode::BITWISE_CONV_1D:
    case OpCode::BITWISE_CONV_2D:
    case OpCode::BITWISE_FFT_ND:
    case OpCode::BITWISE_FFT_1D:
    case OpCode::BITWISE_FFT_2D:
    case OpCode::BITWISE_MUXED:
    case OpCode::BITWISE_DEMUX:
    case OpCode::COMPARATOR:
    case OpCode::ADDER:
    case OpCode::TALLY:
        return true;
    default:
        return false;
    }
}

bool is_kofn(OpCode op) {
    return op == OpCode::BITWISE_ATLEAST || op == OpCode::BITWISE_ATMOST || op == OpCode::BITWISE_EXACTLY;
}

std::uint32_t n_inputs(const Op *op) {
    const auto a = op->inputs() ? op->inputs()->size() : 0;
    const auto b = op->inverted_inputs() ? op->inverted_inputs()->size() : 0;
    return a + b;
}

void check_index(std::vector<Diagnostic> &out, std::uint32_t idx, std::uint32_t bound, const std::string &location,
                 const char *what) {
    if (idx >= bound) {
        out.push_back({Diagnostic::Code::TensorIndexOutOfRange,
                       std::string(what) + " " + std::to_string(idx) + " >= " + std::to_string(bound), location});
    }
}

const char *string_meta(const Circuit *root, const char *key) {
    const auto *md = root->metadata();
    if (!md) {
        return nullptr;
    }
    const auto *kv = md->LookupByKey(key);
    if (!kv || kv->value_type() != MetadataValue::StringValue) {
        return nullptr;
    }
    const auto *sv = kv->value_as_StringValue();
    return (sv && sv->value()) ? sv->value()->c_str() : nullptr;
}

/// §8.8: the literal fields of one query. Indices outside the target's cone are valid: passes may remove
/// events from a cone (spec §5.2), so a list may name samples the target happens to stop depending on.
void validate_literals(std::vector<Diagnostic> &out, const Query *q, const SubCircuit &sc, std::uint32_t n_tensors,
                       std::uint32_t schema_minor, const std::string &qloc) {
    const auto mode = q->literal_mode();
    if (detail::literal_mode_known(mode) == 0) {
        out.push_back({Diagnostic::Code::UnknownLiteralMode,
                       "literal mode " + std::to_string(static_cast<unsigned>(mode)) +
                           " is not defined in pcfb 1.2; a runtime must reject the query",
                       qloc + ".literal_mode"});
    }
    const auto *samples = q->literal_samples();
    const bool non_empty = samples != nullptr && samples->size() != 0;
    const bool non_default_mode = mode != LiteralMode::ATOM_SPLIT;
    if (!non_empty && !non_default_mode) {
        return;
    }
    // The hull kind already requires minor >= 2; check the literal fields on their own account too.
    if (schema_minor < 2) {
        out.push_back({Diagnostic::Code::SchemaVersionTooOld,
                       "query literal fields require schema_version minor >= 2, file declares " +
                           std::to_string(schema_minor),
                       qloc + ".literal_samples"});
    }
    if (q->kind() != QueryKind::MONOTONE_HULL_PROBABILITY) {
        out.push_back({Diagnostic::Code::LiteralFieldsOnNonHullQuery,
                       "literal samples and a literal mode are only meaningful on a MONOTONE_HULL_PROBABILITY query",
                       qloc + ".literal_samples"});
    }
    if (!non_empty) {
        return;
    }
    // A tensor is a valid literal sample iff its producer is a SAMPLE op.
    std::vector<std::uint8_t> from_sample(n_tensors, 0);
    if (const auto *ops = sc.operators()) {
        for (const auto *op : *ops) {
            if (op && op->opcode() == OpCode::SAMPLE && op->output() < n_tensors) {
                from_sample[op->output()] = 1;
            }
        }
    }
    for (flatbuffers::uoffset_t i = 0; i < samples->size(); ++i) {
        const auto id = samples->Get(i);
        const auto sloc = qloc + ".literal_samples[" + std::to_string(i) + "]";
        if (id >= n_tensors) {
            out.push_back({Diagnostic::Code::TensorIndexOutOfRange,
                           "literal sample " + std::to_string(id) + " >= " + std::to_string(n_tensors), sloc});
        } else if (from_sample[id] == 0) {
            out.push_back({Diagnostic::Code::LiteralSampleNotSample,
                           "literal sample " + std::to_string(id) + " is not the output of a SAMPLE operator", sloc});
        }
        if (i > 0 && samples->Get(i - 1) >= id) {
            out.push_back({Diagnostic::Code::LiteralSamplesNotAscending,
                           "literal sample " + std::to_string(id) + " must be strictly greater than " +
                               std::to_string(samples->Get(i - 1)),
                           sloc});
        }
    }
}

void validate_subcircuit(std::vector<Diagnostic> &out, const SubCircuit *sc, std::uint32_t si, std::uint32_t n_buffers,
                         std::uint32_t schema_minor) {
    const auto loc = path_sub(si);
    const std::uint32_t n_tensors = sc->tensors() ? sc->tensors()->size() : 0;

    auto visit_indices = [&](const flatbuffers::Vector<std::uint32_t> *vec, const std::string &at, const char *what) {
        if (!vec) {
            return;
        }
        for (flatbuffers::uoffset_t i = 0; i < vec->size(); ++i) {
            check_index(out, vec->Get(i), n_tensors, at + "[" + std::to_string(i) + "]", what);
        }
    };

    visit_indices(sc->inputs(), loc + ".inputs", "input tensor");
    visit_indices(sc->outputs(), loc + ".outputs", "output tensor");

    std::unordered_set<std::uint32_t> external;
    if (sc->inputs()) {
        for (auto id : *sc->inputs()) {
            external.insert(id);
        }
    }

    std::vector<int> assigned(n_tensors, 0); // 0 = none, 1 = op, 2 = external
    for (auto id : external) {
        if (id < n_tensors) {
            assigned[id] = 2;
        }
    }

    if (sc->tensors()) {
        for (flatbuffers::uoffset_t ti = 0; ti < sc->tensors()->size(); ++ti) {
            const auto *t = sc->tensors()->Get(ti);
            if (!t) {
                continue;
            }
            if (t->buffer_idx() >= n_buffers) {
                out.push_back({Diagnostic::Code::BufferIndexOutOfRange,
                               "buffer_idx " + std::to_string(t->buffer_idx()) + " >= " + std::to_string(n_buffers),
                               loc + ".tensors[" + std::to_string(ti) + "]"});
            }
        }
    }

    std::unordered_set<std::uint32_t> event_ids;
    const auto *ops = sc->operators();
    if (ops) {
        for (flatbuffers::uoffset_t oi = 0; oi < ops->size(); ++oi) {
            const auto *op = ops->Get(oi);
            const auto oloc = path_op(si, oi);
            if (!op) {
                continue;
            }
            const auto opcode = op->opcode();
            if (is_reserved(opcode)) {
                out.push_back({Diagnostic::Code::ReservedOpcode, "opcode is reserved in core-1.1", oloc});
            }

            visit_indices(op->inputs(), oloc + ".inputs", "input tensor");
            visit_indices(op->inverted_inputs(), oloc + ".inverted_inputs", "inverted input tensor");
            check_index(out, op->output(), n_tensors, oloc + ".output", "output tensor");

            auto defined_before = [&](std::uint32_t id) {
                if (id >= n_tensors) {
                    return;
                }
                // Producer must be a previous op (assigned==1) or an external input (2).
                if (assigned[id] == 0) {
                    out.push_back({Diagnostic::Code::NotTopological,
                                   "tensor " + std::to_string(id) + " is used before it is assigned", oloc});
                }
            };
            if (op->inputs()) {
                for (auto id : *op->inputs()) {
                    defined_before(id);
                }
            }
            if (op->inverted_inputs()) {
                for (auto id : *op->inverted_inputs()) {
                    defined_before(id);
                }
            }

            const auto output = op->output();
            if (output < n_tensors) {
                if (assigned[output] != 0) {
                    out.push_back({Diagnostic::Code::MultipleAssignment,
                                   "tensor " + std::to_string(output) + " assigned more than once", oloc + ".output"});
                } else {
                    assigned[output] = 1;
                }
            }

            const auto n = n_inputs(op);
            switch (opcode) {
            case OpCode::BITWISE_ZEROS:
            case OpCode::BITWISE_ONES:
                if (n != 0 || op->args_type() != OpArgs::NONE) {
                    out.push_back({Diagnostic::Code::BadArity, "constant ops take no inputs and no args", oloc});
                }
                break;
            case OpCode::BITWISE_NO_OP:
            case OpCode::BITWISE_NOT:
                if (n != 1 || op->args_type() != OpArgs::NONE) {
                    out.push_back({Diagnostic::Code::BadArity, "unary ops take exactly one input and no args", oloc});
                }
                break;
            case OpCode::BITWISE_AND:
            case OpCode::BITWISE_OR:
            case OpCode::BITWISE_XOR:
            case OpCode::BITWISE_NAND:
            case OpCode::BITWISE_NOR:
            case OpCode::BITWISE_XNOR:
                if (n < 1 || op->args_type() != OpArgs::NONE) {
                    out.push_back({Diagnostic::Code::BadArity, "n-ary folds require ≥1 input and no args", oloc});
                }
                break;
            case OpCode::BITWISE_ATLEAST:
            case OpCode::BITWISE_ATMOST:
            case OpCode::BITWISE_EXACTLY: {
                if (n < 1) {
                    out.push_back({Diagnostic::Code::BadArity, "k-of-n requires ≥1 input", oloc});
                }
                if (op->args_type() != OpArgs::KofNArgs || !op->args_as_KofNArgs()) {
                    out.push_back({Diagnostic::Code::BadArgs, "k-of-n requires KofNArgs", oloc});
                } else {
                    const auto k = op->args_as_KofNArgs()->k();
                    if (k > n) {
                        out.push_back({Diagnostic::Code::BadArgs, "k-of-n requires 0 ≤ k ≤ n", oloc});
                    }
                }
                (void)is_kofn;
                break;
            }
            case OpCode::SAMPLE: {
                if (n != 0) {
                    out.push_back({Diagnostic::Code::BadSample, "SAMPLE takes no inputs", oloc});
                }
                if (op->args_type() != OpArgs::SampleArgs || !op->args_as_SampleArgs()) {
                    out.push_back({Diagnostic::Code::BadArgs, "SAMPLE requires SampleArgs", oloc});
                    break;
                }
                const auto *sa = op->args_as_SampleArgs();
                const double p = sa->probability();
                if (!(p >= 0.0 && p <= 1.0) || !std::isfinite(p)) {
                    out.push_back({Diagnostic::Code::BadSample, "SAMPLE probability must be in [0, 1]", oloc});
                }
                if (sa->always_true()) {
                    if (p != 1.0) {
                        out.push_back({Diagnostic::Code::BadSample, "always_true requires probability == 1", oloc});
                    }
                } else if (p == 1.0) {
                    out.push_back({Diagnostic::Code::BadSample, "probability == 1 requires always_true", oloc});
                } else {
                    const auto expected = probability_to_threshold(p);
                    const auto got = sa->threshold();
                    const auto diff = got > expected ? got - expected : expected - got;
                    if (diff > 1) {
                        out.push_back({Diagnostic::Code::InconsistentThreshold,
                                       "threshold is more than 1 ULP from floor(p·2^64)", oloc});
                    }
                }
                if (!event_ids.insert(sa->event_id()).second) {
                    out.push_back({Diagnostic::Code::DuplicateEventId,
                                   "duplicate event_id " + std::to_string(sa->event_id()), oloc});
                }
                break;
            }
            default:
                break;
            }
        }
    }

    const auto *queries = sc->queries();
    if (!queries || queries->size() == 0) {
        out.push_back({Diagnostic::Code::MissingQuery, "subcircuit has no queries", loc});
    } else {
        for (flatbuffers::uoffset_t qi = 0; qi < queries->size(); ++qi) {
            const auto *q = queries->Get(qi);
            if (!q) {
                continue;
            }
            const auto qloc = loc + ".queries[" + std::to_string(qi) + "]";
            check_index(out, q->target(), n_tensors, qloc + ".target", "query target");
            const auto kind = q->kind();
            const auto needed = detail::query_kind_minor(kind);
            if (needed == 0) {
                out.push_back({Diagnostic::Code::UnknownQueryKind,
                               "query kind " + std::to_string(static_cast<unsigned>(kind)) +
                                   " is not defined in pcfb 1.2; a runtime must reject the file",
                               qloc + ".kind"});
            } else if (schema_minor < needed) {
                out.push_back({Diagnostic::Code::SchemaVersionTooOld,
                               std::string(EnumNameQueryKind(kind)) + " requires schema_version minor >= " +
                                   std::to_string(needed) + ", file declares " + std::to_string(schema_minor),
                               qloc + ".kind"});
            }
            validate_literals(out, q, *sc, n_tensors, schema_minor, qloc);
        }
    }
}

} // namespace

std::vector<Diagnostic> validate(const ir::Circuit &circuit) {
    std::vector<Diagnostic> out;
    if (!circuit.valid() || circuit.data() == nullptr || circuit.size() < 8) {
        out.push_back({Diagnostic::Code::InvalidFile, "circuit view is empty", "circuit"});
        return out;
    }

    if (!flatbuffers::BufferHasIdentifier(circuit.data(), CircuitIdentifier())) {
        out.push_back({Diagnostic::Code::BadIdentifier, "file identifier is not PCF1", "circuit"});
    }

    const auto *root = circuit.get();
    std::uint32_t schema = root->schema_version();
    std::uint32_t major = schema == 0 ? 1u : (schema >> 16U);
    std::uint32_t minor = schema & 0xFFFFU; // 0 for a v1.0 file without schema_version
    if (major != 1) {
        out.push_back({Diagnostic::Code::BadMajorVersion, "schema_version major is " + std::to_string(major) + ", expected 1",
                       "circuit.schema_version"});
    }

    const char *opset = string_meta(root, "pcfb.opset");
    if (!opset) {
        out.push_back({Diagnostic::Code::MissingRequiredMetadata, "missing pcfb.opset", "circuit.metadata"});
    } else if (std::string_view(opset) != kOpset) {
        out.push_back({Diagnostic::Code::UnknownOpset, std::string("unknown pcfb.opset '") + opset + "'",
                       "circuit.metadata[\"pcfb.opset\"]"});
    }
    for (const char *key : {"general.generator", "general.generator_version", "general.endianness"}) {
        if (!string_meta(root, key)) {
            out.push_back({Diagnostic::Code::MissingRequiredMetadata, std::string("missing ") + key,
                           std::string("circuit.metadata[\"") + key + "\"]"});
        }
    }

    const auto *buffers = root->buffers();
    const std::uint32_t n_buffers = buffers ? buffers->size() : 0;
    if (n_buffers == 0) {
        out.push_back({Diagnostic::Code::MissingEmptyBuffer, "buffers[0] is missing", "circuit.buffers"});
    } else {
        const auto *b0 = buffers->Get(0);
        if (!b0 || (b0->data() && b0->data()->size() != 0)) {
            out.push_back({Diagnostic::Code::MissingEmptyBuffer, "buffers[0] must be empty", "circuit.buffers[0]"});
        }
    }

    const auto *subs = root->subcircuits();
    if (!subs || subs->size() == 0) {
        out.push_back({Diagnostic::Code::ExtraSubcircuit, "core-1.1 requires exactly one subcircuit",
                       "circuit.subcircuits"});
        return out;
    }
    if (subs->size() != 1) {
        out.push_back({Diagnostic::Code::ExtraSubcircuit, "core-1.1 uses exactly one subcircuit",
                       "circuit.subcircuits"});
    }
    for (flatbuffers::uoffset_t i = 0; i < subs->size(); ++i) {
        const auto *sc = subs->Get(i);
        if (sc) {
            validate_subcircuit(out, sc, i, n_buffers, minor);
        }
    }
    return out;
}

} // namespace pcfb
