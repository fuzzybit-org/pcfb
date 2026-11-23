#include "exact_eval.h"

#include "pcfb_generated.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pcfb::test {
namespace {

bool eval_op(OpCode opcode, const std::vector<std::uint8_t> &ins, std::uint32_t k) {
    auto fold_and = [&] {
        std::uint8_t v = 1;
        for (auto b : ins) {
            v = static_cast<std::uint8_t>(v & b);
        }
        return v;
    };
    auto fold_or = [&] {
        std::uint8_t v = 0;
        for (auto b : ins) {
            v = static_cast<std::uint8_t>(v | b);
        }
        return v;
    };
    auto fold_xor = [&] {
        std::uint8_t v = 0;
        for (auto b : ins) {
            v = static_cast<std::uint8_t>(v ^ b);
        }
        return v;
    };
    auto count = [&] {
        std::uint32_t n = 0;
        for (auto b : ins) {
            n += b;
        }
        return n;
    };

    switch (opcode) {
    case OpCode::BITWISE_NO_OP:
        return ins.at(0) != 0;
    case OpCode::BITWISE_NOT:
        return ins.at(0) == 0;
    case OpCode::BITWISE_AND:
        return fold_and() != 0;
    case OpCode::BITWISE_OR:
        return fold_or() != 0;
    case OpCode::BITWISE_XOR:
        return fold_xor() != 0;
    case OpCode::BITWISE_NAND:
        return fold_and() == 0;
    case OpCode::BITWISE_NOR:
        return fold_or() == 0;
    case OpCode::BITWISE_XNOR:
        return fold_xor() == 0;
    case OpCode::BITWISE_ATLEAST:
        return count() >= k;
    case OpCode::BITWISE_ATMOST:
        return count() <= k;
    case OpCode::BITWISE_EXACTLY:
        return count() == k;
    case OpCode::BITWISE_ZEROS:
        return false;
    case OpCode::BITWISE_ONES:
        return true;
    default:
        throw std::runtime_error("exact evaluator: reserved opcode");
    }
}

struct RandomEvent {
    std::uint32_t tensor;
    double p;
};

/// Enumerated SAMPLE variables plus SAMPLE outputs substituted by constants.
struct Variables {
    std::vector<RandomEvent> events;
    std::vector<std::int8_t> constant; // per tensor: -1 not a constant SAMPLE, else 0/1
};

const SubCircuit &query_subcircuit(const ir::Circuit &circuit, std::uint32_t query_index) {
    if (!circuit.valid() || !circuit.subcircuits() || circuit.subcircuits()->size() == 0) {
        throw std::runtime_error("exact evaluator: empty circuit");
    }
    const auto *sc = circuit.subcircuits()->Get(0);
    if (!sc || !sc->tensors() || !sc->operators() || !sc->queries() || query_index >= sc->queries()->size()) {
        throw std::runtime_error("exact evaluator: missing query");
    }
    return *sc;
}

/// `ones_are_constant`: Boolean semantics substitute p = 1 by 1; the monotone
/// hull keeps p = 1 SAMPLEs as variables. Both substitute p = 0 by 0.
Variables collect_variables(const SubCircuit &sc, bool ones_are_constant) {
    Variables vars;
    vars.constant.assign(sc.tensors()->size(), -1);
    for (const auto *op : *sc.operators()) {
        if (!op || op->opcode() != OpCode::SAMPLE) {
            continue;
        }
        const auto *sa = op->args_as_SampleArgs();
        if (!sa) {
            throw std::runtime_error("exact evaluator: SAMPLE without SampleArgs");
        }
        const bool one = sa->always_true() || sa->probability() == 1.0;
        if (one && ones_are_constant) {
            vars.constant[op->output()] = 1;
        } else if (!one && sa->probability() == 0.0) {
            vars.constant[op->output()] = 0;
        } else {
            vars.events.push_back({op->output(), one ? 1.0 : sa->probability()});
        }
    }
    if (vars.events.size() > 24) {
        throw std::runtime_error("exact evaluator: more than 24 random events");
    }
    return vars;
}

/// Every SAMPLE output as a variable, including p ∈ {0, 1}. The literal-atom
/// modes need this: a p = 0 event's complement atom is 1 almost surely and
/// droppable (RFC §4.9 Theorem 10(g)), so the event must stay a coordinate.
Variables collect_all_samples(const SubCircuit &sc) {
    Variables vars;
    vars.constant.assign(sc.tensors()->size(), -1);
    for (const auto *op : *sc.operators()) {
        if (!op || op->opcode() != OpCode::SAMPLE) {
            continue;
        }
        const auto *sa = op->args_as_SampleArgs();
        if (!sa) {
            throw std::runtime_error("exact evaluator: SAMPLE without SampleArgs");
        }
        vars.events.push_back({op->output(), sa->always_true() ? 1.0 : sa->probability()});
    }
    return vars;
}

/// Value of `target` for the assignment `mask` over `vars.events`.
bool evaluate(const SubCircuit &sc, const Variables &vars, std::uint32_t mask, std::uint32_t target) {
    std::vector<std::uint8_t> value(sc.tensors()->size(), 0);
    for (std::size_t e = 0; e < vars.events.size(); ++e) {
        value[vars.events[e].tensor] = ((mask >> e) & 1u) != 0 ? 1 : 0;
    }
    for (std::size_t t = 0; t < value.size(); ++t) {
        if (vars.constant[t] >= 0) {
            value[t] = static_cast<std::uint8_t>(vars.constant[t]);
        }
    }
    for (const auto *op : *sc.operators()) {
        if (!op || op->opcode() == OpCode::SAMPLE) {
            continue;
        }
        std::vector<std::uint8_t> ins;
        if (op->inputs()) {
            for (auto id : *op->inputs()) {
                ins.push_back(value[id]);
            }
        }
        if (op->inverted_inputs()) {
            for (auto id : *op->inverted_inputs()) {
                ins.push_back(static_cast<std::uint8_t>(1u - value[id]));
            }
        }
        const std::uint32_t k =
            (op->args_type() == OpArgs::KofNArgs && op->args_as_KofNArgs()) ? op->args_as_KofNArgs()->k() : 0;
        value[op->output()] = eval_op(op->opcode(), ins, k) ? 1 : 0;
    }
    return value[target] != 0;
}

double mass(const Variables &vars, std::uint32_t mask) {
    double m = 1.0;
    for (std::size_t e = 0; e < vars.events.size(); ++e) {
        m *= ((mask >> e) & 1u) != 0 ? vars.events[e].p : (1.0 - vars.events[e].p);
    }
    return m;
}

/// Subset-OR (zeta) transform over one bit of a truth table.
void close_axis(std::vector<std::uint8_t> &table, std::size_t bit) {
    const auto n = table.size();
    const std::size_t step = std::size_t{1} << bit;
    for (std::size_t base = 0; base < n; base += step * 2) {
        for (std::size_t i = 0; i < step; ++i) {
            table[base + step + i] = static_cast<std::uint8_t>(table[base + step + i] | table[base + i]);
        }
    }
}

/// The split circuit of §5.1: event slot per SAMPLE tensor and complement-atom
/// slot per listed SAMPLE tensor. Atoms are indexed [0, n_events) for the
/// events and [n_events, n_events + n_listed) for the complement atoms.
struct SplitLayout {
    std::size_t n_events{};
    std::vector<std::int32_t> event_slot;  // per tensor: event index, or -1
    std::vector<std::int32_t> atom_slot;   // per tensor: complement-atom index, or -1
};

SplitLayout make_split_layout(const SubCircuit &sc, const Variables &vars,
                              const std::vector<std::uint32_t> &listed) {
    SplitLayout lay;
    lay.n_events = vars.events.size();
    lay.event_slot.assign(sc.tensors()->size(), -1);
    lay.atom_slot.assign(sc.tensors()->size(), -1);
    for (std::size_t e = 0; e < vars.events.size(); ++e) {
        lay.event_slot[vars.events[e].tensor] = static_cast<std::int32_t>(e);
    }
    for (std::size_t j = 0; j < listed.size(); ++j) {
        const auto t = listed[j];
        if (t >= lay.event_slot.size() || lay.event_slot[t] < 0) {
            throw std::runtime_error("exact evaluator: literal sample is not a SAMPLE output");
        }
        lay.atom_slot[t] = static_cast<std::int32_t>(lay.n_events + j);
    }
    return lay;
}

/// Value of `target` in the split circuit at the atom assignment `mask`:
/// a listed sample's inverted edges read its complement atom, every other
/// inverted edge is the Boolean complement of its child.
bool evaluate_split(const SubCircuit &sc, const SplitLayout &lay, std::uint32_t mask, std::uint32_t target) {
    std::vector<std::uint8_t> value(sc.tensors()->size(), 0);
    for (std::size_t t = 0; t < value.size(); ++t) {
        const auto slot = lay.event_slot[t];
        if (slot >= 0) {
            value[t] = ((mask >> static_cast<std::size_t>(slot)) & 1u) != 0 ? 1 : 0;
        }
    }
    for (const auto *op : *sc.operators()) {
        if (!op || op->opcode() == OpCode::SAMPLE) {
            continue;
        }
        std::vector<std::uint8_t> ins;
        if (op->inputs()) {
            for (auto id : *op->inputs()) {
                ins.push_back(value[id]);
            }
        }
        if (op->inverted_inputs()) {
            for (auto id : *op->inverted_inputs()) {
                const auto atom = id < lay.atom_slot.size() ? lay.atom_slot[id] : -1;
                if (atom >= 0) {
                    const auto v = ((mask >> static_cast<std::size_t>(atom)) & 1u) != 0 ? 1u : 0u;
                    ins.push_back(static_cast<std::uint8_t>(v));
                } else {
                    ins.push_back(static_cast<std::uint8_t>(1u - value[id]));
                }
            }
        }
        const std::uint32_t k =
            (op->args_type() == OpArgs::KofNArgs && op->args_as_KofNArgs()) ? op->args_as_KofNArgs()->k() : 0;
        value[op->output()] = eval_op(op->opcode(), ins, k) ? 1 : 0;
    }
    return value[target] != 0;
}

} // namespace

double exact_probability(const ir::Circuit &circuit, std::uint32_t query_index) {
    const auto &sc = query_subcircuit(circuit, query_index);
    const auto vars = collect_variables(sc, /*ones_are_constant=*/true);
    const auto target = sc.queries()->Get(query_index)->target();
    const std::uint32_t n_assign = 1u << vars.events.size();
    double total = 0.0;
    for (std::uint32_t mask = 0; mask < n_assign; ++mask) {
        if (evaluate(sc, vars, mask, target)) {
            total += mass(vars, mask);
        }
    }
    return total;
}

double exact_hull_probability(const ir::Circuit &circuit, std::uint32_t query_index) {
    const auto &sc = query_subcircuit(circuit, query_index);
    const auto vars = collect_variables(sc, /*ones_are_constant=*/false);
    const auto target = sc.queries()->Get(query_index)->target();
    const auto n = vars.events.size();
    const std::uint32_t n_assign = 1u << n;
    std::vector<std::uint8_t> hull(n_assign);
    for (std::uint32_t mask = 0; mask < n_assign; ++mask) {
        hull[mask] = evaluate(sc, vars, mask, target) ? 1 : 0;
    }
    // ↑g(x) = OR over y ⊆ x of g(y): subset-OR (zeta) transform, one variable at a time.
    for (std::size_t e = 0; e < n; ++e) {
        close_axis(hull, e);
    }
    double total = 0.0;
    for (std::uint32_t mask = 0; mask < n_assign; ++mask) {
        if (hull[mask] != 0) {
            total += mass(vars, mask);
        }
    }
    return total;
}

double exact_literal_hull_probability(const ir::Circuit &circuit, std::uint32_t query_index) {
    const auto &sc = query_subcircuit(circuit, query_index);
    const auto literals = read_literals(*sc.queries()->Get(query_index));
    return exact_literal_hull_probability(circuit, query_index, literals.mode, literals.samples);
}

double exact_literal_hull_probability(const ir::Circuit &circuit, std::uint32_t query_index, LiteralMode mode,
                                      std::span<const std::uint32_t> literal_samples) {
    const auto &sc = query_subcircuit(circuit, query_index);
    const auto target = sc.queries()->Get(query_index)->target();
    const auto vars = collect_all_samples(sc);

    std::vector<std::uint32_t> listed;
    if (mode != LiteralMode::NEGATION) {
        listed.assign(literal_samples.begin(), literal_samples.end());
        std::sort(listed.begin(), listed.end());
        listed.erase(std::unique(listed.begin(), listed.end()), listed.end());
    }

    if (mode == LiteralMode::ATOM_SPLIT) {
        const auto lay = make_split_layout(sc, vars, listed);
        const std::size_t n_atoms = lay.n_events + listed.size();
        if (n_atoms > 20) {
            throw std::runtime_error("exact evaluator: more than 20 literal atoms");
        }
        const std::uint32_t n_assign = 1u << n_atoms;
        std::vector<std::uint8_t> hull(n_assign);
        for (std::uint32_t mask = 0; mask < n_assign; ++mask) {
            hull[mask] = evaluate_split(sc, lay, mask, target) ? 1 : 0;
        }
        for (std::size_t a = 0; a < n_atoms; ++a) {
            close_axis(hull, a);
        }
        double total = 0.0;
        // Weight the coupled valuation κ(x): every event bit from x, every complement atom its complement.
        for (std::uint32_t x = 0; x < (1u << lay.n_events); ++x) {
            std::uint32_t atoms = 0;
            double weight = 1.0;
            for (std::size_t e = 0; e < lay.n_events; ++e) {
                const bool one = ((x >> e) & 1u) != 0;
                if (one) {
                    atoms |= (1u << e);
                    weight *= vars.events[e].p;
                } else {
                    weight *= (1.0 - vars.events[e].p);
                }
            }
            for (std::size_t j = 0; j < listed.size(); ++j) {
                const auto tensor = listed[j];
                const auto slot = static_cast<std::size_t>(lay.event_slot[tensor]);
                if (((x >> slot) & 1u) == 0) {
                    atoms |= (1u << (lay.n_events + j));
                }
            }
            if (hull[atoms] != 0) {
                total += weight;
            }
        }
        return total;
    }

    // OMEGA_L and NEGATION read the plain target g; OMEGA_L closes only the axes of unlisted events, because a
    // listed event is fixed by ω_L (RFC §3). NEGATION closes every axis, i.e. the plain hull.
    const auto n = vars.events.size();
    if (n > 24) {
        throw std::runtime_error("exact evaluator: more than 24 random events");
    }
    const std::uint32_t n_assign = 1u << n;
    std::vector<std::uint8_t> hull(n_assign);
    for (std::uint32_t mask = 0; mask < n_assign; ++mask) {
        hull[mask] = evaluate(sc, vars, mask, target) ? 1 : 0;
    }
    std::vector<std::uint8_t> is_listed(n, 0);
    for (auto t : listed) {
        for (std::size_t e = 0; e < n; ++e) {
            if (vars.events[e].tensor == t) {
                is_listed[e] = 1;
            }
        }
    }
    for (std::size_t e = 0; e < n; ++e) {
        if (is_listed[e] == 0) {
            close_axis(hull, e);
        }
    }
    double total = 0.0;
    for (std::uint32_t mask = 0; mask < n_assign; ++mask) {
        if (hull[mask] != 0) {
            total += mass(vars, mask);
        }
    }
    return total;
}

} // namespace pcfb::test
