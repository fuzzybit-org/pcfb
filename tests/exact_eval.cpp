#include "exact_eval.h"

#include "pcfb_generated.h"

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
        const std::uint32_t bit = 1u << e;
        for (std::uint32_t mask = 0; mask < n_assign; ++mask) {
            if ((mask & bit) != 0) {
                hull[mask] = static_cast<std::uint8_t>(hull[mask] | hull[mask ^ bit]);
            }
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
