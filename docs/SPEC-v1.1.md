# PCFB v1.1 — Probabilistic Circuit FlatBuffer (draft)

Status: **draft for review**. Normative draft schema: [`schema/drafts/pcfb_v1_1.fbs`](../schema/drafts/pcfb_v1_1.fbs).
Supersedes the informal conventions of `schema/pcfb_v1.fbs` (identifier `PCF1`, unreleased).

## 1. Purpose and position

`.pcfb` is the portable artifact between the **compiler** (model formats → probabilistic circuit) and the **runtimes**
(Monte Carlo, exact, …), in the same role GGUF plays between converters and inference engines:

```
OpenPSA / CAFTA / MAR-D ──convert/compile──▶ model.pcfb ──run──▶ probabilities, intervals, diagnostics
                                                │
                                                └── inspect / validate / diff (any language, MIT reader)
```

Design goals:
1. **One file, zero-copy readable.** Memory-mapped, verified with the FlatBuffers verifier, no parsing pass.
2. **Self-describing.** Key–value metadata records generator, source, compile options, and conventions.
3. **Runtime-neutral.** The file describes *what* to compute, not *how*: no sample counts, kernel partitions,
   bitpack widths, or devices. Those are runtime decisions.
4. **Exact where known.** Probabilities are stored at full precision, plus a precomputed 64-bit sampling threshold.
5. **Evolvable.** FlatBuffers append-only evolution; a major version in the file identifier, a minor version in data.

Non-goals for v1.1: uncertainty distributions on parameters, time/phase dependence, common-cause expansion
(already expanded by the compiler), multi-output operators, executable schedules.

## 2. Versioning and compatibility

- `file_identifier` stays **`PCF1`** (major version 1). Readers MUST reject other identifiers.
- `Circuit.schema_version` encodes `major << 16 | minor`; v1.1 writers set `0x00010001`.
  Readers MUST accept any minor version of major 1 and ignore unknown fields (FlatBuffers forward compatibility).
- Changes from v1 are wire compatible: the only renames (`subciruits` → `subcircuits`, `KofNArgs.atleast` → `k`)
  keep field positions; all new fields, tables, union members, and enum values are appended.
- A file without `schema_version` (0) is treated as v1.0 and MAY be rejected by runtimes that require v1.1 features.

## 3. Data model (core profile)

A circuit is a DAG of Boolean operators over independent random Boolean inputs.

| Concept | Representation |
|---|---|
| Node value | `Tensor` with `type = BIT`, `shape = []` (rank 0): one Boolean per sample. Runtimes choose packing (e.g. 64 samples per `uint64` word). |
| Random input (basic event, collapsed module) | `Op { opcode: SAMPLE, inputs: [], output: t, args: SampleArgs }` |
| Gate | `Op { opcode, inputs, inverted_inputs, output, args? }` — inversion of an input is fused into the op |
| Constants | `BITWISE_ZEROS` / `BITWISE_ONES` ops with no inputs |
| What to compute | `SubCircuit.queries: [Query]` |
| Large constant arrays | `Buffer` (16-byte aligned) referenced by `Tensor.buffer_idx` (reserved for extension profiles) |

### 3.1 Core opset (`pcfb.opset = "core-1.1"`)

| Opcode | Arity | Args | Semantics |
|---|---|---|---|
| `BITWISE_NO_OP` | 1 | — | Z = A |
| `BITWISE_NOT` | 1 | — | Z = ¬A |
| `BITWISE_AND`, `BITWISE_OR`, `BITWISE_XOR` | ≥ 1 | — | n-ary fold over all (possibly inverted) inputs; XOR = odd parity |
| `BITWISE_NAND`, `BITWISE_NOR`, `BITWISE_XNOR` | ≥ 1 | — | negation of the corresponding fold |
| `BITWISE_ATLEAST`, `BITWISE_ATMOST`, `BITWISE_EXACTLY` | ≥ 1 | `KofNArgs { k }` | count of true inputs ≥ / ≤ / == k; 0 ≤ k ≤ n |
| `BITWISE_ZEROS`, `BITWISE_ONES` | 0 | — | constant false / true |
| `SAMPLE` | 0 | `SampleArgs` | independent Bernoulli draw, see §4 |

All other opcodes (`EVENS`, `ODDS`, `CONV_*`, `FFT_*`, `MUXED`, `DEMUX`, `COMPARATOR`, `ADDER`, `TALLY`) are **reserved**:
conforming core-1.1 files MUST NOT use them. `TALLY` is superseded by `queries`.

An input index MAY appear in both `inputs` and `inverted_inputs` only for XOR-family ops (it is then a constant);
writers SHOULD simplify such cases away. Effective input list = `inputs` ∪ `inverted_inputs` (with inversion applied
to the latter); for k-of-n ops, `n = |inputs| + |inverted_inputs|`.

### 3.2 Ordering and single assignment

- `operators` are in **topological (execution) order**: every input tensor of op *i* is the output of an op *j < i*
  or listed in `SubCircuit.inputs`.
- Every tensor is the `output` of **at most one** op (single static assignment).
- `SubCircuit.inputs` are externally supplied Boolean tensors (empty for self-contained models).
- The core profile uses exactly **one** subcircuit (`subcircuits[0]`). Additional subcircuits are reserved.

## 4. Sampling semantics (`SAMPLE`)

```
table SampleArgs {
  probability: float64;   // P(output = true), 0 ≤ p ≤ 1, full precision
  threshold: uint64;      // precomputed: see below
  always_true: bool;      // p == 1
  event_id: uint32;       // stable identity for counter-based PRNGs (reproducibility across runtimes)
  source: SampleSource;   // BASIC_EVENT | COLLAPSED_MODULE | HOUSE_EVENT
  name: string;           // original model identifier (e.g. basic-event name or module root)
}
```

- `threshold = min(floor(p · 2^64), 2^64 − 1)` for `p < 1`, and `always_true = true` for `p = 1`. A sample is true iff
  `u < threshold` for a uniform 64-bit integer `u` (or `always_true`). This resolves probabilities down to 2^-64;
  32-bit quantisation (`floor(p · 2^32)`) silently maps every `p < 2.3e-10` to 0 and is **not** conforming.
- Runtimes MAY use any sampler that draws Bernoulli(`threshold / 2^64`) exactly (e.g. bit-sliced lazy comparison).
- `event_id` values are unique within a subcircuit. Runtimes using counter-based PRNGs SHOULD derive streams from
  `(event_id, sample index, iteration)` so results are reproducible for a given seed.
- `source = COLLAPSED_MODULE` marks an independent subgraph replaced by its exact probability at compile time;
  `name` identifies the module root. `HOUSE_EVENT` with `p ∈ {0, 1}` SHOULD be constant-folded by compilers instead.

## 5. Queries

```
table Query {
  target: uint32;          // tensor index
  kind: QueryKind;         // PROBABILITY (v1.1)
  name: string;            // e.g. top-event or sequence name
  confidence: [float64];   // optional requested two-sided confidence levels, e.g. [0.95, 0.99]
}
```

A runtime answers each query with an estimate (and, for sampling runtimes, interval/diagnostics). Event-tree
sequences or multiple top events are multiple queries over one subcircuit. Additional kinds (importance measures,
joint probabilities) are reserved for later minor versions.

## 6. Metadata (key–value)

`Circuit.metadata: [KeyValue]`, values are a union of string / int64 / float64 / bytes. Keys are lowercase,
dot-separated. Writers MUST set the **required** keys; readers MUST ignore unknown keys.

| Key | Type | Req. | Meaning |
|---|---|---|---|
| `general.generator` | string | ✓ | producing tool, e.g. `fuzzybits` |
| `general.generator_version` | string | ✓ | e.g. `0.3.0+g1a2b3c4` |
| `general.model_name` | string | | model name |
| `general.source_format` | string | | e.g. `openpsa-mef-2.0`, `cafta-ftp`, `saphire-mard` |
| `general.source_sha256` | bytes | | hash of the concatenated source inputs |
| `general.endianness` | string | ✓ | `little` |
| `pcfb.opset` | string | ✓ | `core-1.1` |
| `compile.options` | string | | compiler flags, e.g. `-c 2 --no-kn` |
| `compile.collapsed_modules` | int64 | | number of `COLLAPSED_MODULE` samples |
| `compile.exact_root` | float64 | | present iff the compiler solved the root exactly (no sampling needed) |

## 7. Buffers and alignment

`Buffer.data` is `force_align: 16`. `buffers[0]` MUST be empty (the default `buffer_idx = 0` means "no data").
Core-1.1 node tensors reference no buffers; buffers are reserved for extension profiles (e.g. parameter tables).

## 8. Validation rules (normative)

A conforming reader MUST verify (FlatBuffers verifier) and SHOULD validate:
1. identifier `PCF1`; major version 1; `pcfb.opset` known to the runtime;
2. `buffers[0]` exists and is empty; all `buffer_idx` in range;
3. all tensor indices in `inputs`, `inverted_inputs`, `output`, `SubCircuit.inputs/outputs`, `Query.target` in range;
4. topological order and single assignment (§3.2);
5. arity and args per opcode (§3.1), `0 ≤ k ≤ n`, SAMPLE has no inputs and carries `SampleArgs` with
   `0 ≤ probability ≤ 1` and a consistent `threshold` (±1 ulp of the formula);
6. unique `event_id`s; at least one query.

## 9. Reference API (C++, `pcfb` library)

- `pcfb::io::read(path) / from_bytes(span)` → immutable `pcfb::ir::Circuit` view (existing).
- **New** `pcfb::builder`: `add_sample(name, p, event_id, source) → tensor`, `add_op(opcode, inputs, inverted, k?) → tensor`,
  `add_query(target, kind, name)`, `set(key, value)`, `finish() → std::vector<std::byte>`; sets required metadata,
  computes thresholds, enforces ordering.
- **New** `pcfb::validate(const ir::Circuit&) → diagnostics` implementing §8.
- Python bindings (gguf-py analogue) are planned after v1.1 stabilises.

## 10. Conformance fixtures

`tests/fixtures/`: (1) `and2.pcfb` — AND of p=0.5 and p=0.25, query 0.125; (2) `kofn.pcfb` — ATLEAST(2 of 3);
(3) `inverted.pcfb` — NOR with an inverted input; (4) `chinese.pcfb` — compiled from Aralia `chinese.xml`
(exact P = 1.1705818e-3); (5) negative fixtures for each rule in §8.

## 11. Changes from v1 (summary)

| Area | v1 | v1.1 |
|---|---|---|
| Field name | `subciruits` (typo) | `subcircuits` (same position) |
| k-of-n | `KofNArgs.atleast`, no k for ATMOST/EXACTLY | `KofNArgs.k` for all three |
| Inputs | SAMPLE without parameters | `SampleArgs` (probability, 64-bit threshold, event_id, source, name) |
| Outputs | `TALLY` op | `queries` table; `TALLY` reserved |
| Metadata | name/description only | `schema_version` + key–value metadata with required keys |
| Node type | default `UINT64` | `BIT`, rank-0; packing is a runtime choice |
| Alignment | commented out | `force_align: 16` |
| Opset | everything declared | core-1.1 subset; exotic/multi-output ops reserved |
| Validation | verifier only | normative rules (§8) |
