# PCFB v1.2 — Probabilistic Circuit FlatBuffer

Status: **v1.2**. Canonical schema: [`schema/pcfb.fbs`](../schema/pcfb.fbs)
(v1 frozen at [`schema/history/pcfb_v1.fbs`](../schema/history/pcfb_v1.fbs), v1.1 frozen at
[`schema/history/pcfb_v1_1.fbs`](../schema/history/pcfb_v1_1.fbs); `flatc --conform` must pass against both).
Supersedes [v1.1](SPEC-v1.1.md). The only addition is the query kind `MONOTONE_HULL_PROBABILITY` (§5, §8, §11).

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

Non-goals for v1.2: uncertainty distributions on parameters, time/phase dependence, common-cause expansion
(already expanded by the compiler), multi-output operators, executable schedules.

## 2. Versioning and compatibility

- `file_identifier` stays **`PCF1`** (major version 1). Readers MUST reject other identifiers.
- `Circuit.schema_version` encodes `major << 16 | minor`. A file that uses a v1.2 feature (a
  `MONOTONE_HULL_PROBABILITY` query) MUST set a minor ≥ 2 (`0x00010002`). Writers SHOULD set the lowest minor that
  covers the features they use; the reference builder does, so a file without v1.2 features is `0x00010001` and
  byte-identical to v1.1 builder output.
- Readers MUST accept any minor version of major 1 and ignore unknown fields (FlatBuffers forward compatibility).
  An unknown *enum value* is not an unknown field: a runtime that meets a query kind it does not implement MUST
  reject the file (§8).
- Changes from v1.1 are wire compatible: one enum value is appended (`QueryKind.MONOTONE_HULL_PROBABILITY = 1`).
  Changes from v1 to v1.1 are listed in [SPEC-v1.1 §11](SPEC-v1.1.md#11-changes-from-v1-summary).
- A file without `schema_version` (0) is treated as v1.0 and MAY be rejected by runtimes that require v1.1 features.
- **Compatibility hazard.** v1.1 defined only `PROBABILITY`. A runtime written against v1.1 that does not read
  `Query.kind` would answer a v1.2 hull query as a Boolean probability. Such runtimes MUST add the kind check (§8)
  before they accept files with minor ≥ 2.

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

v1.2 adds no operators; the opset of v1.2 files is still `core-1.1`.

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
- Every `SAMPLE` output is a random variable, including `always_true` ones; `BITWISE_ZEROS` / `BITWISE_ONES` are
  constants. `PROBABILITY` cannot tell an `always_true` `SAMPLE` from `BITWISE_ONES`; `MONOTONE_HULL_PROBABILITY`
  can (§5.2).

## 5. Queries

```
enum QueryKind : ubyte {
  PROBABILITY = 0,               // v1.1
  MONOTONE_HULL_PROBABILITY = 1, // v1.2
}

table Query {
  target: uint32;          // tensor index
  kind: QueryKind;         // default PROBABILITY
  name: string;            // e.g. top-event or sequence name
  confidence: [float64];   // optional requested two-sided confidence levels, e.g. [0.95, 0.99]
}
```

A runtime answers each query with an estimate (and, for sampling runtimes, interval/diagnostics). Event-tree
sequences or multiple top events are multiple queries over one subcircuit. Several queries MAY share a target with
different kinds. Further kinds (importance measures, joint probabilities) are reserved for later minor versions.

### 5.1 Semantics

Let $E$ be the set of `SAMPLE` ops of the subcircuit and $X = (X_e : e \in E)$ their outputs: independent, with
$P(X_e = 1) = p_e$ (§4). For a query, let $g$ be the Boolean function of $x \in \lbrace 0,1 \rbrace^E$ computed at
`target`. For $x, y \in \lbrace 0,1 \rbrace^E$, $y \le x$ means $y_e \le x_e$ for every $e \in E$.

- `PROBABILITY` (v1.1): $P(g(X) = 1)$.
- `MONOTONE_HULL_PROBABILITY` (v1.2): $P(\uparrow g(X) = 1)$, where $\uparrow g(x) = 1$ iff $g(y) = 1$ for some
  $y \le x$. In words: some subset of the `SAMPLE` inputs that are true in the sample satisfies the target when all
  other `SAMPLE` inputs are false.

$\uparrow g$ is the least monotone function above $g$. Its minimal cut sets are the minimal cut sets of the
non-coherent function $g$ in the sense of Rauzy (minimal positive parts of its prime implicants). For a delete-term
target $g = F \wedge \neg D$ with $F$, $D$ monotone, $\uparrow g$ is the union of the minimal cut sets of $F$ that
contain no minimal cut set of $D$: the quantity that cut-set tools approximate with delete terms, here without
truncation and without rare-event or min-cut-upper-bound approximation. Definitions, proofs, an estimator and worked
examples: fuzzybits RFC
[`docs/rfc/monotone-hull-queries.md`](https://github.com/fuzzybit-org/fuzzybits/blob/develop/docs/rfc/monotone-hull-queries.md)
([fuzzybits#174](https://github.com/fuzzybit-org/fuzzybits/issues/174)).

Core files have no `SubCircuit.inputs`; a profile that adds external inputs must define their role in a hull query.

### 5.2 Consequences for writers (informative unless marked)

- $P(g) \le P(\uparrow g)$, with equality when $g$ is monotone (for example when the target's cone has no inverted
  inputs and only `NO_OP`, `AND`, `OR`, `ATLEAST`, constants and `SAMPLE`).
- An `always_true` `SAMPLE` may be dropped by $y$; `BITWISE_ONES` may not. Writers MUST NOT replace an
  `always_true` `SAMPLE` by `BITWISE_ONES` in the cone of a `MONOTONE_HULL_PROBABILITY` target: the answer changes
  (fixture `delete_term.pcfb`, §10: 0.625 becomes 0). Configuration switches that are meant to be constants (house
  events) should be written as constants.
- A `SAMPLE` with `p = 0` may be replaced by `BITWISE_ZEROS` without changing either kind.
- A `COLLAPSED_MODULE` `SAMPLE` in the cone of a hull target must carry $P(\uparrow M)$, where $M$ is the module's
  function (false when all its inputs are false); for a monotone module this is its ordinary probability.
- $\uparrow$ distributes over OR, but the hulls of mutually exclusive sequences are in general not mutually exclusive:
  the hull probability of an end state (an OR of sequences) is its own query, not the sum of the sequence values.

### 5.3 Results (informative)

Result formats are runtime-defined. For `MONOTONE_HULL_PROBABILITY`, sampling runtimes SHOULD report, besides the
estimate and interval, the per-run estimates of the lower and upper bound functions they used and the fraction of
samples that needed an exact check (RFC §7.6).

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
6. unique `event_id`s; at least one query;
7. (v1.2) every `Query.kind` is defined by this specification, and the minor of `schema_version` is at least the
   minor that introduced it (`PROBABILITY`: 1, `MONOTONE_HULL_PROBABILITY`: 2; a missing `schema_version` counts as
   minor 0). A query of either kind is valid on any target tensor; in particular hull targets may contain inverted
   inputs and non-monotone ops.

Runtimes (normative): before evaluating any query, a runtime MUST check that it implements the kind of every query in
the file, and MUST reject the file if it does not (at planning or load time, not after sampling). A runtime MUST NOT
answer a query of one kind with the value of another; in particular it MUST NOT evaluate `MONOTONE_HULL_PROBABILITY`
as `PROBABILITY`. A query kind the runtime does not know (rule 7) is a kind it does not implement.

## 9. Reference API (C++, `pcfb` library)

- `pcfb::io::read(path) / from_bytes(span)` → immutable `pcfb::ir::Circuit` view.
- `pcfb::builder`: `add_sample(name, p, event_id, source) → tensor`, `add_op(opcode, inputs, inverted, k?) → tensor`,
  `add_query(target, kind, name)`, `set(key, value)`, `finish() → std::vector<std::byte>`; sets required metadata,
  computes thresholds, enforces ordering, rejects query kinds it does not define, and writes the lowest
  `schema_version` that covers the query kinds used (`pcfb::kSchemaVersionV11` or `pcfb::kSchemaVersionV12`).
- `pcfb::validate(const ir::Circuit&) → diagnostics` implementing §8; rule 7 reports `UnknownQueryKind` and
  `SchemaVersionTooOld` (codes appended after the v1.1 codes, which keep their values).
- Python bindings (gguf-py analogue) are planned after v1.1 stabilises.

## 10. Conformance fixtures

`tests/fixtures/`: (1) `and2.pcfb` — AND of p=0.5 and p=0.25, query 0.125; (2) `kofn.pcfb` — ATLEAST(2 of 3)
independent p=0.5 events, query 0.5; (3) `inverted.pcfb` — NOR(A, ¬B) with A=0.5, B=0.25, query 0.125;
(4) `chinese.pcfb` — compiled from Aralia `chinese.xml` (public in `fuzzybit-org/fuzzybits`
`input/Aralia/chinese.xml`, 25 basic events) by `fzb compile`
(`fuzzybit-org/fuzzybits@652ceceb0287ad80130b943297244168efe0b89f` (#62)). Published digit string
1.1705818e-3; bit-sliced exhaustive eval yields 1.1705818107586689e-03. The pcfb test-only
evaluator caps at 24 random SAMPLE events, so the fixture test asserts `validate` + structure
(25 `SAMPLE`/`BASIC_EVENT`, 7 AND, 16 OR, query `r1`) rather than exact P; (5) `delete_term.pcfb` (v1.2,
`schema_version` `0x00010002`) — `SEQ = AND(F, ¬DEL)`, `F = OR(A, B)`, `DEL = AND(X, Y)` with A=0.25, B=0.5 and
`always_true` X, Y; query `seq_boolean` (`PROBABILITY`) is exactly 0 and query `seq_hull`
(`MONOTONE_HULL_PROBABILITY`) is exactly 0.625 = P(A ∨ B); (6) negative cases for each rule in §8 live in
`tests/test_validate.cpp` (built with the object API, not committed as `.pcfb` files). Fixtures (1)–(3) stay
`schema_version` `0x00010001`. `tests/exact_eval.cpp` holds test-only exhaustive evaluators for both kinds.

## 11. Changes from v1.1 (summary)

| Area | v1.1 | v1.2 |
|---|---|---|
| Query kinds | `PROBABILITY` | `PROBABILITY`, `MONOTONE_HULL_PROBABILITY` (appended, value 1) |
| `schema_version` | `0x00010001` | `0x00010002` when a v1.2 kind is used; otherwise unchanged |
| Opset | `core-1.1` | `core-1.1` (no new operators) |
| Validation | rules 1–6 | rule 7: known kinds, minimum minor per kind |
| Runtimes | — | MUST reject files with kinds they do not implement; MUST NOT substitute kinds |
| Frozen schemas | v1 | v1, v1.1 (`schema/history/pcfb_v1_1.fbs`) |
