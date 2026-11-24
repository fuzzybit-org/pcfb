# PCFB v1.2 — Probabilistic Circuit FlatBuffer

Status: **v1.2**. Canonical schema: [`schema/pcfb.fbs`](../schema/pcfb.fbs)
(v1 frozen at [`schema/history/pcfb_v1.fbs`](../schema/history/pcfb_v1.fbs), v1.1 frozen at
[`schema/history/pcfb_v1_1.fbs`](../schema/history/pcfb_v1_1.fbs); `flatc --conform` must pass against both).
Supersedes [v1.1](SPEC-v1.1.md). The additions are the query kind `MONOTONE_HULL_PROBABILITY` (§5, §8, §11) and, as an
amendment to 1.2, the per-query literal atoms `literal_mode` and `literal_samples` (§5.1, §5.2, §8, §11).

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
- Changes from v1.1 are wire compatible: one enum value is appended (`QueryKind.MONOTONE_HULL_PROBABILITY = 1`), and
  the literal-atom amendment appends one enum (`LiteralMode`) and two fields to `Query`. Because FlatBuffers appends
  fields and the added enum's default is `ATOM_SPLIT = 0`, a file that does not use them is byte-identical to what the
  same writer produced before the amendment, and a pre-amendment reader ignores them. Changes from v1 to v1.1 are
  listed in [SPEC-v1.1 §11](SPEC-v1.1.md#11-changes-from-v1-summary).
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

enum LiteralMode : ubyte {       // v1.2 amendment
  ATOM_SPLIT = 0,                // default
  OMEGA_L    = 1,
  NEGATION   = 2,
}

table Query {
  target: uint32;                 // tensor index
  kind: QueryKind;                // default PROBABILITY
  name: string;                   // e.g. top-event or sequence name
  confidence: [float64];          // optional requested two-sided confidence levels, e.g. [0.95, 0.99]
  literal_mode: LiteralMode;      // default ATOM_SPLIT; written only when non-default
  literal_samples: [uint32];      // sorted, duplicate-free SAMPLE outputs; written only when non-empty
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

#### 5.1.1 Literal atoms (v1.2 amendment)

Some source formats mark a complemented basic-event operand as an **atomic literal**: cut-set tools built for them
keep `-B` in a cut set as an event of its own (with probability $1 - p_B$) and drop cut sets that contain both $B$ and
`-B`, rather than reading `-B` as the Boolean complement of `B`. The format cannot express that in a Boolean circuit,
so the query names the marked events instead:

- **`literal_samples`** is a **sorted, duplicate-free** list of tensor indices in the same index space as
  `Query.target`. Every listed index MUST be the output of a `SAMPLE` operator. Call the corresponding set of events
  $D$; the list is that set. Indices **outside the cone of `target`** are valid and have no effect, because
  transforms may remove events from a cone (§5.2).
- **`literal_mode`** says how the complemented edges into a listed event are read. Complemented edges into an
  **unlisted** `SAMPLE` are ordinary Boolean complements in every mode; so are complemented edges into gates (including
  single-input gates) in every mode.
- A **missing or empty** `literal_samples` makes every mode the plain hull $\uparrow g$ of §5.1: the query means
  exactly what a v1.2 query meant before this amendment, and files that do not use the fields are unchanged (§2, §10).

For a hull query $q$ with target $g$ and the listed set $D$:

- **`ATOM_SPLIT`** (default). Build $\tilde g$ from the cone of $g$ by replacing every inverted input edge into a
  tensor $s \in D$, within the cone of $g$, with a positive edge into a fresh variable $\bar y_s$; an edge into $s$
  that is not inverted keeps reading $y_s$. Then
  $$\theta = P_X\bigl[\thinspace \exists\, y \le X,\ \exists\, \bar y \le \neg X_D\ :\ \tilde g(y, \bar y) = 1 \thinspace\bigr],$$
  where $\neg X_D$ is the componentwise complement of $X$ on $D$. Equivalently: $\tilde g$ is the plain hull in the
  larger atom space $\mathcal{A} = E \cup \lbrace \bar e : e \in D \rbrace$, read at the coupled valuation
  $\kappa(x)$ with $\kappa(x)_e = x_e$ and $\kappa(x)_{\bar e} = 1 - x_e$. Every sample and every complement atom is
  droppable, with value $X_s$ and $\neg X_s$ respectively.
- **`OMEGA_L`.** $\theta = P_X[\exists\, y \le X \text{ with } y_s = X_s \text{ for all } s \in D\ :\ g(y) = 1]$. The
  listed samples are **fixed**, not droppable; every other sample is droppable. This is Rauzy's widening $\omega_L g$
  with respect to the literals of the events outside $D$ (RFC §3).
- **`NEGATION`.** The list is ignored: $\theta$ is the plain hull $\uparrow g$ of §5.1. It exists so a writer can
  record that it deliberately reads the marked literals as ordinary negations.

In every mode $\theta_{PROBABILITY} \le \theta_{OMEGA\_L} \le \theta_{NEGATION}$ (fixing variables restricts the
existential), and also $\theta_{OMEGA\_L} \le \theta_{ATOM\_SPLIT}$: take an $\omega_L$ witness $y^\* \le X$ with
$y^\*_s = X_s$ on $D$ and set $\bar y_s = \neg X_s = \neg y^\*_s$, so $\tilde g(y^\*, \bar y) = g(y^\*) = 1$. The
chain is therefore $\theta_{PROBABILITY} \le \theta_{OMEGA\_L} \le \theta_{ATOM\_SPLIT}$, with
$\theta_{OMEGA\_L} \le \theta_{NEGATION}$; only `ATOM_SPLIT` against `NEGATION` is unordered in general. An inverted
edge into an unlisted sample is a Boolean complement in every mode, and `always_true` samples and constants behave as
in §4–§5: an `always_true` `SAMPLE` is a droppable event, `BITWISE_ONES` is a constant.

`literal_samples` and a non-default `literal_mode` are only meaningful on a `MONOTONE_HULL_PROBABILITY` query (§8,
rule 8).

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
- **(MUST, `ATOM_SPLIT` with a non-empty list)** A writer MUST NOT create or remove an inverted edge into a listed
  sample. Under `ATOM_SPLIT` each such edge is the atom $\neg x_s$ of its own, droppable independently of $x_s$, so
  removing one (for example by eliminating a single-input gate whose inverted input is a listed `SAMPLE`) or adding one
  (for example by negation normalisation) changes $\tilde g$, hence the answer, even though $g$ is unchanged. A
  transform that only preserves $g$ is not sufficient for these queries; the fuzzybits RFC §7.5 calls the required
  property *atom-preserving*. For `OMEGA_L` and `NEGATION` the listed edges are ordinary complements, so any
  $g$-preserving transform is safe for them.
- The list is part of the query, so it survives transforms that change the circuit: a transform that removes a listed
  event from the cone leaves a valid list entry with no effect (§5.1.1), rather than an invalid file.

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
8. (v1.2 amendment) the literal-atom fields of every `Query` (`UnknownLiteralMode`, `LiteralSampleNotSample`,
   `LiteralSamplesNotAscending`, `LiteralFieldsOnNonHullQuery`, plus `SchemaVersionTooOld` and
   `TensorIndexOutOfRange` where they apply):
   - `literal_mode` is a value of `LiteralMode`;
   - every index in `literal_samples` is in range and is the `output` of a `SAMPLE` operator;
   - `literal_samples` is strictly ascending;
   - `literal_samples` is non-empty, or `literal_mode` is non-default, only on a `MONOTONE_HULL_PROBABILITY` query;
   - a query with either field set declares `schema_version` minor ≥ 2. For a non-empty list this already follows
     from the hull kind (rule 7); the field check is independent so that a non-default mode with an empty list is
     caught too.

   Indices **outside the target's cone are valid** and have no effect: transforms may remove events from a cone
   (§5.2), so a listed sample the target stops depending on is not an error. Runtimes MAY ignore such entries.

Runtimes (normative): before evaluating any query, a runtime MUST check that it implements the kind of every query in
the file, and MUST reject the file if it does not (at planning or load time, not after sampling). A runtime MUST NOT
answer a query of one kind with the value of another; in particular it MUST NOT evaluate `MONOTONE_HULL_PROBABILITY`
as `PROBABILITY`. A query kind the runtime does not know (rule 7) is a kind it does not implement.

The same holds per `literal_mode`, but the obligation is scoped to the queries where the mode can change the answer:

- a query whose `literal_samples` is **empty** is the plain hull $\uparrow g$ whatever its `literal_mode` (§5.1.1).
  A runtime that implements the plain hull MUST answer it, and MUST NOT reject it for a mode it does not implement.
  The same holds for a `PROBABILITY` query, which never carries the fields. A pre-amendment hull file — no
  `literal_samples`, or a non-default `literal_mode` alone — is therefore readable by a plain-hull runtime;
- a runtime **MUST NOT** answer a query with a non-empty `literal_samples` under a mode it does not implement, and
  MUST NOT answer a mode with the value of another mode (in particular not `ATOM_SPLIT` as `NEGATION`);
- a runtime **MAY** accept a query whose listed samples are all outside the target's cone: every mode then equals the
  plain hull (§5.2), so the mode cannot change the answer. Such a runtime answers the plain hull and records that it
  ignored the mode; it does not have to implement it;
- an unknown `literal_mode` value is an unknown *enum* value, not an unknown field, so §2's rule applies to the file:
  reject, do not guess.

`pcfb::literals_require_mode(query)` states the first three rules in one place: it is false when `literal_samples` is
absent or empty, true otherwise, so a runtime rejects exactly when it returns true and it lacks the mode.

## 9. Reference API (C++, `pcfb` library)

- `pcfb::io::read(path) / from_bytes(span)` → immutable `pcfb::ir::Circuit` view. Query readers expose both literal
  fields: `Query::literal_mode()` and `Query::literal_samples()` on the FlatBuffer, and
  `pcfb::read_literals(query)` → `pcfb::query_literals { mode, samples }` (an absent list reads as empty).
- `pcfb::builder`: `add_sample(name, p, event_id, source) → tensor`, `add_op(opcode, inputs, inverted, k?) → tensor`,
  `add_query(target, kind, name, confidence, literals)`, `set(key, value)`, `finish() → std::vector<std::byte>`; sets
  required metadata, computes thresholds, enforces ordering, rejects query kinds it does not define, rejects the
  literal fields rule 8 forbids, and writes the lowest `schema_version` that covers the query kinds used
  (`pcfb::kSchemaVersionV11` or `pcfb::kSchemaVersionV12`). It writes `literal_mode` only when non-default and
  `literal_samples` only when non-empty, so a query without literal atoms serialises exactly as before the amendment.
- `pcfb::validate(const ir::Circuit&) → diagnostics` implementing §8; rule 7 reports `UnknownQueryKind` and
  `SchemaVersionTooOld`, rule 8 reports the five codes named there (all appended after the v1.1 codes, which keep their
  values).
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
`schema_version` `0x00010001`.

(7) `literal_atoms.pcfb` (v1.2 amendment, `schema_version` `0x00010002`) — the literal-atom fixture, built by
`tools/make_fixtures.cpp`. `TOP = AND(C, ¬NO_OP(¬B))`: the inner `¬B` is an inverted input on a single-input
`BITWISE_NO_OP` gate (a gate complement, not a direct complement), the outer `¬` is an inverted input into that gate,
`p_B = 0.25`, `p_C = 0.5`, and `B` is the only entry of `literal_samples`. Four queries share the target:

| Query | Kind | Mode | Δ | Exact value |
|---|---|---|---|---|
| `top_boolean` | `PROBABILITY` | — | — | 0.125 = P(B)P(C) |
| `top_atom_split` | `MONOTONE_HULL_PROBABILITY` | `ATOM_SPLIT` | {B} | 0.5 = P(C) |
| `top_omega_l` | `MONOTONE_HULL_PROBABILITY` | `OMEGA_L` | {B} | 0.125 = P(B)P(C) |
| `top_negation` | `MONOTONE_HULL_PROBABILITY` | `NEGATION` | list ignored | 0.125 = P(B)P(C) |

All four values are exact in binary64. Derivation: $g = c \wedge b$; splitting gives $\tilde g = c \wedge \neg \bar b$,
whose atom hull is $c$ (a witness drops the complement atom), so `ATOM_SPLIT` is $P(c)$; under `OMEGA_L` the event $b$
is fixed and $g$ is monotone in $c$, so $\omega_L g = g$; under `NEGATION` the plain hull of $b \wedge c$ is itself.
It also carries the ordering $\theta_{PROBABILITY} \le \theta_{OMEGA\_L} \le \theta_{NEGATION}$.

`tests/exact_eval.cpp` holds test-only exhaustive evaluators for the plain hull and for all three literal modes.
`tests/test_literal_hull.cpp` checks the two hand-computed vectors, the cross-identities of §5.1 on 500 seeded
circuits (Δ = ∅ gives the plain hull; `OMEGA_L` with Δ = every sample gives `PROBABILITY`; `NEGATION` ignores Δ;
$\theta_{PROBABILITY} \le \theta_{OMEGA\_L} \le \theta_{NEGATION}$), and the worked examples of the fuzzybits RFC §8.11
(`P1`: Boolean 0.2736, `ATOM_SPLIT` 0.316, `OMEGA_L` 0.296, plain hull 0.37; `P4`: 0.3 / 0.5 / 0.3 / 0.3).

Fixtures (1)–(6) are byte-identical to before the amendment: the new `Query` fields are written only when non-default
or non-empty, and `delete_term.pcfb`'s two hull queries still carry neither.

## 11. Changes from v1.1 (summary)

| Area | v1.1 | v1.2 |
|---|---|---|
| Query kinds | `PROBABILITY` | `PROBABILITY`, `MONOTONE_HULL_PROBABILITY` (appended, value 1) |
| `schema_version` | `0x00010001` | `0x00010002` when a v1.2 kind is used; otherwise unchanged |
| Opset | `core-1.1` | `core-1.1` (no new operators) |
| Validation | rules 1–6 | rule 7: known kinds, minimum minor per kind |
| Runtimes | — | MUST reject files with kinds they do not implement; MUST NOT substitute kinds |
| Frozen schemas | v1 | v1, v1.1 (`schema/history/pcfb_v1_1.fbs`) |
| Literal atoms (amendment) | — | `Query.literal_mode` (`LiteralMode`, default `ATOM_SPLIT`) and `Query.literal_samples`, both appended; §5.1.1 |
| Validation (amendment) | — | rule 8: `UnknownLiteralMode`, `LiteralSampleNotSample`, `LiteralSamplesNotAscending`, `LiteralFieldsOnNonHullQuery` |
| Writers (amendment) | — | MUST NOT create or remove inverted edges into listed samples under `ATOM_SPLIT` (§5.2) |
| Runtimes (amendment) | — | MUST reject a `literal_mode` they do not implement **only when `literal_samples` is non-empty**; MUST NOT substitute modes; a query with an empty list is the plain hull whatever its mode, so a plain-hull runtime answers it |
| Fixtures (amendment) | — | `literal_atoms.pcfb` (§10); every earlier fixture byte-identical |
