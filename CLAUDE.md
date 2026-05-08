# loquela

A RISC-V zkVM built on top of [Plonky3](https://github.com/Plonky3/Plonky3) (`p3-*` crates, version 0.5.2).

## Workspace layout

- [crates/vm](crates/vm/) — `loquela-vm`: the executor. Defines `Instruction`, `VMState`, `MemoryOperation`, and the `VM` struct that runs a program and records a trace of `ExecutionStep` (state + instruction word + decoded instruction + memory ops) per step. Supported instructions: `AddI { rd, rs1, imm }` (I-type, opcode 0x13 funct3=0x0), `XorI { rd, rs1, imm }` (I-type, opcode 0x13 funct3=0x4), `Add { rd, rs1, rs2 }` (R-type, opcode 0x33 funct3=0x0 funct7=0x0).
- [crates/air](crates/air/) — `loquela-air`: the proving side. One AIR per concern, wired together through Plonky3's lookup buses. Modules:
  - [boundaries](crates/air/src/boundaries/air.rs) — 4-row AIR enforcing initial and final execution state. Row 0 sends `(pc=0, ts=0)` on the `"trace"` bus; row 3 receives the final `(pc, ts)` from the `"trace"` bus. Uses a preprocessed `(is_first, is_last)` selector pair.
  - [decode](crates/air/src/decode/air.rs) — instruction decoder. Bit-decomposes each instruction word, asserts opcode/funct3/funct7 patterns, sets one-hot `is_addi`/`is_xori`/`is_add` flags, packs `rd` / `rs1` / `imm` / `rs2`. Sends 4 byte lookups to the `"program"` bus and exposes decoded fields on the `"decode"` bus.
  - [instructions/addi](crates/air/src/instructions/addi/air.rs) — ADDI per-instruction AIR. Receives `(pc, ts)` on `"trace"`, reads rs1 and writes rd via `"memory"`, performs sign-extended u32 addition with byte-level carry constraints via `"bytes"`.
  - [instructions/xori](crates/air/src/instructions/xori/air.rs) — XORI per-instruction AIR. Same structure as ADDI but uses the `"bytes_xor"` bus for byte-wise XOR.
  - [instructions/add](crates/air/src/instructions/add/air.rs) — ADD (R-type) per-instruction AIR. Receives `(pc, ts)` on `"trace"`, reads rs1 (ts) and rs2 (ts+1), writes rd (ts+2) via `"memory"`, performs u32 addition with carry via `"bytes"`.
  - [memory](crates/air/src/memory/air.rs) — sorted memory log over `(memory_type, address, timestamp)`. `u32_lt` lookups enforce sort order; a `"memory"` bus exposes individual operations to the instruction AIRs.
  - [program](crates/air/src/program/air.rs) — preprocessed program ROM (`address`, `value` byte, `mult`). Receives byte queries from the `"program"` bus.
  - [primitives](crates/air/src/primitives/) — reusable lookup tables: `byte_lookup` (range-check 0..256), `byte_less_than_lookup` (pairs `x<y`), `xor_lookup`, `u32_less_than_lookup`, `timestamp_less_than_lookup`, plus `bit_decompose` and `u32_ops` helpers.
- [crates/prover](crates/prover/) — `loquela-prover`: wires all AIRs together. Defines `LoquelAir` enum (one variant per AIR), drives `prove_batch` using the Mersenne31 / Circle-STARK / FRI / Keccak Merkle config.
- [guest-programs](guest-programs/) — RISC-V assembly guests (`test.s` / `test.bin`) that run on the VM.

## What is an AIR (in this project)

An **AIR** (Algebraic Intermediate Representation) is a rectangular table of field elements:

- **Columns** are fixed at circuit definition time; each has a semantic role (`pc`, `address`, `is_addi`, …).
- **Rows** are chosen by the prover and **must be a power of two**. The prover picks every cell value freely; only what constraints explicitly forbid is ruled out. **A column called `pc` is just a field element unless a constraint forces it to act like a program counter.**

The native field is `Mersenne31` (`p = 2^31 - 1`). It is a small (31-bit) field, so overflow and wraparound matter. Multi-byte values like `address` and `timestamp` are stored as 4 separate byte limbs (`[F; 4]`) to stay range-checkable.

### Constraint scopes (Plonky3 `AirBuilder`)

Each AIR implements `Air<AB: AirBuilder>::eval`. Constraints are polynomial identities; their meaning depends on **scope**:

- **Default scope** — checked on every row, *including* the cyclic `(last → first)` edge. A constraint reading `next.x` becomes a cyclic adjacent-row constraint. Used here for things like `assert_bool(local.is_memory_type_equal)`.
- **`when_transition()`** — every adjacent pair *except* the `(last, first)` wrap.
- **`when_first_row()`** — boundary at row 0.
- **`when_last_row()`** — boundary at row N-1.
- **`when(x)` / `when_ne(x, y)`** — guards. `when(x)` multiplies the constraint by `x`, so it's enforced wherever `x != 0`. The guard does not have to be boolean for the implication to hold, but if `x` is also used as a bus multiplicity or one-hot selector elsewhere, you still need a separate `assert_bool` on it.

### Trace-row layout pattern

Every AIR in this repo follows the same column pattern: a `#[repr(C)]` struct (e.g. `BoundaryColumns`, `DecodeColumns`, `AddiColumns`, `AddColumns`, `MemoryColumns`, `ProgramColumns`) plus `Borrow`/`BorrowMut` impls so a `&[F]` row slice can be reinterpreted as the typed struct. `NUM_*_COLS` is computed via `size_of::<…<u8>>()`. When adding columns, update the struct and re-check `width()`.

## Lookups: Plonky3's `p3-lookup` interface (NOT OpenVM)

Plonky3 exposes a **single bus model** built around `LookupAir`, not separate `LookupBus` / `PermutationCheckBus` types.

### The `LookupAir` trait

Every AIR that participates in lookups implements:

```rust
impl<F: Field> LookupAir<F> for SomeAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> { /* hands out fresh lookup-column indices */ }
    fn get_lookups(&mut self) -> Vec<Lookup<F>> { /* declares this AIR's lookups */ }
}
```

`get_lookups` builds lookups over a `SymbolicAirBuilder` so they are expressed in terms of symbolic main / preprocessed columns of the AIR itself.

### Declaring a lookup

```rust
self.register_lookup(
    Kind::Global(String::from("bytes")),
    &vec![(
        vec![ /* tuple of value expressions, in this bus's schema */ ],
        /* multiplicity expression */,
        Direction::Send | Direction::Receive,
    )],
);
```

- **`Kind::Global("name")`** — names the bus. All AIRs that reference the same name talk to the same bus. Buses in this repo: `"trace"`, `"bytes"`, `"bytes_lt"`, `"bytes_xor"`, `"u32_lt"`, `"program"`, `"decode"`, `"memory"`.
- **Tuple of values** — the message contents. Every send/receive on a given bus must agree on tuple length and field meaning; mismatches are silent semantic bugs.
- **Multiplicity** — an arbitrary field expression (a column, a constant like `F::ONE`, or a polynomial). It is **not** automatically boolean-constrained.
- **Direction** — `Send` contributes `+multiplicity`, `Receive` contributes `-multiplicity`.

### Soundness rule

The proof system enforces that, for each `Kind`, **the global sum of `±multiplicity` across all rows of all participating AIRs is zero**. In effect: the multiset (with multiplicities) of sent tuples equals the multiset of received tuples. There is no separate "lookup vs. permutation" semantics — both collapse to multiset equality on a shared bus.

Convention used in this repo:

- A *table provider* (preprocessed bus, e.g. `BytesAir`, `LessThanAir`, `ProgramAir`) does **`Direction::Receive`** with a main-trace `mult` column. That column counts how many times the row is consumed.
- A *consumer* does **`Direction::Send`** with multiplicity `F::ONE`, a boolean `enabled` flag, or a logical predicate.
- For sort/range checks (`memory` AIR), the multiplicity is itself a logical column like `is_memory_type_equal` or `is_address_equal`. Boolean-constrain those columns where they are used as `±1` selectors, otherwise the prover can forge interactions.

### What this means in practice

- An unconstrained or non-boolean multiplicity is **not automatically a bug** (it is sound for table-side `Receive` columns). It **is** a bug whenever the multiplicity is a "send a query" flag — those must be boolean.
- Bus-side schema is by position: order and arity of the tuple is the contract. When auditing, line up every `register_lookup` for a given `Kind` and check tuple shape.
- `add_lookup_columns` and `num_lookups` are bookkeeping for column allocation; they are *not* the lookup semantics themselves.

## Proving stack

`loquela-prover` wires all AIRs through `prove_batch`:

- Field: `Mersenne31`, extension `BinomialExtensionField<_, 3>`.
- Hash: `Keccak256Hash` → `SerializingHasher` field hash + `CompressionFunctionFromHasher` Merkle compression.
- MMCS: `MerkleTreeMmcs` (val) + `ExtensionMmcs` (challenge).
- PCS: `CirclePcs` with FRI parameters (`log_blowup=1`, etc.) — note the parameters in-tree are **test-grade**, ~3 bits of security.
- Challenger: `SerializingChallenger32` over `HashChallenger`.
- All AIR variants are bundled in `LoquelAir` enum in `crates/prover/src/lib.rs`; dispatch lives there.

## Testing

Always run tests in release mode: `cargo test --release`. The prover tests involve STARK proving and are prohibitively slow in debug mode.

## Conventions and gotchas

- Always re-derive `NUM_*_COLS = size_of::<…<u8>>()` and the `Borrow` impls when changing a column struct, otherwise the row reinterpretation will silently misalign.
- Trace height must be a power of two — the VM/test harness pads accordingly.
- Memory operations are sorted by `(memory_type, address, timestamp)`; the `MemoryAir` only enforces the sort, the actual log is exposed via the `"memory"` bus to the instruction AIRs.
- `MemoryType` is `Register = 0`, `Ram = 1` — keep these in sync with any AIR that compares the column directly.
- Adding a new instruction requires: extending `Instruction` in the VM, adding an `is_*` flag column + one-hot constraint in the decode AIR, encoding its opcode/funct3/funct7 bit pattern, including it in `instr_type_packed`, creating a new per-instruction AIR under `crates/air/src/instructions/<name>/`, and wiring it into `LoquelAir` in the prover.
- Each per-instruction AIR uses an `is_dummy` column to pad the trace to a power of two without sending spurious bus messages.

## Commits
- Use clear, short commit messages describing what was done. Reference existing commit messages for examples of good style.
- Before committing, always run cargo fmt and cargo clippy to ensure code quality and consistency.