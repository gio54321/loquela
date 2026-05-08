use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_batch_stark::{prove_batch, BatchProof, CommonData, ProverData, StarkInstance};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::{
    extension::BinomialExtensionField, integers::QuotientMap, Field, PrimeCharacteristicRing,
};
use p3_fri::FriParameters;
use p3_keccak::Keccak256Hash;
use p3_lookup::{Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::StarkConfig;

use loquela_air::boundaries::air::BoundariesAir;
use loquela_air::decode::air::DecodeAir;
use loquela_air::instructions::add::air::AddAir;
use loquela_air::instructions::addi::air::AddiAir;
use loquela_air::instructions::and::air::AndInstrAir;
use loquela_air::instructions::sll::air::SllAir;
use loquela_air::instructions::xori::air::XoriAir;
use loquela_air::memory::air::MemoryAir;
use loquela_air::primitives::and_lookup::AndAir;
use loquela_air::primitives::byte_less_than_lookup::LessThanAir;
use loquela_air::primitives::byte_lookup::BytesAir;
use loquela_air::primitives::byte_shift_left_lookup::ByteShiftLeftAir;
use loquela_air::primitives::timestamp_less_than::TimestampLessThanAir;
use loquela_air::primitives::u32_less_than_lookup::U32LessThanAir;
use loquela_air::primitives::xor_lookup::XorAir;
use loquela_air::program::air::ProgramAir;
use loquela_vm::{Instruction, MemoryOperation, VM};

// ── Config ────────────────────────────────────────────────────────────────────

type Val = Mersenne31;
type Challenge = BinomialExtensionField<Val, 3>;
type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

fn build_config() -> MyConfig {
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = MyCompress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress, 1);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 2,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = CirclePcs::new(val_mmcs, fri_params);
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    MyConfig::new(pcs, challenger)
}

// ── LoquelAir enum ────────────────────────────────────────────────────────────

/// Wraps every AIR variant so all instances share a single type for `prove_batch`.
#[derive(Clone)]
pub enum LoquelAir {
    Boundaries(BoundariesAir),
    Decode(DecodeAir),
    Add(AddAir),
    Addi(AddiAir),
    Xori(XoriAir),
    AndInstr(AndInstrAir),
    Sll(SllAir),
    Memory(MemoryAir),
    Program(ProgramAir),
    Bytes(BytesAir),
    Xor(XorAir),
    And(AndAir),
    ByteSll(ByteShiftLeftAir),
    U32Lt(U32LessThanAir),
    TimestampLt(TimestampLessThanAir),
    BytesLt(LessThanAir),
}

impl<F: Field> BaseAir<F> for LoquelAir {
    fn width(&self) -> usize {
        match self {
            LoquelAir::Boundaries(a) => BaseAir::<F>::width(a),
            LoquelAir::Decode(a) => BaseAir::<F>::width(a),
            LoquelAir::Add(a) => BaseAir::<F>::width(a),
            LoquelAir::Addi(a) => BaseAir::<F>::width(a),
            LoquelAir::Xori(a) => BaseAir::<F>::width(a),
            LoquelAir::AndInstr(a) => BaseAir::<F>::width(a),
            LoquelAir::Sll(a) => BaseAir::<F>::width(a),
            LoquelAir::Memory(a) => BaseAir::<F>::width(a),
            LoquelAir::Program(a) => BaseAir::<F>::width(a),
            LoquelAir::Bytes(a) => BaseAir::<F>::width(a),
            LoquelAir::Xor(a) => BaseAir::<F>::width(a),
            LoquelAir::And(a) => BaseAir::<F>::width(a),
            LoquelAir::ByteSll(a) => BaseAir::<F>::width(a),
            LoquelAir::U32Lt(a) => BaseAir::<F>::width(a),
            LoquelAir::TimestampLt(a) => BaseAir::<F>::width(a),
            LoquelAir::BytesLt(a) => BaseAir::<F>::width(a),
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            LoquelAir::Boundaries(a) => a.preprocessed_trace(),
            LoquelAir::Bytes(a) => a.preprocessed_trace(),
            LoquelAir::Xor(a) => a.preprocessed_trace(),
            LoquelAir::And(a) => a.preprocessed_trace(),
            LoquelAir::ByteSll(a) => a.preprocessed_trace(),
            LoquelAir::BytesLt(a) => a.preprocessed_trace(),
            _ => None,
        }
    }
}

impl<AB> Air<AB> for LoquelAir
where
    AB: AirBuilder,
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: Field + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            LoquelAir::Boundaries(a) => a.eval(builder),
            LoquelAir::Decode(a) => a.eval(builder),
            LoquelAir::Add(a) => a.eval(builder),
            LoquelAir::Addi(a) => a.eval(builder),
            LoquelAir::Xori(a) => a.eval(builder),
            LoquelAir::AndInstr(a) => a.eval(builder),
            LoquelAir::Sll(a) => a.eval(builder),
            LoquelAir::Memory(a) => a.eval(builder),
            LoquelAir::Program(a) => a.eval(builder),
            LoquelAir::Bytes(a) => a.eval(builder),
            LoquelAir::Xor(a) => a.eval(builder),
            LoquelAir::And(a) => a.eval(builder),
            LoquelAir::ByteSll(a) => a.eval(builder),
            LoquelAir::U32Lt(a) => a.eval(builder),
            LoquelAir::TimestampLt(a) => a.eval(builder),
            LoquelAir::BytesLt(a) => a.eval(builder),
        }
    }
}

impl<F: Field> LookupAir<F> for LoquelAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        match self {
            LoquelAir::Boundaries(a) => <BoundariesAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Decode(a) => <DecodeAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Add(a) => <AddAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Addi(a) => <AddiAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Xori(a) => <XoriAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::AndInstr(a) => <AndInstrAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Sll(a) => <SllAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Memory(a) => <MemoryAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Program(a) => <ProgramAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Bytes(a) => <BytesAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Xor(a) => <XorAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::And(a) => <AndAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::ByteSll(a) => <ByteShiftLeftAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::U32Lt(a) => <U32LessThanAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::TimestampLt(a) => {
                <TimestampLessThanAir as LookupAir<F>>::add_lookup_columns(a)
            }
            LoquelAir::BytesLt(a) => <LessThanAir as LookupAir<F>>::add_lookup_columns(a),
        }
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        match self {
            LoquelAir::Boundaries(a) => <BoundariesAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Decode(a) => <DecodeAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Add(a) => <AddAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Addi(a) => <AddiAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Xori(a) => <XoriAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::AndInstr(a) => <AndInstrAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Sll(a) => <SllAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Memory(a) => <MemoryAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Program(a) => <ProgramAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Bytes(a) => <BytesAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Xor(a) => <XorAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::And(a) => <AndAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::ByteSll(a) => <ByteShiftLeftAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::U32Lt(a) => <U32LessThanAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::TimestampLt(a) => <TimestampLessThanAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::BytesLt(a) => <LessThanAir as LookupAir<F>>::get_lookups(a),
        }
    }
}

// ── AllTraces ─────────────────────────────────────────────────────────────────

/// All witness matrices for a single execution, bundled with their AIR instances.
pub struct AllTraces {
    pub airs: Vec<LoquelAir>,
    pub boundaries: RowMajorMatrix<Val>,
    pub decode: RowMajorMatrix<Val>,
    pub memory: RowMajorMatrix<Val>,
    pub program: RowMajorMatrix<Val>,
    pub bytes: RowMajorMatrix<Val>,
    pub xor: RowMajorMatrix<Val>,
    pub u32_lt: RowMajorMatrix<Val>,
    pub timestamp_lt: RowMajorMatrix<Val>,
    pub bytes_lt: RowMajorMatrix<Val>,
    /// Present when the program contains ADD instructions.
    pub add: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains ADDI instructions.
    pub addi: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains XORI instructions.
    pub xori: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains AND instructions (instruction AIR).
    pub and_instr: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains AND instructions (bytes_and lookup table).
    pub and_prim: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SLL instructions (instruction AIR).
    pub sll: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SLL instructions (byte_sll lookup table).
    pub byte_sll: Option<RowMajorMatrix<Val>>,
}

impl AllTraces {
    /// Flatten into parallel vecs in the same order as `airs`.
    pub fn into_vecs(self) -> (Vec<LoquelAir>, Vec<RowMajorMatrix<Val>>) {
        let AllTraces {
            airs,
            boundaries,
            decode,
            memory,
            program,
            bytes,
            xor,
            u32_lt,
            timestamp_lt,
            bytes_lt,
            add,
            addi,
            xori,
            and_instr,
            and_prim,
            sll,
            byte_sll,
        } = self;
        let mut traces = vec![
            boundaries,
            decode,
            memory,
            program,
            bytes,
            xor,
            u32_lt,
            timestamp_lt,
            bytes_lt,
        ];
        if let Some(t) = add {
            traces.push(t);
        }
        if let Some(t) = addi {
            traces.push(t);
        }
        if let Some(t) = xori {
            traces.push(t);
        }
        if let Some(t) = and_instr {
            traces.push(t);
        }
        if let Some(t) = and_prim {
            traces.push(t);
        }
        if let Some(t) = sll {
            traces.push(t);
        }
        if let Some(t) = byte_sll {
            traces.push(t);
        }
        (airs, traces)
    }
}

// ── Multiplicity helpers ───────────────────────────────────────────────────────

/// Count byte occurrences across all u32 values that are byte-range-checked.
/// Returns `mults[v]` = number of times byte value `v` appeared.
fn bytes_multiplicities(vals: &[u32]) -> [Val; 256] {
    let mut mults = [Val::ZERO; 256];
    for v in vals {
        for byte in v.to_le_bytes() {
            mults[byte as usize] += Val::ONE;
        }
    }
    mults
}

/// Count XOR triples from XORI steps: row index = x * 256 + y.
fn xor_multiplicities(xori_ops: &[(u32, u32, u32)]) -> Vec<Val> {
    let mut mults = vec![Val::ZERO; 256 * 256];
    for &(x, y, _) in xori_ops {
        mults[x as usize * 256 + y as usize] += Val::ONE;
    }
    mults
}

/// Count AND triples from AND steps: row index = x * 256 + y.
fn and_multiplicities(and_ops: &[(u32, u32, u32)]) -> Vec<Val> {
    let mut mults = vec![Val::ZERO; 256 * 256];
    for &(x, y, _) in and_ops {
        mults[x as usize * 256 + y as usize] += Val::ONE;
    }
    mults
}

/// Collect sorted memory ops and derive all u32_lt and timestamp_lt entries,
/// plus the byte-level lt multiplicities consumed by LessThanAir.
fn memory_lookup_entries(
    all_ops: &[MemoryOperation],
) -> (
    Vec<(u32, u32, Val)>, // u32_lt entries: (addr_x, addr_y, mult)
    Vec<(u32, u32, Val)>, // timestamp_lt entries: (ts_x, ts_y, mult)
    Vec<Val>,             // bytes_lt mults (length <= 32640)
) {
    struct Op {
        memory_type: u8,
        address: u32,
        timestamp: u32,
    }
    let mut ops: Vec<Op> = all_ops
        .iter()
        .map(|op| match op {
            MemoryOperation::Read {
                memory_type,
                address,
                timestamp,
                ..
            } => Op {
                memory_type: *memory_type as u8,
                address: *address,
                timestamp: *timestamp,
            },
            MemoryOperation::Write {
                memory_type,
                address,
                timestamp,
                ..
            } => Op {
                memory_type: *memory_type as u8,
                address: *address,
                timestamp: *timestamp,
            },
        })
        .collect();
    ops.sort_by_key(|op| (op.memory_type, op.address, op.timestamp));

    let mut u32_lt_entries: Vec<(u32, u32, Val)> = Vec::new();
    let mut timestamp_lt_entries: Vec<(u32, u32, Val)> = Vec::new();
    let mut bytes_lt_mults_map = std::collections::HashMap::<(u8, u8), u64>::new();

    for i in 0..ops.len().saturating_sub(1) {
        let cur = &ops[i];
        let nxt = &ops[i + 1];

        let same_type = cur.memory_type == nxt.memory_type;
        if !same_type {
            continue;
        }

        let is_addr_equal = cur.address == nxt.address;
        u32_lt_entries.push((cur.address, nxt.address, Val::ONE));

        if !is_addr_equal {
            let x_bytes = cur.address.to_le_bytes();
            let y_bytes = nxt.address.to_le_bytes();
            for b in (0..4).rev() {
                if x_bytes[b] != y_bytes[b] {
                    let (lo, hi) = (x_bytes[b].min(y_bytes[b]), x_bytes[b].max(y_bytes[b]));
                    *bytes_lt_mults_map.entry((lo, hi)).or_insert(0) += 1;
                    break;
                }
            }
        }

        if is_addr_equal {
            timestamp_lt_entries.push((cur.timestamp, nxt.timestamp, Val::ONE));
        }
    }

    let num_valid = 256 * 255 / 2;
    let mut bytes_lt_mults = vec![Val::ZERO; num_valid];
    for ((x, y), count) in &bytes_lt_mults_map {
        let row = (*y as usize) * (*y as usize - 1) / 2 + (*x as usize);
        bytes_lt_mults[row] = Val::from_u64(*count);
    }

    (u32_lt_entries, timestamp_lt_entries, bytes_lt_mults)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Execute `program` and build all witness matrices (without proving).
///
/// The returned `AllTraces` can be inspected or mutated before passing to
/// `prove_traces`, which is useful for negative testing.
pub fn generate_traces(program: &[u8]) -> AllTraces {
    // 1. Execute the VM.
    let mut vm = VM::new(program.to_vec());
    vm.run().expect("VM execution failed");

    let steps = &vm.trace;
    println!("VM executed in {} steps", steps.len());
    let all_ops: Vec<MemoryOperation> = vm.get_memory_ops().into_iter().cloned().collect();

    // 2. Classify steps by instruction type.
    let has_add = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Add { .. }));
    let has_addi = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::AddI { .. }));
    let has_xori = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::XorI { .. }));
    let has_and = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::And { .. }));
    let has_sll = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Sll { .. }));

    // 3. Build traces.
    println!("Building AIR instances and traces...");

    let final_step = steps.last().expect("no steps");
    let final_pc_bytes = (final_step.state.pc + 4).to_le_bytes();
    let final_pc: [Val; 4] = final_pc_bytes.map(|b| Val::from_u64(b as u64));
    let final_ts = Val::from_u64(vm.timestamp as u64);
    let boundaries = loquela_air::boundaries::air::build_trace(final_pc, final_ts);

    let decode = loquela_air::decode::trace::build_trace::<Val>(steps);

    let add_trace = if has_add {
        Some(loquela_air::instructions::add::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let addi_trace = if has_addi {
        Some(loquela_air::instructions::addi::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let xori_trace = if has_xori {
        Some(loquela_air::instructions::xori::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let and_instr_trace = if has_and {
        Some(loquela_air::instructions::and::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let sll_trace = if has_sll {
        Some(loquela_air::instructions::sll::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };

    let memory = loquela_air::memory::trace::build_trace::<Val>(&all_ops);

    let n_decode_steps = steps.len();
    let num_decode_padding = n_decode_steps
        .next_power_of_two()
        .saturating_sub(n_decode_steps);
    let program_trace =
        loquela_air::program::trace::build_trace::<Val>(program, steps, num_decode_padding);

    let (u32_lt_entries, timestamp_lt_entries, bytes_lt_mults) = memory_lookup_entries(&all_ops);

    // Collect all u32 values that are byte-range-checked by ADDI (rs1, rd_new),
    // ADD (rs1, rs2, rd_new), and SLL (rs2_value, rd_new; plus rs2_shamt_high scalar).
    let mut byte_checked_vals: Vec<u32> = Vec::new();
    for s in steps.iter() {
        match s.memory_ops.as_slice() {
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::AddI { .. }) =>
            {
                byte_checked_vals.push(*rs1);
                byte_checked_vals.push(*rd);
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Add { .. }) =>
            {
                byte_checked_vals.push(*rs1);
                byte_checked_vals.push(*rs2);
                byte_checked_vals.push(*rd);
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Sll { .. }) =>
            {
                // SLL byte-range checks: rs2_value bytes, rd bytes.
                // rs1 bytes are covered by the byte_sll lookup.
                // rs2_shamt_high is a small value (0..7) also range-checked individually.
                byte_checked_vals.push(*rs2);
                byte_checked_vals.push(*rd);
                // rs2_shamt_high: upper 3 bits of rs2 low byte
                let rs2_shamt_high = (*rs2 & 0xFF) >> 5;
                byte_checked_vals.push(rs2_shamt_high);
            }
            _ => {}
        }
    }
    let bytes_mults = bytes_multiplicities(&byte_checked_vals);
    let bytes = loquela_air::primitives::byte_lookup::build_trace::<Val>(&bytes_mults);

    let mut xori_triples: Vec<(u32, u32, u32)> = Vec::new();
    for s in steps.iter() {
        let imm = match s.instruction {
            Instruction::XorI { imm, .. } => imm,
            _ => continue,
        };
        let imm_se = imm as i32 as u32;
        if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }] =
            s.memory_ops.as_slice()
        {
            let rs1_b = rs1.to_le_bytes();
            let imm_b = imm_se.to_le_bytes();
            let rd_b = rd.to_le_bytes();
            for i in 0..4 {
                xori_triples.push((rs1_b[i] as u32, imm_b[i] as u32, rd_b[i] as u32));
            }
        }
    }
    let xor_mults = xor_multiplicities(&xori_triples);
    let xor = loquela_air::primitives::xor_lookup::build_trace::<Val>(&xor_mults);

    let mut and_triples: Vec<(u32, u32, u32)> = Vec::new();
    for s in steps.iter() {
        if !matches!(s.instruction, Instruction::And { .. }) {
            continue;
        }
        if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }] =
            s.memory_ops.as_slice()
        {
            let rs1_b = rs1.to_le_bytes();
            let rs2_b = rs2.to_le_bytes();
            let rd_b = rd.to_le_bytes();
            for i in 0..4 {
                and_triples.push((rs1_b[i] as u32, rs2_b[i] as u32, rd_b[i] as u32));
            }
        }
    }
    let and_prim_trace = if has_and {
        let and_mults = and_multiplicities(&and_triples);
        Some(loquela_air::primitives::and_lookup::build_trace::<Val>(
            &and_mults,
        ))
    } else {
        None
    };

    // Compute byte_sll multiplicities for SLL instructions.
    // For each SLL step, we emit 4 lookups: (rs1_bytes[i], bit_shamt, shifted, carry).
    // Row index in the byte_sll table = byte_val * 8 + bit_shamt.
    let byte_sll_prim_trace = if has_sll {
        let mut byte_sll_mults = vec![Val::ZERO; 256 * 8];
        for s in steps.iter() {
            if !matches!(s.instruction, Instruction::Sll { .. }) {
                continue;
            }
            if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, ..] =
                s.memory_ops.as_slice()
            {
                let shamt = rs2 & 0x1F;
                let bit_shamt = (shamt % 8) as usize;
                let rs1_b = rs1.to_le_bytes();
                for i in 0..4 {
                    let byte_val = rs1_b[i] as usize;
                    let row = byte_val * 8 + bit_shamt;
                    byte_sll_mults[row] += Val::ONE;
                }
            }
        }
        Some(loquela_air::primitives::byte_shift_left_lookup::build_trace::<Val>(&byte_sll_mults))
    } else {
        None
    };

    let u32_lt = loquela_air::primitives::u32_less_than_lookup::build_trace::<Val>(&u32_lt_entries);
    let timestamp_lt =
        loquela_air::primitives::timestamp_less_than::build_trace::<Val>(&timestamp_lt_entries);
    let bytes_lt =
        loquela_air::primitives::byte_less_than_lookup::build_trace::<Val>(&bytes_lt_mults);

    // 4. Assemble AIRs in the same order as the traces vec built in `into_vecs`.
    let mut airs: Vec<LoquelAir> = vec![
        LoquelAir::Boundaries(BoundariesAir::new()),
        LoquelAir::Decode(DecodeAir::new()),
        LoquelAir::Memory(MemoryAir::new()),
        LoquelAir::Program(ProgramAir::new()),
        LoquelAir::Bytes(BytesAir::new()),
        LoquelAir::Xor(XorAir::new()),
        LoquelAir::U32Lt(U32LessThanAir::new()),
        LoquelAir::TimestampLt(TimestampLessThanAir::new()),
        LoquelAir::BytesLt(LessThanAir::new()),
    ];

    if has_add {
        airs.push(LoquelAir::Add(AddAir::new()));
    }
    if has_addi {
        airs.push(LoquelAir::Addi(AddiAir::new()));
    }
    if has_xori {
        airs.push(LoquelAir::Xori(XoriAir::new()));
    }
    if has_and {
        airs.push(LoquelAir::AndInstr(AndInstrAir::new()));
        airs.push(LoquelAir::And(AndAir::new()));
    }
    if has_sll {
        airs.push(LoquelAir::Sll(SllAir::new()));
        airs.push(LoquelAir::ByteSll(ByteShiftLeftAir::new()));
    }

    AllTraces {
        airs,
        boundaries,
        decode,
        memory,
        program: program_trace,
        bytes,
        xor,
        u32_lt,
        timestamp_lt,
        bytes_lt,
        add: add_trace,
        addi: addi_trace,
        xori: xori_trace,
        and_instr: and_instr_trace,
        and_prim: and_prim_trace,
        sll: sll_trace,
        byte_sll: byte_sll_prim_trace,
    }
}

/// Low-level prove: commit to `traces`, run FRI, return a batch proof.
///
/// Callers are responsible for ensuring `airs` and `traces` are parallel and
/// ordered consistently (use `AllTraces::into_vecs` to guarantee this).
fn do_prove(
    config: &MyConfig,
    airs: &[LoquelAir],
    traces: &[RowMajorMatrix<Val>],
) -> (BatchProof<MyConfig>, CommonData<MyConfig>) {
    let trace_refs: Vec<&RowMajorMatrix<Val>> = traces.iter().collect();
    let pvs: Vec<Vec<Val>> = vec![vec![]; airs.len()];

    let initial_instances: Vec<StarkInstance<'_, MyConfig, LoquelAir>> = airs
        .iter()
        .zip(trace_refs.iter())
        .map(|(air, trace)| StarkInstance {
            air,
            trace,
            public_values: vec![],
            lookups: vec![],
        })
        .collect();

    println!("Building prover data ({} AIRs)...", airs.len());
    let prover_data = ProverData::from_instances(config, &initial_instances);
    let instances = StarkInstance::new_multiple(airs, &trace_refs, &pvs, &prover_data.common);

    println!("Proving...");
    let proof = prove_batch(config, &instances, &prover_data);
    (proof, prover_data.common)
}

/// Prove a set of traces produced by `generate_traces` (or a mutation thereof).
pub fn prove_traces(all_traces: AllTraces) -> BatchProof<MyConfig> {
    let (airs, traces) = all_traces.into_vecs();
    let config = build_config();
    do_prove(&config, &airs, &traces).0
}

/// Execute `program`, build all traces, and return a batch STARK proof.
pub fn prove(program: &[u8]) -> BatchProof<MyConfig> {
    prove_traces(generate_traces(program))
}

#[cfg(test)]
mod tests;
