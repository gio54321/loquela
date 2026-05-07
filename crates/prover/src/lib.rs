use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_batch_stark::{prove_batch, BatchProof, ProverData, StarkInstance};
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

use punctum_air::boundaries::air::BoundariesAir;
use punctum_air::decode::air::DecodeAir;
use punctum_air::instructions::addi::air::AddiAir;
use punctum_air::instructions::xori::air::XoriAir;
use punctum_air::memory::air::MemoryAir;
use punctum_air::primitives::byte_less_than_lookup::LessThanAir;
use punctum_air::primitives::byte_lookup::BytesAir;
use punctum_air::primitives::timestamp_less_than::TimestampLessThanAir;
use punctum_air::primitives::u32_less_than_lookup::U32LessThanAir;
use punctum_air::primitives::xor_lookup::XorAir;
use punctum_air::program::air::ProgramAir;
use punctum_vm::{MemoryOperation, VM};

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
    Addi(AddiAir),
    Xori(XoriAir),
    Memory(MemoryAir),
    Program(ProgramAir),
    Bytes(BytesAir),
    Xor(XorAir),
    U32Lt(U32LessThanAir),
    TimestampLt(TimestampLessThanAir),
    BytesLt(LessThanAir),
}

impl<F: Field> BaseAir<F> for LoquelAir {
    fn width(&self) -> usize {
        match self {
            LoquelAir::Boundaries(a) => BaseAir::<F>::width(a),
            LoquelAir::Decode(a) => BaseAir::<F>::width(a),
            LoquelAir::Addi(a) => BaseAir::<F>::width(a),
            LoquelAir::Xori(a) => BaseAir::<F>::width(a),
            LoquelAir::Memory(a) => BaseAir::<F>::width(a),
            LoquelAir::Program(a) => BaseAir::<F>::width(a),
            LoquelAir::Bytes(a) => BaseAir::<F>::width(a),
            LoquelAir::Xor(a) => BaseAir::<F>::width(a),
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
            LoquelAir::Addi(a) => a.eval(builder),
            LoquelAir::Xori(a) => a.eval(builder),
            LoquelAir::Memory(a) => a.eval(builder),
            LoquelAir::Program(a) => a.eval(builder),
            LoquelAir::Bytes(a) => a.eval(builder),
            LoquelAir::Xor(a) => a.eval(builder),
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
            LoquelAir::Addi(a) => <AddiAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Xori(a) => <XoriAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Memory(a) => <MemoryAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Program(a) => <ProgramAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Bytes(a) => <BytesAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Xor(a) => <XorAir as LookupAir<F>>::add_lookup_columns(a),
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
            LoquelAir::Addi(a) => <AddiAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Xori(a) => <XoriAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Memory(a) => <MemoryAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Program(a) => <ProgramAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Bytes(a) => <BytesAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Xor(a) => <XorAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::U32Lt(a) => <U32LessThanAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::TimestampLt(a) => <TimestampLessThanAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::BytesLt(a) => <LessThanAir as LookupAir<F>>::get_lookups(a),
        }
    }
}

// ── Multiplicity helpers ───────────────────────────────────────────────────────

/// Count byte occurrences across all u32 values produced by ADDI steps.
/// Returns `mults[v]` = number of times byte value `v` appeared.
fn bytes_multiplicities(vm_ops: &[(u32, u32, Val)]) -> [Val; 256] {
    let mut mults = [Val::ZERO; 256];
    for &(rs1_val, rd_val, _) in vm_ops {
        for byte in rs1_val
            .to_le_bytes()
            .iter()
            .chain(rd_val.to_le_bytes().iter())
        {
            mults[*byte as usize] += Val::ONE;
        }
    }
    mults
}

/// Count XOR triples from XORI steps: row index = x * 256 + y.
fn xor_multiplicities(xori_ops: &[(u32, u32, u32)]) -> Vec<Val> {
    let mut mults = vec![Val::ZERO; 256 * 256];
    for &(x, y, _) in xori_ops {
        // x and y are single bytes (rs1 byte and imm byte for each limb)
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
    // Normalise into (type, address, timestamp, read, write) tuples and sort.
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

    // bytes_lt: row index = y*(y-1)/2 + x where x < y
    let mut bytes_lt_mults_map = std::collections::HashMap::<(u8, u8), u64>::new();

    for i in 0..ops.len().saturating_sub(1) {
        let cur = &ops[i];
        let nxt = &ops[i + 1];

        let same_type = cur.memory_type == nxt.memory_type;
        if !same_type {
            continue;
        }

        let is_addr_equal = cur.address == nxt.address;

        // u32_lt lookup: (cur.address, nxt.address, is_addr_equal) with mult=1
        u32_lt_entries.push((cur.address, nxt.address, Val::ONE));

        // bytes_lt: the first differing byte limb (if addr differs)
        if !is_addr_equal {
            let x_bytes = cur.address.to_le_bytes();
            let y_bytes = nxt.address.to_le_bytes();
            // Find most significant differing byte (byte index 3 down to 0)
            for b in (0..4).rev() {
                if x_bytes[b] != y_bytes[b] {
                    let (lo, hi) = (x_bytes[b].min(y_bytes[b]), x_bytes[b].max(y_bytes[b]));
                    *bytes_lt_mults_map.entry((lo, hi)).or_insert(0) += 1;
                    break;
                }
            }
        }

        // timestamp_lt: only when addresses are equal
        if is_addr_equal {
            timestamp_lt_entries.push((cur.timestamp, nxt.timestamp, Val::ONE));
        }
    }

    // Convert bytes_lt map to LessThanAir multiplicity slice.
    // LessThanAir row index = y*(y-1)/2 + x (y in 0..256, x in 0..y).
    let num_valid = 256 * 255 / 2;
    let mut bytes_lt_mults = vec![Val::ZERO; num_valid];
    for ((x, y), count) in &bytes_lt_mults_map {
        let row = (*y as usize) * (*y as usize - 1) / 2 + (*x as usize);
        bytes_lt_mults[row] = Val::from_u64(*count);
    }

    (u32_lt_entries, timestamp_lt_entries, bytes_lt_mults)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Execute `program`, build all traces, and return a batch STARK proof.
pub fn prove(program: &[u8]) -> BatchProof<MyConfig> {
    // 1. Execute the VM.
    let mut vm = VM::new(program.to_vec());
    vm.run().expect("VM execution failed");

    let steps = &vm.trace;
    println!("VM executed in {} steps", steps.len());
    let all_ops: Vec<MemoryOperation> = vm.get_memory_ops().into_iter().cloned().collect();

    // 2. Classify steps.
    let has_addi = steps.iter().any(|(state, _)| {
        let pc = state.pc as usize;
        if pc + 4 > program.len() {
            return false;
        }
        let w = u32::from_le_bytes(program[pc..pc + 4].try_into().unwrap());
        w & 0x7F == 0b001_0011 && (w >> 12) & 0x7 == 0b000
    });
    let has_xori = steps.iter().any(|(state, _)| {
        let pc = state.pc as usize;
        if pc + 4 > program.len() {
            return false;
        }
        let w = u32::from_le_bytes(program[pc..pc + 4].try_into().unwrap());
        w & 0x7F == 0b001_0011 && (w >> 12) & 0x7 == 0b100
    });

    // 3. Build traces.

    println!("Building AIR instances and traces...");
    // Boundaries: 2 rows.
    let final_state = steps.last().expect("no steps");
    let final_pc_bytes = (final_state.0.pc + 4).to_le_bytes();
    let final_pc: [Val; 4] = final_pc_bytes.map(|b| Val::from_u64(b as u64));
    let final_ts = Val::from_u64(vm.timestamp as u64);
    let boundaries_trace = punctum_air::boundaries::air::build_trace(final_pc, final_ts);

    // Decode.
    let decode_trace = punctum_air::decode::trace::build_trace::<Val>(program, steps);

    // AddiAir / XoriAir.
    let addi_trace = if has_addi {
        Some(punctum_air::instructions::addi::trace::build_trace::<Val>(
            program, steps,
        ))
    } else {
        None
    };
    let xori_trace = if has_xori {
        Some(punctum_air::instructions::xori::trace::build_trace::<Val>(
            program, steps,
        ))
    } else {
        None
    };

    // Memory.
    let memory_trace = punctum_air::memory::trace::build_trace::<Val>(&all_ops);

    // Program.
    let n_decode_steps = steps.len();
    let n_decode_rows = n_decode_steps.next_power_of_two();
    let num_decode_padding = n_decode_rows - n_decode_steps;
    let program_trace =
        punctum_air::program::trace::build_trace::<Val>(program, steps, num_decode_padding);

    // Lookup table multiplicities.
    let (u32_lt_entries, timestamp_lt_entries, bytes_lt_mults) = memory_lookup_entries(&all_ops);

    // Collect ADDI (rs1_val, rd_val) pairs for bytes bus.
    let addi_byte_ops: Vec<(u32, u32, Val)> = steps
        .iter()
        .filter_map(|(state, ops)| {
            let pc = state.pc as usize;
            if pc + 4 > program.len() { return None; }
            let w = u32::from_le_bytes(program[pc..pc+4].try_into().unwrap());
            if w & 0x7F != 0b001_0011 || (w >> 12) & 0x7 != 0b000 { return None; }
            match ops.as_slice() {
                [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }] => {
                    Some((*rs1, *rd, Val::ONE))
                }
                _ => None,
            }
        })
        .collect();
    let bytes_mults = bytes_multiplicities(&addi_byte_ops);
    let bytes_trace = punctum_air::primitives::byte_lookup::build_trace::<Val>(&bytes_mults);

    // Collect XORI (rs1_byte, imm_byte, result_byte) triples per limb for xor bus.
    let mut xori_triples: Vec<(u32, u32, u32)> = Vec::new();
    for (state, ops) in steps.iter() {
        let pc = state.pc as usize;
        if pc + 4 > program.len() {
            continue;
        }
        let w = u32::from_le_bytes(program[pc..pc + 4].try_into().unwrap());
        if w & 0x7F != 0b001_0011 || (w >> 12) & 0x7 != 0b100 {
            continue;
        }
        let imm_raw = ((w >> 20) & 0xFFF) as u16;
        let imm_se = ((imm_raw as i16) << 4 >> 4) as i32 as u32;
        if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }] =
            ops.as_slice()
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
    let xor_trace = punctum_air::primitives::xor_lookup::build_trace::<Val>(&xor_mults);

    let u32_lt_trace =
        punctum_air::primitives::u32_less_than_lookup::build_trace::<Val>(&u32_lt_entries);
    let timestamp_lt_trace =
        punctum_air::primitives::timestamp_less_than::build_trace::<Val>(&timestamp_lt_entries);
    let bytes_lt_trace =
        punctum_air::primitives::byte_less_than_lookup::build_trace::<Val>(&bytes_lt_mults);

    // 4. Assemble AIRs and traces.
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
    let mut traces: Vec<RowMajorMatrix<Val>> = vec![
        boundaries_trace,
        decode_trace,
        memory_trace,
        program_trace,
        bytes_trace,
        xor_trace,
        u32_lt_trace,
        timestamp_lt_trace,
        bytes_lt_trace,
    ];

    if has_addi {
        airs.push(LoquelAir::Addi(AddiAir::new()));
        traces.push(addi_trace.unwrap());
    }
    if has_xori {
        airs.push(LoquelAir::Xori(XoriAir::new()));
        traces.push(xori_trace.unwrap());
    }

    println!("Built {} AIR instances and traces", airs.len());

    // 5. Build prover data (handles preprocessed traces and lookups).
    let config = build_config();

    let trace_refs: Vec<&RowMajorMatrix<Val>> = traces.iter().collect();
    let pvs: Vec<Vec<Val>> = vec![vec![]; airs.len()];

    // Build initial instances (lookups will be filled from common data).
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

    let prover_data = ProverData::from_instances(&config, &initial_instances);

    // Rebuild instances with correct lookups from common data.
    let instances = StarkInstance::new_multiple(&airs, &trace_refs, &pvs, &prover_data.common);

    println!("Prepared prover data with {} instances", instances.len());
    println!("proving...");
    // 6. Prove.
    prove_batch(&config, &instances, &prover_data)
}

#[cfg(test)]
mod tests {
    use super::prove;

    fn encode_addi(rd: u8, rs1: u8, imm: i16) -> [u8; 4] {
        let word =
            ((imm as u32 & 0xFFF) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0b001_0011;
        word.to_le_bytes()
    }

    fn encode_xori(rd: u8, rs1: u8, imm: i16) -> [u8; 4] {
        let word = ((imm as u32 & 0xFFF) << 20)
            | ((rs1 as u32) << 15)
            | (0b100 << 12)
            | ((rd as u32) << 7)
            | 0b001_0011;
        word.to_le_bytes()
    }

    /// Single ADDI: x1 = x0 + 1.  Exercises the bytes and trace buses.
    #[test]
    fn prove_single_addi() {
        let program = encode_addi(1, 0, 1).to_vec();
        prove(&program);
    }

    /// Single XORI: x1 = x0 ^ 0xFF.  Exercises the bytes_xor bus.
    #[test]
    fn prove_single_xori() {
        let program = encode_xori(1, 0, 0xFF).to_vec();
        prove(&program);
    }

    /// Multiple ADDI steps touching different registers; exercises the u32_lt
    /// and timestamp_lt buses through the memory sort.
    #[test]
    fn prove_addi_chain() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 5)); // x1 = 5
        program.extend_from_slice(&encode_addi(2, 1, 3)); // x2 = 8
        program.extend_from_slice(&encode_addi(3, 2, -1)); // x3 = 7
        program.extend_from_slice(&encode_addi(3, 2, -1)); // x3 = 7
        prove(&program);
    }

    /// Mixed ADDI + XORI program, identical to the guest test.s fixture.
    #[test]
    fn prove_mixed_addi_xori() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, -1i16)); // x1 = 0xFFFF_FFFF
        program.extend_from_slice(&encode_addi(2, 0, 0)); // x2 = 0
        program.extend_from_slice(&encode_addi(3, 0, 1)); // x3 = 1
        program.extend_from_slice(&encode_xori(4, 3, -1i16)); // x4 = x3 ^ 0xFFFF_FFFF
        program.extend_from_slice(&encode_xori(5, 2, 0x55)); // x5 = x2 ^ 0x55
        prove(&program);
    }

    /// Overwriting the same register twice; checks that the memory AIR handles
    /// multiple writes at the same address in sorted order.
    #[test]
    fn prove_overwrite_register() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 10)); // x1 = 10
        program.extend_from_slice(&encode_addi(1, 0, 20)); // x1 = 20
        program.extend_from_slice(&encode_addi(1, 1, 5)); // x1 = 25
        prove(&program);
    }
}
