use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;
use punctum_vm::{MemoryOperation, VMState};

use super::air::{XoriColumns, NUM_XORI_COLS};

struct XoriStep {
    pc: u32,
    timestamp: u32,
    rd: u8,
    rs1: u8,
    /// Unsigned 12-bit immediate (bits 20–31 of the instruction word).
    imm: u16,
    rs1_value: u32,
    old_rd_value: u32,
    rd_new_value: u32,
}

/// Decode the instruction at `pc` in `program`, returning `(rd, rs1, imm_unsigned_12)`
/// if it is a XORI, or `None` otherwise.
fn decode_xori(program: &[u8], pc: u32) -> Option<(u8, u8, u16)> {
    let off = pc as usize;
    if off + 4 > program.len() {
        return None;
    }
    let word = u32::from_le_bytes(program[off..off + 4].try_into().unwrap());
    if word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b100 {
        let rd = ((word >> 7) & 0x1F) as u8;
        let rs1 = ((word >> 15) & 0x1F) as u8;
        let imm = ((word >> 20) & 0xFFF) as u16;
        Some((rd, rs1, imm))
    } else {
        None
    }
}

/// Collect all XORI execution steps from the VM trace.
fn extract_xori_steps(program: &[u8], steps: &[(VMState, Vec<MemoryOperation>)]) -> Vec<XoriStep> {
    steps
        .iter()
        .filter_map(|(state, ops)| {
            let (rd, rs1, imm) = decode_xori(program, state.pc)?;
            // XORI emits exactly [Read(rs1), Write(rd)].
            let (timestamp, rs1_value, old_rd_value, rd_new_value) = match ops.as_slice() {
                [MemoryOperation::Read {
                    timestamp, value, ..
                }, MemoryOperation::Write {
                    old_value,
                    new_value,
                    ..
                }] => (*timestamp, *value, *old_value, *new_value),
                _ => return None,
            };
            Some(XoriStep {
                pc: state.pc,
                timestamp,
                rd,
                rs1,
                imm,
                rs1_value,
                old_rd_value,
                rd_new_value,
            })
        })
        .collect()
}

fn u32_to_limbs<F: PrimeCharacteristicRing>(v: u32) -> [F; 4] {
    let b = v.to_le_bytes();
    [
        F::from_u64(b[0] as u64),
        F::from_u64(b[1] as u64),
        F::from_u64(b[2] as u64),
        F::from_u64(b[3] as u64),
    ]
}

/// Carry bits produced when computing `pc + 4` byte by byte (matching `u32_plus_four`).
/// Returns `[carry_out_byte0, carry_out_byte1, carry_out_byte2]`.
fn pc_plus4_carries(pc: u32) -> [u8; 3] {
    let b = pc.to_le_bytes();
    let s0 = b[0] as u32 + 4;
    let c0 = (s0 >> 8) as u8;
    let s1 = b[1] as u32 + c0 as u32;
    let c1 = (s1 >> 8) as u8;
    let s2 = b[2] as u32 + c1 as u32;
    let c2 = (s2 >> 8) as u8;
    [c0, c1, c2]
}

fn fill_row<F: PrimeCharacteristicRing>(row: &mut XoriColumns<F>, step: &XoriStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rd = F::from_u64(step.rd as u64);
    row.rs1 = F::from_u64(step.rs1 as u64);
    row.imm = F::from_u64(step.imm as u64);

    // Bit decomposition of bits 8–11 of imm (the high nibble).
    let high_nibble = (step.imm >> 8) & 0xF;
    row.imm_high_bits = [
        F::from_u64(((high_nibble >> 0) & 1) as u64),
        F::from_u64(((high_nibble >> 1) & 1) as u64),
        F::from_u64(((high_nibble >> 2) & 1) as u64),
        F::from_u64(((high_nibble >> 3) & 1) as u64),
    ];

    // Sign-extend the 12-bit immediate to 32 bits.
    let imm_se = ((step.imm as i16) << 4 >> 4) as i32 as u32;
    row.imm_se_bytes = u32_to_limbs(imm_se);

    row.rs1_value = u32_to_limbs(step.rs1_value);
    row.old_rd_value = u32_to_limbs(step.old_rd_value);
    row.rd_new_value = u32_to_limbs(step.rd_new_value);

    row.is_dummy = F::ONE;

    row.next_pc = u32_to_limbs(step.pc + 4);
    let carries = pc_plus4_carries(step.pc);
    row.next_pc_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
    ];
}

/// Fill a padding row that satisfies all `eval` constraints when all semantic
/// values are zero. `u32_plus_four(0, 4, [0,0,0])` holds since `4 + 0·256 = 0 + 4`.
fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut XoriColumns<F>) {
    *row = XoriColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rd: F::ZERO,
        rs1: F::ZERO,
        imm: F::ZERO,
        imm_high_bits: [F::ZERO; 4],
        imm_se_bytes: [F::ZERO; 4],
        rs1_value: [F::ZERO; 4],
        old_rd_value: [F::ZERO; 4],
        rd_new_value: [F::ZERO; 4],
        next_pc: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        next_pc_carries: [F::ZERO; 3],
        is_dummy: F::ZERO,
    };
}

/// Build the XORI execution trace from the raw VM trace.
///
/// Filters all XORI steps from `steps`, fills one row per step, and pads
/// to the next power of two.  The input `program` is needed to re-decode
/// each instruction and verify it is a XORI.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    program: &[u8],
    steps: &[(VMState, Vec<MemoryOperation>)],
) -> DenseMatrix<F> {
    let xori_steps = extract_xori_steps(program, steps);
    assert!(!xori_steps.is_empty(), "no XORI steps found in trace");

    let num_rows = xori_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_XORI_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<XoriColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(xori_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(xori_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_XORI_COLS)
}
