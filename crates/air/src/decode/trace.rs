use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use punctum_vm::{MemoryOperation, VMState};
use super::air::{DecodeColumns, Instruction, NUM_DECODE_COLS};

fn u32_to_limbs<F: PrimeCharacteristicRing>(v: u32) -> [F; 4] {
    let b = v.to_le_bytes();
    [
        F::from_u64(b[0] as u64),
        F::from_u64(b[1] as u64),
        F::from_u64(b[2] as u64),
        F::from_u64(b[3] as u64),
    ]
}

fn fill_row<F: PrimeCharacteristicRing>(row: &mut DecodeColumns<F>, pc: u32, program: &[u8]) {
    let off = pc as usize;
    let word = u32::from_le_bytes(program[off..off + 4].try_into().unwrap());

    row.pc = u32_to_limbs(pc);

    for i in 0..4 {
        let byte = (word >> (i * 8)) & 0xFF;
        row.instruction[i] = F::from_u64(byte as u64);
        for j in 0..8 {
            row.decompositions[i][j] = F::from_u64(((byte >> j) & 1) as u64);
        }
    }

    let rd = ((word >> 7) & 0x1F) as u8;
    let rs1 = ((word >> 15) & 0x1F) as u8;
    let imm = ((word >> 20) & 0xFFF) as u16;

    row.rd = F::from_u64(rd as u64);
    row.rs1 = F::from_u64(rs1 as u64);
    row.imm = F::from_u64(imm as u64);

    let is_addi = word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b000;
    let is_xori = word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b100;

    row.instr_type = Instruction {
        is_addi: F::from_bool(is_addi),
        is_xori: F::from_bool(is_xori),
    };
    // instr_type_packed: 0 for ADDI, 1 for XORI (matches the eval constraint)
    row.instr_type_packed = if is_xori { F::ONE } else { F::ZERO };
    // Each row is consumed once by its instruction AIR.
    row.mult = F::ONE;
}

/// Build the decode trace from the VM execution steps.
///
/// One row per step. The `mult` column is always 1 (each decode is consumed
/// once by the corresponding instruction AIR).
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    program: &[u8],
    steps: &[(VMState, Vec<MemoryOperation>)],
) -> RowMajorMatrix<F> {
    let num_steps = steps.len();
    assert!(num_steps > 0, "decode trace requires at least one step");

    let num_rows = num_steps.next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_DECODE_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<DecodeColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, (state, _)) in rows.iter_mut().zip(steps.iter()) {
        fill_row(row, state.pc, program);
    }
    // Padding rows: reuse pc=0 with mult=0 so the decode bus stays balanced.
    for row in rows.iter_mut().skip(num_steps) {
        fill_row(row, 0, program);
        row.mult = F::ZERO;
    }

    RowMajorMatrix::new(values, NUM_DECODE_COLS)
}
