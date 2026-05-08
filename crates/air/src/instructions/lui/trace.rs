use loquela_vm::{ExecutionStep, Instruction, MemoryOperation};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

use super::air::{LuiColumns, NUM_LUI_COLS};

struct LuiStep {
    pc: u32,
    timestamp: u32,
    rd: u8,
    /// Lower 8 bits of the raw 20-bit immediate (bits 19:12 of instruction word).
    imm_low8: u32,
    /// Upper 12 bits of the raw 20-bit immediate (bits 31:20 of instruction word).
    imm_high12: u32,
    /// rd_val = imm_raw << 12
    rd_val: u32,
    old_rd_value: u32,
}

/// Collect all LUI execution steps from the VM trace.
fn extract_lui_steps(steps: &[ExecutionStep]) -> Vec<LuiStep> {
    steps
        .iter()
        .filter_map(|step| {
            let (rd, imm) = match step.instruction {
                Instruction::Lui { rd, imm } => (rd, imm),
                _ => return None,
            };
            let (timestamp, old_rd_value, rd_val) = match step.memory_ops.as_slice() {
                [MemoryOperation::Write {
                    timestamp,
                    old_value,
                    new_value,
                    ..
                }] => (*timestamp, *old_value, *new_value),
                _ => return None,
            };
            // imm in VM is the raw upper-20 bits (bits 31:12 of instruction).
            let imm_raw = imm as u32;
            let imm_low8 = imm_raw & 0xFF;
            let imm_high12 = (imm_raw >> 8) & 0xFFF;
            Some(LuiStep {
                pc: step.state.pc,
                timestamp,
                rd,
                imm_low8,
                imm_high12,
                rd_val,
                old_rd_value,
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

fn fill_row<F: PrimeCharacteristicRing>(row: &mut LuiColumns<F>, step: &LuiStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rd = F::from_u64(step.rd as u64);
    row.imm_low8 = F::from_u64(step.imm_low8 as u64);
    row.imm_high12 = F::from_u64(step.imm_high12 as u64);
    row.rd_val = u32_to_limbs(step.rd_val);
    row.old_rd_value = u32_to_limbs(step.old_rd_value);

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
/// values are zero.
fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut LuiColumns<F>) {
    *row = LuiColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rd: F::ZERO,
        imm_low8: F::ZERO,
        imm_high12: F::ZERO,
        rd_val: [F::ZERO; 4],
        old_rd_value: [F::ZERO; 4],
        next_pc: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        next_pc_carries: [F::ZERO; 3],
        is_dummy: F::ZERO,
    };
}

/// Build the LUI execution trace from the VM execution steps.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let lui_steps = extract_lui_steps(steps);
    assert!(!lui_steps.is_empty(), "no LUI steps found in trace");

    let num_rows = lui_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_LUI_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<LuiColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(lui_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(lui_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_LUI_COLS)
}
