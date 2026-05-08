use loquela_vm::{ExecutionStep, Instruction, MemoryOperation};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

use super::air::{SltuColumns, NUM_SLTU_COLS};

struct SltuStep {
    pc: u32,
    timestamp: u32,
    rd: u8,
    rs1: u8,
    rs2: u8,
    rs1_value: u32,
    rs2_value: u32,
    old_rd_value: u32,
}

fn extract_sltu_steps(steps: &[ExecutionStep]) -> Vec<SltuStep> {
    steps
        .iter()
        .filter_map(|step| {
            let (rd, rs1, rs2) = match step.instruction {
                Instruction::Sltu { rd, rs1, rs2 } => (rd, rs1, rs2),
                _ => return None,
            };
            let (timestamp, rs1_value, rs2_value, old_rd_value) =
                match step.memory_ops.as_slice() {
                    [MemoryOperation::Read {
                        timestamp,
                        value: v1,
                        ..
                    }, MemoryOperation::Read { value: v2, .. }, MemoryOperation::Write {
                        old_value,
                        ..
                    }] => (*timestamp, *v1, *v2, *old_value),
                    _ => return None,
                };
            Some(SltuStep {
                pc: step.state.pc,
                timestamp,
                rd,
                rs1,
                rs2,
                rs1_value,
                rs2_value,
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

/// Compute borrow chain for rs1 - rs2.
/// borrow[i] = 1 iff byte i requires borrowing.
/// Final borrow[3] = 1 iff rs1 < rs2 (unsigned).
fn u32_sub_borrows(rs1: u32, rs2: u32) -> ([u8; 4], [u8; 4]) {
    let a = rs1.to_le_bytes();
    let b = rs2.to_le_bytes();
    let mut borrows = [0u8; 4];
    let mut diff = [0u8; 4];
    let mut borrow_in = 0u32;
    for i in 0..4 {
        let sub = a[i] as i32 - b[i] as i32 - borrow_in as i32;
        if sub < 0 {
            diff[i] = (sub + 256) as u8;
            borrows[i] = 1;
        } else {
            diff[i] = sub as u8;
            borrows[i] = 0;
        }
        borrow_in = borrows[i] as u32;
    }
    (diff, borrows)
}

fn fill_row<F: PrimeCharacteristicRing>(row: &mut SltuColumns<F>, step: &SltuStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rd = F::from_u64(step.rd as u64);
    row.rs1 = F::from_u64(step.rs1 as u64);
    row.rs2 = F::from_u64(step.rs2 as u64);

    row.rs1_bytes = u32_to_limbs(step.rs1_value);
    row.rs2_bytes = u32_to_limbs(step.rs2_value);
    row.old_rd_value = u32_to_limbs(step.old_rd_value);

    let (diff, borrows) = u32_sub_borrows(step.rs1_value, step.rs2_value);
    row.diff_bytes = diff.map(|b| F::from_u64(b as u64));
    row.borrow = borrows.map(|b| F::from_u64(b as u64));

    let lt_result = if step.rs1_value < step.rs2_value {
        F::ONE
    } else {
        F::ZERO
    };
    row.lt_result = lt_result;

    row.is_dummy = F::ONE;

    row.next_pc = u32_to_limbs(step.pc + 4);
    let carries = pc_plus4_carries(step.pc);
    row.next_pc_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
    ];
}

fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut SltuColumns<F>) {
    *row = SltuColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rd: F::ZERO,
        rs1: F::ZERO,
        rs2: F::ZERO,
        rs1_bytes: [F::ZERO; 4],
        rs2_bytes: [F::ZERO; 4],
        old_rd_value: [F::ZERO; 4],
        diff_bytes: [F::ZERO; 4],
        borrow: [F::ZERO; 4],
        lt_result: F::ZERO,
        next_pc: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        next_pc_carries: [F::ZERO; 3],
        is_dummy: F::ZERO,
    };
}

/// Build the SLTU execution trace from the VM execution steps.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let sltu_steps = extract_sltu_steps(steps);
    assert!(!sltu_steps.is_empty(), "no SLTU steps found in trace");

    let num_rows = sltu_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_SLTU_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<SltuColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(sltu_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(sltu_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_SLTU_COLS)
}
