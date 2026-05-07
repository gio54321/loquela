use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;
use punctum_vm::{ExecutionStep, Instruction, MemoryOperation};

use super::air::{AddColumns, NUM_ADD_COLS};

struct AddStep {
    pc: u32,
    timestamp: u32,
    rd: u8,
    rs1: u8,
    rs2: u8,
    rs1_value: u32,
    rs2_value: u32,
    old_rd_value: u32,
    rd_new_value: u32,
}

fn extract_add_steps(steps: &[ExecutionStep]) -> Vec<AddStep> {
    steps
        .iter()
        .filter_map(|step| {
            let (rd, rs1, rs2) = match step.instruction {
                Instruction::Add { rd, rs1, rs2 } => (rd, rs1, rs2),
                _ => return None,
            };
            let (timestamp, rs1_value, rs2_value, old_rd_value, rd_new_value) =
                match step.memory_ops.as_slice() {
                    [MemoryOperation::Read {
                        timestamp, value: v1, ..
                    }, MemoryOperation::Read {
                        value: v2, ..
                    }, MemoryOperation::Write {
                        old_value,
                        new_value,
                        ..
                    }] => (*timestamp, *v1, *v2, *old_value, *new_value),
                    _ => return None,
                };
            Some(AddStep {
                pc: step.state.pc,
                timestamp,
                rd,
                rs1,
                rs2,
                rs1_value,
                rs2_value,
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

fn u32_add_carries(x: u32, y: u32) -> [u8; 4] {
    let xb = x.to_le_bytes();
    let yb = y.to_le_bytes();
    let mut carries = [0u8; 4];
    let mut carry = 0u32;
    for i in 0..4 {
        let sum = xb[i] as u32 + yb[i] as u32 + carry;
        carries[i] = (sum >> 8) as u8;
        carry = carries[i] as u32;
    }
    carries
}

fn fill_row<F: PrimeCharacteristicRing>(row: &mut AddColumns<F>, step: &AddStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rd = F::from_u64(step.rd as u64);
    row.rs1 = F::from_u64(step.rs1 as u64);
    row.rs2 = F::from_u64(step.rs2 as u64);

    row.rs1_value = u32_to_limbs(step.rs1_value);
    row.rs2_value = u32_to_limbs(step.rs2_value);
    row.old_rd_value = u32_to_limbs(step.old_rd_value);
    row.rd_new_value = u32_to_limbs(step.rd_new_value);

    let carries = u32_add_carries(step.rs1_value, step.rs2_value);
    row.add_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
        F::from_u64(carries[3] as u64),
    ];

    row.is_dummy = F::ONE;

    row.next_pc = u32_to_limbs(step.pc + 4);
    let carries = pc_plus4_carries(step.pc);
    row.next_pc_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
    ];
}

fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut AddColumns<F>) {
    *row = AddColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rd: F::ZERO,
        rs1: F::ZERO,
        rs2: F::ZERO,
        rs1_value: [F::ZERO; 4],
        rs2_value: [F::ZERO; 4],
        old_rd_value: [F::ZERO; 4],
        rd_new_value: [F::ZERO; 4],
        add_carries: [F::ZERO; 4],
        next_pc: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        next_pc_carries: [F::ZERO; 3],
        is_dummy: F::ZERO,
    };
}

/// Build the ADD execution trace from the VM execution steps.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let add_steps = extract_add_steps(steps);
    assert!(!add_steps.is_empty(), "no ADD steps found in trace");

    let num_rows = add_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_ADD_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<AddColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(add_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(add_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_ADD_COLS)
}
