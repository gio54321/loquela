use loquela_vm::{ExecutionStep, Instruction, MemoryOperation};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

use super::air::{BgeuColumns, NUM_BGEU_COLS};

struct BgeuStep {
    pc: u32,
    timestamp: u32,
    rs1: u8,
    rs2: u8,
    imm: i32,
    rs1_value: u32,
    rs2_value: u32,
}

fn extract_steps(steps: &[ExecutionStep]) -> Vec<BgeuStep> {
    steps
        .iter()
        .filter_map(|step| {
            let (rs1, rs2, imm) = match step.instruction {
                Instruction::Bgeu { rs1, rs2, imm } => (rs1, rs2, imm),
                _ => return None,
            };
            let (timestamp, rs1_value, rs2_value) = match step.memory_ops.as_slice() {
                [MemoryOperation::Read {
                    timestamp,
                    value: v1,
                    ..
                }, MemoryOperation::Read { value: v2, .. }] => (*timestamp, *v1, *v2),
                _ => return None,
            };
            Some(BgeuStep {
                pc: step.state.pc,
                timestamp,
                rs1,
                rs2,
                imm,
                rs1_value,
                rs2_value,
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

fn u32_sub_borrow(x: u32, y: u32) -> [u8; 4] {
    let xb = x.to_le_bytes();
    let yb = y.to_le_bytes();
    let mut borrows = [0u8; 4];
    let mut borrow = 0i32;
    for i in 0..4 {
        let diff = xb[i] as i32 - yb[i] as i32 - borrow;
        borrows[i] = if diff < 0 { 1 } else { 0 };
        borrow = borrows[i] as i32;
    }
    borrows
}

fn fill_row<F: PrimeCharacteristicRing>(row: &mut BgeuColumns<F>, step: &BgeuStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rs1 = F::from_u64(step.rs1 as u64);
    row.rs2 = F::from_u64(step.rs2 as u64);

    let imm_u = step.imm as u32;
    let imm12 = (imm_u >> 12) & 1;
    let imm10_5 = (imm_u >> 5) & 0x3F;
    let imm4_1 = (imm_u >> 1) & 0xF;
    let imm11 = (imm_u >> 11) & 1;

    let imm_top7 = imm10_5 | (imm12 << 6);
    row.imm_top7 = F::from_u64(imm_top7 as u64);
    let imm_lo5 = imm4_1 | (imm11 << 4);
    row.imm_lo5 = F::from_u64(imm_lo5 as u64);

    for i in 0..7 {
        row.imm_top7_bits[i] = F::from_u64(((imm_top7 >> i) & 1) as u64);
    }
    for i in 0..5 {
        row.imm_lo5_bits[i] = F::from_u64(((imm_lo5 >> i) & 1) as u64);
    }

    let imm_se = step.imm as u32;
    row.imm_b = u32_to_limbs(imm_se);

    let jmp_target = step.pc.wrapping_add(imm_se);
    row.jmp_target = u32_to_limbs(jmp_target);
    let jmp_c = u32_add_carries(step.pc, imm_se);
    row.jmp_carries = [
        F::from_u64(jmp_c[0] as u64),
        F::from_u64(jmp_c[1] as u64),
        F::from_u64(jmp_c[2] as u64),
        F::from_u64(jmp_c[3] as u64),
    ];

    let pc_p4 = step.pc.wrapping_add(4);
    row.pc_plus4 = u32_to_limbs(pc_p4);
    let p4c = pc_plus4_carries(step.pc);
    row.pc_plus4_carries = [
        F::from_u64(p4c[0] as u64),
        F::from_u64(p4c[1] as u64),
        F::from_u64(p4c[2] as u64),
    ];

    row.rs1_bytes = u32_to_limbs(step.rs1_value);
    row.rs2_bytes = u32_to_limbs(step.rs2_value);

    let diff = step.rs1_value.wrapping_sub(step.rs2_value);
    row.diff_bytes = u32_to_limbs(diff);
    let borrows = u32_sub_borrow(step.rs1_value, step.rs2_value);
    row.borrow = [
        F::from_u64(borrows[0] as u64),
        F::from_u64(borrows[1] as u64),
        F::from_u64(borrows[2] as u64),
        F::from_u64(borrows[3] as u64),
    ];

    // taken = 1 iff rs1 >= rs2 (unsigned) = 1 - borrow[3].
    let taken = step.rs1_value >= step.rs2_value;
    row.taken = if taken { F::ONE } else { F::ZERO };

    let next_pc = if taken { jmp_target } else { pc_p4 };
    row.next_pc = u32_to_limbs(next_pc);

    row.is_dummy = F::ONE;
}

fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut BgeuColumns<F>) {
    *row = BgeuColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rs1: F::ZERO,
        rs2: F::ZERO,
        imm_top7: F::ZERO,
        imm_lo5: F::ZERO,
        imm_top7_bits: [F::ZERO; 7],
        imm_lo5_bits: [F::ZERO; 5],
        imm_b: [F::ZERO; 4],
        jmp_target: [F::ZERO; 4],
        jmp_carries: [F::ZERO; 4],
        pc_plus4: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        pc_plus4_carries: [F::ZERO; 3],
        rs1_bytes: [F::ZERO; 4],
        rs2_bytes: [F::ZERO; 4],
        diff_bytes: [F::ZERO; 4],
        borrow: [F::ZERO; 4],
        // taken=1: 0 >= 0 is true, borrow[3]=0, taken=1-0=1
        taken: F::ONE,
        next_pc: [F::ZERO; 4],
        is_dummy: F::ZERO,
    };
}

pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let branch_steps = extract_steps(steps);
    assert!(!branch_steps.is_empty(), "no BGEU steps found in trace");

    let num_rows = branch_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_BGEU_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<BgeuColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(branch_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(branch_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_BGEU_COLS)
}
