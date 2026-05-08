use loquela_vm::{ExecutionStep, Instruction, MemoryOperation};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

use super::air::{LhColumns, NUM_LH_COLS};

struct LhStep {
    pc: u32,
    timestamp: u32,
    rd: u8,
    rs1: u8,
    imm: i32,
    rs1_value: u32,
    addr: u32,
    loaded_val: u32,
    old_rd_value: u32,
    rd_val: u32,
}

fn extract_lh_steps(steps: &[ExecutionStep]) -> Vec<LhStep> {
    steps
        .iter()
        .filter_map(|step| {
            let (rd, rs1, imm) = match step.instruction {
                Instruction::Lh { rd, rs1, imm } => (rd, rs1, imm),
                _ => return None,
            };
            let (timestamp, rs1_value, loaded_val, old_rd_value) =
                match step.memory_ops.as_slice() {
                    [MemoryOperation::Read {
                        timestamp,
                        value: rs1_val,
                        ..
                    }, MemoryOperation::Read {
                        value: ram_val, ..
                    }, MemoryOperation::Write {
                        old_value, ..
                    }] => (*timestamp, *rs1_val, *ram_val, *old_value),
                    _ => return None,
                };
            let addr = rs1_value.wrapping_add(imm as u32);
            let half = (loaded_val & 0xFFFF) as u16;
            let rd_val = (half as i16 as i32) as u32;
            Some(LhStep {
                pc: step.state.pc,
                timestamp,
                rd,
                rs1,
                imm,
                rs1_value,
                addr,
                loaded_val,
                old_rd_value,
                rd_val,
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

fn fill_row<F: PrimeCharacteristicRing>(row: &mut LhColumns<F>, step: &LhStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rd = F::from_u64(step.rd as u64);
    row.rs1 = F::from_u64(step.rs1 as u64);

    let imm_u12 = (step.imm as u32) & 0xFFF;
    row.imm = F::from_u64(imm_u12 as u64);

    let high_nibble = (imm_u12 >> 8) & 0xF;
    row.imm_high_bits = [
        F::from_u64(((high_nibble >> 0) & 1) as u64),
        F::from_u64(((high_nibble >> 1) & 1) as u64),
        F::from_u64(((high_nibble >> 2) & 1) as u64),
        F::from_u64(((high_nibble >> 3) & 1) as u64),
    ];

    let imm_se = step.imm as u32;
    row.imm_se_bytes = u32_to_limbs(imm_se);

    row.rs1_value = u32_to_limbs(step.rs1_value);
    row.addr_bytes = u32_to_limbs(step.addr);

    let carries = u32_add_carries(step.rs1_value, imm_se);
    row.addr_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
        F::from_u64(carries[3] as u64),
    ];

    row.loaded_val = u32_to_limbs(step.loaded_val);
    row.old_rd_value = u32_to_limbs(step.old_rd_value);
    row.rd_val = u32_to_limbs(step.rd_val);

    // Sign bit extraction from loaded_val[1] (byte 1 = bits 15:8 of halfword).
    let byte1 = ((step.loaded_val >> 8) & 0xFF) as u8;
    let sign_bit = (byte1 >> 7) & 1;
    let byte1_low7 = byte1 & 0x7F;
    row.sign_bit = F::from_u64(sign_bit as u64);
    row.byte1_low7 = F::from_u64(byte1_low7 as u64);

    row.next_pc = u32_to_limbs(step.pc + 4);
    let carries = pc_plus4_carries(step.pc);
    row.next_pc_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
    ];

    row.is_dummy = F::ONE;
}

fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut LhColumns<F>) {
    *row = LhColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rd: F::ZERO,
        rs1: F::ZERO,
        imm: F::ZERO,
        imm_high_bits: [F::ZERO; 4],
        imm_se_bytes: [F::ZERO; 4],
        rs1_value: [F::ZERO; 4],
        addr_bytes: [F::ZERO; 4],
        addr_carries: [F::ZERO; 4],
        loaded_val: [F::ZERO; 4],
        old_rd_value: [F::ZERO; 4],
        rd_val: [F::ZERO; 4],
        sign_bit: F::ZERO,
        byte1_low7: F::ZERO,
        next_pc: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        next_pc_carries: [F::ZERO; 3],
        is_dummy: F::ZERO,
    };
}

pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let lh_steps = extract_lh_steps(steps);
    assert!(!lh_steps.is_empty(), "no LH steps found in trace");

    let num_rows = lh_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_LH_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<LhColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(lh_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(lh_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_LH_COLS)
}
