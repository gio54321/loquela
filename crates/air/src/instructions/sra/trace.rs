use loquela_vm::{ExecutionStep, Instruction, MemoryOperation};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

use super::air::{SraColumns, NUM_SRA_COLS};

struct SraStep {
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

fn extract_sra_steps(steps: &[ExecutionStep]) -> Vec<SraStep> {
    steps
        .iter()
        .filter_map(|step| {
            let (rd, rs1, rs2) = match step.instruction {
                Instruction::Sra { rd, rs1, rs2 } => (rd, rs1, rs2),
                _ => return None,
            };
            let (timestamp, rs1_value, rs2_value, old_rd_value, rd_new_value) =
                match step.memory_ops.as_slice() {
                    [MemoryOperation::Read {
                        timestamp,
                        value: v1,
                        ..
                    }, MemoryOperation::Read { value: v2, .. }, MemoryOperation::Write {
                        old_value,
                        new_value,
                        ..
                    }] => (*timestamp, *v1, *v2, *old_value, *new_value),
                    _ => return None,
                };
            Some(SraStep {
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

fn fill_row<F: PrimeCharacteristicRing>(row: &mut SraColumns<F>, step: &SraStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rd = F::from_u64(step.rd as u64);
    row.rs1 = F::from_u64(step.rs1 as u64);
    row.rs2 = F::from_u64(step.rs2 as u64);

    row.rs1_bytes = u32_to_limbs(step.rs1_value);
    row.rs2_value = u32_to_limbs(step.rs2_value);
    row.old_rd_value = u32_to_limbs(step.old_rd_value);
    row.rd_bytes = u32_to_limbs(step.rd_new_value);

    // Compute shamt and its decomposition.
    let shamt = step.rs2_value & 0x1F;
    let bit_shamt = shamt % 8;
    let byte_shamt = shamt / 8;
    let rs2_shamt_high = (step.rs2_value & 0xFF) >> 5;

    row.bit_shamt = F::from_u64(bit_shamt as u64);
    row.byte_shamt = F::from_u64(byte_shamt as u64);
    row.rs2_shamt_high = F::from_u64(rs2_shamt_high as u64);

    // One-hot byte_shamt selectors.
    row.is_bs0 = F::from_bool(byte_shamt == 0);
    row.is_bs1 = F::from_bool(byte_shamt == 1);
    row.is_bs2 = F::from_bool(byte_shamt == 2);
    row.is_bs3 = F::from_bool(byte_shamt == 3);

    // Compute per-byte shift results (right shift).
    let rs1_b = step.rs1_value.to_le_bytes();
    for i in 0..4 {
        let byte_val = rs1_b[i] as u32;
        let shifted = byte_val >> bit_shamt;
        let carry = if bit_shamt == 0 {
            0u32
        } else {
            (byte_val << (8 - bit_shamt)) & 0xFF
        };
        row.shifted_bytes[i] = F::from_u64(shifted as u64);
        row.carry_bytes[i] = F::from_u64(carry as u64);
    }

    // Sign bit: MSB of rs1_bytes[3] (bit 31 of rs1).
    let sign_bit = (step.rs1_value >> 31) & 1;
    row.sign_bit = F::from_u64(sign_bit as u64);
    row.fill_byte = F::from_u64((sign_bit * 255) as u64);
    row.rs1_byte3_low7 = F::from_u64((rs1_b[3] as u32 & 0x7F) as u64);

    // 0xFF >> bit_shamt and its carry; the carry is the top-bits mask used
    // for sign extension within the high byte (see eval).
    let srl_ff_shifted = 0xFFu32 >> bit_shamt;
    let srl_ff_carry = if bit_shamt == 0 {
        0u32
    } else {
        (0xFFu32 << (8 - bit_shamt)) & 0xFF
    };
    row.srl_ff_shifted = F::from_u64(srl_ff_shifted as u64);
    row.srl_ff_carry = F::from_u64(srl_ff_carry as u64);

    row.is_dummy = F::ONE;

    row.next_pc = u32_to_limbs(step.pc + 4);
    let carries = pc_plus4_carries(step.pc);
    row.next_pc_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
    ];
}

fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut SraColumns<F>) {
    *row = SraColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rd: F::ZERO,
        rs1: F::ZERO,
        rs2: F::ZERO,
        rs1_bytes: [F::ZERO; 4],
        rs2_value: [F::ZERO; 4],
        old_rd_value: [F::ZERO; 4],
        rd_bytes: [F::ZERO; 4],
        bit_shamt: F::ZERO,
        byte_shamt: F::ZERO,
        rs2_shamt_high: F::ZERO,
        is_bs0: F::ONE, // byte_shamt == 0 for dummy rows
        is_bs1: F::ZERO,
        is_bs2: F::ZERO,
        is_bs3: F::ZERO,
        shifted_bytes: [F::ZERO; 4],
        carry_bytes: [F::ZERO; 4],
        sign_bit: F::ZERO,
        fill_byte: F::ZERO,
        rs1_byte3_low7: F::ZERO,
        srl_ff_shifted: F::ZERO,
        srl_ff_carry: F::ZERO,
        next_pc: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        next_pc_carries: [F::ZERO; 3],
        is_dummy: F::ZERO,
    };
}

/// Build the SRA execution trace from the VM execution steps.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let sra_steps = extract_sra_steps(steps);
    assert!(!sra_steps.is_empty(), "no SRA steps found in trace");

    let num_rows = sra_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_SRA_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<SraColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(sra_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(sra_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_SRA_COLS)
}
