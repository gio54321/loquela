use loquela_vm::{ExecutionStep, Instruction, MemoryOperation};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

use super::air::{BeqColumns, NUM_BEQ_COLS};

struct BeqStep {
    pc: u32,
    timestamp: u32,
    rs1: u8,
    rs2: u8,
    imm: i32,
    rs1_value: u32,
    rs2_value: u32,
}

fn extract_beq_steps(steps: &[ExecutionStep]) -> Vec<BeqStep> {
    steps
        .iter()
        .filter_map(|step| {
            let (rs1, rs2, imm) = match step.instruction {
                Instruction::Beq { rs1, rs2, imm } => (rs1, rs2, imm),
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
            Some(BeqStep {
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

fn fill_row<F: PrimeCharacteristicRing>(row: &mut BeqColumns<F>, step: &BeqStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rs1 = F::from_u64(step.rs1 as u64);
    row.rs2 = F::from_u64(step.rs2 as u64);

    // B-type immediate encoding reconstruction:
    // imm_top7 = bits 31:25 of the instruction = {imm[12], imm[10:5]}
    // The instruction bits are derived from the immediate value.
    // imm[10:5] = bits 10:5 of imm (6 bits)
    // imm[12] = bit 12 of imm (sign)
    let imm_u = step.imm as u32;
    let imm12 = (imm_u >> 12) & 1;
    let imm10_5 = (imm_u >> 5) & 0x3F;
    let imm4_1 = (imm_u >> 1) & 0xF;
    let imm11 = (imm_u >> 11) & 1;

    // imm_top7: {imm[12], imm[10:5]} = imm12 * 64 + imm10_5
    let imm_top7 = imm10_5 | (imm12 << 6);
    row.imm_top7 = F::from_u64(imm_top7 as u64);

    // imm_lo5: {imm[4:1], imm[11]} encoded as bits in instruction positions 11:7
    // instruction bit 7 = imm[11], instruction bits 11:8 = imm[4:1]
    // When stored as a 5-bit value: bit0..3 = imm[4:1], bit4 = imm[11]
    let imm_lo5 = imm4_1 | (imm11 << 4);
    row.imm_lo5 = F::from_u64(imm_lo5 as u64);

    // Bit decompositions.
    for i in 0..7 {
        row.imm_top7_bits[i] = F::from_u64(((imm_top7 >> i) & 1) as u64);
    }
    for i in 0..5 {
        row.imm_lo5_bits[i] = F::from_u64(((imm_lo5 >> i) & 1) as u64);
    }

    // Sign-extended B-type immediate (already i32 from VM).
    let imm_se = step.imm as u32;
    row.imm_b = u32_to_limbs(imm_se);

    // jmp_target = pc + imm_b.
    let jmp_target = step.pc.wrapping_add(imm_se);
    row.jmp_target = u32_to_limbs(jmp_target);
    let jmp_c = u32_add_carries(step.pc, imm_se);
    row.jmp_carries = [
        F::from_u64(jmp_c[0] as u64),
        F::from_u64(jmp_c[1] as u64),
        F::from_u64(jmp_c[2] as u64),
        F::from_u64(jmp_c[3] as u64),
    ];

    // pc + 4.
    let pc_p4 = step.pc.wrapping_add(4);
    row.pc_plus4 = u32_to_limbs(pc_p4);
    let p4c = pc_plus4_carries(step.pc);
    row.pc_plus4_carries = [
        F::from_u64(p4c[0] as u64),
        F::from_u64(p4c[1] as u64),
        F::from_u64(p4c[2] as u64),
    ];

    // rs1 and rs2 byte values.
    row.rs1_bytes = u32_to_limbs(step.rs1_value);
    row.rs2_bytes = u32_to_limbs(step.rs2_value);

    // diff_bytes = rs1 - rs2 (wrapping).
    let diff = step.rs1_value.wrapping_sub(step.rs2_value);
    row.diff_bytes = u32_to_limbs(diff);
    let borrows = u32_sub_borrow(step.rs1_value, step.rs2_value);
    row.borrow = [
        F::from_u64(borrows[0] as u64),
        F::from_u64(borrows[1] as u64),
        F::from_u64(borrows[2] as u64),
        F::from_u64(borrows[3] as u64),
    ];

    // Zero-check per diff byte.
    let diff_b = diff.to_le_bytes();
    for i in 0..4 {
        if diff_b[i] == 0 {
            row.diff_byte_inv[i] = F::ZERO;
            row.byte_is_zero[i] = F::ONE;
        } else {
            // inverse in Mersenne31: use the field's inverse via Fermat's little theorem
            // We'll store it as a u64 and let the field do the conversion.
            // Actually we can't easily compute field inverse in trace building without
            // access to the field's inverse operation. Use a workaround:
            // For the constraint: byte_is_zero = 1 - diff * diff_inv = 0 when diff != 0
            // We just need diff * diff_inv = 1, so diff_inv = diff^{-1}.
            // We'll compute it as: diff_inv = 1/diff using field arithmetic.
            // Since F: PrimeCharacteristicRing, we need F::inverse or similar.
            // Mersenne31 inverse: x^{-1} = x^{p-2} mod p = x^{2^31 - 3} mod p.
            // For trace building, compute the modular inverse using extended Euclidean.
            let d = diff_b[i] as u64;
            let p = (1u64 << 31) - 1; // Mersenne31
                                      // Extended GCD to find inverse.
            let inv = mod_inverse(d, p);
            row.diff_byte_inv[i] = F::from_u64(inv);
            row.byte_is_zero[i] = F::ZERO;
        }
    }

    // taken = 1 iff rs1 == rs2 (iff diff == 0).
    let taken = step.rs1_value == step.rs2_value;
    row.taken = if taken { F::ONE } else { F::ZERO };

    // next_pc selection.
    let next_pc = if taken { jmp_target } else { pc_p4 };
    row.next_pc = u32_to_limbs(next_pc);

    row.is_dummy = F::ONE;
}

/// Compute modular inverse of a mod p using extended Euclidean algorithm.
/// Returns 0 if a == 0.
fn mod_inverse(a: u64, p: u64) -> u64 {
    if a == 0 {
        return 0;
    }
    // Extended Euclidean algorithm.
    let mut old_r = a as i128;
    let mut r = p as i128;
    let mut old_s: i128 = 1;
    let mut s: i128 = 0;
    while r != 0 {
        let q = old_r / r;
        let tmp_r = r;
        r = old_r - q * r;
        old_r = tmp_r;
        let tmp_s = s;
        s = old_s - q * s;
        old_s = tmp_s;
    }
    // old_r is gcd (should be 1), old_s is the inverse.
    ((old_s % p as i128 + p as i128) % p as i128) as u64
}

fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut BeqColumns<F>) {
    *row = BeqColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rs1: F::ZERO,
        rs2: F::ZERO,
        imm_top7: F::ZERO,
        imm_lo5: F::ZERO,
        imm_top7_bits: [F::ZERO; 7],
        imm_lo5_bits: [F::ZERO; 5],
        imm_b: [F::ZERO; 4],
        jmp_target: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        jmp_carries: [F::ZERO; 4],
        pc_plus4: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        pc_plus4_carries: [F::ZERO; 3],
        rs1_bytes: [F::ZERO; 4],
        rs2_bytes: [F::ZERO; 4],
        diff_bytes: [F::ZERO; 4],
        borrow: [F::ZERO; 4],
        diff_byte_inv: [F::ZERO; 4],
        byte_is_zero: [F::ONE; 4],
        taken: F::ONE, // dummy: 0==0 so taken=1, next_pc = jmp_target = 0+0 = 0
        next_pc: [F::ZERO; 4],
        is_dummy: F::ZERO,
    };
}

pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let beq_steps = extract_beq_steps(steps);
    assert!(!beq_steps.is_empty(), "no BEQ steps found in trace");

    let num_rows = beq_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_BEQ_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<BeqColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(beq_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(beq_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_BEQ_COLS)
}
