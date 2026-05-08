use loquela_vm::{ExecutionStep, Instruction, MemoryOperation};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::DenseMatrix;

use super::air::{JalColumns, NUM_JAL_COLS};

struct JalStep {
    pc: u32,
    timestamp: u32,
    rd: u8,
    /// Upper 12 bits of instruction word (bits 31:20): {imm[20], imm[10:1], imm[11]}.
    imm_high12: u32,
    /// Lower 8 bits of the J-type immediate (bits 19:12 of instruction word): imm[19:12].
    imm_lo8: u32,
    /// The sign-extended J-type immediate.
    imm_j: i32,
    /// Return address = pc + 4.
    rd_val: u32,
    old_rd_value: u32,
}

fn extract_jal_steps(steps: &[ExecutionStep]) -> Vec<JalStep> {
    let mut out = Vec::new();
    let mut running_ts: u32 = 0;
    for step in steps {
        let entry_ts = running_ts;
        // Each step consumes one timestamp per memory op, plus one extra for
        // JAL with rd=0 (no memory op but the VM still increments timestamp).
        let consumed = if step.memory_ops.is_empty()
            && matches!(step.instruction, Instruction::Jal { rd: 0, .. })
        {
            1
        } else {
            step.memory_ops.len() as u32
        };

        if let Instruction::Jal { rd, imm } = step.instruction {
            // For rd != 0 the VM emits one Write op with the canonical
            // (timestamp, old, new) triple. For rd == 0 there is no Write
            // op (x0 is silently skipped), so derive rd_val from pc and use
            // 0 for old_rd_value (x0's invariant).
            let (timestamp, old_rd_value, rd_val) = match step.memory_ops.as_slice() {
                [MemoryOperation::Write {
                    timestamp,
                    old_value,
                    new_value,
                    ..
                }] => (*timestamp, *old_value, *new_value),
                [] if rd == 0 => (entry_ts, 0, step.state.pc.wrapping_add(4)),
                _ => {
                    running_ts += consumed;
                    continue;
                }
            };
            // Reconstruct the raw instruction word fields from the immediate.
            // J-type immediate: {imm[20], imm[10:1], imm[11], imm[19:12]}.
            // imm_high12 = bits 31:20 of instruction = {imm[20], imm[10:1], imm[11]}.
            // imm_lo8    = bits 19:12 of instruction = imm[19:12].
            let imm_u = imm as u32;
            let imm20 = (imm_u >> 20) & 1;
            let imm10_1 = (imm_u >> 1) & 0x3FF;
            let imm11 = (imm_u >> 11) & 1;
            let imm19_12 = (imm_u >> 12) & 0xFF;
            // imm_high12 = {imm[20], imm[10:1], imm[11]} = imm10_1 | (imm11 << 10) | (imm20 << 11)
            let imm_high12 = imm10_1 | (imm11 << 10) | (imm20 << 11);
            let imm_lo8 = imm19_12;
            out.push(JalStep {
                pc: step.state.pc,
                timestamp,
                rd,
                imm_high12,
                imm_lo8,
                imm_j: imm,
                rd_val,
                old_rd_value,
            });
        }
        running_ts += consumed;
    }
    out
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

fn fill_row<F: Field>(row: &mut JalColumns<F>, step: &JalStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rd = F::from_u64(step.rd as u64);
    row.imm_high12 = F::from_u64(step.imm_high12 as u64);
    row.imm_lo8 = F::from_u64(step.imm_lo8 as u64);

    // Bit-decompose imm_high12 (12 bits).
    for i in 0..12 {
        row.imm_high12_bits[i] = F::from_u64(((step.imm_high12 >> i) & 1) as u64);
    }

    // Bit-decompose imm_lo8 (8 bits).
    for i in 0..8 {
        row.imm_lo8_bits[i] = F::from_u64(((step.imm_lo8 >> i) & 1) as u64);
    }

    // imm_j as 4 byte limbs (sign-extended 32-bit value).
    let imm_j_u32 = step.imm_j as u32;
    row.imm_j = u32_to_limbs(imm_j_u32);

    // next_pc = pc + imm_j (wrapping).
    let next_pc = step.pc.wrapping_add(imm_j_u32);
    row.next_pc = u32_to_limbs(next_pc);

    // Carry bits for pc + imm_j.
    let carries = u32_add_carries(step.pc, imm_j_u32);
    row.jmp_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
        F::from_u64(carries[3] as u64),
    ];

    // rd_val = pc + 4.
    row.rd_val = u32_to_limbs(step.rd_val);
    row.old_rd_value = u32_to_limbs(step.old_rd_value);

    let carries = pc_plus4_carries(step.pc);
    row.rd_val_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
    ];

    // rd_is_zero indicator and rd_inv witness.
    if step.rd == 0 {
        row.rd_is_zero = F::ONE;
        row.rd_inv = F::ZERO; // unused but constraint accepts any value when rd=0
    } else {
        row.rd_is_zero = F::ZERO;
        row.rd_inv = F::from_u64(step.rd as u64).inverse();
    }

    row.is_dummy = F::ONE;
}

fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut JalColumns<F>) {
    *row = JalColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rd: F::ZERO,
        imm_high12: F::ZERO,
        imm_lo8: F::ZERO,
        imm_high12_bits: [F::ZERO; 12],
        imm_lo8_bits: [F::ZERO; 8],
        imm_j: [F::ZERO; 4],
        jmp_carries: [F::ZERO; 4],
        rd_val: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        rd_val_carries: [F::ZERO; 3],
        old_rd_value: [F::ZERO; 4],
        next_pc: [F::ZERO; 4],
        // Padding rows have rd=0, so rd_is_zero must be 1 to satisfy the
        // is-zero indicator constraints.
        rd_is_zero: F::ONE,
        rd_inv: F::ZERO,
        is_dummy: F::ZERO,
    };
}

pub fn build_trace<F: Field + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let jal_steps = extract_jal_steps(steps);
    assert!(!jal_steps.is_empty(), "no JAL steps found in trace");

    let num_rows = jal_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_JAL_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<JalColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(jal_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(jal_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_JAL_COLS)
}
